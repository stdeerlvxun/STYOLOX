"""Bounded problem-only comparisons: train-group CV, validation cutoff, fixed-test audit."""
from __future__ import annotations
import argparse
import hashlib
import io
import json
from pathlib import Path
import time
import numpy as np
from PIL import Image, ImageOps
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.transforms import functional as TF
from torchvision.models import convnext_tiny
from vision_classifier import (WORK, TRAINING_POLICY, RULES, read_json, write_json,
                               validate_training_dataset, image_tensor, load_backbone, features)

ENCODER_META = WORK / 'models' / 'torch-hub' / 'convnext-tiny.json'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def crop_tensors(row):
    raw = Path(row['full_path']).read_bytes()
    if hashlib.sha256(raw).hexdigest() != row['sha256']:
        raise ValueError('Source image changed after dataset freeze')
    with Image.open(io.BytesIO(raw)) as original:
        image = ImageOps.exif_transpose(original).convert('RGB')
    # Full image is always retained; no automatic crop is treated as a box annotation.
    full = ImageOps.pad(image, (224, 224), color=(124, 116, 104))
    width, height = image.size
    side = max(1, int(min(width, height) * .75))
    origins = [(0, 0), (width-side, 0), (0, height-side), (width-side, height-side),
               ((width-side)//2, (height-side)//2)]
    crops = [full] + [image.crop((x, y, x+side, y+side)).resize((224, 224), Image.Resampling.BILINEAR) for x, y in origins]
    return torch.stack([TF.normalize(TF.to_tensor(crop), [.485, .456, .406], [.229, .224, .225]) for crop in crops])


def classification_encoder(device):
    meta = read_json(ENCODER_META)
    if digest(meta['file']) != meta['sha256']:
        raise ValueError('ConvNeXt pretrained weights changed')
    model = convnext_tiny(weights=None)
    model.load_state_dict(torch.load(meta['file'], map_location='cpu', weights_only=True), strict=True)
    model.classifier[-1] = nn.Identity()
    return model.eval().to(device), meta


def pack_views(vectors):
    global_vector = vectors[0]
    local = vectors[1:]
    parts = [global_vector, local.mean(0), local.max(0).values]
    multi = torch.cat([F.normalize(part, dim=0) for part in parts])
    return global_vector, multi


def extract_classification_features(source_run, output):
    dataset = read_json(source_run / 'dataset.json')
    validate_training_dataset(dataset)
    target = output / 'convnext-features.pt'
    manifest_hash = digest(source_run / 'dataset.json')
    if target.exists():
        cache = torch.load(target, weights_only=True)
        if cache['manifest_hash'] != manifest_hash:
            raise ValueError('Mismatched dataset feature cache')
        return cache
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.set_num_threads(4)
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats()
    model, meta = classification_encoder(device)
    start, globals_, multis = time.perf_counter(), [], []
    with torch.inference_mode():
        for index, row in enumerate(dataset['records']):
            vectors = model(crop_tensors(row).to(device)).cpu()
            global_vector, multi = pack_views(vectors)
            globals_.append(global_vector); multis.append(multi)
            if index % 20 == 0 or index + 1 == len(dataset['records']):
                progress = {'stage':'classification_features', 'done':index+1, 'total':len(dataset['records']),
                            'seconds':round(time.perf_counter()-start,1)}
                write_json(output / 'progress.json', progress)
                print(json.dumps(progress), flush=True)
    result = {'global':torch.stack(globals_), 'multi':torch.stack(multis), 'manifest_hash':manifest_hash,
              'encoder_sha256':meta['sha256'], 'encoder_file':meta['file'],
              'seconds':time.perf_counter()-start,
              'peak_allocated_mib':torch.cuda.max_memory_allocated()/2**20 if device.type=='cuda' else 0}
    torch.save(result, target)
    del model
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    return result


def summary(truth, scores, threshold):
    truth = np.asarray(truth, dtype=bool)
    guess = np.asarray(scores) >= threshold
    tp, fp, fn = int((truth & guess).sum()), int((~truth & guess).sum()), int((truth & ~guess).sum())
    precision, recall = tp/max(1,tp+fp), tp/max(1,tp+fn)
    covered = np.all(~truth | guess,axis=1)
    return {'images':len(truth), 'tp':tp, 'fp':fp, 'fn':fn, 'precision':precision, 'recall':recall,
            'f1':2*precision*recall/max(1e-12,precision+recall),
            'f2':5*precision*recall/max(1e-12,4*precision+recall),
            'all_issues_covered_images':int(covered.sum()), 'missed_images':int((~covered).sum()),
            'candidates_per_image':float(guess.sum()/max(1,len(truth))),
            'exact_images':int(np.all(truth==guess,axis=1).sum()), 'exact_rate':float(np.all(truth==guess,axis=1).mean())}


def recall_selection_key(metrics):
    precision,recall=metrics['precision'],metrics['recall']
    return (5*precision*recall/max(1e-12,4*precision+recall),recall,precision)


def select_threshold(truth, scores, beta=2):
    choices = []
    for threshold in [.01,.02,.03,.05,.075,.1,.125,.15,.175,.2,.25,.3,.35,.4,.45,.5,.55,.6,.65,.7,.75,.8,.85,.9]:
        metrics = summary(truth, scores, threshold)
        choices.append({'threshold':threshold, **metrics})
    if beta==1:
        return max(choices, key=lambda r:(r['f1'],r['precision'],r['threshold']))
    if beta!=2:raise ValueError('Supported review objectives are F1 or recall-first F2')
    return max(choices, key=lambda r:(*recall_selection_key(r),r['threshold']))


def corrected_logits(logits, positives, training_size):
    weight = ((training_size - positives) / positives.clamp_min(1)).clamp(1,10)
    return (logits - weight.log()).sigmoid()


def fit_ridge(raw, truth, regularization):
    center = raw.mean(0)
    vectors = F.normalize(raw-center, dim=1)
    prior = truth.mean(0)
    kernel = vectors @ vectors.T
    alpha = torch.linalg.solve(kernel + regularization*torch.eye(len(raw)), truth-prior)
    return {'center':center, 'training_vectors':vectors, 'alpha':alpha, 'prior':prior,
            'regularization':regularization}


def ridge_scores(model, raw):
    vectors = F.normalize(raw-model['center'], dim=1)
    return (vectors @ model['training_vectors'].T @ model['alpha'] + model['prior']).clamp(0,1)


def micro_ap(truth, scores):
    labels = truth.flatten().numpy().astype(bool)
    order = np.argsort(-scores.flatten().numpy(), kind='stable')
    labels = labels[order]
    if not labels.sum():
        return 0.
    precision = np.cumsum(labels)/np.arange(1,len(labels)+1)
    return float(precision[labels].mean())


def choose_regularization(raw, truth, groups):
    # Only training rows enter group CV. Rare categories remain as output columns.
    unique = sorted(set(groups))
    rng = np.random.default_rng(42); rng.shuffle(unique)
    assignment = {group:i%3 for i,group in enumerate(unique)}
    folds = torch.tensor([assignment[group] for group in groups])
    trials = []
    for regularization in [.1,1.,10.]:
        out = torch.zeros_like(truth)
        for fold in range(3):
            held = folds==fold
            model = fit_ridge(raw[~held], truth[~held], regularization)
            out[held] = ridge_scores(model, raw[held])
        trials.append({'regularization':regularization, 'train_group_oof_micro_ap':micro_ap(truth,out)})
    chosen = max(trials, key=lambda r:(r['train_group_oof_micro_ap'], r['regularization']))
    return chosen['regularization'], trials


def per_class(truth, scores, names, threshold):
    records = []
    for i, name in enumerate(names):
        value = summary(truth[:,i:i+1], scores[:,i:i+1], threshold)
        value.update({'class':name, 'support':int(truth[:,i].sum())})
        if not value['support']:
            value['recall'] = None
        records.append(value)
    return records


def compare(source_run, output):
    dataset = read_json(source_run / 'dataset.json'); validate_training_dataset(dataset)
    output.mkdir(parents=True, exist_ok=True)
    if (output / 'selection.json').exists():
        raise ValueError('Comparison is frozen; use a new output directory')
    torch.set_num_threads(4)
    rows, names = dataset['records'], dataset['summary']['problem_classes']
    masks = {part:torch.tensor([r['split']==part for r in rows]) for part in ['train','validation','test']}
    truth = torch.tensor([[float(name in r['defects']) for name in names] for r in rows])
    train_mask, val_mask = masks['train'], masks['validation']
    yolo = torch.load(source_run/'features.pt', map_location='cpu', weights_only=True)
    if yolo['manifest_hash'] != digest(source_run/'dataset.json'):
        raise ValueError('YOLO feature cache does not match the frozen dataset')
    conv = extract_classification_features(source_run, output)
    baseline = torch.load(source_run/'candidate-b'/'classifier.pt', map_location='cpu', weights_only=True)
    x = (yolo['features']-baseline['feature_mean']) / baseline['feature_std']
    head = nn.Linear(x.shape[1],len(names)); head.load_state_dict(baseline['problem_state_dict']);head.eval()
    with torch.inference_mode():
        logits = head(x)
    candidates = []
    for corrected in [False, True]:
        score = corrected_logits(logits, truth[train_mask].sum(0), int(train_mask.sum())) if corrected else logits.sigmoid()
        selection = select_threshold(truth[val_mask].numpy(), score[val_mask].numpy())
        candidates.append({'name':'yolox_loss_corrected' if corrected else 'yolox_threshold_only',
            'selection':selection, 'scores':score, 'payload':{'classifier_kind':'corrected_yolox_head',
            'head_checkpoint':baseline, 'loss_weight_correction':corrected,
            'training_positive_counts':truth[train_mask].sum(0), 'training_size':int(train_mask.sum())}})
    group_ids = [r['group'] for r in rows if r['split']=='train']
    cv_reports = {}
    for name, raw in [('yolox_ridge',yolo['features']), ('convnext_global_ridge',conv['global']), ('convnext_local_ridge',conv['multi'])]:
        regularization, cv_reports[name] = choose_regularization(raw[train_mask], truth[train_mask], group_ids)
        model = fit_ridge(raw[train_mask], truth[train_mask], regularization)
        score = ridge_scores(model,raw)
        selection = select_threshold(truth[val_mask].numpy(), score[val_mask].numpy())
        candidates.append({'name':name, 'selection':selection, 'scores':score,
            'payload':{'classifier_kind':'centered_kernel_ridge', 'feature_kind':name, 'ridge':model}})
    best = max(candidates, key=lambda r:recall_selection_key(r['selection']))
    selection = {'training_policy':TRAINING_POLICY, 'source_manifest_hash':digest(source_run/'dataset.json'),
                 'criterion':'validation_micro_f2_then_recall_then_precision', 'test_used_for_selection':False,
                 'all_classes_retained':names, 'cv_policy':'regularization chosen by 3-fold grouped TRAINING-only micro AP',
                 'train_cv':cv_reports, 'candidates':[{'name':c['name'],**c['selection']} for c in candidates],
                 'selected':best['name'], 'threshold':best['selection']['threshold']}
    # Persist selection before accessing test labels/scores for reporting.
    write_json(output/'selection.json',selection)
    checkpoint = {'schema_version':3, 'training_policy':TRAINING_POLICY, 'problems':names,
                  'threshold':best['selection']['threshold'], 'rules':RULES, 'source_run':str(source_run),
                  'source_manifest_hash':selection['source_manifest_hash'], 'encoder_sha256':conv['encoder_sha256'],
                  'yolox_encoder_sha256':yolo['base_sha256'],
                  **best['payload']}
    torch.save(checkpoint,output/'classifier.pt')
    report = {'status':'experimental_completed', 'production_ready':False, 'training_policy':TRAINING_POLICY,
              'selected':best['name'], 'threshold':checkpoint['threshold'], 'selection':selection,
              'feature_extraction':{k:conv[k] for k in ['seconds','peak_allocated_mib','encoder_sha256']},
              'counts':dataset['summary'], 'evaluations':{}}
    for part, mask in masks.items():
        target, score = truth[mask].numpy(), best['scores'][mask].numpy()
        baseline_score = logits[mask].sigmoid().numpy()
        majority = np.zeros_like(score); majority[:,int(truth[train_mask].sum(0).argmax())]=1
        report['evaluations'][part] = {'selected':summary(target,score,checkpoint['threshold']),
            'previous':summary(target,baseline_score,.5), 'majority_only_baseline':summary(target,majority,.5),
            'per_class':per_class(target,score,names,checkpoint['threshold'])}
    predictions = []
    for i,row in enumerate(rows):
        if row['split']=='test':
            scores = best['scores'][i].tolist()
            predictions.append({'image_id':row['image_id'],'relative_path':row['relative_path'],
                'true_problems':row['defects'], 'candidate_problems':[n for n,s in zip(names,scores) if s>=checkpoint['threshold']],
                'scores':dict(zip(names,scores))})
    write_json(output/'metrics.json',report); write_json(output/'test-predictions.json',predictions)
    print(json.dumps({'selected':report['selected'],'test':report['evaluations']['test'],'model':str(output/'classifier.pt')},ensure_ascii=False),flush=True)


def predict(output, image_path, result_path):
    checkpoint = torch.load(output/'classifier.pt',map_location='cpu',weights_only=True)
    if checkpoint.get('training_policy') != TRAINING_POLICY:
        raise ValueError('Superseded training policy')
    row = {'image_id':'external','full_path':str(image_path),'sha256':digest(image_path)}
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu');torch.set_num_threads(4)
    kind = checkpoint.get('feature_kind','')
    with torch.inference_mode():
        if checkpoint['classifier_kind']=='convnext_mil_finetuned':
            from finetune_classifier import predict_finetuned
            score=predict_finetuned(checkpoint,row,device)
        elif kind.startswith('convnext'):
            encoder,meta = classification_encoder(device)
            if meta['sha256']!=checkpoint['encoder_sha256']:
                raise ValueError('Classification encoder changed')
            global_vector,multi = pack_views(encoder(crop_tensors(row).to(device)).cpu())
            raw = (multi if kind=='convnext_local_ridge' else global_vector).unsqueeze(0)
        else:
            encoder,base_hash=load_backbone(device)
            raw=features(encoder,image_tensor(row,640).unsqueeze(0).to(device)).cpu()
            expected = checkpoint.get('head_checkpoint',{}).get('base_sha256') or checkpoint['yolox_encoder_sha256']
            if expected and base_hash!=expected:
                raise ValueError('YOLOX encoder changed')
        if checkpoint['classifier_kind']=='convnext_mil_finetuned':
            pass
        elif checkpoint['classifier_kind']=='centered_kernel_ridge':
            score=ridge_scores(checkpoint['ridge'],raw)[0].tolist()
        else:
            head_meta=checkpoint['head_checkpoint'];x=(raw-head_meta['feature_mean'])/head_meta['feature_std']
            head=nn.Linear(x.shape[1],len(checkpoint['problems']));head.load_state_dict(head_meta['problem_state_dict'])
            logits=head(x)
            score=(corrected_logits(logits,checkpoint['training_positive_counts'],checkpoint['training_size']) if checkpoint['loss_weight_correction'] else logits.sigmoid())[0].tolist()
    names=checkpoint['problems'];candidates=[n for n,s in zip(names,score) if s>=checkpoint['threshold']]
    result={'image':str(image_path),'sha256':row['sha256'],'training_policy':TRAINING_POLICY,
            'candidate_problems':candidates,'problem_scores':dict(zip(names,score)),'threshold':checkpoint['threshold'],
            'requires_human_review':True,'probabilities_calibrated':False,
            'objective':'find_problem_points_for_human_review'}
    write_json(result_path,result);print(json.dumps(result,ensure_ascii=False),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=['compare','predict'])
    parser.add_argument('--source-run',type=Path);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--image',type=Path);parser.add_argument('--result',type=Path)
    args=parser.parse_args()
    if args.command=='compare':
        compare(args.source_run,args.output)
    else:
        predict(args.output,args.image,args.result or args.output/'prediction.json')
