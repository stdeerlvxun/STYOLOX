"""One bounded ConvNeXt last-stage MIL fine-tune on the fixed problem-only split."""
from pathlib import Path
import argparse
import copy
import json
import time
import torch
from torch import nn
from torch.nn import functional as F
from improve_classifier import (classification_encoder, crop_tensors, digest, summary, per_class,
                                micro_ap, select_threshold, recall_selection_key)
from vision_classifier import read_json, write_json, validate_training_dataset, TRAINING_POLICY


class MILTail(nn.Module):
    def __init__(self, tail, norm, classes):
        super().__init__();self.tail=tail;self.norm=norm;self.dropout=nn.Dropout(.25)
        self.head=nn.Linear(2304,classes)

    def forward(self, prefix):
        batch,views,channels,height,width=prefix.shape
        x=self.tail(prefix.reshape(batch*views,channels,height,width))
        v=self.norm(F.adaptive_avg_pool2d(x,1)).flatten(1).reshape(batch,views,768)
        pooled=torch.cat([F.normalize(part,dim=1) for part in
                         [v[:,0],v[:,1:].mean(1),v[:,1:].max(1).values]],dim=1)
        return self.head(self.dropout(pooled))


def recall_first_loss(logits, target, positive_weights):
    """Weighted BCE plus soft micro F2; FN's penalty coefficient is four times FP's.

    Scores use the same class-weight correction as inference. No candidate
    cutoff enters this differentiable learning objective.
    """
    bce=F.binary_cross_entropy_with_logits(logits,target,pos_weight=positive_weights)
    scores=(logits-positive_weights.log()).sigmoid()
    tp=(scores*target).sum();fn=((1-scores)*target).sum();fp=(scores*(1-target)).sum()
    soft_f2=(5*tp+1e-6)/(5*tp+4*fn+fp+1e-6)
    return bce+1-soft_f2


def previous_validation(previous):
    selection=read_json(previous/'selection.json')
    if 'validation' in selection:return selection['validation']
    return next(c for c in selection['candidates'] if c['name']==selection['selected'])


def prefix_cache(source,output,dataset,device):
    target=output/'prefix.pt';manifest=digest(source/'dataset.json')
    if target.exists():
        cache=torch.load(target,map_location='cpu',weights_only=True)
        if cache['manifest_hash']!=manifest:raise ValueError('Prefix cache mismatch')
        return cache
    encoder,meta=classification_encoder(device);prefix=encoder.features[:7].eval()
    values=[];started=time.perf_counter()
    with torch.inference_mode():
        for index,row in enumerate(dataset['records']):
            values.append(prefix(crop_tensors(row).to(device)).cpu().half())
            if index%40==0 or index+1==len(dataset['records']):
                print(json.dumps({'stage':'frozen_prefix','done':index+1,'total':len(dataset['records']),
                                  'seconds':round(time.perf_counter()-started,1)}),flush=True)
    cache={'prefix':torch.stack(values),'manifest_hash':manifest,'encoder_sha256':meta['sha256']}
    torch.save(cache,target);del encoder,prefix
    if device.type=='cuda':torch.cuda.empty_cache()
    return cache


def train(source,output,previous,epochs=12,recall_first=False):
    dataset=read_json(source/'dataset.json');validate_training_dataset(dataset)
    output.mkdir(parents=True,exist_ok=True)
    if (output/'family-selection.json').exists():raise ValueError('Fine-tuning run is frozen')
    config={'max_epochs':epochs,'early_stop_patience':4,'tail_lr':1e-5,'head_lr':3e-4,'batch_bags':2,
            'views_per_image':6,'dropout':.25,'weight_decay':.05,'seed':42,'precision':'float32',
            'prefix_storage':'float16; cast back to float32 for training and inference',
            'trainable':'ConvNeXt last stage, normalization and multilabel head only',
            'epoch_selection':'validation_micro_AP','family_selection':'validation_micro_F2_then_recall_then_precision',
            'test_used_for_selection':False,'training_policy':TRAINING_POLICY}
    if recall_first:
        config.update({'early_stop_patience':8,'minimum_epochs':12,'tail_lr':3e-5,'head_lr':1e-3,
            'batch_bags':8,'loss':'positive_weighted_BCE_plus_one_minus_soft_micro_F2',
            'positive_weight':'sqrt(train_negative/train_positive), clipped to [1,8]',
            'soft_F2_false_negative_coefficient':4,'soft_F2_false_positive_coefficient':1,
            'epoch_selection':'validation_micro_F2_then_recall_then_precision',
            'objective':'find_problem_points_for_human_review','responsibility_output':False})
    write_json(output/'planned-config.json',config)
    torch.set_num_threads(4);torch.manual_seed(42)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cache=prefix_cache(source,output,dataset,device)
    encoder,meta=classification_encoder(device)
    if cache['encoder_sha256']!=meta['sha256']:raise ValueError('Prefix encoder mismatch')
    names=dataset['summary']['problem_classes'];rows=dataset['records']
    truth=torch.tensor([[float(name in r['defects']) for name in names] for r in rows])
    indices={p:torch.tensor([i for i,r in enumerate(rows) if r['split']==p]) for p in ['train','validation','test']}
    positives=truth[indices['train']].sum(0);negative=len(indices['train'])-positives
    weights=(negative/positives.clamp_min(1)).sqrt().clamp(1,8 if recall_first else 4)
    model=MILTail(encoder.features[7],encoder.classifier[0],len(names)).to(device);del encoder
    nn.init.normal_(model.head.weight,std=.002)
    with torch.no_grad():model.head.bias.copy_(torch.log(positives*weights/negative.clamp_min(1)).to(device))
    initial=model.tail[0].block[0].weight.detach().cpu().clone()
    initial_head=model.head.weight.detach().cpu().clone()
    optimizer=torch.optim.AdamW([{'params':model.tail.parameters(),'lr':config['tail_lr']},
        {'params':model.norm.parameters(),'lr':config['tail_lr']},
        {'params':model.head.parameters(),'lr':config['head_lr']}],weight_decay=.05)
    criterion=nn.BCEWithLogitsLoss(pos_weight=weights.to(device))
    weights_device=weights.to(device)
    weight_log=weights.log().to(device);prefix=cache['prefix'];start=time.perf_counter()
    if device.type=='cuda':torch.cuda.reset_peak_memory_stats()

    def infer(part):
        model.eval();outputs=[]
        with torch.inference_mode():
            for ids in indices[part].split(config['batch_bags']):
                logits=model(prefix[ids].to(device).float())-weight_log
                outputs.append(logits.sigmoid().cpu())
        return torch.cat(outputs)

    initial_scores=infer('validation')
    initial_validation=select_threshold(truth[indices['validation']].numpy(),initial_scores.numpy())
    write_json(output/'untrained-validation.json',initial_validation)
    best_key=None;best_epoch=0;stale=0;history=[];optimizer_steps=0
    for epoch in range(1,epochs+1):
        model.train();total=0
        order=indices['train'][torch.randperm(len(indices['train']))]
        for ids in order.split(config['batch_bags']):
            optimizer.zero_grad(set_to_none=True)
            logits=model(prefix[ids].to(device).float());target=truth[ids].to(device)
            loss=recall_first_loss(logits,target,weights_device) if recall_first else criterion(logits,target)
            if not torch.isfinite(loss):raise ValueError('Non-finite fine-tuning loss')
            loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step();optimizer_steps+=1
            total+=float(loss.detach())*len(ids)
        scores=infer('validation');ap=micro_ap(truth[indices['validation']],scores)
        validation=select_threshold(truth[indices['validation']].numpy(),scores.numpy())
        key=recall_selection_key(validation) if recall_first else (ap,)
        info={'epoch':epoch,'train_loss':total/len(order),'validation_micro_ap':ap,
              'validation':validation,'seconds':round(time.perf_counter()-start,1)};history.append(info)
        write_json(output/'progress.json',info);print(json.dumps(info),flush=True)
        if best_key is None or key>best_key:
            best_key=key;best_epoch=epoch;stale=0
            best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        else:stale+=1
        if epoch>=config.get('minimum_epochs',0) and stale>=config['early_stop_patience']:break
    model.load_state_dict(best_state)
    validation_scores=infer('validation');choice=select_threshold(truth[indices['validation']].numpy(),validation_scores.numpy())
    prev=previous_validation(previous)
    chosen='convnext_mil_finetuned' if recall_selection_key(choice)>recall_selection_key(prev) else 'previous_model'
    family={'criterion':'validation_micro_f2_then_recall_then_precision','test_used_for_selection':False,'previous_validation':prev,
            'finetuned_validation':choice,'selected':chosen,'selected_run':str(output if chosen=='convnext_mil_finetuned' else previous),
            'best_epoch':best_epoch,'epochs_run':epoch}
    write_json(output/'family-selection.json',family)
    checkpoint={'schema_version':3,'training_policy':TRAINING_POLICY,'classifier_kind':'convnext_mil_finetuned',
        'problems':names,'threshold':choice['threshold'],'model_state_dict':best_state,
        'encoder_sha256':cache['encoder_sha256'],'positive_weights':weights,'config':config,
        'source_manifest_hash':cache['manifest_hash'],'source_run':str(source),'best_epoch':best_epoch,
        'weights_retrained':True,'responsibility_output':False,'objective':'find_problem_points_for_human_review'}
    torch.save(checkpoint,output/'classifier.pt')
    report={'status':'experimental_completed','production_ready':False,'training_policy':TRAINING_POLICY,'config':config,
        'family_selection':family,'threshold':choice['threshold'],'best_epoch':best_epoch,'epochs_run':epoch,
        'tail_weight_change_l2':float((model.tail[0].block[0].weight.detach().cpu()-initial).norm()),
        'head_weight_change_l2':float((model.head.weight.detach().cpu()-initial_head).norm()),
        'optimizer_steps':optimizer_steps,'weights_retrained':True,'responsibility_output':False,
        'untrained_validation':initial_validation,
        'training_seconds':time.perf_counter()-start,
        'peak_allocated_mib':torch.cuda.max_memory_allocated()/2**20 if device.type=='cuda' else 0,'evaluations':{}}
    # Selection is frozen BEFORE test evaluation. Even a rejected trained model
    # is reported honestly; an old-model fallback is never described as retraining.
    predictions=[]
    for part in indices:
        scores=infer(part);target=truth[indices[part]].numpy()
        report['evaluations'][part]={'selected':summary(target,scores.numpy(),choice['threshold']),
            'at_previous_threshold_0_125':summary(target,scores.numpy(),.125),
            'at_original_threshold_0_3':summary(target,scores.numpy(),.3),
            'micro_ap':micro_ap(truth[indices[part]],scores),
            'per_class':per_class(target,scores.numpy(),names,choice['threshold'])}
        if part=='test':
            for index,score in zip(indices[part].tolist(),scores.tolist()):
                row=rows[index];predictions.append({'image_id':row['image_id'],'relative_path':row['relative_path'],
                    'true_problems':row['defects'],'candidate_problems':[n for n,s in zip(names,score) if s>=choice['threshold']],
                    'scores':dict(zip(names,score))})
    write_json(output/'test-predictions.json',predictions)
    write_json(output/'metrics.json',report);write_json(output/'training-history.json',history)
    print(json.dumps({'family_selection':family,'test':report['evaluations'].get('test'),
                      'training_seconds':report['training_seconds']},ensure_ascii=False),flush=True)


def predict_finetuned(checkpoint,row,device):
    encoder,meta=classification_encoder(device)
    if meta['sha256']!=checkpoint['encoder_sha256']:raise ValueError('Encoder changed')
    model=MILTail(encoder.features[7],encoder.classifier[0],len(checkpoint['problems'])).to(device)
    model.load_state_dict(checkpoint['model_state_dict'],strict=True);model.eval()
    with torch.inference_mode():
        prefix=encoder.features[:7](crop_tensors(row).to(device)).half().float()
        logits=model(prefix.unsqueeze(0))-checkpoint['positive_weights'].to(device).log()
        return logits.sigmoid()[0].cpu().tolist()


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--source-run',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--previous',type=Path,required=True)
    parser.add_argument('--epochs',type=int,default=12)
    parser.add_argument('--recall-first',action='store_true');args=parser.parse_args()
    train(args.source_run,args.output,args.previous,args.epochs,args.recall_first)
