"""Bounded recall-cost retraining with an unchanged candidate cutoff.

Keep the original visual features. Refit all category coefficients using
positive-weighted least squares; choose weights on validation, then test once.
"""
from pathlib import Path
import argparse
import json
import time
import torch
from torch.nn import functional as F
from improve_classifier import digest, ridge_scores, summary, per_class, recall_selection_key
from vision_classifier import TRAINING_POLICY, validate_training_dataset, read_json, write_json


def fit_recall_ridge(raw, truth, regularization, positive_cost):
    if positive_cost<=0 or regularization<=0:raise ValueError('Costs must be positive')
    center=raw.mean(0);vectors=F.normalize(raw-center,dim=1);kernel=vectors@vectors.T
    weights=1+(positive_cost-1)*truth
    prior=(weights*truth).sum(0)/weights.sum(0)
    # Minimize (y-b-Ka)' W (y-b-Ka) + lambda a' K a.
    # This changes image-dependent coefficients, not only the class intercept.
    alpha=torch.stack([torch.linalg.solve(kernel+regularization*torch.diag(1/weights[:,c]),
        truth[:,c]-prior[c]) for c in range(truth.shape[1])],dim=1)
    return {'center':center,'training_vectors':vectors,'alpha':alpha,'prior':prior,
            'regularization':regularization,'positive_cost':positive_cost}


def choose_candidate(candidates, max_candidates=4.):
    allowed=[c for c in candidates if c['validation']['candidates_per_image']<=max_candidates]
    if not allowed:raise ValueError('No candidate satisfies the predefined review budget')
    return max(allowed,key=lambda c:(c['validation']['recall'],
        *recall_selection_key(c['validation']),-c['positive_cost']))


def train(source,previous,output):
    dataset=read_json(source/'dataset.json');validate_training_dataset(dataset)
    output.mkdir(parents=True,exist_ok=True)
    if (output/'selection.json').exists():raise ValueError('Run is frozen')
    old=torch.load(previous/'classifier.pt',map_location='cpu',weights_only=True)
    if old['training_policy']!=TRAINING_POLICY or old['feature_kind']!='convnext_local_ridge':
        raise ValueError('Expected the fixed local-view ridge baseline')
    names=dataset['summary']['problem_classes'];rows=dataset['records']
    if old['problems']!=names:raise ValueError('Category order changed')
    config={'training_policy':TRAINING_POLICY,'objective':'find_problem_points_for_human_review',
        'loss':'positive_weighted_squared_error_plus_kernel_L2',
        'positive_cost_candidates':[2.,4.,8.], 'regularization':old['ridge']['regularization'],
        'threshold':old['threshold'],'threshold_changed':False,
        'selection':'validation_recall_then_F2_then_precision_under_average_4_candidates',
        'max_validation_candidates_per_image':4.,'test_used_for_selection':False,
        'all_classes_retained':names,'responsibility_output':False,
        'initial_tail_trial_rejected':'validation recall decreased; test is not used to select this trial'}
    write_json(output/'planned-config.json',config)
    cache=torch.load(source/'improvement-v1'/'convnext-features.pt',map_location='cpu',weights_only=True)
    if cache['manifest_hash']!=digest(source/'dataset.json'):raise ValueError('Feature cache mismatch')
    if cache['encoder_sha256']!=old['encoder_sha256']:raise ValueError('Encoder mismatch')
    raw=cache['multi']
    truth=torch.tensor([[float(n in r['defects']) for n in names] for r in rows])
    indices={p:torch.tensor([i for i,r in enumerate(rows) if r['split']==p]) for p in ['train','validation','test']}
    torch.set_num_threads(4);started=time.perf_counter();candidates=[];models={}
    for cost in config['positive_cost_candidates']:
        model=fit_recall_ridge(raw[indices['train']],truth[indices['train']],config['regularization'],cost)
        models[cost]=model
        val=summary(truth[indices['validation']],ridge_scores(model,raw[indices['validation']]),old['threshold'])
        candidates.append({'positive_cost':cost,'validation':val})
    chosen=choose_candidate(candidates,config['max_validation_candidates_per_image'])
    previous_val=summary(truth[indices['validation']],ridge_scores(old['ridge'],raw[indices['validation']]),old['threshold'])
    accepted=(chosen['validation']['recall']>=previous_val['recall'] and
              recall_selection_key(chosen['validation'])>=recall_selection_key(previous_val))
    selection={**config,'candidates':candidates,'positive_cost':chosen['positive_cost'],
        'validation':chosen['validation'],'previous_validation':previous_val,
        'accepted_by_validation':accepted,'source_manifest_hash':cache['manifest_hash']}
    # Save the decision before any computation or inspection of test predictions.
    write_json(output/'selection.json',selection)
    model=models[chosen['positive_cost']]
    checkpoint={**old,'ridge':model,'threshold':config['threshold'],'weights_retrained':True,
        'objective':config['objective'],'responsibility_output':False,'training_config':config,
        'source_run':str(source),'source_manifest_hash':cache['manifest_hash']}
    checkpoint.pop('rules',None)
    torch.save(checkpoint,output/'classifier.pt')
    report={'training_policy':TRAINING_POLICY,'weights_retrained':True,'production_ready':False,
        'selection':selection,'training_seconds':time.perf_counter()-started,
        'alpha_change_l2':float((model['alpha']-old['ridge']['alpha']).norm()),
        'prior_change_l2':float((model['prior']-old['ridge']['prior']).norm()),'evaluations':{}}
    predictions=[]
    for part,ids in indices.items():
        scores=ridge_scores(model,raw[ids]);before=ridge_scores(old['ridge'],raw[ids]);target=truth[ids]
        report['evaluations'][part]={'current':summary(target,scores,config['threshold']),
            'previous':summary(target,before,config['threshold']),
            'per_class':per_class(target.numpy(),scores.numpy(),names,config['threshold'])}
        if part=='test':
            for i,new,prev in zip(ids.tolist(),scores.tolist(),before.tolist()):
                row=rows[i];actual=set(row['defects'])
                predicted={n for n,s in zip(names,new) if s>=config['threshold']}
                earlier={n for n,s in zip(names,prev) if s>=config['threshold']}
                entry={'image_id':row['image_id'],'relative_path':row['relative_path'],
                    'true_problems':row['defects'],'candidate_problems':sorted(predicted),
                    'missing_problems':sorted(actual-predicted),'extra_problems':sorted(predicted-actual),
                    'newly_recovered_problems':sorted((actual&predicted)-earlier),
                    'regressed_problems':sorted((actual&earlier)-predicted),'scores':dict(zip(names,new))}
                predictions.append(entry)
    for key in ['newly_recovered_problems','regressed_problems','missing_problems','extra_problems']:
        report[key]=sum(len(p[key]) for p in predictions)
    write_json(output/'test-predictions.json',predictions);write_json(output/'metrics.json',report)
    print(json.dumps({'accepted_by_validation':accepted,'positive_cost':chosen['positive_cost'],
        'validation':chosen['validation'],'test':report['evaluations']['test']['current'],
        'newly_recovered_problems':report['newly_recovered_problems'],
        'regressed_problems':report['regressed_problems'],'alpha_change_l2':report['alpha_change_l2']},ensure_ascii=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--source-run',type=Path,required=True)
    parser.add_argument('--previous',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();train(args.source_run,args.previous,args.output)
