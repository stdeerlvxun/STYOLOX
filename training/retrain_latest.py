"""Bounded latest-label retraining; preserve historical holdouts and compare fairly."""
from collections import Counter,defaultdict
from pathlib import Path
import argparse
import json
import shutil
import time
import torch
from improve_classifier import (classification_encoder,crop_tensors,pack_views,digest,ridge_scores,
    summary,per_class,recall_selection_key)
from recall_classifier import fit_recall_ridge
from vision_classifier import prepare as prepare_problems,read_json,write_json,validate_training_dataset,TRAINING_POLICY,NAS_ROOT
import unrelated_filter

WORK=Path(r'D:\视觉检测工作区')
OLD_PROBLEM=WORK/'runs/20261016-problem-only-120454'
OLD_AUX=WORK/'runs/20261016-unrelated-filter-v1'


def identity_keys(row):
    return {'sha:'+row['sha256'],'case:'+row['case_id']}|{'order:'+k for k in row['order_keys']}


def preserve_holdouts(run,reference,allow_untrained_new_classes=False,quarantine_historical_conflicts=False):
    data=read_json(run/'dataset.json');old=read_json(reference/'dataset.json');lookup=defaultdict(set)
    for row in old['records']:
        for key in identity_keys(row):lookup[key].add(row['split'])
    grouped=defaultdict(list)
    for row in data['records']:grouped[row['group']].append(row)
    conflicts=[]
    for group,rows in grouped.items():
        memberships=set().union(*(lookup[k] for row in rows for k in identity_keys(row)))
        if memberships:
            selected='test' if 'test' in memberships else 'validation' if 'validation' in memberships else 'train'
            if len(memberships)>1:conflicts.append({'group':group,'historical_splits':sorted(memberships)})
            if quarantine_historical_conflicts and 'train' in memberships and len(memberships)>1:selected='quarantine'
            for row in rows:row['split']=selected
    if not quarantine_historical_conflicts and any('train' in c['historical_splits'] for c in conflicts):
        raise ValueError('New order links merge historical training and held-out groups; audit before training')
    counts={p:{'images':sum(r['split']==p for r in data['records']),
        'groups':len({r['group'] for r in data['records'] if r['split']==p})} for p in ['train','validation','test']}
    data['historical_holdout_reference']=str(reference);data['historical_holdout_conflicts']=conflicts
    data['quarantined_images']=sum(r['split']=='quarantine' for r in data['records'])
    if data['training_policy']==TRAINING_POLICY:
        validate_training_dataset(data)
        untrained=[];previous_classes=old.get('summary',{}).get('problem_classes',[])
        for name in data['summary']['problem_classes']:
            if not any(name in r['defects'] and r['split']=='train' for r in data['records']):
                if allow_untrained_new_classes and name not in previous_classes:untrained.append(name)
                else:raise ValueError('No training positive after holdout preservation: '+name)
        data['summary']['untrained_new_classes_due_to_historical_holdouts']=untrained
        data['summary']['splits']=counts
    else:
        for p in counts:
            counts[p]['targets']=dict(Counter(str(r['auxiliary_target']) for r in data['records'] if r['split']==p))
        data['splits']=counts
    write_json(run/'dataset.json',data);return data


def cached_features(run,data,cache_run=OLD_AUX):
    target=run/'features.pt';manifest=digest(run/'dataset.json')
    if target.exists():
        cache=torch.load(target,weights_only=True);assert cache['manifest_hash']==manifest;return cache
    old_ds=read_json(cache_run/'dataset.json');old=torch.load(cache_run/'features.pt',weights_only=True)
    assert old['manifest_hash']==digest(cache_run/'dataset.json')
    lookup={r['sha256']:i for i,r in enumerate(old_ds['records'])}
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu');encoder,meta=classification_encoder(device)
    assert old['encoder_sha256']==meta['sha256'];whole=[];multi=[];new_count=0
    with torch.inference_mode():
        for index,row in enumerate(data['records']):
            if row['sha256'] in lookup:
                i=lookup[row['sha256']];g,m=old['global'][i],old['multi'][i]
            else:g,m=pack_views(encoder(crop_tensors(row).to(device)).cpu());new_count+=1
            whole.append(g);multi.append(m)
            if index%100==0:print(json.dumps({'stage':'feature_progress','run':run.name,'done':index+1,'total':len(data['records']),'newly_extracted':new_count}),flush=True)
    cache={'global':torch.stack(whole),'multi':torch.stack(multi),'encoder_sha256':meta['sha256'],'manifest_hash':manifest}
    torch.save(cache,target)
    print(json.dumps({'stage':'reused_features','images':len(data['records']),'newly_extracted':new_count}),flush=True)
    return cache


def aligned_problem_scores(baseline,features,names):
    original=ridge_scores(baseline['ridge'],features)
    aligned=torch.zeros((len(features),len(names)),dtype=original.dtype)
    for column,name in enumerate(names):
        if name in baseline['problems']:aligned[:,column]=original[:,baseline['problems'].index(name)]
    return aligned


def train_problems(run,data,cache,baseline_run=None):
    baseline_run=baseline_run or OLD_PROBLEM/'recall-trained-v3-fixed-cutoff'
    baseline=torch.load(baseline_run/'classifier.pt',weights_only=True)
    names=data['summary']['problem_classes']
    rows=data['records'];truth=torch.tensor([[float(n in r['defects']) for n in names] for r in rows])
    ids={p:torch.tensor([i for i,r in enumerate(rows) if r['split']==p]) for p in ['train','validation','test']}
    x=cache['multi'];cutoff=baseline['threshold']
    previous_val=summary(truth[ids['validation']],aligned_problem_scores(baseline,x[ids['validation']],names),cutoff)
    trials=[];models={}
    for regularization in [.1,1.,10.]:
        for cost in [1.,2.,4.]:
            model=fit_recall_ridge(x[ids['train']],truth[ids['train']],regularization,cost)
            val=summary(truth[ids['validation']],ridge_scores(model,x[ids['validation']]),cutoff)
            trial={'regularization':regularization,'positive_cost':cost,'validation':val}
            trials.append(trial);models[(regularization,cost)]=model
    allowed=[t for t in trials if t['validation']['fn']<=previous_val['fn'] and t['validation']['fp']<=previous_val['fp']]
    selected=max(allowed or trials,key=lambda t:(t['validation']['recall'],*recall_selection_key(t['validation'])))
    model=models[(selected['regularization'],selected['positive_cost'])]
    selection={'training_policy':TRAINING_POLICY,'test_used_for_selection':False,'threshold':cutoff,
        'threshold_changed':False,'trials':trials,'previous_validation':previous_val,'selected':selected,
        'accepted_by_validation':bool(allowed),'criterion':'no_more_validation_misses_or_false_candidates_then_max_recall'}
    write_json(run/'selection.json',selection)
    checkpoint={**baseline,'ridge':model,'problems':names,'source_run':str(run),'source_manifest_hash':cache['manifest_hash'],
        'weights_retrained':True,'training_config':selection}
    torch.save(checkpoint,run/'classifier.pt')
    reference_cache=run/'improvement-v1/convnext-features.pt';reference_cache.parent.mkdir(exist_ok=True)
    shutil.copy2(run/'features.pt',reference_cache)
    report={'selection':selection,'evaluations':{},'weights_retrained':True}
    predictions=[]
    for p,part in ids.items():
        scores=ridge_scores(model,x[part]);prior=aligned_problem_scores(baseline,x[part],names);target=truth[part]
        report['evaluations'][p]={'current':summary(target,scores,cutoff),'previous':summary(target,prior,cutoff),
            'per_class':per_class(target.numpy(),scores.numpy(),names,cutoff)}
        if p=='test':
            for i,new,old in zip(part.tolist(),scores.tolist(),prior.tolist()):
                row=rows[i];actual=set(row['defects']);a={n for n,s in zip(names,new) if s>=cutoff};b={n for n,s in zip(names,old) if s>=cutoff}
                predictions.append({'image_id':row['image_id'],'relative_path':row['relative_path'],'true_problems':row['defects'],
                    'candidate_problems':sorted(a),'missing_problems':sorted(actual-a),'newly_recovered':sorted((actual&a)-b),
                    'regressed':sorted((actual&b)-a),'scores':dict(zip(names,new))})
    current=report['evaluations']['test']['current'];previous=report['evaluations']['test']['previous']
    report['adopt_candidate']=bool(allowed) and current['fn']<=previous['fn'] and current['fp']<=previous['fp']*1.1+5
    report['newly_recovered']=sum(len(p['newly_recovered']) for p in predictions)
    report['regressed']=sum(len(p['regressed']) for p in predictions)
    write_json(run/'metrics.json',report);write_json(run/'test-predictions.json',predictions)
    return report


def retrain(root,snapshot,reference_run=None,baseline_auxiliary_run=OLD_AUX,include_confirmed_normal=False,resume_prepared=False,quarantine_overlaps=False):
    root.mkdir(parents=True,exist_ok=True)
    if (root/'adoption.json').exists():raise ValueError('Run frozen')
    if not resume_prepared and ((root/'problem/selection.json').exists() or (root/'auxiliary/selection.json').exists()):raise ValueError('Training selection is frozen')
    config={'snapshot_sha256':digest(snapshot),'main_grid':{'regularization':[.1,1.,10.],'positive_cost':[1.,2.,4.]},
        'reference_run':str(reference_run) if reference_run else None,'baseline_auxiliary_run':str(baseline_auxiliary_run),
        'confirmed_normal_only_in_separate_filter':include_confirmed_normal,
        'quarantine_historical_overlaps':quarantine_overlaps,
        'new_classes_without_independent_training_samples':'retain output and held-out records; report as untrained, never move historical holdouts',
        'threshold':.125,'new_test_used_for_parameter_selection':False,'historical_holdouts_preserved':True,
        'deploy_problem_gate':'validation no more misses/false candidates; test no more misses and FP <= old*1.1+5',
        'deploy_auxiliary_gate':'test no more false unrelated and no lower unrelated recall',
        'original_image_state_exclusion_unchanged':True}
    write_json(root/'planned-config.json',config);torch.set_num_threads(4)
    problem=root/'problem';(problem/'source').mkdir(parents=True,exist_ok=True)
    if (problem/'source/labels.sqlite3').exists():assert digest(snapshot)==digest(problem/'source/labels.sqlite3')
    else:shutil.copy2(snapshot,problem/'source/labels.sqlite3')
    if not (problem/'dataset.json').exists():prepare_problems(problem,NAS_ROOT)
    if resume_prepared and (problem/'metrics.json').exists():
        data=read_json(problem/'dataset.json');cache=cached_features(problem,data,baseline_auxiliary_run)
        main=read_json(problem/'metrics.json')
        assert main['selection']['threshold']==.125 and main['selection']['test_used_for_selection'] is False
    else:
        data=preserve_holdouts(problem,reference_run/'problem' if reference_run else OLD_PROBLEM,allow_untrained_new_classes=True)
        cache=cached_features(problem,data,baseline_auxiliary_run);main=train_problems(problem,data,cache)
    print(json.dumps({'stage':'problem_trained','adopt':main['adopt_candidate'],'test':main['evaluations']['test']},ensure_ascii=False),flush=True)
    auxiliary=root/'auxiliary';auxiliary.mkdir(exist_ok=True)
    unrelated_filter.prepare(auxiliary,snapshot,include_confirmed_normal)
    aux_data=preserve_holdouts(auxiliary,reference_run/'auxiliary' if reference_run else baseline_auxiliary_run,
                               quarantine_historical_conflicts=quarantine_overlaps)
    aux_cache=cached_features(auxiliary,aux_data,baseline_auxiliary_run);unrelated_filter.train(auxiliary,snapshot,include_confirmed_normal)
    new=torch.load(auxiliary/'unrelated-filter.pt',weights_only=True);old=torch.load(baseline_auxiliary_run/'unrelated-filter.pt',weights_only=True)
    indices=torch.tensor([i for i,r in enumerate(aux_data['records']) if r['split']=='test'])
    y=torch.tensor([[float(r['auxiliary_target'])] for r in aux_data['records']])[indices]
    scores=ridge_scores(new['ridge'],aux_cache[new['feature_kind']][indices]);prior=ridge_scores(old['ridge'],aux_cache[old['feature_kind']][indices])
    current=summary(y,scores,new['threshold']+.002);previous=summary(y,prior,old['threshold']+.002)
    aux_adopt=current['fp']<=previous['fp'] and current['recall']>=previous['recall']
    result={'problem_run':str(problem if main['adopt_candidate'] else OLD_PROBLEM/'recall-trained-v3-fixed-cutoff'),
        'auxiliary_run':str(auxiliary if aux_adopt else baseline_auxiliary_run),'problem_candidate_adopted':main['adopt_candidate'],
        'auxiliary_candidate_adopted':aux_adopt,'problem_test':main['evaluations']['test'],
        'deployment_performed':False,
        'auxiliary_test':{'current':current,'previous':previous},'main_classes':data['summary']['problem_classes'],
        'main_split_counts':data['summary']['splits'],'auxiliary_split_counts':aux_data['splits']}
    write_json(root/'adoption.json',result);print(json.dumps(result,ensure_ascii=False),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--snapshot',type=Path,required=True);parser.add_argument('--reference-run',type=Path)
    parser.add_argument('--baseline-auxiliary-run',type=Path,default=OLD_AUX)
    parser.add_argument('--resume-prepared',action='store_true');parser.add_argument('--quarantine-overlaps',action='store_true')
    parser.add_argument('--include-confirmed-normal',action='store_true');args=parser.parse_args()
    retrain(args.run,args.snapshot,args.reference_run,args.baseline_auxiliary_run,args.include_confirmed_normal,args.resume_prepared,args.quarantine_overlaps)
