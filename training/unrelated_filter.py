"""Separately authorized unrelated-image filter. Never trains the problem head."""
from collections import Counter,defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import argparse
import functools
import hashlib
import io
import json
import sqlite3
import time
import cv2
import numpy as np
from PIL import Image,ImageOps
import torch
from torch.nn import functional as F
from improve_classifier import classification_encoder,crop_tensors,pack_views,digest,ridge_scores,micro_ap,summary
from vision_classifier import read_json,write_json,assign_groups,split_groups,training_exclusion_reason,NAS_ROOT

POLICY='unrelated_auxiliary_v1'
NORMAL_POLICY='unrelated_auxiliary_v2_with_confirmed_normal'


def auxiliary_target(label,has_draft,include_confirmed_normal=False):
    if has_draft:return None
    if label.get('image_state')=='unrelated' and not label.get('defects'):return 1
    if include_confirmed_normal and label.get('image_state')=='normal' and not label.get('defects'):return 0
    if not training_exclusion_reason(label):return 0
    return None


@functools.lru_cache(maxsize=2048)
def order_keys(path):
    case=read_json(path);keys=[]
    for order in case.get('related_orders',[]):
        for field in ['h3_object_id','production_no','crm_no','contract_no']:
            if order.get(field):keys.append(field+':'+str(order[field]).strip())
    keys+=['h3_object_id:'+str(v) for v in case.get('order_link',{}).get('ids',[]) if v]
    return sorted(set(keys))


def inspect(row):
    relative=Path(row['relative_path'])
    if relative.is_absolute() or '..' in relative.parts:raise ValueError('Invalid relative path')
    path=NAS_ROOT/relative;raw=path.read_bytes()
    with Image.open(io.BytesIO(raw)) as original:
        image=ImageOps.exif_transpose(original).convert('RGB');width,height=image.size
        gray=np.asarray(image.convert('L').resize((32,32)),dtype=np.float32)
    dct=cv2.dct(gray)[:8,:8].flatten()[1:]
    phash=sum(int(value>np.median(dct))<<bit for bit,value in enumerate(dct))
    return {**row,'full_path':str(path),'sha256':hashlib.sha256(raw).hexdigest(),
        'width':width,'height':height,'aspect':width/height,'phash':phash,
        'order_keys':order_keys(path.parent.parent.parent/'case.json')}


def prepare(run,snapshot,include_confirmed_normal=False):
    if (run/'dataset.json').exists():return read_json(run/'dataset.json')
    excluded=[];candidates=[]
    with sqlite3.connect(snapshot.resolve().as_uri()+'?mode=ro',uri=True) as db:
        db.row_factory=sqlite3.Row
        for row in db.execute('SELECT i.id AS image_id,i.relative_path,i.case_id,l.data,'
            'd.image_id IS NOT NULL AS has_draft FROM labels l JOIN images i ON i.id=l.image_id '
            'LEFT JOIN drafts d ON d.image_id=i.id ORDER BY i.id'):
            row=dict(row);label=json.loads(row.pop('data'));target=auxiliary_target(label,row['has_draft'],include_confirmed_normal)
            if target is None:excluded.append({'image_id':row['image_id'],'reason':'draft_or_other_state'});continue
            candidates.append({**row,'image_state':label.get('image_state',''),'defects':label['defects'],
                'auxiliary_target':target})
    def safe(row):
        try:return inspect(row),None
        except (OSError,ValueError,KeyError) as exc:return None,{'image_id':row['image_id'],'reason':'source_error','error':str(exc)}
    checked=[];start=time.perf_counter()
    with ThreadPoolExecutor(max_workers=4) as pool:
        for i,(record,error) in enumerate(pool.map(safe,candidates)):
            if record:checked.append(record)
            if error:excluded.append(error)
            if i%100==0:print(json.dumps({'stage':'source_audit','done':i+1,'total':len(candidates),'seconds':round(time.perf_counter()-start,1)}),flush=True)
    by_sha=defaultdict(list)
    for row in checked:by_sha[row['sha256']].append(row)
    records=[]
    for duplicates in by_sha.values():
        if len({r['auxiliary_target'] for r in duplicates})>1:
            excluded.extend({'image_id':r['image_id'],'reason':'conflicting_duplicate_state'} for r in duplicates);continue
        kept=duplicates[0];kept['alias_image_ids']=[r['image_id'] for r in duplicates[1:]]
        kept['order_keys']=sorted({key for r in duplicates for key in r['order_keys']}|
                                 {'case_alias:'+r['case_id'] for r in duplicates})
        records.append(kept)
        excluded.extend({'image_id':r['image_id'],'reason':'exact_duplicate'} for r in duplicates[1:])
    assign_groups(records)
    proxy=[{**r,'defects':['unrelated' if r['auxiliary_target'] else 'related']} for r in records]
    split_groups(proxy);splits={r['image_id']:r['split'] for r in proxy}
    for row in records:row['split']=splits[row['image_id']];row.pop('phash')
    counts={p:{'images':sum(r['split']==p for r in records),
        'groups':len({r['group'] for r in records if r['split']==p}),
        'targets':dict(Counter(str(r['auxiliary_target']) for r in records if r['split']==p))}
        for p in ['train','validation','test']}
    dataset={'training_policy':NORMAL_POLICY if include_confirmed_normal else POLICY,
        'authorization':'User allowed separate unrelated-image training and confirmed normal comparison samples on 2026-10-16',
        'confirmed_normal_related_negatives':include_confirmed_normal,
        'problem_model_training_policy_unchanged':True,'snapshot_sha256':digest(snapshot),'records':records,
        'splits':counts,'excluded':excluded,'excluded_counts':dict(Counter(r['reason'] for r in excluded))}
    write_json(run/'dataset.json',dataset)
    print(json.dumps({'stage':'dataset_frozen','splits':counts,'excluded':dataset['excluded_counts']},ensure_ascii=False),flush=True)
    return dataset


def extract(run,dataset):
    target=run/'features.pt';manifest=digest(run/'dataset.json')
    if target.exists():
        cached=torch.load(target,weights_only=True);assert cached['manifest_hash']==manifest;return cached
    old=Path(r'D:\视觉检测工作区\runs\20261016-problem-only-120454')
    old_ds=read_json(old/'dataset.json');old_cache=torch.load(old/'improvement-v1/convnext-features.pt',weights_only=True)
    reusable={r['sha256']:i for i,r in enumerate(old_ds['records'])}
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu');encoder,meta=classification_encoder(device)
    assert old_cache['encoder_sha256']==meta['sha256'];whole=[];multi=[];start=time.perf_counter();reused=0
    with torch.inference_mode():
        for i,row in enumerate(dataset['records']):
            if row['sha256'] in reusable:
                index=reusable[row['sha256']];g=old_cache['global'][index];m=old_cache['multi'][index];reused+=1
            else:g,m=pack_views(encoder(crop_tensors(row).to(device)).cpu())
            whole.append(g);multi.append(m)
            if i%50==0 or i+1==len(dataset['records']):
                info={'stage':'features','done':i+1,'total':len(dataset['records']),'reused':reused,'seconds':round(time.perf_counter()-start,1)}
                write_json(run/'progress.json',info);print(json.dumps(info),flush=True)
    cached={'global':torch.stack(whole),'multi':torch.stack(multi),'encoder_sha256':meta['sha256'],'manifest_hash':manifest}
    torch.save(cached,target);return cached


def fit(raw,target,regularization):
    center=raw.mean(0);vectors=F.normalize(raw-center,dim=1);kernel=vectors@vectors.T
    # Mistaking a related image for unrelated is penalized four times as much.
    weights=torch.where(target[:,0]>0,1.,4.);prior=(target[:,0]*weights).sum()/weights.sum()
    alpha=torch.linalg.solve(kernel+regularization*torch.diag(1/weights),target-prior)
    return {'center':center,'training_vectors':vectors,'alpha':alpha,'prior':prior.reshape(1),'regularization':regularization}


def conservative_cutoff(truth,scores):
    y=truth[:,0].numpy().astype(bool);s=scores[:,0].numpy()
    choices=[]
    for threshold in sorted(set([.5,.6,.7,.8,.9,.95,1.0001]+[float(v)+1e-5 for v in s[~y]])):
        m=summary(y[:,None],s[:,None],threshold)
        if m['fp']==0 and m['tp']>0:choices.append({'threshold':threshold,**m})
    if not choices:return {'threshold':1.0001,**summary(truth,scores,1.0001)}
    return max(choices,key=lambda m:(m['recall'],m['precision'],m['threshold']))


def train(run,snapshot,include_confirmed_normal=False):
    run.mkdir(parents=True,exist_ok=True)
    if (run/'selection.json').exists():raise ValueError('Frozen run')
    policy=NORMAL_POLICY if include_confirmed_normal else POLICY
    config={'policy':policy,'negative_related_cost':4,'regularizations':[.1,1.,10.],
        'features':['global','multi'],'selection':'training_only_3_fold_grouped_OOF_average_precision',
        'threshold_selection':'validation_zero_false_unrelated_then_max_recall',
        'seed':42,'test_used_for_selection':False,'automatic_skip':False,'problem_candidates_suppressed':False}
    write_json(run/'planned-config.json',config);torch.set_num_threads(4)
    dataset=prepare(run,snapshot,include_confirmed_normal);assert dataset['training_policy']==policy
    cache=extract(run,dataset);rows=dataset['records']
    target=torch.tensor([[float(r['auxiliary_target'])] for r in rows])
    ids={p:torch.tensor([i for i,r in enumerate(rows) if r['split']==p]) for p in ['train','validation','test']}
    groups=sorted({rows[i]['group'] for i in ids['train'].tolist()})
    rng=np.random.default_rng(42);rng.shuffle(groups);foldmap={g:i%3 for i,g in enumerate(groups)}
    folds=torch.tensor([foldmap[rows[i]['group']] for i in ids['train'].tolist()]);yt=target[ids['train']]
    candidates=[];oofs={}
    for feature in config['features']:
        x=cache[feature][ids['train']]
        for reg in config['regularizations']:
            oof=torch.zeros_like(yt)
            for fold in range(3):
                held=folds==fold;model=fit(x[~held],yt[~held],reg);oof[held]=ridge_scores(model,x[held])
            candidate={'feature':feature,'regularization':reg,'training_group_oof_ap':micro_ap(yt,oof)}
            candidates.append(candidate);oofs[(feature,reg)]=oof
    chosen=max(candidates,key=lambda c:(c['training_group_oof_ap'],c['regularization']))
    feature=chosen['feature'];model=fit(cache[feature][ids['train']],yt,chosen['regularization'])
    val_scores=ridge_scores(model,cache[feature][ids['validation']]);choice=conservative_cutoff(target[ids['validation']],val_scores)
    selection={**config,'candidates':candidates,'selected':chosen,'validation':choice,
        'source_manifest_hash':cache['manifest_hash'],'all_categories_independent_of_problem_head':True}
    write_json(run/'selection.json',selection)
    checkpoint={'training_policy':policy,'feature_kind':feature,'ridge':model,'threshold':choice['threshold'],
        'encoder_sha256':cache['encoder_sha256'],'source_manifest_hash':cache['manifest_hash'],
        'requires_human_confirmation':True,'problem_model_unchanged':True}
    torch.save(checkpoint,run/'unrelated-filter.pt')
    report={'training_policy':policy,'selection':selection,'split_counts':dataset['splits'],'evaluations':{}}
    test=[]
    for p,part in ids.items():
        scores=ridge_scores(model,cache[feature][part]);report['evaluations'][p]=summary(target[part],scores,choice['threshold'])
        if p=='test':
            for i,score in zip(part.tolist(),scores[:,0].tolist()):
                row=rows[i];test.append({**row,'score':score,'suggest_unrelated':score>=choice['threshold']})
    selected_oof=oofs[(feature,chosen['regularization'])][:,0]
    reviews=[]
    for i,score in zip(ids['train'].tolist(),selected_oof.tolist()):
        row=rows[i];reviews.append({'image_id':row['image_id'],'relative_path':row['relative_path'],
            'label_is_unrelated':bool(row['auxiliary_target']),'oof_score':score,
            'disagreement':abs(score-row['auxiliary_target']),'review_only_not_auto_relabelled':True})
    reviews.sort(key=lambda r:-r['disagreement'])
    write_json(run/'training-label-review-candidates.json',reviews[:30])
    write_json(run/'test-predictions.json',test);write_json(run/'metrics.json',report)
    print(json.dumps({'stage':'completed','selection':chosen,'validation':choice,'test':report['evaluations']['test']},ensure_ascii=False),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--snapshot',type=Path,required=True);parser.add_argument('--include-confirmed-normal',action='store_true')
    args=parser.parse_args();train(args.run,args.snapshot,args.include_confirmed_normal)
