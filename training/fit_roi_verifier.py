"""Learn a conservative local false-positive check from confirmed train regions only."""
import argparse
import copy
import json
from pathlib import Path
import time
import numpy as np
import torch
from train_detector import load_detector, read_rgb, sha256, write_json, compare_validation
from roi_verifier import extract_roi_features, fit_verifier


def fit(dataset_path, base_run, run):
    started=time.monotonic();run=Path(run);run.mkdir(parents=True,exist_ok=False)
    dataset_path=Path(dataset_path);dataset=json.loads(dataset_path.read_text(encoding='utf-8-sig'))
    dataset_sha=sha256(dataset_path)
    records=dataset['records'];splits={name:{row['group'] for row in records if row['split'] in aliases}
        for name,aliases in [('train',('train',)),('val',('val','validation')),('test',('test',))]}
    if any(splits[a]&splits[b] for a,b in [('train','val'),('train','test'),('val','test')]):
        raise ValueError('ROI verifier split leakage')
    positive=[row for row in records if row['split']=='train']
    negative=dataset.get('negative_feedback_records',[])
    if not positive or not negative:
        raise ValueError('Confirmed positive boxes and accepted local negative regions are both required')
    for row in positive+negative:
        if row.get('has_draft') or row.get('image_state') not in (None,'','defect') or not row.get('defects'):
            raise ValueError('ROI verifier violates unchanged image-state and problem-label policy')
    for row in positive:
        if row.get('annotation_status')!='confirmed' or row.get('annotation_outcome')!='boxed':
            raise ValueError('Only confirmed positive boxes may train the ROI verifier')
    for row in negative:
        if row.get('split')!='train' or row.get('feedback_only') is not True or not row.get('negative_rois'):
            raise ValueError('Only prepared explicit train-only negative regions are allowed')
        if any(row['sha256']==held['sha256'] for held in records if held['split']!='train'):
            raise ValueError('Negative ROI overlaps a held-out source')
    torch.set_num_threads(2)
    detector=load_detector(base_run,device='cuda');detector.roi_verifier=None
    vectors={'positive':[],'negative':[]};source_hashes={}
    for kind,rows in [('positive',positive),('negative',negative)]:
        for row in rows:
            if sha256(row['full_path'])!=row['sha256']:
                raise ValueError('ROI training image changed')
            rgb=read_rgb(row['full_path'])
            boxes=[box['xyxy'] for box in row['boxes']] if kind=='positive' else row['negative_rois']
            if boxes:vectors[kind].append(extract_roi_features(detector,rgb,boxes,context_scale=1.0))
            source_hashes[str(row['image_id'])]=row['sha256']
    metadata=fit_verifier(np.concatenate(vectors['positive']),np.concatenate(vectors['negative']),
        source_hashes=source_hashes,context_scale=1.0)
    metadata.update(dataset_sha256=dataset_sha,source_checkpoint_sha256=sha256(Path(base_run)/'detector.pt'),
        positive_regions=sum(len(row['boxes']) for row in positive),
        negative_regions=sum(len(row['negative_rois']) for row in negative),
        image_policy='unchanged confirmed problem images only; no image-state negatives or unclear',
        supervision='confirmed positive boxes and explicitly confirmed local false-positive regions only')
    assert sha256(dataset_path)==dataset_sha
    checkpoint=torch.load(Path(base_run)/'detector.pt',map_location='cpu',weights_only=False)
    checkpoint=copy.deepcopy(checkpoint)
    if checkpoint['threshold'] != .05:
        raise ValueError('This fixed comparison requires the retained 0.05 threshold')
    checkpoint['config']['fixed_threshold']=.05
    checkpoint['config']['roi_verifier']=metadata
    checkpoint['config']['dataset_sha256']=dataset_sha
    torch.save(checkpoint,run/'detector.pt')
    write_json(run/'roi-verifier.json',metadata)
    del detector
    torch.cuda.empty_cache()
    comparison=compare_validation(dataset_path,run,base_run)
    write_json(run/'roi-vs-input-comparison.json',comparison)
    metrics={'experiment':'frozen detector plus learned conservative local false-positive verifier',
        'training_policy':dataset['training_policy'],'dataset_sha256':dataset_sha,
        'source_model_sha256':metadata['source_checkpoint_sha256'],
        'model_sha256':sha256(run/'detector.pt'),'test':None,'test_evaluation_pending':True,
        'selected_epoch':checkpoint.get('epoch',0),'completed_epochs':0,
        'validation':{key:comparison['candidate'][key] for key in ('class_and_location','location_only','whole_image_pass')},
        'training_seconds':time.monotonic()-started,
        'positive_regions':metadata['positive_regions'],'negative_regions':metadata['negative_regions'],
        'verifier_iterations':metadata.get('iterations',80),
        'existing_detector_weights_changed':False,'source_images_copied':False}
    write_json(run/'metrics.json',metrics)
    print(json.dumps(metrics,ensure_ascii=False),flush=True)
    return metrics


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset',type=Path,required=True);parser.add_argument('--base-run',type=Path,required=True)
    parser.add_argument('--run',type=Path,required=True);args=parser.parse_args()
    fit(args.dataset,args.base_run,args.run)
