"""Generate independent review-only boxes with a supervised human-box detector."""
from __future__ import annotations
import argparse
from collections import Counter
import datetime as dt
import hashlib
import io
import json
import math
from pathlib import Path
import sqlite3
import time
import uuid
from PIL import Image,ImageOps


def normalize_predictions(predictions,allowed,max_boxes=10,class_agnostic=False):
    # A location-only detector must never invent one of several issue types.
    if class_agnostic and len(allowed)!=1:return []
    boxes=[]
    for prediction in sorted(predictions,key=lambda p:p['score'],reverse=True):
        problem=next(iter(allowed)) if class_agnostic else prediction.get('problem')
        if problem not in allowed:continue
        values=[float(v) for v in prediction['xywh']]
        score=float(prediction['score'])
        if not all(math.isfinite(v) for v in values+[score]):continue
        x,y,w,h=values
        x1,y1=max(0.,x),max(0.,y);x2,y2=min(1.,x+w),min(1.,y+h)
        if x2<=x1 or y2<=y1 or (x2-x1)*(y2-y1)>.9:continue
        if class_agnostic:
            duplicate=False
            for previous in boxes:
                overlap=max(0.,min(x2,previous['x']+previous['w'])-max(x1,previous['x']))*max(0.,min(y2,previous['y']+previous['h'])-max(y1,previous['y']))
                union=(x2-x1)*(y2-y1)+previous['w']*previous['h']-overlap
                if union>0 and overlap/union>.5:duplicate=True;break
            if duplicate:continue
        boxes.append({'id':'detector-'+uuid.uuid4().hex,'problem':problem,
            'x':x1,'y':y1,'w':x2-x1,'h':y2-y1,'score':score,'origin':'model'})
        if len(boxes)>=max_boxes:break
    return boxes


def select_generation_rows(rows,existing,refresh_pending=False):
    """Only unedited, current pending suggestions can be explicitly refreshed."""
    selected=[];preserved=0
    for row in rows:
        previous=existing.get(row['image_id'])
        if previous is None:
            selected.append(dict(row));continue
        if (refresh_pending and previous['status']=='pending'
            and previous.get('label_fingerprint')==row['label_fingerprint']):
            selected.append({**row,'replaces_proposal_id':previous['proposal_id']})
        else:preserved+=1
    return selected,preserved


def generate(snapshot,run,output,limit=None,device='cuda',source_root=None,refresh_pending=False,location_first=False):
    from generate_box_proposals import select_images,safe_source_path
    from vision_classifier import NAS_ROOT,write_json
    from train_detector import load_detector
    snapshot,run,output=Path(snapshot),Path(run),Path(output)
    if output.exists():raise ValueError('Output exists; choose a new file')
    root=Path(source_root) if source_root else NAS_ROOT
    rows,selection=select_images(snapshot)
    db=sqlite3.connect(snapshot.resolve().as_uri()+'?mode=ro',uri=True)
    try:
        existing={r[0]:{'proposal_id':r[1],'status':r[2],
            'label_fingerprint':json.loads(r[3]).get('label_fingerprint')}
            for r in db.execute('SELECT image_id,proposal_id,status,data FROM annotation_proposals')}
    finally:db.close()
    rows,preserved=select_generation_rows(rows,existing,refresh_pending)
    selection['existing_machine_suggestions_preserved']=preserved
    selection['pending_machine_suggestions_selected_for_refresh']=sum('replaces_proposal_id' in row for row in rows)
    if limit is not None:rows=rows[:limit]
    model=load_detector(run,device=device)
    detector_class_agnostic=bool(getattr(model,'class_agnostic',False))
    class_agnostic=detector_class_agnostic or bool(location_first)
    checkpoint=run/'detector.pt';model_sha=hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    snapshot_sha=hashlib.sha256(snapshot.read_bytes()).hexdigest()
    result={'schema_version':1,'model_id':('yolox-location-' if class_agnostic else 'yolox-human-boxes-')+model_sha[:12],
        'method':'yolox_problem_location_v2' if class_agnostic else 'yolox_supervised_human_boxes_v1','source_checkpoint_sha256':model_sha,
        'class_agnostic':detector_class_agnostic,'location_first_suggestions':class_agnostic,
        'problem_name_source':'existing_single_human_image_label' if class_agnostic else 'detector_class_prediction',
        'source_snapshot_sha256':snapshot_sha,'is_trained_detector':True,
        'requires_human_review':True,'unreviewed_proposals_training_eligible':False,
        'trial_only':True,'quality_warning':'Small-sample detector trial; localization is not reliable enough for automatic acceptance.',
        'classification_results_modified':False,'source_images_modified':False,
        'coordinate_space':'exif_transposed_normalized_xywh','probabilities_calibrated':False,
        'refresh_pending':bool(refresh_pending),
        'created_at':dt.datetime.now(dt.timezone.utc).isoformat(),'selection':selection,'items':[],'errors':[]}
    reasons=Counter();started=time.perf_counter()
    for index,row in enumerate(rows):
        try:
            path=safe_source_path(root,row['relative_path'])
            before=path.stat();raw=path.read_bytes();after=path.stat()
            if (before.st_size,before.st_mtime_ns)!=(after.st_size,after.st_mtime_ns):raise ValueError('Source changed during read')
            sha=hashlib.sha256(raw).hexdigest()
            if row.get('sha256') and row['sha256']!=sha:raise ValueError('Source hash changed from snapshot')
            with Image.open(io.BytesIO(raw)) as image:source=ImageOps.exif_transpose(image).convert('RGB')
            predictions=model.predict_image(source)
            boxes=normalize_predictions(predictions,set(row['problems']),class_agnostic=class_agnostic)
            if class_agnostic and len(set(row['problems']))!=1:
                reasons['multiple_problem_types_require_human_assignment']+=1
            elif not boxes:reasons['no_matching_problem_box_above_validation_cutoff']+=1
            item={'image_id':row['image_id'],'relative_path':row['relative_path'],
                'label_fingerprint':row['label_fingerprint'],'source':{'sha256':sha,'width':source.width,'height':source.height},
                'boxes':boxes,'note':('模型仅建议问题位置；问题名称沿用这张图唯一的人工问题分类，并非模型重新判断类型。请确认位置，已有分类保持不变。'
                    if class_agnostic else 'YOLOX 试验位置建议，需逐框审核；已有问题分类保持不变。')}
            if row.get('replaces_proposal_id'):
                item.update(replaces_proposal_id=row['replaces_proposal_id'],withdraw_if_empty=not boxes)
            result['items'].append(item)
        except (OSError,ValueError,KeyError) as error:
            result['errors'].append({'image_id':row['image_id'],'error':str(error)})
        if (index+1)%25==0:
            write_json(output.with_suffix('.progress.json'),{'processed':index+1,'total':len(rows),'seconds':round(time.perf_counter()-started,2)})
    result['summary']={'attempted_images':len(rows),'processed_images':len(result['items']),
        'images_with_proposals':sum(bool(r['boxes']) for r in result['items']),
        'proposed_boxes':sum(len(r['boxes']) for r in result['items']),
        'images_without_proposals':sum(not r['boxes'] for r in result['items']),
        'abstentions':dict(reasons),'errors':len(result['errors']),'seconds':round(time.perf_counter()-started,3)}
    if hashlib.sha256(snapshot.read_bytes()).hexdigest()!=snapshot_sha:raise RuntimeError('Frozen snapshot changed')
    write_json(output,result)
    print(json.dumps(result['summary'],ensure_ascii=False))
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--snapshot',type=Path,required=True);parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--limit',type=int)
    parser.add_argument('--device',default='cuda');parser.add_argument('--source-root',type=Path)
    parser.add_argument('--refresh-pending',action='store_true',help='Explicitly refresh only current unedited pending machine suggestions')
    parser.add_argument('--location-first',action='store_true',help='Use locations from any detector issue class; require one existing human issue type and retain it as context')
    args=parser.parse_args();generate(args.snapshot,args.run,args.output,args.limit,args.device,args.source_root,args.refresh_pending,args.location_first)
