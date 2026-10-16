"""Fill untouched pending box types locally, without moving boxes or writing labels."""
from __future__ import annotations

import argparse
from collections import Counter
import datetime as dt
import hashlib
import io
import json
from pathlib import Path
import sqlite3

from PIL import Image,ImageOps
import correction_cycle as worker
from review_refill import file_hash,safe_source_path
from typed_proposals import enrich_boxes


def select_pending(snapshot,model_sha):
    db=sqlite3.connect(Path(snapshot).resolve().as_uri()+'?mode=ro',uri=True);db.row_factory=sqlite3.Row
    try:
        root=int(db.execute("SELECT value FROM settings WHERE key='root'").fetchone()[0])
        rows=db.execute('SELECT i.id image_id,i.relative_path,i.size,i.mtime_ns,i.available,p.proposal_id,p.data,'
            'p.label_data,p.status,l.data current_label,'
            'EXISTS(SELECT 1 FROM drafts d WHERE d.image_id=i.id) has_draft,'
            'EXISTS(SELECT 1 FROM annotation_documents a WHERE a.image_id=i.id) has_annotation,'
            'EXISTS(SELECT 1 FROM annotation_history h WHERE h.image_id=i.id) has_history,'
            'EXISTS(SELECT 1 FROM annotation_feedback f WHERE f.image_id=i.id) has_feedback,'
            'EXISTS(SELECT 1 FROM annotation_proposal_history h WHERE h.image_id=i.id) has_proposal_history,'
            "EXISTS(SELECT 1 FROM error_reviews e WHERE e.image_id=i.id AND e.status='reviewed') has_review,"
            "EXISTS(SELECT 1 FROM model_reviews m WHERE m.image_id=i.id AND m.decision='unrelated' AND m.status='accepted') has_unrelated_review "
            'FROM annotation_proposals p JOIN images i ON i.id=p.image_id LEFT JOIN labels l ON l.image_id=i.id '
            'WHERE i.root_id=? ORDER BY i.reported_at DESC,i.relative_path DESC,i.id DESC',(root,)).fetchall()
        selected=[];excluded=Counter()
        for record in rows:
            row=dict(record);raw=row['current_label'] or '';label=json.loads(raw) if raw else {};proposal=json.loads(row['data'])
            reason=('not_pending' if row['status']!='pending'
                else 'human_work' if any(row[key] for key in ('has_draft','has_annotation','has_history','has_feedback','has_proposal_history','has_review','has_unrelated_review'))
                else 'image_state' if label.get('image_state','') not in ('','defect')
                else 'classification_changed' if row['label_data']!=raw or proposal.get('label_fingerprint')!=hashlib.sha256(raw.encode()).hexdigest()
                else 'different_location_model' if proposal.get('source_checkpoint_sha256')!=model_sha
                else 'no_missing_types' if not any(not box.get('problem') for box in proposal.get('boxes',[]))
                else 'source_changed' if not row['available'] or any(row[key]!=proposal.get('source',{}).get(key) for key in ('size','mtime_ns'))
                else '')
            if reason:excluded[reason]+=1;continue
            row.update(proposal=proposal,expected_proposal_fingerprint=hashlib.sha256(row['data'].encode()).hexdigest())
            selected.append(row)
        return selected,{'candidates':len(selected),'excluded':dict(excluded)}
    finally:db.close()


def generate(config,snapshot,output,model=None,limit=200):
    if type(limit) is not int or not 1<=limit<=200:raise ValueError('Enrichment limit must be 1..200')
    output=Path(output)
    if output.exists():raise ValueError('Enrichment output already exists')
    run=Path(config['current_detector_run']);model_sha=file_hash(run/'detector.pt')
    metrics=worker.read_json(run/'metrics.json')
    if metrics['model_sha256']!=model_sha:raise ValueError('Detector checkpoint differs from validated metadata')
    rows,selection=select_pending(snapshot,model_sha)
    result={'schema_version':1,'enrichment_only':True,'requires_human_review':True,
        'model_id':'yolox-human-boxes-'+model_sha[:12],'method':'yolox_typed_problem_suggestions_v3',
        'source_checkpoint_sha256':model_sha,'source_snapshot_sha256':file_hash(snapshot),
        'created_at':dt.datetime.now(dt.timezone.utc).isoformat(),'selection':selection,
        'classification_results_modified':False,'source_images_modified':False,
        'geometry_modified':False,'unreviewed_proposals_training_eligible':False,'items':[],'errors':[]}
    if rows and model is None:
        from train_detector import load_detector
        model=load_detector(run,device='cuda')
    if rows and getattr(model,'class_agnostic',False):
        raise ValueError('A typed detector is required; retain the current issue-class model before enrichment')
    changed=0;attempted=0;remaining=0
    for row in rows[:limit]:
        attempted+=1
        try:
            source=row['proposal']['source'];path=safe_source_path(config['source_root'],row['relative_path'])
            before=path.stat();raw=path.read_bytes();after=path.stat()
            if (before.st_size,before.st_mtime_ns)!=(after.st_size,after.st_mtime_ns):raise ValueError('Source changed during read')
            if hashlib.sha256(raw).hexdigest()!=source['sha256']:raise ValueError('Source hash no longer matches pending proposal')
            with Image.open(io.BytesIO(raw)) as opened:image=ImageOps.exif_transpose(opened).convert('RGB')
            if (image.width,image.height)!=(source['width'],source['height']):raise ValueError('Source dimensions changed')
            predictions=model.predict_image(image,**({'threshold':config['fixed_threshold']} if 'fixed_threshold' in config else {}))
            boxes,count=enrich_boxes(row['proposal']['boxes'],predictions,model_sha,
                class_agnostic=bool(getattr(model,'class_agnostic',False)))
            final=path.stat()
            if (after.st_size,after.st_mtime_ns)!=(final.st_size,final.st_mtime_ns) or file_hash(path)!=source['sha256']:
                raise ValueError('Source changed during inference')
            remaining+=sum(not box.get('problem') for box in boxes)
            if count:
                result['items'].append({'image_id':row['image_id'],'relative_path':row['relative_path'],
                    'proposal_id':row['proposal_id'],'expected_proposal_fingerprint':row['expected_proposal_fingerprint'],
                    'label_fingerprint':row['proposal']['label_fingerprint'],'source':source,'boxes':boxes})
                changed+=count
        except (OSError,ValueError,KeyError) as error:result['errors'].append({'image_id':row['image_id'],'error':str(error)[:500]})
    if file_hash(run/'detector.pt')!=model_sha or file_hash(snapshot)!=result['source_snapshot_sha256']:
        raise ValueError('Checkpoint or frozen snapshot changed during inference')
    result['summary']={'attempted_images':attempted,'enriched_images':len(result['items']),'typed_boxes':changed,
        'unmatched_blank_boxes':remaining,'errors':len(result['errors']),'remaining_candidates':max(0,len(rows)-attempted)}
    worker.atomic_json(output,result)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True);parser.add_argument('--snapshot',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--limit',type=int,default=200)
    args=parser.parse_args()
    result=generate(worker.validate_config(worker.read_json(args.config)),args.snapshot,args.output,limit=args.limit)
    print(json.dumps(result['summary'],ensure_ascii=False),flush=True)
