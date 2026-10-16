"""Exercise real automatic-box review APIs on a disposable production backup."""
import argparse
import copy
import json
from pathlib import Path
import sys
import threading
from urllib.request import Request,urlopen
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from app import make_server
from snapshot import snapshot,digest,annotation_digest
from import_box_proposals import install


def verify(data,manifest,output):
    live=Path('/var/lib/after-sales-image-labeler')
    if data.resolve()==live.resolve():raise ValueError('QA database must be separate from production')
    snapshot(live/'labels.sqlite3',data/'labels.sqlite3')
    imported=install(data,manifest)
    if not imported['imported']:raise ValueError('No source-verified proposals to exercise')
    server=make_server(0,data,nas_root='/mnt/stdeer_analysis')
    threading.Thread(target=server.serve_forever,daemon=True).start()
    store=server.app.store;base='http://127.0.0.1:'+str(server.server_port)
    def get(route):
        with urlopen(base+route,timeout=20) as response:return json.load(response)
    token=get('/api/state')['token']
    def post(route,body):
        request=Request(base+route,json.dumps(body).encode(),headers={'Content-Type':'application/json','X-Local-Token':token})
        with urlopen(request,timeout=20) as response:return json.load(response)
    before=digest(store.db);annotations_before=annotation_digest(store.db)
    row=get('/api/images?status=auto_boxes&limit=1')['items'][0]
    detail=get('/api/annotations?id='+str(row['id']));proposal=detail['proposal']
    assert proposal and proposal['status']=='pending'
    assert get('/api/annotations/export')['items']==[]
    assert digest(store.db)==before and annotation_digest(store.db)==annotations_before
    body={'id':row['id'],'source_sha256':detail['source']['sha256'],
        'label_fingerprint':detail['label_fingerprint'],'expected_version':detail['version'],
        'proposal_id':proposal['proposal_id'],'boxes':copy.deepcopy(proposal['boxes']),
        'status':'draft','outcome':'boxed'}
    box=body['boxes'][0];box['x']=max(0,min(1-box['w'],box['x']+.01))
    draft=post('/api/annotations',body)
    assert get('/api/annotation-proposals/state')['accepted']==0
    assert any(r['image_id']==row['id'] and not r['training_eligible_by_label_policy'] for r in get('/api/annotations/export')['items'])
    body.update(status='confirmed',expected_version=draft['version'])
    confirmed=post('/api/annotations',body)
    assert get('/api/annotation-proposals/state')['accepted']==1
    exported=next(r for r in get('/api/annotations/export')['items'] if r['image_id']==row['id'])
    assert exported['training_eligible_by_label_policy']
    assert exported['annotation']['origin']=='machine_assisted_human_confirmed'
    assert exported['annotation']['boxes'][0]['proposal_id']==proposal['proposal_id']
    body.update(expected_version=confirmed['version'])
    post('/api/annotations',body)  # Existing approved boxes remain editable.
    remaining=get('/api/images?status=auto_boxes&limit=1')['items']
    rejected=0
    if remaining:
        second=get('/api/annotations?id='+str(remaining[0]['id']))
        annotation_checkpoint=annotation_digest(store.db)
        post('/api/annotation-proposals/reject',{'id':remaining[0]['id'],'proposal_id':second['proposal']['proposal_id']})
        assert annotation_digest(store.db)==annotation_checkpoint;rejected=1
    assert digest(store.db)==before
    report={'status':'passed','isolated_database':str(data/'labels.sqlite3'),
        'imported':imported['imported'],'source_verified_real_image_id':row['id'],
        'suggestions_do_not_create_human_annotations':True,'draft_is_not_training_eligible':True,
        'explicit_confirmation_required':True,'confirmed_model_origin_retained':True,
        'confirmed_boxes_remain_editable':True,'rejection_preserves_annotations':bool(rejected),
        'human_labels_drafts_tags_history_unchanged':True,'production_database_not_written':True}
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    server.shutdown();server.server_close();print(json.dumps(report,ensure_ascii=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--data',type=Path,required=True)
    parser.add_argument('--manifest',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();verify(args.data,args.manifest,args.output)
