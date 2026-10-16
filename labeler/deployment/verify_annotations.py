"""Real NAS localization/API checks on an isolated online-backup database."""
import argparse
import json
from pathlib import Path
import sys
import threading
from urllib.request import Request,urlopen
from urllib.error import HTTPError
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from app import make_server
from snapshot import snapshot,digest,annotation_digest
from import_error_review import install


def verify(data,manifest,output):
    assert data.resolve()!=Path('/var/lib/after-sales-image-labeler').resolve()
    snapshot(Path('/var/lib/after-sales-image-labeler/labels.sqlite3'),data/'labels.sqlite3')
    imported=install(data,manifest)
    server=make_server(0,data,nas_root='/mnt/stdeer_analysis');threading.Thread(target=server.serve_forever,daemon=True).start()
    store=server.app.store;base='http://127.0.0.1:'+str(server.server_port)
    def get(route):
        with urlopen(base+route,timeout=15) as response:return json.load(response)
    token=get('/api/state')['token']
    def post(body,route='/api/annotations'):
        request=Request(base+route,json.dumps(body).encode(),
            headers={'Content-Type':'application/json','X-Local-Token':token})
        with urlopen(request,timeout=15) as response:return json.load(response)
    def rejected(body,text):
        try:post(body)
        except HTTPError as exc:
            assert exc.code==400 and text in json.load(exc)['error'];return
        raise AssertionError('Expected conflict rejection')
    row=get('/api/images?status=error_review&review_kind=problem_miss&limit=1')['items'][0]
    image_id=row['id'];detail=get('/api/annotations?id='+str(image_id));before=digest(store.db)
    body={'id':image_id,'source_sha256':detail['source']['sha256'],'expected_version':detail['version'],
        'label_fingerprint':detail['label_fingerprint'],'review_run':detail['review']['run_id'],
        'status':'draft','outcome':'boxed','boxes':[{'id':'qa-box','problem':'崩边','x':.2,'y':.25,'w':.3,'h':.35}]}
    saved=post(body);assert before==digest(store.db)
    body.update(expected_version=saved['version'],status='confirmed');confirmed=post(body)
    assert confirmed['labels_unchanged'] and before==digest(store.db)
    restored=get('/api/annotations?id='+str(image_id));assert restored['document']['boxes']==body['boxes']
    assert restored['review']['status']=='reviewed'
    rejected({**body,'expected_version':0},'其他页面更新')
    export=get('/api/annotations/export');assert any(r['image_id']==image_id for r in export['items'])
    links=get('/api/responsibility-links');assert links['其他破损']['suggestion']=='运输'
    assert links['崩边']['suggestion']=='生产' and links['其他破损']['requires_confirmation']
    # All following human-classification mutation occurs only in this disposable QA copy.
    original=store.detail(image_id)['label'];store.commit([(image_id,{**original,'note':'isolated concurrent QA edit'})])
    rejected({**body,'expected_version':confirmed['version']},'人工分类已更新')
    pending=get('/api/images?status=error_review&limit=1')['items'][0]
    current=get('/api/item?id='+str(pending['id']))
    run=current['error_review']['run_id'];annotation_before=annotation_digest(store.db)
    # Reproduce the final pending image in an isolated database, through the real right-side API.
    with store.lock,store.db:
        store.db.execute("UPDATE error_reviews SET status='reviewed' WHERE run_id=? AND image_id!=?",(run,current['id']))
    result=post({'id':current['id'],'label':current['label'],'review_run':run},'/api/commit')
    assert result['review_completed']==1
    assert get('/api/state')['error_review_counts']['pending']==0
    assert get('/api/images?status=error_review&limit=1')['total']==0
    assert annotation_digest(store.db)==annotation_before
    post({},'/api/undo')
    assert get('/api/state')['error_review_counts']['pending']==1
    report={'status':'passed','isolated_database':str(data/'labels.sqlite3'),'review_cases_imported':imported['cases'],
        'real_image_id':image_id,'real_source_size':detail['source'],'draft_and_confirmation_roundtrip':True,
        'localization_never_changed_human_labels_drafts_tags_history':True,'stale_annotation_version_rejected':True,
        'concurrent_human_classification_guard':True,'review_case_completed':True,'normalized_export_roundtrip':True,
        'transport_prior_and_existing_rules_verified':True,'production_database_not_written':True,
        'right_side_classification_completes_last_review_without_fake_boxes':True,'undo_restores_pending_review':True}
    output.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    server.shutdown();server.server_close();print(json.dumps(report,ensure_ascii=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--data',type=Path,required=True)
    parser.add_argument('--manifest',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();verify(args.data,args.manifest,args.output)
