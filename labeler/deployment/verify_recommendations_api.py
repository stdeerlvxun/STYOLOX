"""Real NAS/API roundtrip using an isolated COPY of the 101 label database."""
import argparse
import json
import sqlite3
from pathlib import Path
import sys
import threading
import time
from urllib.request import Request,urlopen
from urllib.error import HTTPError
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from app import make_server
from snapshot import snapshot,digest


def verify(data, model, output):
    snapshot(Path('/var/lib/after-sales-image-labeler/labels.sqlite3'),data/'labels.sqlite3')
    server=make_server(0,data,nas_root='/mnt/stdeer_analysis',recommendation_model=model)
    thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
    base=f'http://127.0.0.1:{server.server_port}'
    def get(route):
        with urlopen(base+route,timeout=10) as response:return json.load(response)
    token=get('/api/state')['token']
    def post(route,value):
        request=Request(base+route,json.dumps(value).encode(),headers={'Content-Type':'application/json','X-Local-Token':token})
        with urlopen(request,timeout=10) as response:return json.load(response)
    store=server.app.store
    def human_digest():
        with sqlite3.connect(store.db_path) as db:
            db.execute('BEGIN');return digest(db)
    before=human_digest()
    protected={}
    for table in ['labels','drafts']:
        row=store.one('SELECT image_id FROM '+table+' LIMIT 1')
        if row:protected[table]=get('/api/recommendation?id='+str(row['image_id']))['status']
    assert all(status=='protected' for status in protected.values())
    image_id=store.one('SELECT i.id FROM images i LEFT JOIN labels l ON l.image_id=i.id '
        'LEFT JOIN drafts d ON d.image_id=i.id WHERE l.image_id IS NULL AND d.image_id IS NULL '
        'AND i.available=1 ORDER BY i.reported_at DESC,i.relative_path LIMIT 1')['id']
    started=time.perf_counter()
    for attempt in range(120):
        result=get('/api/recommendation?id='+str(image_id))
        if result['status'] not in ['queued','running']:break
        time.sleep(.1)
    assert result['status']=='ready',result
    assert human_digest()==before,'Merely showing a suggestion changed human data'
    assert result['candidates'],'Use a nonempty suggestion for the isolated confirmation check'
    problem=result['candidates'][0]['name']
    label={'defects':[problem],'image_state':'defect','note':'isolated API acceptance test',
        'responsibilities':['待核'],'problem_responsibilities':{problem:['待核']}}
    saved=post('/api/draft',{'id':image_id,'label':label,'only_unclassified':True})
    assert get('/api/item?id='+str(image_id))['has_draft']
    post('/api/commit',{'id':image_id,'label':label,'only_unclassified':True,'expected_draft':saved['draft_fingerprint']})
    confirmed=get('/api/item?id='+str(image_id))['label']
    assert confirmed==label
    assert get('/api/recommendation?id='+str(image_id))['status']=='protected'
    guard_passed=False
    try:post('/api/commit',{'id':image_id,'label':{**label,'note':'must not replace'},'only_unclassified':True})
    except HTTPError as exc:guard_passed=exc.code==400
    assert guard_passed and get('/api/item?id='+str(image_id))['label']==confirmed
    auxiliary_confirmed=None
    meta=json.loads((model/'model.json').read_text(encoding='utf-8'))
    if meta.get('state_filter'):
        references=json.loads((model/'auxiliary-references.json').read_text(encoding='utf-8'))
        example=next(r for r in references if r['is_unrelated'] and r['unrelated_score']>meta['state_filter']['threshold']+.05)
        auxiliary_id=example['image_id']
        # Only the copied QA store is changed to simulate an unclassified image.
        assert store.db_path.resolve()==(data/'labels.sqlite3').resolve()
        assert store.db_path.resolve()!=Path('/var/lib/after-sales-image-labeler/labels.sqlite3').resolve()
        with store.lock,store.db:
            store.db.execute('DELETE FROM labels WHERE image_id=?',(auxiliary_id,))
            store.db.execute('DELETE FROM drafts WHERE image_id=?',(auxiliary_id,))
        auxiliary_before=human_digest()
        store.review_model=server.app.recommendations.meta['model_id']
        with store.lock,store.db:
            store.db.execute("INSERT OR REPLACE INTO settings VALUES('review_model',?)",(store.review_model,))
        for attempt in range(100):
            suggested=get('/api/recommendation?id='+str(auxiliary_id))
            if suggested['status'] not in ['queued','running']:break
            time.sleep(.1)
        assert suggested['status']=='ready' and suggested['image_state_suggestion']['state']=='unrelated'
        assert 'candidates' in suggested and human_digest()==auxiliary_before
        assert auxiliary_id in [r['id'] for r in get('/api/images?status=model_unrelated&limit=100')['items']]
        post('/api/review/unrelated',{'ids':[auxiliary_id],'accept':False})
        assert human_digest()==auxiliary_before
        state_label={'defects':[],'responsibilities':['待核'],'image_state':'unrelated',
                     'note':'isolated explicit unrelated confirmation','problem_responsibilities':{}}
        saved=post('/api/draft',{'id':auxiliary_id,'label':state_label,'only_unclassified':True})
        post('/api/commit',{'id':auxiliary_id,'label':state_label,'only_unclassified':True,'expected_draft':saved['draft_fingerprint']})
        assert get('/api/item?id='+str(auxiliary_id))['label']==state_label
        assert get('/api/recommendation?id='+str(auxiliary_id))['status']=='protected'
        auxiliary_confirmed={'image_id':auxiliary_id,'explicit_state_confirmation':'passed','automatic_label_writes':0}
        second=next(r for r in references if r['image_id']!=auxiliary_id and
                    r['unrelated_score']>=server.app.recommendations.meta['state_filter']['threshold'])
        second_id=second['image_id']
        with store.lock,store.db:
            store.db.execute('DELETE FROM labels WHERE image_id=?',(second_id,))
            store.db.execute('DELETE FROM drafts WHERE image_id=?',(second_id,))
        for attempt in range(100):
            queued=get('/api/recommendation?id='+str(second_id))
            if queued['status'] not in ['queued','running']:break
            time.sleep(.1)
        assert queued['status']=='ready' and get('/api/item?id='+str(second_id))['model_unrelated']
        reviewed=post('/api/review/unrelated',{'ids':[second_id],'accept':True})
        assert reviewed['saved']==1 and get('/api/item?id='+str(second_id))['label']['image_state']=='unrelated'
        auxiliary_confirmed['separate_review_queue_reject_and_confirm']='passed'
    report={'status':'passed','isolated_database':str(data/'labels.sqlite3'),'image_id':image_id,
        'model_id':result['model_id'],'suggestions':[c['name'] for c in result['candidates']],
        'display_did_not_change_labels_drafts_tags_history':True,
        'existing_label_and_draft_protected':protected,'explicit_confirmation_roundtrip':True,
        'concurrent_overwrite_guard':True,'production_database_not_written':True,
        'first_request_seconds':round(time.perf_counter()-started,3)}
    if auxiliary_confirmed:report['auxiliary_state_workflow']=auxiliary_confirmed
    output.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    server.shutdown();server.server_close()
    print(json.dumps(report,ensure_ascii=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--data',type=Path,required=True)
    parser.add_argument('--model',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();verify(args.data,args.model,args.output)
