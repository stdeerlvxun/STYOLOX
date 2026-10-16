"""Verify a new detector batch while preserving existing human annotations in QA."""
import argparse
import copy
import json
from pathlib import Path
import sys
import threading
from urllib.request import Request,urlopen
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from app import Store,make_server
from snapshot import digest,annotation_digest


def verify(data,manifest_path,output):
    if data.resolve()==Path('/var/lib/after-sales-image-labeler').resolve():raise ValueError('Use an isolated QA backup')
    manifest=json.loads(manifest_path.read_text(encoding='utf-8'))
    store=Store(data)
    before=digest(store.db);positions=annotation_digest(store.db)
    imported=store.annotations.install_proposals(manifest)
    assert digest(store.db)==before and annotation_digest(store.db)==positions
    row=store.one("SELECT image_id FROM annotation_proposals WHERE json_extract(data,'$.model_id')=? AND status='pending' LIMIT 1",(manifest['model_id'],))
    if not row:raise ValueError('No eligible candidate was imported')
    image_id=row['image_id'];store.db.close()
    server=make_server(0,data,nas_root='/mnt/stdeer_analysis')
    threading.Thread(target=server.serve_forever,daemon=True).start()
    store=server.app.store;base='http://127.0.0.1:'+str(server.server_port)
    def get(path):
        with urlopen(base+path,timeout=20) as response:return json.load(response)
    token=get('/api/state')['token']
    def post(body):
        request=Request(base+'/api/annotations',json.dumps(body).encode(),headers={'Content-Type':'application/json','X-Local-Token':token})
        with urlopen(request,timeout=20) as response:return json.load(response)
    existing=store.query('SELECT * FROM annotation_documents WHERE image_id!=? ORDER BY image_id',(image_id,))
    detail=get('/api/annotations?id='+str(image_id));proposal=detail['proposal']
    assert proposal['model_id']==manifest['model_id'] and not detail['document']['boxes']
    body={'id':image_id,'source_sha256':detail['source']['sha256'],'label_fingerprint':detail['label_fingerprint'],
        'expected_version':detail['version'],'proposal_id':proposal['proposal_id'],
        'boxes':copy.deepcopy(proposal['boxes']),'outcome':'boxed','status':'draft'}
    draft=post(body)
    exported=next(r for r in get('/api/annotations/export')['items'] if r['image_id']==image_id)
    assert not exported['training_eligible_by_label_policy']
    body.update(status='confirmed',expected_version=draft['version']);post(body)
    confirmed=next(r for r in get('/api/annotations/export')['items'] if r['image_id']==image_id)
    assert confirmed['training_eligible_by_label_policy']
    assert confirmed['annotation']['origin']=='machine_assisted_human_confirmed'
    assert existing==store.query('SELECT * FROM annotation_documents WHERE image_id!=? ORDER BY image_id',(image_id,))
    assert digest(store.db)==before
    report={'status':'passed','imported_images':imported['imported'],'skipped':imported['skipped'],
        'model_id':manifest['model_id'],'human_classifications_unchanged':True,
        'existing_human_annotation_documents_unchanged':True,'draft_not_training_eligible':True,
        'explicit_confirmation_and_source_provenance_verified':True,'production_database_not_written':True}
    output.parent.mkdir(parents=True,exist_ok=True);output.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    server.shutdown();server.server_close();print(json.dumps(report,ensure_ascii=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--data',type=Path,required=True)
    parser.add_argument('--manifest',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();verify(args.data,args.manifest,args.output)
