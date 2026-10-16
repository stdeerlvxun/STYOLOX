"""Import a frozen model-review manifest without editing human classifications."""
import argparse
import hashlib
import json
from pathlib import Path
import sqlite3
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from app import Store
from snapshot import digest


def install(data,manifest_path):
    manifest=json.loads(manifest_path.read_text(encoding='utf-8'))
    run_id=manifest['run_id']
    if not isinstance(run_id,str) or not run_id or len(run_id)>100:raise ValueError('Invalid run ID')
    store=Store(data);rows=[];ids=set()
    for item in manifest['cases']:
        image=store.image(int(item['image_id']))
        if image['id'] in ids:raise ValueError('Duplicate review image')
        ids.add(image['id'])
        if image['root_id']!=store.current_root()['id'] or image['relative_path']!=item['relative_path']:
            raise ValueError('Review source identity changed')
        if hashlib.sha256(store.path(image).read_bytes()).hexdigest()!=item['source_sha256']:
            raise ValueError('Review source content changed')
        rows.append((image['id'],run_id,json.dumps(item,ensure_ascii=False,separators=(',',':'))))
    with store.lock,store.db:
        store.db.execute('BEGIN IMMEDIATE');before=digest(store.db)
        for image_id,run_id,raw in rows:
            previous=store.one('SELECT data FROM error_reviews WHERE image_id=? AND run_id=?',(image_id,run_id))
            if previous and previous['data']!=raw:raise ValueError('Frozen review manifest changed')
            store.db.execute('INSERT OR IGNORE INTO error_reviews(image_id,run_id,data) VALUES(?,?,?)',(image_id,run_id,raw))
        store.db.execute("INSERT OR REPLACE INTO settings VALUES('error_review_run',?)",(run_id,))
        assert before==digest(store.db),'Human data changed'
        report={'status':'imported','run_id':run_id,'cases':len(rows),'human_labels_drafts_tags_history_unchanged':True}
    store.db.close();return report


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--data',type=Path,required=True)
    parser.add_argument('--manifest',type=Path,required=True);args=parser.parse_args()
    print(json.dumps(install(args.data,args.manifest),ensure_ascii=False))
