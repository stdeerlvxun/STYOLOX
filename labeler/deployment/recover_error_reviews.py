"""Recover review progress from exact, active human classification confirmations."""
import argparse
from contextlib import closing
import json
from pathlib import Path
import sqlite3
from snapshot import digest,annotation_digest


def recover(data, run_id, after_history_id, apply=False, expected_count=None):
    path=(Path(data)/'labels.sqlite3').resolve(strict=True)
    with closing(sqlite3.connect(path.as_uri()+('?mode=rw' if apply else '?mode=ro'),uri=True)) as db:
        db.row_factory=sqlite3.Row
        try:
            db.execute('BEGIN IMMEDIATE' if apply else 'BEGIN')
            active=db.execute("SELECT value FROM settings WHERE key='error_review_run'").fetchone()
            if not active or active['value']!=run_id:raise ValueError('Review run changed')
            root=db.execute("SELECT value FROM settings WHERE key='root'").fetchone()[0]
            latest={}
            for row in db.execute("SELECT id,data,created_at FROM history WHERE id>? AND root_id=? "
                    "AND kind='commit' AND undone=0 ORDER BY id",(after_history_id,root)):
                for change in json.loads(row['data']):
                    latest[change['id']]={'label':change['after'],'history_id':row['id'],'time':row['created_at']}
            recovered=[];skipped=[]
            rows=db.execute("SELECT e.image_id,l.data label_data,d.image_id has_draft FROM error_reviews e "
                "JOIN images i ON i.id=e.image_id LEFT JOIN labels l ON l.image_id=e.image_id "
                "LEFT JOIN drafts d ON d.image_id=e.image_id WHERE e.run_id=? AND e.status='pending' AND i.root_id=?",
                (run_id,root)).fetchall()
            for row in rows:
                proof=latest.get(row['image_id'])
                if not proof or row['has_draft'] is not None or not row['label_data'] or json.loads(row['label_data'])!=proof['label']:
                    skipped.append(row['image_id']);continue
                recovered.append({'image_id':row['image_id'],'history_id':proof['history_id'],'reviewed_at':proof['time']})
            if expected_count is not None and len(recovered)!=expected_count:
                raise ValueError('Recoverable count changed: '+str(len(recovered)))
            before=digest(db);annotations_before=annotation_digest(db)
            if apply:
                for item in recovered:
                    db.execute("UPDATE error_reviews SET status='reviewed',reviewed_at=? WHERE image_id=? AND run_id=? AND status='pending'",
                        (item['reviewed_at'],item['image_id'],run_id))
            assert digest(db)==before,'Human classifications changed'
            assert annotation_digest(db)==annotations_before,'Position annotations changed'
            report={'applied':apply,'run_id':run_id,'after_history_id':after_history_id,
                'recoverable':len(recovered),'skipped':len(skipped),'records':recovered,
                'human_data_unchanged':True,'annotation_data_unchanged':True,
                'human_digest':before,'annotation_digest':annotations_before}
            if apply:db.commit()
            else:db.rollback()
            return report
        except Exception:
            db.rollback();raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',type=Path,required=True)
    parser.add_argument('--run-id',required=True)
    parser.add_argument('--after-history-id',type=int,required=True)
    parser.add_argument('--expected-count',type=int)
    parser.add_argument('--apply',action='store_true')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    report=recover(args.data,args.run_id,args.after_history_id,args.apply,args.expected_count)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k!='records'},ensure_ascii=False))
