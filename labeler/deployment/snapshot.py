"""SQLite online backup, plus a readable copy of labels, drafts and tag vocabulary."""
import argparse
import datetime as dt
import hashlib
import json
import sqlite3
from pathlib import Path

TABLES = ("labels", "drafts", "tags", "history")


def digest(db):
    contents = {table: [tuple(row) for row in db.execute("SELECT * FROM " + table + " ORDER BY 1").fetchall()] for table in TABLES}
    return hashlib.sha256(json.dumps(contents, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def annotation_digest(db):
    contents={}
    for table in ('annotation_documents','annotation_history'):
        exists=db.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",(table,)).fetchone()
        contents[table]=[tuple(row) for row in db.execute('SELECT * FROM '+table+' ORDER BY 1').fetchall()] if exists else []
    return hashlib.sha256(json.dumps(contents,ensure_ascii=False,sort_keys=True).encode()).hexdigest()


def snapshot(source, output):
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise ValueError("Snapshot already exists")
    with sqlite3.connect(source) as live, sqlite3.connect(output) as saved:
        live.backup(saved)
        assert saved.execute("PRAGMA quick_check").fetchone()[0] == "ok"
        info = {"checked_at": dt.datetime.now(dt.timezone.utc).isoformat(),
                **{table: saved.execute("SELECT COUNT(*) FROM " + table).fetchone()[0]
                   for table in ("images", *TABLES)},
                "data_digest": digest(saved), "roots": saved.execute("SELECT * FROM roots").fetchall()}
        info['annotation_data_digest']=annotation_digest(saved)
        readable = {"roots": info["roots"], "tags": saved.execute("SELECT * FROM tags").fetchall()}
        for table in ("labels", "drafts"):
            rows = saved.execute("SELECT a.image_id,i.relative_path,a.data,a.updated_at FROM " +
                                 table + " a JOIN images i ON i.id=a.image_id ORDER BY a.image_id").fetchall()
            readable[table] = [{"image_id": r[0], "relative_path": r[1],
                                "label": json.loads(r[2]), "updated_at": r[3]} for r in rows]
    output.with_suffix(".report.json").write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
    output.with_suffix(".labels.json").write_text(json.dumps(readable, ensure_ascii=False, indent=2), encoding="utf-8")
    return info


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    print(json.dumps(snapshot(args.source, args.output), ensure_ascii=False))
