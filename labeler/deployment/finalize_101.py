"""Import the final frozen snapshot without altering any annotation or image identifier."""
import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path
import shutil
import sqlite3
import subprocess
import sys

BASE = Path("/opt/after-sales-image-labeler/current")
DATA = Path("/var/lib/after-sales-image-labeler")
NAS = Path("/mnt/stdeer_analysis")


def finalize(source):
    sys.path.insert(0, str(BASE))
    from deployment.snapshot import digest
    from app import PathMapper
    destination = DATA / "labels.sqlite3"
    if destination.exists():
        raise RuntimeError("The production database already exists; refusing to overwrite labels")
    stopped = subprocess.run(["systemctl", "stop", "after-sales-image-labeler-staging"],
                             capture_output=True, text=True)
    if stopped.returncode not in (0, 5):  # A completed transient unit may already be unloaded.
        raise RuntimeError(stopped.stderr)
    backups = DATA / "migration-backups"
    backups.mkdir(parents=True, exist_ok=True)
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    shutil.copy2(source, backups / ("original-" + stamp + ".sqlite3"))
    temporary = DATA / ("incoming-" + stamp + ".sqlite3")
    shutil.copy2(source, temporary)
    mapper = PathMapper(NAS)
    with sqlite3.connect(temporary) as db:
        original_digest = digest(db)
        original_counts = {table: db.execute("SELECT COUNT(*) FROM " + table).fetchone()[0]
                           for table in ("images", "labels", "drafts", "tags", "history")}
        original_identity = db.execute("SELECT id,root_id,relative_path FROM images ORDER BY id").fetchall()
        for root_id, path in db.execute("SELECT id,path FROM roots").fetchall():
            db.execute("UPDATE roots SET path=? WHERE id=?", (mapper.server(path), root_id))
        columns = {r[1] for r in db.execute("PRAGMA table_info(images)")}
        for name in ("reported_at", "date_source"):
            if name not in columns:
                db.execute("ALTER TABLE images ADD COLUMN " + name + " TEXT NOT NULL DEFAULT ''")
        # Only enrich dates from the staged index; final labels/drafts/history always come from the final snapshot.
        db.execute("ATTACH DATABASE ? AS stage", (str(DATA / "staging/labels.sqlite3"),))
        db.execute("""UPDATE images SET reported_at=(SELECT s.reported_at FROM stage.images s
          WHERE s.root_id=images.root_id AND s.relative_path=images.relative_path),
          date_source=(SELECT s.date_source FROM stage.images s
          WHERE s.root_id=images.root_id AND s.relative_path=images.relative_path)
          WHERE EXISTS (SELECT 1 FROM stage.images s WHERE s.root_id=images.root_id
          AND s.relative_path=images.relative_path)""")
        assert digest(db) == original_digest, "Annotation data changed"
        assert db.execute("SELECT id,root_id,relative_path FROM images ORDER BY id").fetchall() == original_identity
        assert db.execute("PRAGMA quick_check").fetchone()[0] == "ok"
        report = {"imported_at": stamp, **original_counts, "data_digest": original_digest,
                  "labels_drafts_tags_history_identical": True, "image_ids_identical": True,
                  "server_roots": db.execute("SELECT * FROM roots").fetchall(),
                  "sort": "after_sales.reported_at descending"}
    temporary.replace(destination)
    (DATA / "migration-report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    subprocess.run(["systemctl", "enable", "--now", "after-sales-image-labeler.service"], check=True)
    print(json.dumps(report, ensure_ascii=True))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("snapshot", type=Path)
    args = parser.parse_args()
    finalize(args.snapshot)
