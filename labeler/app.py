"""Local, read-only source-image browser and label store. Python 3.11 + Pillow."""
from __future__ import annotations
import argparse
import csv
import datetime as dt
import hashlib
import io
import json
import mimetypes
import os
from pathlib import Path
import queue
import secrets
import sqlite3
import subprocess
import threading
import time
from concurrent.futures import Future
from contextlib import closing
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlsplit, quote

from PIL import Image, ImageOps

BASE = Path(__file__).resolve().parent
DEFAULT_ROOT = r"\\stdeer\home\本地分析图片\cases"
DEFAULT_MANIFEST = r"\\stdeer\home\本地分析图片\人工分类\批次清单\第001批_300张_20261016.json"
EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
DEFAULT_TAGS = {
    "defect": ["崩边", "缺角", "开胶脱胶", "划伤", "其他破损"],
    "responsibility": ["生产", "运输", "安装", "其他", "待核", "原材料"],
}
RESPONSIBILITY_RULES = {
    **dict.fromkeys(["缺角", "工艺问题", "残胶", "pet边高", "边高", "软成型工艺",
                     "崩边", "开胶脱胶", "尺寸不符", "封边色差"], "生产"),
    **dict.fromkeys(["板件色差", "五金质量"], "原材料"),
}
STATES = {"", "defect", "normal", "unclear", "unrelated"}


class PathMapper:
    """Map the existing NAS share to its already-mounted server directory."""
    def __init__(self, nas_root=None):
        self.nas_root = Path(nas_root) if nas_root else None

    def server(self, raw):
        value = str(raw)
        normal = value.replace("\\", "/")
        for prefix in ("//stdeer/home", "//192.168.0.230/home"):
            if self.nas_root and (normal.lower() == prefix or normal.lower().startswith(prefix + "/")):
                return str(self.nas_root / normal[len(prefix):].lstrip("/"))
        return value

    def client(self, raw):
        if self.nas_root:
            try:
                return "\\\\stdeer\\home\\" + str(Path(raw).relative_to(self.nas_root)).replace("/", "\\")
            except ValueError:
                pass
        return str(raw)


def business_date(metadata):
    for raw, source in ((metadata.get("after_sales", {}).get("reported_at"), "reported_at"),
                        (metadata.get("source", {}).get("source_created_at"), "source_created_at")):
        if not raw:
            continue
        try:
            value = dt.datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
            zone = dt.timezone(dt.timedelta(hours=8))
            if value.tzinfo is None:
                value = value.replace(tzinfo=zone)
            return value.astimezone(zone).isoformat(timespec="seconds"), source
        except (ValueError, TypeError):
            pass
    return "", ""


def now():
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def dumps(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def clean_label(data, complete=False):
    result = {}
    for field in ("defects", "responsibilities"):
        values = data.get(field, [])
        if not isinstance(values, list) or len(values) > 40:
            raise ValueError("标签格式不正确")
        result[field] = list(dict.fromkeys(str(v).strip()[:50] for v in values if str(v).strip()))
    result["image_state"] = data.get("image_state", "")
    if result["image_state"] not in STATES:
        raise ValueError("图片状态不正确")
    if result["defects"]:
        if result["image_state"] == "normal":
            raise ValueError("无可见破损不能同时选择问题点")
        result["image_state"] = "defect"
    if complete and (not result["image_state"] or
                     (result["image_state"] == "defect" and not result["defects"])):
        raise ValueError("请选择问题点，或标记无破损、看不清、无关图片")
    # Optional per-problem data keeps legacy records unchanged until explicitly edited.
    if "problem_responsibilities" in data:
        mapping = data["problem_responsibilities"]
        if not isinstance(mapping, dict) or len(mapping) > 40:
            raise ValueError("问题点责任格式不正确")
        resolved = {}
        for problem in result["defects"]:
            values = mapping.get(problem, [RESPONSIBILITY_RULES[problem]] if problem in RESPONSIBILITY_RULES else [])
            if not isinstance(values, list) or len(values) > 40:
                raise ValueError("问题点责任格式不正确")
            values = list(dict.fromkeys(str(v).strip()[:50] for v in values if str(v).strip()))
            if "待核" in values and len(values) > 1:
                values.remove("待核")
            if complete and not values:
                raise ValueError("请为“" + problem + "”选择责任；不确定可选“待核”")
            resolved[problem] = values
        result["problem_responsibilities"] = resolved
        if resolved:
            result["responsibilities"] = list(dict.fromkeys(v for values in resolved.values() for v in values))
    if complete and not result["responsibilities"]:
        result["responsibilities"] = ["待核"]
    if ("problem_responsibilities" not in result and
            len(result["responsibilities"]) > 1 and "待核" in result["responsibilities"]):
        result["responsibilities"].remove("待核")
    result["note"] = str(data.get("note", ""))[:4000]
    return result


class Store:
    def __init__(self, data_dir):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = self.data_dir / "labels.sqlite3"
        self.lock = threading.RLock()
        self.db = sqlite3.connect(self.db_path, check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA foreign_keys=ON")
        self.db.executescript("""
        CREATE TABLE IF NOT EXISTS settings(key TEXT PRIMARY KEY,value TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS roots(id INTEGER PRIMARY KEY,path TEXT UNIQUE NOT NULL,last_image INTEGER);
        CREATE TABLE IF NOT EXISTS images(
          id INTEGER PRIMARY KEY,root_id INTEGER NOT NULL REFERENCES roots(id),
          relative_path TEXT NOT NULL,file_name TEXT NOT NULL,case_id TEXT,attachment_id TEXT,
          size INTEGER NOT NULL,mtime_ns INTEGER NOT NULL,sha256 TEXT,
          available INTEGER NOT NULL DEFAULT 1,last_seen TEXT,UNIQUE(root_id,relative_path));
        CREATE INDEX IF NOT EXISTS images_root_path ON images(root_id,relative_path);
        CREATE TABLE IF NOT EXISTS labels(image_id INTEGER PRIMARY KEY REFERENCES images(id),
          data TEXT NOT NULL,updated_at TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS drafts(image_id INTEGER PRIMARY KEY REFERENCES images(id),
          data TEXT NOT NULL,updated_at TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS tags(group_name TEXT NOT NULL,name TEXT NOT NULL,position INTEGER NOT NULL,
          UNIQUE(group_name,name));
        CREATE TABLE IF NOT EXISTS history(id INTEGER PRIMARY KEY,root_id INTEGER NOT NULL,
          kind TEXT NOT NULL,data TEXT NOT NULL,undone INTEGER NOT NULL DEFAULT 0,created_at TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS model_reviews(image_id INTEGER NOT NULL REFERENCES images(id),
          model_id TEXT NOT NULL,decision TEXT NOT NULL,score REAL,source_size INTEGER NOT NULL,
          source_mtime INTEGER NOT NULL,source_sha256 TEXT,status TEXT NOT NULL DEFAULT 'pending',
          created_at TEXT NOT NULL,PRIMARY KEY(image_id,model_id));
        """)
        self.review_model = None
        columns = {r["name"] for r in self.db.execute("PRAGMA table_info(images)")}
        for name in ("reported_at", "date_source"):
            if name not in columns:
                self.db.execute("ALTER TABLE images ADD COLUMN " + name + " TEXT NOT NULL DEFAULT ''")
        self.db.execute("CREATE INDEX IF NOT EXISTS images_recent ON images(root_id,reported_at DESC,relative_path)")
        for group, names in DEFAULT_TAGS.items():
            for i, name in enumerate(names):
                self.db.execute("INSERT OR IGNORE INTO tags VALUES(?,?,?)", (group, name, i))
        self.db.commit()
        self.backup()
        from annotations import Annotations
        self.annotations=Annotations(self)
        from automatic_classification import AutomaticClassifications
        self.automatic = AutomaticClassifications(self)

    def backup(self):
        folder = self.data_dir / "backups"
        folder.mkdir(exist_ok=True)
        target = folder / (dt.date.today().isoformat() + ".sqlite3")
        if not target.exists():
            with self.lock, closing(sqlite3.connect(target)) as dest:
                self.db.backup(dest)
        for old in sorted(folder.glob("*.sqlite3"))[:-7]:
            old.unlink()

    def query(self, sql, args=()):
        with self.lock:
            return [dict(r) for r in self.db.execute(sql, args).fetchall()]

    def one(self, sql, args=()):
        rows = self.query(sql, args)
        return rows[0] if rows else None

    def current_root(self):
        return self.one("SELECT r.* FROM roots r JOIN settings s ON s.key='root' AND r.id=s.value")

    def set_root(self, raw_path):
        path = os.path.abspath(os.path.expanduser(str(raw_path).strip().strip('"')))
        if not raw_path or not os.path.isabs(path):
            raise ValueError("请输入完整的原始图片目录")
        with self.lock, self.db:
            self.db.execute("INSERT OR IGNORE INTO roots(path) VALUES(?)", (path,))
            root = dict(self.db.execute("SELECT * FROM roots WHERE path=?", (path,)).fetchone())
            self.db.execute("INSERT OR REPLACE INTO settings VALUES('root',?)", (str(root["id"]),))
        return root

    def image(self, image_id):
        result = self.one("SELECT i.*,r.path AS root_path FROM images i JOIN roots r ON r.id=i.root_id WHERE i.id=?",
                          (image_id,))
        if not result:
            raise ValueError("没有找到这张图片，请重新扫描目录")
        return result

    @staticmethod
    def path(record):
        relative = Path(record["relative_path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("图片路径不正确")
        return Path(record["root_path"]) / relative

    def tags(self):
        result = {key: [] for key in DEFAULT_TAGS}
        for row in self.query("SELECT * FROM tags ORDER BY position,name"):
            result[row["group_name"]].append(row["name"])
        return result

    def add_tag(self, group, name):
        name = str(name).strip()
        if group not in DEFAULT_TAGS or not name or len(name) > 50:
            raise ValueError("标签名称应为 1～50 个字符")
        with self.lock, self.db:
            self.db.execute("INSERT OR IGNORE INTO tags VALUES(?,?,(SELECT COUNT(*) FROM tags WHERE group_name=?))",
                            (group, name, group))
        return self.tags()

    def detail(self, image_id):
        record = self.image(image_id)
        committed = self.one("SELECT * FROM labels WHERE image_id=?", (image_id,))
        draft = self.one("SELECT * FROM drafts WHERE image_id=?", (image_id,))
        record["label"] = json.loads((draft or committed)["data"]) if (draft or committed) else {
            "defects": [], "responsibilities": ["待核"], "image_state": "", "note": "",
            "problem_responsibilities": {}}
        record["has_draft"] = bool(draft)
        record["draft_fingerprint"] = hashlib.sha256(draft["data"].encode()).hexdigest() if draft else None
        record["classified"] = bool(committed)
        record['automatic_classification'] = self.automatic.current(record)
        record['classification_source'] = 'human' if committed else 'machine' if record['automatic_classification'] else 'none'
        if not committed and not draft and record['automatic_classification']:
            record['label'] = record['automatic_classification']['label']
        record['has_annotations']=bool(self.one('SELECT image_id FROM annotation_documents WHERE image_id=?',(image_id,)))
        record['error_review']=self.annotations.review(image_id)
        record["model_unrelated"] = bool(not committed and not draft and self.one(
            'SELECT i.id FROM images i WHERE i.id=? AND '+self.review_match(),(image_id,)))
        record["full_path"] = str(self.path(record))
        with self.lock, self.db:
            self.db.execute("UPDATE roots SET last_image=? WHERE id=?", (image_id, record["root_id"]))
        return record

    def protect_recommendation_save(self, image_id, expected_draft):
        if self.one("SELECT image_id FROM labels WHERE image_id=?", (image_id,)):
            raise ValueError("图片已有人工分类，建议未保存，请刷新后查看原分类")
        draft = self.one("SELECT data FROM drafts WHERE image_id=?", (image_id,))
        if draft and hashlib.sha256(draft["data"].encode()).hexdigest() != expected_draft:
            raise ValueError("图片已有其他暂存，建议未覆盖，请刷新后查看原编辑")

    def save_draft(self, image_id, data, only_unclassified=False, expected_draft=None):
        label = clean_label(data)
        self.image(image_id)
        with self.lock, self.db:
            if only_unclassified:self.protect_recommendation_save(image_id, expected_draft)
            self.db.execute("INSERT OR REPLACE INTO drafts VALUES(?,?,?)", (image_id, dumps(label), now()))
        return {"saved": True, "draft_fingerprint": hashlib.sha256(dumps(label).encode()).hexdigest()}

    def commit(self, entries, kind="commit", only_unclassified=False, expected_draft=None, review_run=None):
        if not entries or len(entries) > 1000:
            raise ValueError("一次操作请选择 1～1000 张图片")
        root = self.current_root()
        changes = []
        with self.lock, self.db:
            if review_run:
                active=self.one("SELECT value FROM settings WHERE key='error_review_run'")
                if not active or active['value']!=review_run:
                    raise ValueError('错例复核批次已变化，请重新读取；分类未覆盖')
            for image_id, data in entries:
                image = self.image(image_id)
                if image["root_id"] != root["id"]:
                    raise ValueError("所选图片不属于当前目录")
                value = clean_label(data, complete=True)
                before = self.one("SELECT * FROM labels WHERE image_id=?", (image_id,))
                if only_unclassified:self.protect_recommendation_save(image_id, expected_draft)
                change={"id": image_id, "before": before, "after": value}
                if review_run:
                    review=self.one('SELECT status,reviewed_at FROM error_reviews WHERE image_id=? AND run_id=?',(image_id,review_run))
                    if not review:raise ValueError('图片不属于当前错例复核批次')
                    change['error_review_before']={**review,'run_id':review_run}
                    self.db.execute("UPDATE error_reviews SET status='reviewed',reviewed_at=? WHERE image_id=? AND run_id=?",
                        (now(),image_id,review_run))
                changes.append(change)
                self.db.execute("INSERT OR REPLACE INTO labels VALUES(?,?,?)", (image_id, dumps(value), now()))
                self.db.execute("DELETE FROM drafts WHERE image_id=?", (image_id,))
                for group, field in (("defect", "defects"), ("responsibility", "responsibilities")):
                    for name in value[field]:
                        self.db.execute("INSERT OR IGNORE INTO tags VALUES(?,?,999)", (group, name))
            self.db.execute("INSERT INTO history(root_id,kind,data,created_at) VALUES(?,?,?,?)",
                            (root["id"], kind, dumps(changes), now()))
        return {"saved": len(changes), "image_id": changes[0]["id"], "review_completed":len(changes) if review_run else 0}

    def batch(self, ids, additions):
        entries = []
        additions = clean_label(additions)
        for image_id in dict.fromkeys(ids):
            value = self.detail(int(image_id))["label"]
            if ((additions["defects"] and value["image_state"] in {"normal", "unclear", "unrelated"}) or
                (additions["image_state"] in {"normal", "unclear", "unrelated"} and
                 (value["defects"] or value["image_state"] not in {"", additions["image_state"]}))):
                raise ValueError("批量标签与部分图片状态冲突，原有标签已保留，请逐张复核")
            if "problem_responsibilities" in value or "problem_responsibilities" in additions:
                mapping = value.get("problem_responsibilities", {
                    problem: value["responsibilities"][:] for problem in value["defects"]})
                appended = additions.get("problem_responsibilities", {
                    problem: additions["responsibilities"][:] for problem in additions["defects"]})
                for problem, responsibilities in appended.items():
                    mapping[problem] = list(dict.fromkeys(mapping.get(problem, []) + responsibilities))
                value["problem_responsibilities"] = mapping
            for field in ("defects", "responsibilities"):
                value[field] = list(dict.fromkeys(value[field] + additions[field]))
            if additions["defects"]:
                value["image_state"] = "defect"
            elif additions["image_state"]:
                value["image_state"] = additions["image_state"]
                if value["image_state"] != "defect":
                    value["defects"] = []
            if additions["note"]:
                value["note"] = (value["note"] + "\n" + additions["note"]).strip()
            entries.append((int(image_id), value))
        return self.commit(entries, "batch")

    def undo(self):
        root = self.current_root()
        with self.lock, self.db:
            row = self.one("SELECT * FROM history WHERE root_id=? AND undone=0 ORDER BY id DESC LIMIT 1",
                           (root["id"],))
            if not row:
                raise ValueError("没有可以撤销的确认记录")
            changes = json.loads(row["data"])
            for item in changes:
                self.db.execute("DELETE FROM labels WHERE image_id=?", (item["id"],))
                if item["before"]:
                    previous = item["before"]
                    self.db.execute("INSERT INTO labels VALUES(?,?,?)",
                                    (item["id"], previous["data"], previous["updated_at"]))
                # Keep the just-confirmed choices visible as a draft for easy correction.
                self.db.execute("INSERT OR REPLACE INTO drafts VALUES(?,?,?)",
                                (item["id"], dumps(item["after"]), now()))
                review=item.get('error_review_before')
                if review:
                    self.db.execute('UPDATE error_reviews SET status=?,reviewed_at=? WHERE image_id=? AND run_id=?',
                        (review['status'],review['reviewed_at'],item['id'],review['run_id']))
                else:
                    self.db.execute("UPDATE error_reviews SET status='pending',reviewed_at=NULL WHERE image_id=? "
                        "AND run_id=(SELECT value FROM settings WHERE key='error_review_run')",(item['id'],))
            self.db.execute("UPDATE history SET undone=1 WHERE id=?", (row["id"],))
        return {"undone": len(changes), "image_id": changes[0]["id"]}

    def where(self, params):
        root = self.current_root()
        parts, values = ["i.root_id=?"], [root["id"]]
        status = params.get("status", ["all"])[0]
        machine = self.automatic.match()
        if status == 'automatic':
            parts.extend(['l.image_id IS NULL', machine])
        elif status == 'inference_pending':
            parts.extend(['l.image_id IS NULL', 'd.image_id IS NULL', 'i.available=1',
                          "NOT EXISTS(SELECT 1 FROM annotation_documents a WHERE a.image_id=i.id AND json_extract(a.data,'$.status')='confirmed')",
                          '(m.image_id IS NULL OR NOT ('+machine+'))'])
        elif status == 'machine_unresolved':
            parts.extend(['l.image_id IS NULL', machine, "json_extract(m.data,'$.outcome')='unresolved'"])
        elif status == "unclassified":
            parts.append("l.image_id IS NULL")
            if self.review_model:
                parts.append("(d.image_id IS NOT NULL OR NOT " + self.review_match() + ")")
        elif status == "all_unclassified":
            parts.append("l.image_id IS NULL")
        elif status == "model_unrelated":
            parts.extend(["l.image_id IS NULL", "d.image_id IS NULL", self.review_match()])
        elif status == "classified":
            parts.append("l.image_id IS NOT NULL")
        elif status == "unrelated":
            parts.append("json_extract(l.data,'$.image_state')='unrelated'")
        elif status == "normal":
            parts.append("json_extract(l.data,'$.image_state')='normal'")
        elif status in ('error_review','error_reviewed'):
            parts.append("EXISTS (SELECT 1 FROM error_reviews e WHERE e.image_id=i.id AND "
                "e.run_id=(SELECT value FROM settings WHERE key='error_review_run') AND e.status=?)")
            values.append('reviewed' if status=='error_reviewed' else 'pending')
        elif status == 'annotation_pending':
            parts.append("json_extract(l.data,'$.image_state')='defect' AND NOT EXISTS "
                "(SELECT 1 FROM annotation_documents a WHERE a.image_id=i.id AND json_extract(a.data,'$.status')='confirmed')")
        elif status == 'annotated':
            parts.append('EXISTS (SELECT 1 FROM annotation_documents a WHERE a.image_id=i.id)')
        elif status == 'auto_boxes':
            parts.append('EXISTS (SELECT 1 FROM annotation_proposals p WHERE p.image_id=i.id AND '+self.annotations.proposal_match()+')')
        elif status == "pending":
            parts.append("(json_extract(l.data,'$.image_state')='unclear' OR EXISTS "
                         "(SELECT 1 FROM json_each(l.data,'$.responsibilities') WHERE value='待核'))")
        elif status == "unavailable":
            parts.append("i.available=0")
        for key, field in (("defect", "defects"), ("responsibility", "responsibilities")):
            tags = params.get(key, [])
            if tags:
                effective = "COALESCE(l.data,CASE WHEN "+machine+" THEN json_extract(m.data,'$.label') END)"
                parts.append("EXISTS (SELECT 1 FROM json_each("+effective+",'$." + field + "') WHERE value IN (" +
                             ",".join("?" for _ in tags) + "))")
                values.extend(tags)
        search = params.get("q", [""])[0].strip()
        if params.get('review_kind',[''])[0]:
            parts.append("EXISTS (SELECT 1 FROM error_reviews e,json_each(e.data,'$.kinds') k WHERE e.image_id=i.id "
                "AND e.run_id=(SELECT value FROM settings WHERE key='error_review_run') AND k.value=?)")
            values.append(params['review_kind'][0])
        if search:
            parts.append("(instr(lower(i.relative_path),lower(?))>0 OR instr(COALESCE(l.data,''),?)>0)")
            values.extend([search, search])
        return " AND ".join(parts), values

    def review_match(self):
        if not self.review_model:return '0'
        # Model ID is never interpolated into SQL. The active ID is a bound settings value.
        return ("EXISTS (SELECT 1 FROM model_reviews m WHERE m.image_id=i.id "
                "AND m.model_id=(SELECT value FROM settings WHERE key='review_model') "
                "AND m.decision='unrelated' AND m.status='pending' "
                "AND m.source_size=i.size AND m.source_mtime=i.mtime_ns AND i.available=1)")

    def record_model_review(self, image, result, stamp):
        if not self.review_model or result['model_id'] != self.review_model:return
        suggestion = result.get('image_state_suggestion')
        with self.lock, self.db:
            if self.one('SELECT image_id FROM labels WHERE image_id=? UNION ALL SELECT image_id FROM drafts WHERE image_id=?',
                        (image['id'], image['id'])):return
            self.db.execute('''INSERT INTO model_reviews VALUES(?,?,?,?,?,?,?,'pending',?)
                ON CONFLICT(image_id,model_id) DO UPDATE SET decision=excluded.decision,score=excluded.score,
                source_size=excluded.source_size,source_mtime=excluded.source_mtime,source_sha256=excluded.source_sha256,
                status=CASE WHEN model_reviews.status='rejected' THEN 'rejected' ELSE 'pending' END
                ''', (image['id'], result['model_id'], ('unrelated' if suggestion else 'retain') if result.get('status')=='ready' else 'error',
                      suggestion.get('score') if suggestion else None, stamp[0], stamp[1],
                      result.get('source_sha256'), now()))

    def review_unrelated(self, ids, accept):
        ids = list(dict.fromkeys(int(i) for i in ids))
        if not ids or len(ids)>1000:raise ValueError('请选择 1～1000 张待审核图片')
        entries=[]
        with self.lock, self.db:
            for image_id in ids:
                self.protect_recommendation_save(image_id, None)
                image=self.image(image_id)
                if image['root_id'] != self.current_root()['id']:raise ValueError('图片不属于当前目录')
                review=self.one("SELECT * FROM model_reviews WHERE image_id=? AND model_id=? AND decision='unrelated' AND status='pending'",
                                (image_id,self.review_model))
                if not review:raise ValueError('图片已退出待审核队列，请刷新列表')
                stat=self.path(image).stat()
                if not image['available'] or (stat.st_size,stat.st_mtime_ns)!=(review['source_size'],review['source_mtime']):
                    raise ValueError('原图发生变化，请重新判断')
                entries.append((image_id, {'defects':[], 'responsibilities':['待核'],
                    'image_state':'unrelated', 'note':'', 'problem_responsibilities':{}}))
            result=self.commit(entries,'review-unrelated',only_unclassified=True) if accept else {'returned':len(ids)}
            self.db.executemany('UPDATE model_reviews SET status=? WHERE image_id=? AND model_id=?',
                [('accepted' if accept else 'rejected',i,self.review_model) for i in ids])
        return result

    def review_counts(self):
        if not self.review_model:return {'enabled':False,'pending':0}
        root=self.current_root()
        return {'enabled':True, 'pending':self.listing({'status':['model_unrelated'],'limit':['1']})['total'],
            'evaluated':self.one('SELECT COUNT(*) n FROM model_reviews m JOIN images i ON i.id=m.image_id '
                'WHERE m.model_id=? AND i.root_id=? AND m.source_size=i.size AND m.source_mtime=i.mtime_ns',
                (self.review_model,root['id']))['n']}

    def listing(self, params, export=False):
        where, values = self.where(params)
        join = (" FROM images i LEFT JOIN labels l ON l.image_id=i.id LEFT JOIN drafts d ON d.image_id=i.id "
                "LEFT JOIN automatic_classifications m ON m.image_id=i.id ")
        total = self.one("SELECT COUNT(*) AS n" + join + "WHERE " + where, values)["n"]
        offset = max(0, int(params.get("offset", ["0"])[0]))
        limit = min(100, max(1, int(params.get("limit", ["40"])[0])))
        position = None
        if params.get("locate"):
            image = self.image(int(params["locate"][0]))
            exists = self.one("SELECT i.id" + join + "WHERE " + where + " AND i.id=?", values + [image["id"]])
            position = self.one("SELECT COUNT(*) AS n" + join + "WHERE " + where +
                                " AND (i.reported_at>? OR (i.reported_at=? AND i.relative_path<?))",
                                values + [image["reported_at"], image["reported_at"],
                                          image["relative_path"]])["n"] if exists else -1
            if position >= 0:
                offset = position // limit * limit
        sql = ("SELECT i.*,l.data AS label_data,l.updated_at,d.image_id IS NOT NULL AS has_draft" +
               join + "WHERE " + where + " ORDER BY i.reported_at DESC,i.relative_path")
        if not export:
            sql += " LIMIT ? OFFSET ?"
            values += [limit, offset]
        rows = self.query(sql, values)
        for row in rows:
            row["label"] = json.loads(row.pop("label_data")) if row["label_data"] else None
            row['automatic_classification'] = self.automatic.current(row)
            row['display_label'] = row['label'] or (row['automatic_classification'] or {}).get('label')
            row['classification_source'] = 'human' if row['label'] else 'machine' if row['automatic_classification'] else 'none'
        return {"items": rows, "total": total, "offset": offset, "position": position}

    def counts(self):
        root = self.current_root()
        row = self.one("""SELECT COUNT(*) total,SUM(l.image_id IS NOT NULL) classified,
          SUM(d.image_id IS NOT NULL) drafts,SUM(i.available=0) unavailable
          FROM images i LEFT JOIN labels l ON l.image_id=i.id LEFT JOIN drafts d ON d.image_id=i.id
          WHERE i.root_id=?""", (root["id"],))
        return {key: value or 0 for key, value in row.items()}


class Scanner:
    def __init__(self, store):
        self.store = store
        self.status = {"running": False, "found": 0, "error": "", "errors": 0}
        self.lock = threading.Lock()

    def start(self, root, recent=False):
        with self.lock:
            if self.status["running"]:
                raise ValueError("目录正在扫描，请稍后再试")
            self.status = {"running": True, "found": 0, "error": "", "errors": 0,
                           "root_id": root["id"], "started_at": now(), 'scope':'recent' if recent else 'full'}
        threading.Thread(target=self.run, args=(root,recent), daemon=True).start()

    def start_periodic(self, interval=120):
        def loop():
            last_full = time.monotonic()
            while True:
                time.sleep(interval)
                if self.status['running']:continue
                full = time.monotonic()-last_full >= 86400
                try:
                    self.start(self.store.current_root(), recent=not full)
                    if full:last_full=time.monotonic()
                except ValueError:
                    pass
        threading.Thread(target=loop,daemon=True,name='automatic-image-index').start()

    def run(self, root, recent=False):
        stamp = secrets.token_hex(12)
        pending = []

        def flush():
            if not pending:
                return
            with self.store.lock, self.store.db:
                self.store.db.executemany("""INSERT INTO images
                  (root_id,relative_path,file_name,case_id,attachment_id,size,mtime_ns,last_seen,reported_at,date_source)
                  VALUES(?,?,?,?,?,?,?,?,?,?) ON CONFLICT(root_id,relative_path) DO UPDATE SET
                  sha256=CASE WHEN images.size=excluded.size AND images.mtime_ns=excluded.mtime_ns
                    THEN images.sha256 ELSE NULL END,size=excluded.size,mtime_ns=excluded.mtime_ns,
                  last_seen=excluded.last_seen,available=1,reported_at=excluded.reported_at,
                  date_source=excluded.date_source""", pending)
            pending.clear()

        try:
            base = Path(root["path"])
            if not base.is_dir():
                raise ValueError("原始目录暂时无法访问，请确认 NAS 连接后重新扫描")
            targets=[base]
            if recent:
                today=dt.datetime.now(dt.timezone(dt.timedelta(hours=8))).date()
                previous=today.replace(day=1)-dt.timedelta(days=1)
                months=sorted({today.strftime('%Y-%m'),previous.strftime('%Y-%m')})
                targets=[base/month for month in months if (base/month).is_dir()] or [base]
            stack = [(folder, {"date": "", "source": ""}) for folder in targets]
            while stack:
                folder, inherited_date = stack.pop()
                date_info = inherited_date.copy()
                files = []
                try:
                    with os.scandir(folder) as entries:
                        for entry in sorted(entries, key=lambda entry: entry.name):
                            if entry.is_dir(follow_symlinks=False):
                                stack.append((Path(entry.path), date_info))
                            elif entry.name == "case.json" and entry.is_file(follow_symlinks=False):
                                try:
                                    if entry.stat().st_size < 2 * 1024 * 1024:
                                        metadata = json.loads(Path(entry.path).read_text(encoding="utf-8-sig"))
                                        date_info["date"], date_info["source"] = business_date(metadata)
                                except (OSError, ValueError, TypeError, AttributeError):
                                    pass
                            elif entry.is_file(follow_symlinks=False) and Path(entry.name).suffix.lower() in EXTENSIONS:
                                stat = entry.stat(follow_symlinks=False)
                                relative = Path(entry.path).relative_to(base).as_posix()
                                files.append((entry.name, stat, relative))
                        for name, stat, relative in files:
                                parts = Path(relative).parts
                                case_id = next((p for p in parts if len(p) == 36 and p.count("-") == 4), None)
                                attachment_id = name.split("__")[0] if "__" in name else None
                                reported_at, date_source = date_info["date"], date_info["source"]
                                if not reported_at:
                                    month = next((part for part in parts if len(part) == 7 and
                                                  part[4] == "-" and part[:4].isdigit() and part[5:].isdigit()), "")
                                    if month:
                                        reported_at, date_source = month + "-01T00:00:00+08:00", "archive_month"
                                    else:
                                        reported_at = dt.datetime.fromtimestamp(stat.st_mtime,
                                          dt.timezone(dt.timedelta(hours=8))).isoformat(timespec="seconds")
                                        date_source = "file_modified"
                                pending.append((root["id"], relative, name, case_id, attachment_id,
                                                stat.st_size, stat.st_mtime_ns, stamp, reported_at, date_source))
                                self.status["found"] += 1
                                if len(pending) >= 80:
                                    flush()
                except OSError:
                    self.status["errors"] += 1
            flush()
            if not self.status["errors"] and not recent:
                with self.store.lock, self.store.db:
                    self.store.db.execute("UPDATE images SET available=0 WHERE root_id=? AND last_seen<>?",
                                          (root["id"], stamp))
            elif self.status["errors"]:
                self.status["error"] = "部分子目录暂时无法读取，已有标签已保留；可重新扫描"
        except Exception as exc:
            self.status["error"] = str(exc)
        finally:
            self.status["running"] = False
            self.status["finished_at"] = now()


class ImageQueue:
    """Two NAS readers with current-image priority, stale-view cancellation and bounded previews."""
    def __init__(self, store, max_cache_mb=256):
        self.store = store
        self.cache = store.data_dir / "cache"
        self.cache.mkdir(exist_ok=True)
        self.cap = max_cache_mb * 1024 * 1024
        self.jobs = queue.PriorityQueue(maxsize=200)
        self.views = {}
        self.counter = 0
        self.lock = threading.Lock()
        self.last_trim = 0
        for _ in range(2):
            threading.Thread(target=self.worker, daemon=True).start()

    def view(self, client, view, focus):
        with self.lock:
            self.views[client] = (view, focus, time.time())
            if len(self.views) > 100:
                cutoff = time.time() - 3600
                self.views = {k: v for k, v in self.views.items() if v[2] > cutoff}

    def valid(self, request):
        state = self.views.get(request["client"])
        return not state or (str(state[0]) == request["view"] and
                             (request["kind"] == "thumb" or str(state[1]) == request["focus"]))

    def get(self, record, kind, params):
        if kind not in {"thumb", "preview", "original"}:
            raise ValueError("图片尺寸不正确")
        request = {"record": record, "kind": kind, "client": params.get("c", [""])[0],
                   "view": params.get("v", [""])[0], "focus": params.get("f", [""])[0],
                   "future": Future()}
        priority = 20 if params.get("prefetch") else (10 if kind == "thumb" else 0)
        with self.lock:
            self.counter += 1
            serial = self.counter
        self.jobs.put_nowait((priority, serial, request))
        return request["future"].result(timeout=60)

    def worker(self):
        while True:
            _, _, request = self.jobs.get()
            try:
                if not self.valid(request):
                    raise InterruptedError("已取消旧的图片读取")
                record, kind = request["record"], request["kind"]
                key = hashlib.sha256(f"{record['root_id']}|{record['relative_path']}|"
                                     f"{record['size']}|{record['mtime_ns']}|{kind}".encode()).hexdigest()
                cached = self.cache / (key + ".jpg")
                cached_content = None
                if kind != "original":
                    try:
                        cached_content = cached.read_bytes()
                        os.utime(cached, None)
                    except FileNotFoundError:
                        pass
                if cached_content is not None:
                    content, mime = cached_content, "image/jpeg"
                else:
                    path = self.store.path(record)
                    if path.stat().st_size > 64 * 1024 * 1024:
                        raise ValueError("图片超过 64MB，请使用“定位原文件”查看")
                    content = path.read_bytes()
                    if not self.valid(request):
                        raise InterruptedError("已取消旧的图片读取")
                    digest = hashlib.sha256(content).hexdigest()
                    with self.store.lock, self.store.db:
                        self.store.db.execute("UPDATE images SET sha256=?,available=1 WHERE id=? AND mtime_ns=?",
                                              (digest, record["id"], record["mtime_ns"]))
                    mime = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
                    if kind != "original":
                        with Image.open(io.BytesIO(content)) as opened:
                            image = ImageOps.exif_transpose(opened)
                            image.thumbnail((320, 240) if kind == "thumb" else (1600, 1600))
                            if image.mode != "RGB":
                                background = Image.new("RGB", image.size, "white")
                                if "A" in image.getbands():
                                    background.paste(image, mask=image.getchannel("A"))
                                else:
                                    background.paste(image.convert("RGB"))
                                image = background
                            output = io.BytesIO()
                            image.save(output, format="JPEG", quality=84)
                            content, mime = output.getvalue(), "image/jpeg"
                        temporary = cached.with_suffix("." + secrets.token_hex(6) + ".tmp")
                        temporary.write_bytes(content)
                        temporary.replace(cached)
                try:
                    self.trim()
                except OSError:
                    pass
                request["future"].set_result((content, mime))
            except Exception as exc:
                if isinstance(exc, FileNotFoundError):
                    with self.store.lock, self.store.db:
                        self.store.db.execute("UPDATE images SET available=0 WHERE id=?",
                                              (request["record"]["id"],))
                if not request["future"].done():
                    request["future"].set_exception(exc)
            finally:
                self.jobs.task_done()

    def trim(self):
        with self.lock:
            if time.monotonic() - self.last_trim < 20:
                return
            self.last_trim = time.monotonic()
            files = sorted(self.cache.glob("*.jpg"), key=lambda p: p.stat().st_mtime)
            size = sum(p.stat().st_size for p in files)
            for path in files:
                if size <= self.cap:
                    break
                length = path.stat().st_size
                path.unlink(missing_ok=True)
                size -= length


class Application:
    def __init__(self, data_dir, initial_root=None, nas_root=None, recommendation_model=None, unrelated_review=False):
        self.paths = PathMapper(nas_root)
        self.store = Store(data_dir)
        if initial_root or not self.store.current_root():
            self.store.set_root(self.paths.server(initial_root or DEFAULT_ROOT))
        from review_batches import ReviewBatches
        self.review_batches = ReviewBatches(self.store)
        from review_quality import ReviewQuality
        self.review_quality = ReviewQuality(self.store)
        from training_rounds import TrainingRounds
        self.training_rounds = TrainingRounds(self.store)
        self.review_batches.register('legacy-'+dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'),
                                     metadata={'source':'existing_pending'})
        self.scanner = Scanner(self.store)
        self.images = ImageQueue(self.store)
        from recommendations import Recommendations
        self.recommendations = Recommendations(self.store, self.images, recommendation_model)
        if self.recommendations.enabled:self.recommendations.start_review_scan()
        self.token = secrets.token_urlsafe(32)
        self.scanner.start(self.store.current_root())
        if self.recommendations.enabled:self.scanner.start_periodic()
        self.import_preview = None

    def state(self):
        from annotations import responsibility_links
        root = self.store.current_root()
        root["display_path"] = self.paths.client(root["path"])
        return {"root": root, "counts": self.store.counts(),
                "scan": self.scanner.status.copy(), "tags": self.store.tags(), "token": self.token,
                "default_manifest": DEFAULT_MANIFEST, "cache_limit_mb": 256,
                "responsibility_rules": RESPONSIBILITY_RULES,
                "responsibility_links": responsibility_links(self.store,RESPONSIBILITY_RULES),
                "annotation_counts": self.store.one('SELECT COUNT(*) documents,SUM(json_extract(a.data,\'$.status\')=\'confirmed\') confirmed '
                    'FROM annotation_documents a JOIN images i ON i.id=a.image_id WHERE i.root_id=?',(root['id'],)),
                "annotation_proposal_counts":self.store.annotations.proposal_state(),
                "review_batches":self.review_state(),
                "error_review_counts": self.store.one("SELECT COUNT(*) total,SUM(e.status='pending') pending FROM error_reviews e JOIN images i "
                    "ON i.id=e.image_id WHERE i.root_id=? AND e.run_id=(SELECT value FROM settings WHERE key='error_review_run')",(root['id'],)),
                "recommendations": self.recommendations.state(),
                "automation": self.training_rounds.runtime_state(),
                "model_review": self.store.review_counts(),
                "server_mode": bool(self.paths.nas_root), "sort": "reported_at_desc"}

    def review_state(self):
        self.review_batches.sync()
        state = self.review_batches.state()
        row = self.store.one('SELECT value FROM settings WHERE key=?', ('review_refill_worker:'+str(self.store.current_root()['id']),))
        state['worker'] = json.loads(row['value']) if row else {'enabled':False,'status':'not_configured'}
        return state

    def update_review_worker(self, body):
        allowed = {'status','detail','pending','generated_images','generated_boxes','eligible_images',
                   'attempted_images','batch_id','enabled','low_water','batch_size'}
        if not isinstance(body,dict) or set(body)-allowed:raise ValueError('自动续批状态格式不正确')
        value = {key:body[key] for key in body}
        for key in allowed-{'status','detail','batch_id','enabled'}:
            if key in value and (type(value[key]) is not int or not 0<=value[key]<=1000000):raise ValueError('自动续批计数不正确')
        for key in ('status','detail','batch_id'):
            if key in value and (not isinstance(value[key],str) or len(value[key])>1000):raise ValueError('自动续批说明不正确')
        if 'enabled' in value and type(value['enabled']) is not bool:raise ValueError('自动续批开关不正确')
        value['checked_at']=now()
        with self.store.lock,self.store.db:
            self.store.db.execute('INSERT OR REPLACE INTO settings(key,value) VALUES(?,?)',
                ('review_refill_worker:'+str(self.store.current_root()['id']),dumps(value)))
        return {'saved':True}

    def context(self, image_id):
        record = self.store.image(image_id)
        root = Path(record["root_path"])
        folder = self.store.path(record).parent
        while folder == root or root in folder.parents:
            path = folder / "case.json"
            if path.is_file() and path.stat().st_size < 2 * 1024 * 1024:
                return json.loads(path.read_text(encoding="utf-8-sig"))
            if folder == root:
                break
            folder = folder.parent
        return {"说明": "当前目录内没有关联的 case.json"}

    def prepare_import(self, manifest_path):
        path = Path(self.paths.server(manifest_path))
        if path.suffix.lower() != ".json" or not path.is_file() or path.stat().st_size > 20 * 1024 * 1024:
            raise ValueError("请选择已有批次清单 JSON（不超过 20MB）")
        manifest = json.loads(path.read_text(encoding="utf-8-sig"))
        directory = Path(self.paths.server(manifest["classification_directory"]))
        known = {item["filename"]: item for item in manifest["items"]}
        root = Path(self.store.current_root()["path"])
        mapped, unknown, conflicts = {}, set(), []
        category_map = {
            "崩边": ("defects", "崩边"), "缺角": ("defects", "缺角"),
            "开胶脱胶": ("defects", "开胶脱胶"), "划伤": ("defects", "划伤"),
            "其他破损": ("defects", "其他破损"), "正常": ("image_state", "normal"),
            "非板件": ("image_state", "unrelated"), "看不清": ("image_state", "unclear"),
        }
        for folder in directory.iterdir():
            if not folder.is_dir() or "待分类" in folder.name:
                continue
            category = next((value for word, value in category_map.items() if word in folder.name), None)
            if category is None:
                # Explicit responsibility folder names are safe to import as custom tags.
                if "责任" in folder.name:
                    category = ("responsibilities", folder.name.split("_", 1)[-1].replace("责任", "").strip())
                else:
                    unknown.add(folder.name)
                    continue
            for file in folder.iterdir():
                item = known.get(file.name)
                if not file.is_file() or not item:
                    continue
                original = directory.parent.parent / item["relative_path"]
                try:
                    relative = original.relative_to(root).as_posix()
                except ValueError:
                    continue
                image = self.store.one("SELECT id FROM images WHERE root_id=? AND relative_path=?",
                                       (self.store.current_root()["id"], relative))
                if not image:
                    continue
                image_id = image["id"]
                if image_id not in mapped:
                    mapped[image_id] = self.store.detail(image_id)["label"]
                value = mapped[image_id]
                field, tag = category
                if field == "image_state":
                    if value["defects"] or (value["image_state"] and value["image_state"] != tag):
                        conflicts.append(image_id)
                    value["image_state"] = tag
                else:
                    if field == "defects":
                        if value["image_state"] in {"normal", "unrelated", "unclear"}:
                            conflicts.append(image_id)
                        value["image_state"] = "defect"
                    value[field] = list(dict.fromkeys(value[field] + [tag]))
        for image_id in set(conflicts):
            mapped.pop(image_id, None)
        # Responsibility-only imports remain drafts until an image state is supplied.
        complete, drafts = [], []
        for image_id, value in mapped.items():
            (complete if value["image_state"] else drafts).append((image_id, value))
        key = secrets.token_urlsafe(16)
        self.import_preview = {"key": key, "root_id": self.store.current_root()["id"],
                               "complete": complete, "drafts": drafts}
        return {"key": key, "completed_count": len(complete), "draft_count": len(drafts),
                "unknown_folders": sorted(unknown), "conflicts": len(set(conflicts)),
                "notice": "仅写入标签；原图和已有分类文件夹均保留"}

    def apply_import(self, key):
        preview = self.import_preview
        if not preview or preview["key"] != key or preview["root_id"] != self.store.current_root()["id"]:
            raise ValueError("导入预览已过期，请重新预览")
        if preview["complete"]:
            self.store.commit(preview["complete"], "import")
        for image_id, value in preview["drafts"]:
            self.store.save_draft(image_id, value)
        self.import_preview = None
        return {"imported": len(preview["complete"]), "drafts": len(preview["drafts"])}


class Handler(BaseHTTPRequestHandler):
    server_version = "LocalImageLabeler/1.0"

    @property
    def app(self):
        return self.server.app

    def log_message(self, fmt, *args):
        if args and str(args[1] if len(args) > 1 else "").startswith("5"):
            super().log_message(fmt, *args)

    def send(self, content, mime="application/json; charset=utf-8", code=200, headers=None):
        if not isinstance(content, bytes):
            content = dumps(content).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", mime)
        self.send_header("Content-Length", str(len(content)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Content-Security-Policy",
                         "default-src 'self'; img-src 'self' blob:; style-src 'self'; "
                         "script-src 'self'; connect-src 'self'; frame-ancestors 'none'")
        for key, value in (headers or {}).items():
            self.send_header(key, value)
        self.end_headers()
        self.wfile.write(content)

    def check_request(self, mutation=False):
        expected = self.headers.get("Host", "")
        if expected not in self.server.allowed_hosts:
            raise PermissionError("请使用配置的分类工具地址访问")
        origin = self.headers.get("Origin")
        if origin and origin != "http://" + expected:
            raise PermissionError("请从分类工具页面操作")
        if mutation and self.headers.get("X-Local-Token") != self.app.token:
            raise PermissionError("页面连接已更新，请刷新页面")

    def do_GET(self):
        try:
            self.check_request()
            url = urlsplit(self.path)
            params = parse_qs(url.query)
            path = url.path
            static = {"/": "index.html", "/app.js": "app.js", "/style.css": "style.css",'/annotations.js':'annotations.js','/automation.js':'automation.js'}
            if path in static:
                file = BASE / "web" / static[path]
                return self.send(file.read_bytes(), mimetypes.guess_type(file.name)[0] or "text/plain")
            if path == "/api/state":
                return self.send(self.app.state())
            if path == "/api/images":
                return self.send(self.app.store.listing(params))
            if path == "/api/item":
                record = self.app.store.detail(int(params["id"][0]))
                record["full_path"] = self.app.paths.client(record["full_path"])
                return self.send(record)
            if path == "/api/context":
                return self.send(self.app.context(int(params["id"][0])))
            if path == '/api/annotations':
                return self.send(self.app.store.annotations.detail(int(params['id'][0])))
            if path == '/api/annotations/export':
                return self.send(self.app.store.annotations.export(),headers={'Content-Disposition':'attachment; filename="problem-localizations.json"'})
            if path == '/api/annotation-proposals/state':
                return self.send(self.app.store.annotations.proposal_state())
            if path == '/api/review-batches/state':
                return self.send(self.app.review_state())
            if path == '/api/review-quality/state':
                return self.send(self.app.review_quality.state())
            if path == '/api/training-rounds/state':
                return self.send(self.app.training_rounds.state())
            if path == '/api/review-batches/archive':
                content=self.app.review_batches.archive_content(params.get('batch',[''])[0])
                return self.send(content.encode('utf-8'),headers={'Content-Disposition':'attachment; filename="review-archive.json"'})
            if path == '/api/annotation-feedback/state':
                return self.send(self.app.store.annotations.feedback_state())
            if path == '/api/annotation-feedback/export':
                return self.send(self.app.store.annotations.feedback_export(),headers={'Content-Disposition':'attachment; filename="annotation-feedback.json"'})
            if path == '/api/responsibility-links':
                from annotations import responsibility_links
                return self.send(responsibility_links(self.app.store,RESPONSIBILITY_RULES))
            if path == "/api/recommendation":
                return self.send(self.app.recommendations.get(int(params["id"][0]), params))
            if path == "/api/image":
                record = self.app.store.image(int(params["id"][0]))
                content, mime = self.app.images.get(record, params.get("kind", ["preview"])[0], params)
                return self.send(content, mime)
            if path == "/api/export":
                return self.export(params)
            self.send({"error": "页面不存在"}, code=404)
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            pass
        except InterruptedError as exc:
            self.safe_error(exc, 409)
        except PermissionError as exc:
            self.safe_error(exc, 403)
        except (ValueError, KeyError, FileNotFoundError, OSError) as exc:
            self.safe_error(exc, 400)
        except Exception as exc:
            self.safe_error(exc, 500)

    def safe_error(self, exc, code):
        try:
            self.send({"error": str(exc)}, code=code)
        except (BrokenPipeError, ConnectionError, OSError):
            pass

    def do_POST(self):
        try:
            self.check_request(mutation=True)
            length = int(self.headers.get("Content-Length", "0"))
            if length > 2 * 1024 * 1024 or length < 2:
                raise ValueError("请求内容过大或不完整")
            body = json.loads(self.rfile.read(length))
            path = urlsplit(self.path).path
            store = self.app.store
            if path == "/api/root":
                if self.app.scanner.status["running"]:
                    raise ValueError("目录正在扫描，请稍后切换")
                root = store.set_root(self.app.paths.server(body["path"]))
                self.app.scanner.start(root)
                result = self.app.state()
            elif path == "/api/scan":
                self.app.scanner.start(store.current_root())
                result = {"started": True}
            elif path == "/api/view":
                self.app.images.view(body["client"], body["view"], body["focus"])
                result = {"ok": True}
            elif path == "/api/draft":
                result = store.save_draft(int(body["id"]), body["label"], bool(body.get("only_unclassified")), body.get("expected_draft"))
            elif path == "/api/commit":
                result = store.commit([(int(body["id"]), body["label"])], only_unclassified=bool(body.get("only_unclassified")), expected_draft=body.get("expected_draft"), review_run=body.get('review_run'))
            elif path == '/api/annotations':
                result = store.annotations.save(int(body['id']),body)
            elif path == '/api/review-batches/register':
                result = self.app.review_batches.register(body['batch_id'],body.get('proposal_ids'),body.get('metadata'))
            elif path == '/api/review-batches/worker':
                result = self.app.update_review_worker(body)
            elif path == '/api/training-rounds/record':
                result = self.app.training_rounds.record(body)
            elif path == '/api/automation/report':
                result = self.app.training_rounds.record_runtime(body)
            elif path == '/api/annotation-proposals/reject':
                result=store.annotations.reject_proposal(int(body['id']),body['proposal_id'],body)
            elif path == '/api/annotation-feedback':
                result=store.annotations.save_feedback(int(body['id']),body)
            elif path == "/api/batch":
                result = store.batch(body["ids"], body["label"])
            elif path == "/api/review/unrelated":
                if not isinstance(body.get('accept'),bool):raise ValueError('请选择确认或判错')
                result = store.review_unrelated(body['ids'],body['accept'])
            elif path == "/api/undo":
                result = store.undo()
            elif path == "/api/tag":
                result = store.add_tag(body["group"], body["name"])
            elif path == "/api/locate":
                record = store.image(int(body["id"]))
                file = store.path(record)
                if not file.is_file():
                    raise ValueError("原文件暂时无法访问，请检查 NAS 连接")
                if os.name == "nt":
                    subprocess.Popen(["explorer.exe", "/select," + str(file)])
                result = {"path": self.app.paths.client(file), "mode": "explorer" if os.name == "nt" else "copy"}
            elif path == "/api/import/preview":
                result = self.app.prepare_import(body["path"])
            elif path == "/api/import/apply":
                result = self.app.apply_import(body["key"])
            else:
                return self.send({"error": "接口不存在"}, code=404)
            self.send(result)
        except PermissionError as exc:
            self.safe_error(exc, 403)
        except (ValueError, KeyError, TypeError, OSError) as exc:
            self.safe_error(exc, 400)
        except Exception as exc:
            self.safe_error(exc, 500)

    def export(self, params):
        items = self.app.store.listing(params, export=True)["items"]
        root = self.app.store.current_root()
        rows = []
        for item in items:
            effective=item['display_label'] or {'defects':[], 'responsibilities':[], 'image_state':'', 'note':''}
            rows.append({"image_id": item["id"], "relative_path": item["relative_path"],
                         "case_id": item["case_id"], "attachment_id": item["attachment_id"],
                         "sha256": item["sha256"], "source_available": bool(item["available"]),
                         "classified": item["label"] is not None,
                         "classification_source":item['classification_source'],
                         "machine_model_id":(item['automatic_classification'] or {}).get('model_id'),
                         "reported_at": item["reported_at"], "date_source": item["date_source"],
                         **effective,
                         "updated_at": item["updated_at"]})
        kind = params.get("format", ["json"])[0]
        filename = "image-labels-" + dt.datetime.now().strftime("%Y%m%d-%H%M%S") + "." + kind
        if kind == "json":
            content = json.dumps({"schema_version": 1, "source_root": self.app.paths.client(root["path"]), "exported_at": now(),
                                  "items": rows}, ensure_ascii=False, indent=2).encode("utf-8")
            mime = "application/json; charset=utf-8"
        elif kind == "csv":
            output = io.StringIO(newline="")
            fields = ["image_id", "relative_path", "case_id", "attachment_id", "sha256", "source_available",
                      "classified", "classification_source", "machine_model_id", "reported_at", "date_source", "defects", "responsibilities",
                      "problem_responsibilities", "image_state", "note", "updated_at"]
            writer = csv.DictWriter(output, fieldnames=fields)
            writer.writeheader()
            for row in rows:
                safe = {k: (dumps(v) if isinstance(v, (list, dict)) else v) for k, v in row.items()}
                for key, value in safe.items():
                    if isinstance(value, str) and value.startswith(("=", "+", "-", "@", "\t", "\r")):
                        safe[key] = "'" + value
                writer.writerow(safe)
            content, mime = output.getvalue().encode("utf-8-sig"), "text/csv; charset=utf-8"
        else:
            raise ValueError("只支持 JSON 或 CSV")
        self.send(content, mime, headers={"Content-Disposition": 'attachment; filename="' + filename + '"'})


def make_server(port, data_dir, root=None, *, host="127.0.0.1", allowed_hosts=(), nas_root=None, recommendation_model=None, unrelated_review=False):
    server = ThreadingHTTPServer((host, port), Handler)
    server.daemon_threads = True
    server.allowed_hosts = {f"127.0.0.1:{server.server_port}", *allowed_hosts}
    server.app = Application(data_dir, root, nas_root, recommendation_model, unrelated_review)
    return server


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8878)
    parser.add_argument("--data-dir", type=Path, default=BASE / "data")
    parser.add_argument("--root")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--allow-host", action="append", default=[])
    parser.add_argument("--nas-root")
    parser.add_argument("--recommendation-model", type=Path)
    parser.add_argument("--unrelated-review", action='store_true')
    args = parser.parse_args()
    server = make_server(args.port, args.data_dir, args.root, host=args.host,
                         allowed_hosts=args.allow_host, nas_root=args.nas_root,
                         recommendation_model=args.recommendation_model,unrelated_review=args.unrelated_review)
    print(f"Image labeler: http://{args.host}:{server.server_port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
