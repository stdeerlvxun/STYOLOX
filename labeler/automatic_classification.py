"""Durable machine classifications; human labels and boxes always take precedence."""
import json
import time


class AutomaticClassifications:
    def __init__(self, store):
        self.store = store
        with store.lock, store.db:
            store.db.execute('''CREATE TABLE IF NOT EXISTS automatic_classifications(
                image_id INTEGER PRIMARY KEY REFERENCES images(id), model_id TEXT NOT NULL,
                data TEXT NOT NULL, status TEXT NOT NULL, source_size INTEGER NOT NULL,
                source_mtime INTEGER NOT NULL, source_sha256 TEXT, updated_at REAL NOT NULL,
                retry_after REAL NOT NULL DEFAULT 0, error TEXT NOT NULL DEFAULT '')''')

    @staticmethod
    def match(alias='m'):
        return (f"{alias}.status='ready' AND {alias}.source_size=i.size "
                f"AND {alias}.source_mtime=i.mtime_ns AND i.available=1")

    def record(self, image, result, stamp, rules, error=''):
        ready = result.get('status') == 'ready' and not error
        candidates = result.get('candidates', []) if ready else []
        unrelated = bool(result.get('image_state_suggestion')) if ready else False
        defects = list(dict.fromkeys(c['name'] for c in candidates)) if not unrelated else []
        roles = {name: [rules.get(name, '待核')] for name in defects}
        label = {'defects': defects, 'responsibilities': list(dict.fromkeys(
            role for values in roles.values() for role in values)) or ['待核'],
            'problem_responsibilities': roles, 'note': '',
            'image_state': 'unrelated' if unrelated else 'defect' if defects else ''}
        payload = {'label': label, 'outcome': 'unrelated' if unrelated else 'problems' if defects else 'unresolved',
            'candidates': candidates, 'source': 'machine', 'model_id': result['model_id'],
            'probabilities_calibrated': False, 'state_suggestion': result.get('image_state_suggestion'),
            'training_eligible': False, 'human_confirmation_blocks_automation': False}
        store = self.store
        with store.lock, store.db:
            if store.one('SELECT image_id FROM labels WHERE image_id=? UNION ALL '
                         'SELECT image_id FROM drafts WHERE image_id=?', (image['id'], image['id'])):
                return False
            if store.one("SELECT image_id FROM annotation_documents WHERE image_id=? "
                         "AND json_extract(data,'$.status')='confirmed'", (image['id'],)):
                return False
            current = store.image(image['id'])
            if not current['available'] or (current['size'], current['mtime_ns']) != tuple(stamp):
                return False
            store.db.execute('''INSERT INTO automatic_classifications VALUES(?,?,?,?,?,?,?,?,?,?)
                ON CONFLICT(image_id) DO UPDATE SET model_id=excluded.model_id,data=excluded.data,
                status=excluded.status,source_size=excluded.source_size,source_mtime=excluded.source_mtime,
                source_sha256=excluded.source_sha256,updated_at=excluded.updated_at,
                retry_after=excluded.retry_after,error=excluded.error''',
                (image['id'], result['model_id'], json.dumps(payload, ensure_ascii=False),
                 'ready' if ready else 'error', stamp[0], stamp[1], result.get('source_sha256'),
                 time.time(), 0 if ready else time.time()+300, str(error)[:500]))
        return True

    def current(self, image):
        row = self.store.one('SELECT m.* FROM automatic_classifications m JOIN images i ON i.id=m.image_id '
                             'WHERE i.id=? AND '+self.match(), (image['id'],))
        if not row:
            return None
        return {**json.loads(row['data']), 'updated_at': row['updated_at'], 'model_id': row['model_id']}

    def next_image(self, model_id):
        return self.store.one('''SELECT i.* FROM images i LEFT JOIN labels l ON l.image_id=i.id
            LEFT JOIN drafts d ON d.image_id=i.id
            LEFT JOIN automatic_classifications m ON m.image_id=i.id
            WHERE i.root_id=? AND i.available=1 AND l.image_id IS NULL AND d.image_id IS NULL
            AND NOT EXISTS(SELECT 1 FROM annotation_documents a WHERE a.image_id=i.id
                AND json_extract(a.data,'$.status')='confirmed')
            AND (m.image_id IS NULL OR m.source_size<>i.size OR m.source_mtime<>i.mtime_ns
                OR m.model_id<>? OR (m.status='error' AND m.retry_after<=?))
            ORDER BY i.reported_at DESC,i.id DESC LIMIT 1''',
            (self.store.current_root()['id'], model_id, time.time()))

    def counts(self):
        row = self.store.one('''SELECT SUM(CASE WHEN '''+self.match()+''' THEN 1 ELSE 0 END) completed,
            SUM(CASE WHEN '''+self.match()+''' AND json_extract(m.data,'$.outcome')='unresolved' THEN 1 ELSE 0 END) unresolved,
            SUM(m.status='error') failed,
            SUM(CASE WHEN m.image_id IS NULL OR m.source_size<>i.size OR m.source_mtime<>i.mtime_ns
                THEN 1 ELSE 0 END) waiting
            FROM images i LEFT JOIN automatic_classifications m ON m.image_id=i.id
            LEFT JOIN labels l ON l.image_id=i.id WHERE i.root_id=? AND i.available=1 AND l.image_id IS NULL
            AND NOT EXISTS(SELECT 1 FROM drafts d WHERE d.image_id=i.id)
            AND NOT EXISTS(SELECT 1 FROM annotation_documents a WHERE a.image_id=i.id
                AND json_extract(a.data,'$.status')='confirmed')''',
            (self.store.current_root()['id'],))
        return {key: value or 0 for key, value in row.items()}
