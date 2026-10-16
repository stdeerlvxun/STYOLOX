"""Versioned localization supplements and model error review; never rewrite labels."""
from collections import Counter, defaultdict
import hashlib
import io
import json
import math
import datetime as dt
from PIL import Image, ImageOps


def fingerprint(value):
    return hashlib.sha256((value or '').encode('utf-8')).hexdigest()


class Annotations:
    def __init__(self, store):
        self.store = store
        with store.lock, store.db:
            store.db.executescript('''
                CREATE TABLE IF NOT EXISTS annotation_documents(
                  image_id INTEGER PRIMARY KEY REFERENCES images(id),data TEXT NOT NULL,
                  version INTEGER NOT NULL,updated_at TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS annotation_history(
                  id INTEGER PRIMARY KEY,image_id INTEGER NOT NULL,version INTEGER NOT NULL,
                  before_data TEXT,after_data TEXT NOT NULL,created_at TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS error_reviews(
                  image_id INTEGER NOT NULL REFERENCES images(id),run_id TEXT NOT NULL,
                  data TEXT NOT NULL,status TEXT NOT NULL DEFAULT 'pending',reviewed_at TEXT,
                  PRIMARY KEY(image_id,run_id));
            ''')

    def source(self, image_id):
        image = self.store.image(image_id)
        if image['root_id'] != self.store.current_root()['id']:
            raise ValueError('图片不属于当前目录')
        if not image['available']:raise ValueError('原图暂不可用')
        path = self.store.path(image);before = path.stat();raw = path.read_bytes();after = path.stat()
        if (before.st_size,before.st_mtime_ns)!=(after.st_size,after.st_mtime_ns):
            raise ValueError('原图读取期间发生变化，请重试')
        with Image.open(io.BytesIO(raw)) as original:
            width,height = ImageOps.exif_transpose(original).size
        return {'sha256':hashlib.sha256(raw).hexdigest(),'width':width,'height':height,
                'size':after.st_size,'mtime_ns':after.st_mtime_ns}

    def review(self, image_id):
        row = self.store.one("SELECT e.* FROM error_reviews e WHERE e.image_id=? AND "
            "e.run_id=(SELECT value FROM settings WHERE key='error_review_run')",(image_id,))
        return {**json.loads(row['data']),'status':row['status'],'run_id':row['run_id']} if row else None

    def detail(self, image_id):
        source = self.source(image_id)
        row = self.store.one('SELECT * FROM annotation_documents WHERE image_id=?',(image_id,))
        label = self.store.one('SELECT data FROM labels WHERE image_id=?',(image_id,))
        data = json.loads(row['data']) if row else {'boxes':[],'status':'draft','outcome':'boxed'}
        return {'image_id':image_id,'source':source,'document':data,'version':row['version'] if row else 0,
                'source_changed':bool(row and data['source']['sha256']!=source['sha256']),
                'label_fingerprint':fingerprint(label['data'] if label else ''),'review':self.review(image_id)}

    def save(self, image_id, body):
        source = self.source(image_id)
        if body.get('source_sha256')!=source['sha256']:raise ValueError('原图已变化，请重新读取后补标')
        version = body.get('expected_version')
        if type(version) is not int or version<0:raise ValueError('补标版本不正确')
        status,outcome = body.get('status'),body.get('outcome','boxed')
        if status not in ('draft','confirmed') or outcome not in ('boxed','labels_correct','needs_correction','unclear'):
            raise ValueError('请选择补标和复核结果')
        boxes = body.get('boxes')
        if not isinstance(boxes,list) or len(boxes)>100:raise ValueError('每张图片最多100个问题框')
        known = set(self.store.tags()['defect']);clean=[];identities=set()
        for box in boxes:
            if not isinstance(box,dict):raise ValueError('问题框格式不正确')
            identity,problem = box.get('id'),box.get('problem')
            if not isinstance(identity,str) or not identity or len(identity)>80 or identity in identities:
                raise ValueError('问题框标识不正确')
            if problem not in known:raise ValueError('请为问题框选择已有问题点')
            values=[box.get(k) for k in ('x','y','w','h')]
            if any(type(v) not in (int,float) or not math.isfinite(v) for v in values):raise ValueError('框坐标不正确')
            x,y,w,h=values
            if min(x,y)<0 or min(w,h)<=0 or x+w>1.000001 or y+h>1.000001:raise ValueError('问题框必须位于原图内')
            clean.append(dict(id=identity,problem=problem,x=x,y=y,w=min(w,1-x),h=min(h,1-y)))
            identities.add(identity)
        if status=='confirmed' and outcome=='boxed' and not clean:raise ValueError('请框选问题，或选择其他复核结果')
        with self.store.lock,self.store.db:
            row=self.store.one('SELECT * FROM annotation_documents WHERE image_id=?',(image_id,))
            if (row['version'] if row else 0)!=version:raise ValueError('补标已被其他页面更新，请重新读取；原结果未覆盖')
            label=self.store.one('SELECT data FROM labels WHERE image_id=?',(image_id,))
            label_hash=fingerprint(label['data'] if label else '')
            if status=='confirmed' and body.get('label_fingerprint')!=label_hash:
                raise ValueError('人工分类已更新，请重新读取后确认补标')
            document={'source':source,'coordinate_space':'exif_transposed_normalized_xywh',
                'boxes':clean,'status':status,'outcome':outcome,'label_fingerprint':label_hash,
                'review_run':body.get('review_run')}
            raw=json.dumps(document,ensure_ascii=False,separators=(',',':'))
            stamp=dt.datetime.now(dt.timezone.utc).isoformat();next_version=version+1
            self.store.db.execute('INSERT OR REPLACE INTO annotation_documents VALUES(?,?,?,?)',
                (image_id,raw,next_version,stamp))
            self.store.db.execute('INSERT INTO annotation_history(image_id,version,before_data,after_data,created_at) VALUES(?,?,?,?,?)',
                (image_id,next_version,row['data'] if row else None,raw,stamp))
            if status=='confirmed':
                self.store.db.execute("UPDATE error_reviews SET status='reviewed',reviewed_at=? WHERE image_id=? AND run_id=?",
                    (stamp,image_id,body.get('review_run')))
        return {'saved':True,'version':next_version,'document':document,'labels_unchanged':True}

    def export(self):
        rows=self.store.query('SELECT a.*,i.relative_path,l.data label_data,d.image_id has_draft FROM annotation_documents a '
            'JOIN images i ON i.id=a.image_id LEFT JOIN labels l ON l.image_id=i.id '
            'LEFT JOIN drafts d ON d.image_id=i.id WHERE i.root_id=?',(self.store.current_root()['id'],))
        result=[]
        for row in rows:
            document=json.loads(row['data']);label=json.loads(row['label_data']) if row['label_data'] else None
            eligible=bool(label and label.get('image_state')=='defect' and label.get('defects') and not row['has_draft']
                and document['status']=='confirmed' and document['outcome']=='boxed'
                and document['label_fingerprint']==fingerprint(row['label_data']))
            result.append({'image_id':row['image_id'],'relative_path':row['relative_path'],'version':row['version'],
                'annotation':document,'original_classification':label,'training_eligible_by_label_policy':eligible})
        return {'schema_version':1,'coordinate_space':'exif_transposed_normalized_xywh','source_images_copied':False,
                'source_hash_must_be_reverified_before_training':True,'items':result}


def responsibility_links(store, approved_rules):
    """Count each after-sales case once per problem; never infer responsibility from pixels."""
    groups=defaultdict(lambda:defaultdict(set));images=Counter()
    rows=store.query('SELECT i.id,i.case_id,l.data FROM labels l JOIN images i ON i.id=l.image_id '
        'LEFT JOIN drafts d ON d.image_id=i.id WHERE i.root_id=? AND d.image_id IS NULL',
        (store.current_root()['id'],))
    for row in rows:
        label=json.loads(row['data'])
        if label.get('image_state')!='defect':continue
        for problem in label.get('defects',[]):
            images[problem]+=1;mapping=label.get('problem_responsibilities')
            roles=mapping.get(problem,[]) if isinstance(mapping,dict) else label.get('responsibilities',[]) if len(label['defects'])==1 else []
            groups[problem][row['case_id'] or 'image:'+str(row['id'])].update(r for r in roles if r!='待核')
    result={}
    for problem in store.tags()['defect']:
        cases=groups[problem];resolved=[v for v in cases.values() if v];counts=Counter(r for v in resolved for r in v)
        ranked=counts.most_common();suggestion=approved_rules.get(problem);basis='approved_rule' if suggestion else None
        if not suggestion and problem=='其他破损':suggestion='运输';basis='user_experience_prior'
        if not suggestion and ranked and len(resolved)>=5 and ranked[0][1]/len(resolved)>=.8:
            suggestion=ranked[0][0];basis='observed_case_association'
        result[problem]={'suggestion':suggestion,'basis':basis,'resolved_cases':len(resolved),
            'unresolved_cases':sum(not v for v in cases.values()),'image_count':images[problem],
            'responsibility_case_counts':dict(counts),'requires_confirmation':True,
            'suggested_case_share':counts[suggestion]/len(resolved) if suggestion and resolved else None}
    return result
