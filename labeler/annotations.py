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
                CREATE TABLE IF NOT EXISTS annotation_proposals(
                  image_id INTEGER PRIMARY KEY REFERENCES images(id),proposal_id TEXT UNIQUE NOT NULL,
                  data TEXT NOT NULL,label_data TEXT NOT NULL,status TEXT NOT NULL DEFAULT 'pending',
                  created_at TEXT NOT NULL,reviewed_at TEXT);
            ''')

    def proposal_match(self, alias='p', include_annotations=False):
        # Raw classification identity and file version invalidate stale suggestions without hashing all NAS files.
        condition=(f"{alias}.status='pending' AND {alias}.label_data=l.data AND d.image_id IS NULL AND i.available=1 "
            "AND COALESCE(json_extract(l.data,'$.image_state'),'') IN ('','defect') "
            "AND json_array_length(json_extract(l.data,'$.defects'))>0 "
            f"AND i.size=json_extract({alias}.data,'$.source.size') AND i.mtime_ns=json_extract({alias}.data,'$.source.mtime_ns') ")
        if not include_annotations:
            condition+=("AND NOT EXISTS (SELECT 1 FROM annotation_documents a WHERE a.image_id=i.id AND "
                f"(json_extract(a.data,'$.status')='confirmed' OR COALESCE(json_extract(a.data,'$.proposal_id'),'')!={alias}.proposal_id))")
        return condition

    def proposal_state(self):
        root=self.store.current_root()['id']
        rows=self.store.query('SELECT p.status,COUNT(*) count FROM annotation_proposals p JOIN images i ON i.id=p.image_id '
            'WHERE i.root_id=? GROUP BY p.status',(root,))
        counts={name:0 for name in ('pending','accepted','rejected','stale')}
        counts.update({r['status']:r['count'] for r in rows})
        pending=self.store.one('SELECT COUNT(*) count FROM annotation_proposals p JOIN images i ON i.id=p.image_id '
            'LEFT JOIN labels l ON l.image_id=i.id LEFT JOIN drafts d ON d.image_id=i.id WHERE i.root_id=? AND '+
            self.proposal_match(),(root,))['count']
        return {**counts,'total':sum(counts.values()),'pending':pending,'stale':counts['stale']+counts['pending']-pending}

    def proposal(self, image_id, source=None, include_annotations=False):
        row=self.store.one('SELECT p.* FROM annotation_proposals p JOIN images i ON i.id=p.image_id '
            'LEFT JOIN labels l ON l.image_id=i.id LEFT JOIN drafts d ON d.image_id=i.id '
            'WHERE i.id=? AND i.root_id=? AND '+self.proposal_match(include_annotations=include_annotations),(image_id,self.store.current_root()['id']))
        if not row:return None
        data=json.loads(row['data'])
        if source is not None and data['source']['sha256']!=source['sha256']:
            with self.store.lock,self.store.db:
                self.store.db.execute("UPDATE annotation_proposals SET status='stale' WHERE image_id=? AND proposal_id=? AND status='pending'",(image_id,row['proposal_id']))
            return None
        return {**data,'proposal_id':row['proposal_id'],'status':row['status'],
                'source_sha256':data['source']['sha256'],'requires_human_review':True}

    def install_proposals(self, manifest):
        if manifest.get('schema_version')!=1 or manifest.get('requires_human_review') is not True:
            raise ValueError('自动框必须标记为待人工审核')
        model_id,method=manifest.get('model_id'),manifest.get('method')
        if any(not isinstance(v,str) or not v or len(v)>160 for v in (model_id,method)):
            raise ValueError('自动框模型来源不正确')
        items=manifest.get('items')
        if not isinstance(items,list) or len(items)>10000:raise ValueError('自动框清单格式不正确')
        ids=[item.get('image_id') for item in items if isinstance(item,dict)]
        if len(ids)!=len(items) or any(type(v) is not int for v in ids) or len(set(ids))!=len(ids):raise ValueError('自动框图片标识重复或格式不正确')
        prepared=[];skipped=0
        for item in items:
            try:
                image_id=item['image_id']
                if type(image_id) is not int:raise ValueError('图片标识不正确')
                image=self.store.image(image_id)
                if image['relative_path']!=item['relative_path']:raise ValueError('原图路径不一致')
                label=self.store.one('SELECT data FROM labels WHERE image_id=?',(image_id,))
                value=json.loads(label['data']) if label else {}
                if (not label or value.get('image_state','') not in ('','defect') or not value.get('defects')
                    or self.store.one('SELECT image_id FROM drafts WHERE image_id=?',(image_id,))
                    or self.store.one('SELECT image_id FROM annotation_documents WHERE image_id=?',(image_id,))):
                    raise ValueError('已有人工编辑或图片状态，保留原数据')
                if item['label_fingerprint']!=fingerprint(label['data']):raise ValueError('人工分类已变化')
                source=self.source(image_id)
                if any(item['source'][k]!=source[k] for k in ('sha256','width','height')):
                    raise ValueError('自动框原图不一致')
                boxes=item.get('boxes')
                if not isinstance(boxes,list) or not boxes or len(boxes)>100:raise ValueError('没有有效自动框')
                clean=[];box_ids=set()
                for box in boxes:
                    identity=box.get('id');problem=box.get('problem');values=[box.get(k) for k in ('x','y','w','h')]
                    if (box.get('origin')!='model' or not isinstance(identity,str) or not identity or len(identity)>80 or identity in box_ids
                        or problem not in value['defects'] or problem not in self.store.tags()['defect']):
                        raise ValueError('框类型或标识不正确')
                    if any(type(v) not in (int,float) or not math.isfinite(v) for v in values):raise ValueError('框坐标不正确')
                    x,y,w,h=values
                    if min(x,y)<0 or min(w,h)<=0 or x+w>1.000001 or y+h>1.000001:raise ValueError('框超出原图')
                    score=box.get('score')
                    if score is not None and (type(score) not in (int,float) or not math.isfinite(score)):raise ValueError('分数不正确')
                    clean.append({'id':identity,'problem':problem,'x':x,'y':y,'w':min(w,1-x),'h':min(h,1-y),
                                  'score':score,'origin':'model'})
                    box_ids.add(identity)
                data={'model_id':model_id,'method':method,'source':source,'label_fingerprint':item['label_fingerprint'],
                      'boxes':clean,'note':str(item.get('note','粗定位建议，需人工逐框审核。'))[:1000],
                      'source_checkpoint_sha256':manifest.get('source_checkpoint_sha256'),
                      'coordinate_space':'exif_transposed_normalized_xywh','requires_human_review':True}
                identity=fingerprint(json.dumps(data,sort_keys=True,ensure_ascii=False))[:32]
                for box in clean:box['proposal_id']=identity
                prepared.append((image_id,identity,data,label['data']))
            except (ValueError,KeyError,TypeError,OSError):
                skipped+=1
        imported=0
        with self.store.lock,self.store.db:
            self.store.db.execute('BEGIN IMMEDIATE')
            from deployment.snapshot import digest,annotation_digest
            before=digest(self.store.db);annotations_before=annotation_digest(self.store.db)
            for image_id,identity,data,label_raw in prepared:
                current=self.store.one('SELECT data FROM labels WHERE image_id=?',(image_id,))
                if (not current or current['data']!=label_raw
                    or self.store.one('SELECT image_id FROM drafts WHERE image_id=?',(image_id,))
                    or self.store.one('SELECT image_id FROM annotation_documents WHERE image_id=?',(image_id,))
                    or self.store.one('SELECT image_id FROM annotation_proposals WHERE image_id=?',(image_id,))):
                    skipped+=1;continue
                self.store.db.execute('INSERT INTO annotation_proposals(image_id,proposal_id,data,label_data,created_at) VALUES(?,?,?,?,?)',
                    (image_id,identity,json.dumps(data,ensure_ascii=False,separators=(',',':')),label_raw,dt.datetime.now(dt.timezone.utc).isoformat()))
                imported+=1
            assert before==digest(self.store.db),'Human classifications changed'
            assert annotations_before==annotation_digest(self.store.db),'Human position annotations changed'
        return {'imported':imported,'skipped':skipped,'human_classifications_unchanged':True,'human_annotations_unchanged':True,
                'human_digest':before,'annotation_digest':annotations_before}

    def reject_proposal(self, image_id, proposal_id):
        proposal=self.proposal(image_id,self.source(image_id))
        if not proposal or proposal['proposal_id']!=proposal_id:raise ValueError('建议已失效或已审核，请重新读取')
        with self.store.lock,self.store.db:
            self.store.db.execute("UPDATE annotation_proposals SET status='rejected',reviewed_at=? WHERE image_id=? AND proposal_id=? AND status='pending'",
                (dt.datetime.now(dt.timezone.utc).isoformat(),image_id,proposal_id))
        return {'rejected':True,'labels_unchanged':True,'annotations_unchanged':True}

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
                'label_fingerprint':fingerprint(label['data'] if label else ''),'review':self.review(image_id),
                'proposal':self.proposal(image_id,source)}

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
        proposal_id=body.get('proposal_id')
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
            item=dict(id=identity,problem=problem,x=x,y=y,w=min(w,1-x),h=min(h,1-y))
            if box.get('origin')=='model':
                if not proposal_id or box.get('proposal_id')!=proposal_id:raise ValueError('模型框来源不正确')
                item.update(origin='model',proposal_id=proposal_id)
            clean.append(item)
            identities.add(identity)
        if status=='confirmed' and outcome=='boxed' and not clean:raise ValueError('请框选问题，或选择其他复核结果')
        with self.store.lock,self.store.db:
            row=self.store.one('SELECT * FROM annotation_documents WHERE image_id=?',(image_id,))
            if (row['version'] if row else 0)!=version:raise ValueError('补标已被其他页面更新，请重新读取；原结果未覆盖')
            if proposal_id:
                previous=json.loads(row['data']) if row else None
                proposal=self.proposal(image_id,source,include_annotations=True)
                if not proposal and previous and previous.get('proposal_id')==proposal_id:
                    accepted=self.store.one("SELECT data FROM annotation_proposals WHERE image_id=? AND proposal_id=? AND status='accepted'",(image_id,proposal_id))
                    if accepted:
                        data=json.loads(accepted['data'])
                        if data['source']['sha256']==source['sha256']:proposal={**data,'proposal_id':proposal_id}
                if not proposal or proposal['proposal_id']!=proposal_id:raise ValueError('自动框建议已变化或已审核，请重新读取')
                if previous and previous.get('proposal_id')!=proposal_id:
                    current_boxes={box['id']:box for box in clean}
                    if any(current_boxes.get(box['id'])!=box for box in previous['boxes']):
                        raise ValueError('已有独立人工补标，自动框不能覆盖原结果')
            label=self.store.one('SELECT data FROM labels WHERE image_id=?',(image_id,))
            label_hash=fingerprint(label['data'] if label else '')
            if status=='confirmed' and body.get('label_fingerprint')!=label_hash:
                raise ValueError('人工分类已更新，请重新读取后确认补标')
            document={'source':source,'coordinate_space':'exif_transposed_normalized_xywh',
                'boxes':clean,'status':status,'outcome':outcome,'label_fingerprint':label_hash,
                'review_run':body.get('review_run')}
            if proposal_id:
                document.update(proposal_id=proposal_id,model_id=proposal['model_id'],method=proposal['method'],
                    origin='machine_assisted_human_confirmed' if status=='confirmed' else 'machine_assisted_draft')
            raw=json.dumps(document,ensure_ascii=False,separators=(',',':'))
            stamp=dt.datetime.now(dt.timezone.utc).isoformat();next_version=version+1
            self.store.db.execute('INSERT OR REPLACE INTO annotation_documents VALUES(?,?,?,?)',
                (image_id,raw,next_version,stamp))
            self.store.db.execute('INSERT INTO annotation_history(image_id,version,before_data,after_data,created_at) VALUES(?,?,?,?,?)',
                (image_id,next_version,row['data'] if row else None,raw,stamp))
            if status=='confirmed':
                if proposal_id:
                    self.store.db.execute('UPDATE annotation_proposals SET status=?,reviewed_at=? WHERE image_id=? AND proposal_id=?',
                        ('accepted' if outcome=='boxed' and clean else 'rejected',stamp,image_id,proposal_id))
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
