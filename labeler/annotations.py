"""Versioned localization supplements and explicit, append-only box classification."""
from collections import Counter, defaultdict
import hashlib
import io
import json
import math
import datetime as dt
import re
from PIL import Image, ImageOps


def fingerprint(value):
    return hashlib.sha256((value or '').encode('utf-8')).hexdigest()


def prediction_metadata(box):
    """Validate genuine model type provenance; inherited labels are not predictions."""
    provenance=box.get('type_provenance')
    if provenance is None:
        if box.get('problem_source')=='model_prediction':raise ValueError('问题类型缺少模型来源')
        return {}
    if not isinstance(provenance,dict):raise ValueError('问题类型模型来源不正确')
    source=provenance.get('source');sha=provenance.get('model_sha256');score=provenance.get('score')
    if source not in ('detector_class_prediction','image_classifier_prediction'):
        raise ValueError('问题类型模型来源不正确')
    if not isinstance(sha,str) or not re.fullmatch('[0-9a-fA-F]{64}',sha):raise ValueError('问题类型模型指纹不正确')
    if type(score) not in (int,float) or not math.isfinite(score):raise ValueError('问题类型模型分数不正确')
    if source=='detector_class_prediction' and not 0<=score<=1:raise ValueError('问题类型模型分数不正确')
    if source=='image_classifier_prediction' and provenance.get('scope')!='whole_image_context':
        raise ValueError('图级问题类型必须注明辅助范围')
    clean={'source':source,'model_sha256':sha.lower(),'score':score,
        'scope':'whole_image_context' if source=='image_classifier_prediction' else 'box'}
    if 'match_iou' in provenance:
        overlap=provenance['match_iou']
        if type(overlap) not in (int,float) or not math.isfinite(overlap) or not 0<=overlap<=1:
            raise ValueError('问题类型匹配位置不正确')
        clean['match_iou']=overlap
    return {'type_provenance':clean,'problem_source':'model_prediction','problem_model_id':sha.lower(),
        'type_score':score,'type_score_kind':'raw_classifier_score' if source=='image_classifier_prediction' else 'detector_score'}


def clean_negative_regions(regions):
    if not isinstance(regions,list) or len(regions)>100:raise ValueError('每张图片最多100个纠错区域')
    result=[];identities=set()
    for region in regions:
        if not isinstance(region,dict):raise ValueError('纠错区域格式不正确')
        identity=region.get('id');reason=region.get('reason')
        if not isinstance(identity,str) or not identity or len(identity)>80 or identity in identities:
            raise ValueError('纠错区域标识不正确')
        if reason not in ('process_hole','normal_structure','false_positive'):raise ValueError('纠错原因不正确')
        values=[region.get(k) for k in ('x','y','w','h')]
        if any(type(v) not in (int,float) or not math.isfinite(v) for v in values):raise ValueError('纠错坐标不正确')
        x,y,w,h=values
        if min(x,y)<0 or min(w,h)<=0 or x+w>1.000001 or y+h>1.000001:raise ValueError('纠错区域必须位于原图内')
        item=dict(id=identity,x=x,y=y,w=min(w,1-x),h=min(h,1-y),reason=reason)
        source_id=region.get('source_box_id')
        if source_id is not None:
            if not isinstance(source_id,str) or not source_id or len(source_id)>80:raise ValueError('原建议框标识不正确')
            item['source_box_id']=source_id
        if 'normal_provenance' in region:
            provenance=region['normal_provenance']
            action={'process_hole':'process_hole','normal_structure':'normal_structure','false_positive':'confirmed_false_positive'}[reason]
            if (not isinstance(provenance,dict) or provenance.get('source')!='human_explicit_normal'
                or provenance.get('action')!=action or provenance.get('confirmed') is not True):
                raise ValueError('正常区域必须由人工明确确认，纠错原因与确认操作需一致')
            item['normal_provenance']={'source':'human_explicit_normal','action':action,'confirmed':True}
        result.append(item);identities.add(identity)
    return result


def reject_conflicting_regions(boxes, regions):
    for positive in boxes:
        for negative in regions:
            width=max(0,min(positive['x']+positive['w'],negative['x']+negative['w'])-max(positive['x'],negative['x']))
            height=max(0,min(positive['y']+positive['h'],negative['y']+negative['h'])-max(positive['y'],negative['y']))
            if width*height/min(positive['w']*positive['h'],negative['w']*negative['h'])>=.5:
                raise ValueError('问题框与正常结构纠错区域明显重叠，请调整后确认')


def normal_confirmation(region):
    """Never upgrade an old generic delete into an explicit normal observation."""
    if region.get('normal_provenance',{}).get('source')=='human_explicit_normal':return 'explicit_ui_action'
    if region.get('reason') in ('process_hole','normal_structure'):return 'legacy_named_reason'
    return None


def correction_events(proposal, boxes, regions):
    original={box['id']:box for box in proposal.get('boxes',[])}
    final={box.get('source_box_id',box['id']):box for box in boxes};negative={r.get('source_box_id'):r for r in regions}
    result=[]
    for identity,box in original.items():
        current=final.get(identity)
        if current is None:
            region=negative.get(identity,{})
            confirmed_normal=normal_confirmation(region)
            # An overlap only links a possible replacement; it never creates a negative ROI.
            replacements=[]
            if not confirmed_normal:
                for other_id,other in final.items():
                    if other_id in original:continue
                    width=max(0,min(box['x']+box['w'],other['x']+other['w'])-max(box['x'],other['x']))
                    height=max(0,min(box['y']+box['h'],other['y']+other['h'])-max(box['y'],other['y']))
                    if width*height/min(box['w']*box['h'],other['w']*other['h'])>=.3:replacements.append(other['id'])
            result.append({'kind':'deleted','source_box_id':identity,'before':box,
                'reason':region.get('reason','not_confirmed_as_defect'),
                'correction_semantics':['normal' if confirmed_normal else 'replacement' if replacements else 'ambiguous_delete'],
                'normal_confirmation':confirmed_normal,
                **({'normal_provenance':region['normal_provenance']} if 'normal_provenance' in region else {}),
                **({'replacement_box_ids':replacements,'replacement_inferred':True} if replacements else {})})
            continue
        changes=[]
        if current['problem']!=box['problem']:changes.append('reclassified' if box['problem'] else 'assigned_problem')
        if any(abs(current[k]-box[k])>1e-6 for k in ('x','y','w','h')):changes.append('moved')
        result.append({'kind':'corrected' if changes else 'accepted','source_box_id':identity,
            'changes':changes,'correction_semantics':(['geometry'] if 'moved' in changes else [])+
                (['type'] if any(c in changes for c in ('reclassified','assigned_problem')) else []),
            'before':box,'after':current})
    result.extend({'kind':'added','correction_semantics':['positive_addition'],'after':box} for identity,box in final.items() if identity not in original)
    return result


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
                CREATE TABLE IF NOT EXISTS annotation_proposal_history(
                  id INTEGER PRIMARY KEY,image_id INTEGER NOT NULL,proposal_id TEXT NOT NULL,
                  data TEXT NOT NULL,label_data TEXT NOT NULL,status TEXT NOT NULL,
                  created_at TEXT NOT NULL,reviewed_at TEXT,archived_at TEXT NOT NULL,replaced_by TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS annotation_feedback(
                  id INTEGER PRIMARY KEY,image_id INTEGER NOT NULL REFERENCES images(id),
                  identity TEXT UNIQUE NOT NULL,data TEXT NOT NULL,created_at TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS annotation_feedback_training(
                  id INTEGER PRIMARY KEY,feedback_id INTEGER NOT NULL REFERENCES annotation_feedback(id),
                  run_id TEXT NOT NULL,model_sha256 TEXT NOT NULL,created_at TEXT NOT NULL,
                  UNIQUE(feedback_id,run_id));
                CREATE INDEX IF NOT EXISTS annotation_history_image ON annotation_history(image_id);
                CREATE INDEX IF NOT EXISTS annotation_feedback_image ON annotation_feedback(image_id);
                CREATE INDEX IF NOT EXISTS annotation_proposal_history_image_status ON annotation_proposal_history(image_id,status);
            ''')

    def proposal_match(self, alias='p', include_annotations=False):
        # Raw classification identity and file version invalidate stale suggestions without hashing all NAS files.
        condition=(f"{alias}.status='pending' AND {alias}.label_data=COALESCE(l.data,'') AND d.image_id IS NULL AND i.available=1 "
            "AND COALESCE(json_extract(l.data,'$.image_state'),'') IN ('','defect') "
            "AND NOT EXISTS (SELECT 1 FROM annotation_history ah WHERE ah.image_id=i.id AND json_extract(ah.after_data,'$.status')='confirmed') "
            "AND NOT EXISTS (SELECT 1 FROM annotation_feedback af WHERE af.image_id=i.id AND json_extract(af.data,'$.status')='confirmed') "
            "AND NOT EXISTS (SELECT 1 FROM error_reviews er WHERE er.image_id=i.id AND er.status='reviewed') "
            "AND NOT EXISTS (SELECT 1 FROM annotation_proposal_history ph WHERE ph.image_id=i.id AND ph.status IN ('accepted','rejected')) "
            f"AND i.size=json_extract({alias}.data,'$.source.size') AND i.mtime_ns=json_extract({alias}.data,'$.source.mtime_ns') ")
        if not include_annotations:
            condition+=("AND NOT EXISTS (SELECT 1 FROM annotation_documents a WHERE a.image_id=i.id AND "
                f"(json_extract(a.data,'$.status')='confirmed' OR COALESCE(json_extract(a.data,'$.proposal_id'),'')!={alias}.proposal_id))")
        return condition

    def _has_human_review(self, image_id):
        """A model version must never reopen a person's completed review or old position work."""
        queries=(
            'SELECT 1 FROM annotation_history WHERE image_id=? LIMIT 1',
            "SELECT 1 FROM annotation_feedback WHERE image_id=? AND json_extract(data,'$.status')='confirmed' LIMIT 1",
            "SELECT 1 FROM error_reviews WHERE image_id=? AND status='reviewed' LIMIT 1",
            "SELECT 1 FROM annotation_proposal_history WHERE image_id=? AND status IN ('accepted','rejected') LIMIT 1",
        )
        return any(self.store.one(query,(image_id,)) for query in queries)

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
                'source_sha256':data['source']['sha256'],'requires_human_review':True,
                'content_fingerprint':fingerprint(row['data'])}

    def install_proposals(self, manifest, refresh_pending=False):
        if manifest.get('enrichment_only') is True:return self.enrich_proposals(manifest)
        refresh_pending=manifest.get('refresh_pending',refresh_pending)
        if type(refresh_pending) is not bool:raise ValueError('刷新建议参数不正确')
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
                replaces=item.get('replaces_proposal_id')
                if replaces is not None and (not refresh_pending or not isinstance(replaces,str) or not replaces or len(replaces)>80):
                    raise ValueError('需要指定待替换建议版本')
                if type(image_id) is not int:raise ValueError('图片标识不正确')
                image=self.store.image(image_id)
                if image['relative_path']!=item['relative_path']:raise ValueError('原图路径不一致')
                label=self.store.one('SELECT data FROM labels WHERE image_id=?',(image_id,))
                value=json.loads(label['data']) if label else {}
                if (value.get('image_state','') not in ('','defect')
                    or self.store.one('SELECT image_id FROM drafts WHERE image_id=?',(image_id,))
                    or self.store.one('SELECT image_id FROM annotation_documents WHERE image_id=?',(image_id,))
                    or self._has_human_review(image_id)):
                    raise ValueError('已有人工编辑或图片状态，保留原数据')
                label_raw=label['data'] if label else ''
                if item['label_fingerprint']!=fingerprint(label_raw):raise ValueError('人工分类已变化')
                source=self.source(image_id)
                if any(item['source'][k]!=source[k] for k in ('sha256','width','height')):
                    raise ValueError('自动框原图不一致')
                boxes=item.get('boxes')
                withdraw=bool(refresh_pending and replaces and item.get('withdraw_if_empty') is True and boxes==[])
                if not isinstance(boxes,list) or (not boxes and not withdraw) or len(boxes)>100:raise ValueError('没有有效自动框')
                clean=[];box_ids=set()
                for box in boxes:
                    identity=box.get('id');problem=box.get('problem');values=[box.get(k) for k in ('x','y','w','h')]
                    prediction=prediction_metadata(box)
                    if (box.get('origin')!='model' or not isinstance(identity,str) or not identity or len(identity)>80 or identity in box_ids
                        or not isinstance(problem,str)
                        or (problem!='' and (problem not in self.store.tags()['defect'] or (not prediction and problem not in value.get('defects',[]))))
                        or (prediction and not problem)):
                        raise ValueError('框类型或标识不正确')
                    if any(type(v) not in (int,float) or not math.isfinite(v) for v in values):raise ValueError('框坐标不正确')
                    x,y,w,h=values
                    if min(x,y)<0 or min(w,h)<=0 or x+w>1.000001 or y+h>1.000001:raise ValueError('框超出原图')
                    score=box.get('score')
                    if score is not None and (type(score) not in (int,float) or not math.isfinite(score)):raise ValueError('分数不正确')
                    clean.append({'id':identity,'problem':problem,'x':x,'y':y,'w':min(w,1-x),'h':min(h,1-y),
                                  'score':score,'origin':'model',**prediction})
                    if problem and not prediction:clean[-1]['problem_source']='human_label'
                    box_ids.add(identity)
                data={'model_id':model_id,'method':method,'source':source,'label_fingerprint':item['label_fingerprint'],
                      'boxes':clean,'note':str(item.get('note','粗定位建议，需人工逐框审核。'))[:1000],
                      'source_checkpoint_sha256':manifest.get('source_checkpoint_sha256'),
                      'coordinate_space':'exif_transposed_normalized_xywh','requires_human_review':True}
                identity=fingerprint(json.dumps(data,sort_keys=True,ensure_ascii=False))[:32]
                for box in clean:box['proposal_id']=identity
                prepared.append((image_id,identity,data,label_raw,replaces))
            except (ValueError,KeyError,TypeError,OSError):
                skipped+=1
        imported=0;refreshed=0;withdrawn=0;imported_ids=[];imported_images=[]
        with self.store.lock,self.store.db:
            self.store.db.execute('BEGIN IMMEDIATE')
            from deployment.snapshot import digest,annotation_digest
            before=digest(self.store.db);annotations_before=annotation_digest(self.store.db)
            for image_id,identity,data,label_raw,replaces in prepared:
                current=self.store.one('SELECT data FROM labels WHERE image_id=?',(image_id,))
                if ((current['data'] if current else '')!=label_raw
                    or self.store.one('SELECT image_id FROM drafts WHERE image_id=?',(image_id,))
                    or self.store.one('SELECT image_id FROM annotation_documents WHERE image_id=?',(image_id,))
                    or self._has_human_review(image_id)):
                    skipped+=1;continue
                old=self.store.one('SELECT * FROM annotation_proposals WHERE image_id=?',(image_id,))
                if old:
                    old_data=json.loads(old['data'])
                    if (not refresh_pending or not replaces or old['proposal_id']!=replaces or old['status']!='pending'
                        or old['label_data']!=label_raw or old_data['source']['sha256']!=data['source']['sha256']):
                        skipped+=1;continue
                    stamp=dt.datetime.now(dt.timezone.utc).isoformat()
                    self.store.db.execute('INSERT INTO annotation_proposal_history(image_id,proposal_id,data,label_data,status,created_at,reviewed_at,archived_at,replaced_by) VALUES(?,?,?,?,?,?,?,?,?)',
                        (image_id,old['proposal_id'],old['data'],old['label_data'],old['status'],old['created_at'],old['reviewed_at'],stamp,identity))
                    if not data['boxes']:
                        self.store.db.execute("UPDATE annotation_proposals SET status='stale',reviewed_at=? WHERE image_id=? AND proposal_id=? AND status='pending'",(stamp,image_id,replaces))
                        withdrawn+=1;continue
                    self.store.db.execute("UPDATE annotation_proposals SET proposal_id=?,data=?,label_data=?,status='pending',created_at=?,reviewed_at=NULL WHERE image_id=? AND proposal_id=? AND status='pending'",
                        (identity,json.dumps(data,ensure_ascii=False,separators=(',',':')),label_raw,stamp,image_id,replaces))
                    refreshed+=1;continue
                if replaces or not data['boxes']:
                    skipped+=1;continue
                self.store.db.execute('INSERT INTO annotation_proposals(image_id,proposal_id,data,label_data,created_at) VALUES(?,?,?,?,?)',
                    (image_id,identity,json.dumps(data,ensure_ascii=False,separators=(',',':')),label_raw,dt.datetime.now(dt.timezone.utc).isoformat()))
                imported+=1;imported_ids.append(identity);imported_images.append(image_id)
            assert before==digest(self.store.db),'Human classifications changed'
            assert annotations_before==annotation_digest(self.store.db),'Human position annotations changed'
        return {'imported':imported,'refreshed':refreshed,'withdrawn':withdrawn,'skipped':skipped,'human_classifications_unchanged':True,'human_annotations_unchanged':True,
                'human_digest':before,'annotation_digest':annotations_before,
                'imported_proposal_ids':imported_ids,'imported_image_ids':imported_images}

    def enrich_proposals(self,manifest):
        """Only fill untouched blank machine types; keep position IDs and human data immutable."""
        if manifest.get('schema_version')!=1 or manifest.get('requires_human_review') is not True:
            raise ValueError('自动框必须标记为待人工审核')
        items=manifest.get('items')
        if not isinstance(items,list) or len(items)>10000:raise ValueError('自动框清单格式不正确')
        ids=[item.get('image_id') for item in items if isinstance(item,dict)]
        if len(ids)!=len(items) or any(type(v) is not int for v in ids) or len(set(ids))!=len(ids):
            raise ValueError('自动框图片标识重复或格式不正确')
        prepared=[];skipped=0
        for item in items:
            try:
                image_id=item['image_id'];image=self.store.image(image_id)
                if image['relative_path']!=item['relative_path']:raise ValueError('原图路径不一致')
                old=self.store.one("SELECT * FROM annotation_proposals WHERE image_id=? AND proposal_id=? AND status='pending'",
                    (image_id,item['proposal_id']))
                if not old:raise ValueError('建议已审核或变化')
                if item.get('expected_proposal_fingerprint')!=fingerprint(old['data']):raise ValueError('建议版本已变化')
                old_data=json.loads(old['data']);source=self.source(image_id)
                if any(source.get(k)!=item['source'].get(k) or source.get(k)!=old_data['source'].get(k)
                       for k in ('sha256','width','height','size','mtime_ns')):raise ValueError('自动框原图不一致')
                if item['label_fingerprint']!=fingerprint(old['label_data']):raise ValueError('人工分类已变化')
                if not self.proposal(image_id,source) or self._has_human_review(image_id):raise ValueError('已有人工编辑')
                if self.store.one('SELECT image_id FROM annotation_documents WHERE image_id=?',(image_id,)):
                    raise ValueError('已有人工编辑')
                if self.store.one('SELECT image_id FROM annotation_feedback WHERE image_id=? LIMIT 1',(image_id,)):
                    raise ValueError('已有人工纠错')
                boxes=item.get('boxes')
                if not isinstance(boxes,list) or len(boxes)!=len(old_data['boxes']):raise ValueError('补充类型不能改变框')
                incoming={box['id']:box for box in boxes}
                if len(incoming)!=len(boxes) or set(incoming)!={box['id'] for box in old_data['boxes']}:
                    raise ValueError('补充类型不能改变框')
                enriched=[];changes=[]
                for before in old_data['boxes']:
                    box=incoming[before['id']]
                    if any(box.get(k)!=before[k] for k in ('x','y','w','h')):raise ValueError('补充类型不能改变框')
                    if before['problem']:
                        if box.get('problem')!=before['problem']:raise ValueError('已有问题类型不能替换')
                        enriched.append(dict(before));continue
                    if not box.get('problem'):enriched.append(dict(before));continue
                    if box['problem'] not in self.store.tags()['defect']:raise ValueError('请选择已有问题点')
                    prediction=prediction_metadata(box)
                    if not prediction:raise ValueError('补充问题类型必须有模型预测来源')
                    enriched.append({**before,'problem':box['problem'],**prediction})
                    changes.append({'box_id':before['id'],'before_problem':'','after_problem':box['problem'],
                        'type_provenance':prediction['type_provenance']})
                if not changes:raise ValueError('没有新的问题类型建议')
                data={**old_data,'boxes':enriched,'type_enrichment':{
                    'at':dt.datetime.now(dt.timezone.utc).isoformat(),'previous_content_fingerprint':fingerprint(old['data']),
                    'changes':changes,'requires_fresh_review':True}}
                prepared.append((image_id,old,data))
            except (ValueError,KeyError,TypeError,OSError):skipped+=1
        enriched_ids=[];enriched_images=[]
        with self.store.lock,self.store.db:
            self.store.db.execute('BEGIN IMMEDIATE')
            from deployment.snapshot import digest,annotation_digest
            before_digest=digest(self.store.db);annotations_before=annotation_digest(self.store.db)
            has_batches=self.store.one("SELECT 1 FROM sqlite_master WHERE type='table' AND name='review_batch_items'")
            for image_id,old,data in prepared:
                current=self.store.one('SELECT * FROM annotation_proposals WHERE image_id=?',(image_id,))
                label=self.store.one('SELECT data FROM labels WHERE image_id=?',(image_id,))
                image=self.store.image(image_id)
                if (not current or current['proposal_id']!=old['proposal_id'] or current['status']!='pending'
                    or current['reviewed_at'] is not None or current['data']!=old['data']
                    or current['label_data']!=old['label_data'] or (label['data'] if label else '')!=old['label_data']
                    or not image['available'] or any(image[k]!=data['source'][k] for k in ('size','mtime_ns'))
                    or self.store.one('SELECT image_id FROM drafts WHERE image_id=?',(image_id,))
                    or self.store.one('SELECT image_id FROM annotation_documents WHERE image_id=?',(image_id,))
                    or self.store.one('SELECT image_id FROM annotation_feedback WHERE image_id=? LIMIT 1',(image_id,))
                    or self._has_human_review(image_id)):
                    skipped+=1;continue
                batch=None
                if has_batches:
                    batch=self.store.one('SELECT b.*,r.completed_at FROM review_batch_items b JOIN review_batches r ON r.batch_id=b.batch_id WHERE b.proposal_id=?',(old['proposal_id'],))
                    if batch and (batch['decision'] is not None or batch['reviewed_at'] is not None or batch['result_data'] is not None or batch['completed_at'] is not None
                        or batch['proposal_data']!=old['data']):skipped+=1;continue
                raw=json.dumps(data,ensure_ascii=False,separators=(',',':'))
                self.store.db.execute("UPDATE annotation_proposals SET data=? WHERE image_id=? AND proposal_id=? AND status='pending' AND data=?",
                    (raw,image_id,old['proposal_id'],old['data']))
                if batch:self.store.db.execute('UPDATE review_batch_items SET proposal_data=? WHERE proposal_id=? AND proposal_data=? AND decision IS NULL',
                    (raw,old['proposal_id'],old['data']))
                enriched_ids.append(old['proposal_id']);enriched_images.append(image_id)
            assert before_digest==digest(self.store.db),'Human classifications changed'
            assert annotations_before==annotation_digest(self.store.db),'Human position annotations changed'
        return {'imported':0,'refreshed':0,'withdrawn':0,'enriched':len(enriched_ids),'skipped':skipped,
            'enriched_proposal_ids':enriched_ids,'enriched_image_ids':enriched_images,
            'imported_proposal_ids':[],'imported_image_ids':[],
            'human_classifications_unchanged':True,'human_annotations_unchanged':True,
            'human_digest':before_digest,'annotation_digest':annotations_before}

    def reject_proposal(self, image_id, proposal_id, feedback=None):
        source=self.source(image_id)
        with self.store.lock,self.store.db:
            if not self.store.db.in_transaction:self.store.db.execute('BEGIN IMMEDIATE')
            proposal=self.proposal(image_id,source)
            if not proposal or proposal['proposal_id']!=proposal_id:raise ValueError('建议已失效或已审核，请重新读取')
            expected=(feedback or {}).get('expected_proposal_fingerprint')
            if (expected is not None or proposal.get('type_enrichment')) and expected!=proposal['content_fingerprint']:
                raise ValueError('自动框已补充或更新问题类型，请重新读取后审核；原结果未覆盖')
            if feedback and feedback.get('negative_regions'):
                self.save_feedback(image_id,{**feedback,'proposal_id':proposal_id,'status':'confirmed'})
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
        data = json.loads(row['data']) if row else {'boxes':[],'negative_regions':[],'status':'draft','outcome':'boxed'}
        data.setdefault('negative_regions',[])
        return {'image_id':image_id,'source':source,'document':data,'version':row['version'] if row else 0,
                'source_changed':bool(row and data['source']['sha256']!=source['sha256']),
                'label_fingerprint':fingerprint(label['data'] if label else ''),'review':self.review(image_id),
                'proposal':self.proposal(image_id,source)}

    def _box_roles(self, values, required=False):
        if not isinstance(values,list) or len(values)>40 or any(not isinstance(v,str) for v in values):
            raise ValueError('问题框责任格式不正确')
        values=list(dict.fromkeys(v.strip() for v in values if v.strip()))
        if any(v not in self.store.tags()['responsibility'] for v in values):
            raise ValueError('请选择已有责任类型')
        if '待核' in values and len(values)>1:values.remove('待核')
        if required and not values:raise ValueError('请为问题点选择责任；不确定可选“待核”')
        return values

    def _classification_append(self, image_id, body, boxes, previous, label, status):
        """Prepare append-only problems and explicit responsibility corrections atomically."""
        from app import RESPONSIBILITY_RULES
        updates=body['classification_updates']
        if not isinstance(updates,dict):raise ValueError('问题点追加格式不正确')
        if any(k not in ('defects','responsibilities','problem_responsibilities','responsibility_overrides') for k in updates):
            raise ValueError('框选仅支持追加问题点和责任')
        problems=updates.get('defects',[])
        if (not isinstance(problems,list) or len(problems)>40 or
            any(not isinstance(p,str) or p not in self.store.tags()['defect'] for p in problems)):
            raise ValueError('请选择已有问题点')
        if 'responsibilities' in updates:self._box_roles(updates['responsibilities'])
        mapping=updates.get('problem_responsibilities',{})
        if not isinstance(mapping,dict) or any(p not in problems for p in mapping):
            raise ValueError('问题点责任格式不正确')
        mapping={p:self._box_roles(v) for p,v in mapping.items()}
        requested=list(dict.fromkeys(problems+[b['problem'] for b in boxes if b['problem']]))
        overrides=updates.get('responsibility_overrides',[])
        if (not isinstance(overrides,list) or len(overrides)>40 or
            any(not isinstance(p,str) or p not in requested for p in overrides)):
            raise ValueError('人工责任纠正格式不正确')
        overrides=set(overrides)|{b['problem'] for b in boxes if b.get('responsibility_override') is True}
        committed=json.loads(label['data']) if label else None
        draft=self.store.one('SELECT * FROM drafts WHERE image_id=?',(image_id,))
        visible=json.loads(draft['data']) if draft else None
        if status=='confirmed':
            expected=body.get('expected_classification_draft')
            actual=fingerprint(draft['data']) if draft else None
            if expected!=actual:
                raise ValueError('分类暂存已变化，请重新读取；原结果未覆盖')
            if any(v and v.get('image_state','') in ('normal','unclear','unrelated') for v in (committed,visible)):
                raise ValueError('图片已有状态，请先明确复核原分类；框选不能自动改变图片状态')
        base=dict(committed or {'defects':[],'responsibilities':[],'image_state':'','note':''})
        base['defects']=list(base.get('defects',[]));base['responsibilities']=list(base.get('responsibilities',[]))
        resolved={p:list(v) for p,v in base.get('problem_responsibilities',{}).items()}
        if not resolved and len(base['defects'])==1:
            resolved[base['defects'][0]]=list(base['responsibilities'])
        if visible:
            base['defects']=list(dict.fromkeys(base['defects']+visible.get('defects',[])))
            base['responsibilities']=list(dict.fromkeys(base['responsibilities']+visible.get('responsibilities',[])))
            visible_mapping=visible.get('problem_responsibilities',{})
            if not visible_mapping and len(visible.get('defects',[]))==1:
                visible_mapping={visible['defects'][0]:visible.get('responsibilities',[])}
            for p,roles in visible_mapping.items():resolved[p]=list(dict.fromkeys(resolved.get(p,[])+roles))
            base['note']=visible.get('note','')
        old_boxes={b['id']:b for b in (previous or {}).get('boxes',[])}
        # Preserve unattributed legacy responsibility values; corrections replace only
        # the old responsibility values attributable to the explicitly edited issue.
        attributed={r for roles in resolved.values() for r in roles}
        legacy_roles=[r for r in base['responsibilities'] if r not in attributed]
        box_mapping={}
        for box in boxes:
            problem=box['problem'];old=old_boxes.get(box['id'],{})
            if not problem:
                box['responsibilities']=[]
                continue
            roles=box.get('responsibilities',mapping.get(problem))
            if roles is None and old.get('problem')==problem:roles=old.get('responsibilities')
            if roles is None:roles=resolved.get(problem,[])
            roles=self._box_roles(roles)
            if not roles and problem not in overrides and problem in RESPONSIBILITY_RULES:
                roles=[RESPONSIBILITY_RULES[problem]]
            if status=='confirmed' and not roles:
                raise ValueError('请为“'+problem+'”选择责任；不确定可选“待核”')
            box['responsibilities']=roles
            box_mapping[problem]=list(dict.fromkeys(box_mapping.get(problem,[])+roles))
        for problem in requested:
            roles=box_mapping.get(problem,mapping.get(problem,[]))
            if not roles and problem not in overrides:roles=resolved.get(problem,[])
            if not roles and problem not in overrides and problem in RESPONSIBILITY_RULES:
                roles=[RESPONSIBILITY_RULES[problem]]
            if status=='confirmed' and not roles:
                raise ValueError('请为“'+problem+'”选择责任；不确定可选“待核”')
            resolved[problem]=roles if problem in overrides else list(dict.fromkeys(resolved.get(problem,[])+roles))
            mapping[problem]=roles
        base['defects']=list(dict.fromkeys(base['defects']+requested))
        base['problem_responsibilities']=resolved
        base['responsibilities']=list(dict.fromkeys(legacy_roles+[r for roles in resolved.values() for r in roles]))
        if len(base['responsibilities'])>1 and '待核' in base['responsibilities']:base['responsibilities'].remove('待核')
        if base['defects']:base['image_state']='defect'
        if len(base['defects'])>40:raise ValueError('每张图片最多40个问题点')
        return base,draft,{'defects':requested,
            'problem_responsibilities':mapping,'responsibility_overrides':[p for p in requested if p in overrides]}

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
            if problem not in known and not (status=='draft' and problem==''):
                raise ValueError('请为问题框选择已有问题点')
            values=[box.get(k) for k in ('x','y','w','h')]
            if any(type(v) not in (int,float) or not math.isfinite(v) for v in values):raise ValueError('框坐标不正确')
            x,y,w,h=values
            if min(x,y)<0 or min(w,h)<=0 or x+w>1.000001 or y+h>1.000001:raise ValueError('问题框必须位于原图内')
            item=dict(id=identity,problem=problem,x=x,y=y,w=min(w,1-x),h=min(h,1-y))
            if 'responsibilities' in box:item['responsibilities']=self._box_roles(box['responsibilities'])
            if 'responsibility_override' in box:
                if type(box['responsibility_override']) is not bool:raise ValueError('人工责任纠正标记不正确')
                item['responsibility_override']=box['responsibility_override']
            if box.get('origin')=='model':
                if not proposal_id or box.get('proposal_id')!=proposal_id:raise ValueError('模型框来源不正确')
                item.update(origin='model',proposal_id=proposal_id)
                if box.get('source_box_id') is not None:
                    if not isinstance(box['source_box_id'],str) or not box['source_box_id'] or len(box['source_box_id'])>80:
                        raise ValueError('原建议框标识不正确')
                    item['source_box_id']=box['source_box_id']
            clean.append(item)
            identities.add(identity)
        if status=='confirmed' and outcome=='boxed' and not clean:raise ValueError('请框选问题，或选择其他复核结果')
        with self.store.lock,self.store.db:
            if not self.store.db.in_transaction:self.store.db.execute('BEGIN IMMEDIATE')
            row=self.store.one('SELECT * FROM annotation_documents WHERE image_id=?',(image_id,))
            if (row['version'] if row else 0)!=version:raise ValueError('补标已被其他页面更新，请重新读取；原结果未覆盖')
            previous=json.loads(row['data']) if row else None
            previous_boxes={b['id']:b for b in (previous or {}).get('boxes',[])}
            for box in clean:
                old=previous_boxes.get(box['id'],{})
                if 'responsibilities' not in box and old.get('problem')==box['problem'] and 'responsibilities' in old:
                    box['responsibilities']=list(old['responsibilities'])
                if 'responsibility_override' not in box and old.get('problem')==box['problem'] and 'responsibility_override' in old:
                    box['responsibility_override']=old['responsibility_override']
            regions=clean_negative_regions(body.get('negative_regions',previous.get('negative_regions',[]) if previous else []))
            if status=='confirmed':reject_conflicting_regions(clean,regions)
            if proposal_id:
                proposal=self.proposal(image_id,source,include_annotations=True)
                if not proposal and previous and previous.get('proposal_id')==proposal_id:
                    accepted=self.store.one("SELECT data FROM annotation_proposals WHERE image_id=? AND proposal_id=? AND status='accepted'",(image_id,proposal_id))
                    if accepted:
                        data=json.loads(accepted['data'])
                        if data['source']['sha256']==source['sha256']:proposal={**data,'proposal_id':proposal_id,'content_fingerprint':fingerprint(accepted['data'])}
                if not proposal or proposal['proposal_id']!=proposal_id:raise ValueError('自动框建议已变化或已审核，请重新读取')
                expected_proposal=body.get('expected_proposal_fingerprint')
                if (expected_proposal is not None or (proposal.get('type_enrichment') and proposal.get('status')=='pending')):
                    if expected_proposal!=proposal['content_fingerprint']:
                        raise ValueError('自动框已补充或更新问题类型，请重新读取后审核；原结果未覆盖')
                original_boxes={box['id']:box for box in proposal['boxes']}
                for box in clean:
                    original=original_boxes.get(box.get('source_box_id',box['id']))
                    if box['problem']=='' and box.get('origin')=='model':
                        if not original or original['problem']!='':raise ValueError('待选问题框来源不正确')
                    if original and original.get('type_provenance'):
                        box.update(prediction_metadata(original),predicted_problem=original['problem'])
                if (status=='confirmed' and clean and any(box['problem']=='' or box.get('type_provenance') for box in proposal['boxes'])
                    and 'classification_updates' not in body):
                    raise ValueError('请为待选问题框选择问题点和责任后确认')
                if previous and previous.get('proposal_id')!=proposal_id:
                    current_boxes={box['id']:box for box in clean}
                    if any(current_boxes.get(box['id'])!=box for box in previous['boxes']):
                        raise ValueError('已有独立人工补标，自动框不能覆盖原结果')
            label=self.store.one('SELECT * FROM labels WHERE image_id=?',(image_id,))
            label_hash=fingerprint(label['data'] if label else '')
            if status=='confirmed' and body.get('label_fingerprint')!=label_hash:
                raise ValueError('人工分类已更新，请重新读取后确认补标')
            classification=None;draft=None;updates=None;labels_changed=False
            if 'classification_updates' in body:
                classification,draft,updates=self._classification_append(image_id,body,clean,previous,label,status)
                if status=='confirmed':
                    label_raw=json.dumps(classification,ensure_ascii=False,separators=(',',':'))
                    labels_changed=not label or label_raw!=label['data']
                    stamp=dt.datetime.now(dt.timezone.utc).isoformat()
                    if labels_changed:
                        self.store.db.execute('INSERT OR REPLACE INTO labels VALUES(?,?,?)',(image_id,label_raw,stamp))
                        changes=[{'id':image_id,'before':label,'after':classification}]
                        self.store.db.execute('INSERT INTO history(root_id,kind,data,created_at) VALUES(?,?,?,?)',
                            (self.store.current_root()['id'],'annotation_append',json.dumps(changes,ensure_ascii=False,separators=(',',':')),stamp))
                    if draft:self.store.db.execute('DELETE FROM drafts WHERE image_id=?',(image_id,))
                    label_hash=fingerprint(label_raw)
            document={'source':source,'coordinate_space':'exif_transposed_normalized_xywh',
                'boxes':clean,'negative_regions':regions,'status':status,'outcome':outcome,'label_fingerprint':label_hash,
                'review_run':body.get('review_run')}
            if updates is not None:document['classification_updates']=updates
            if proposal_id:
                document.update(proposal_id=proposal_id,model_id=proposal['model_id'],method=proposal['method'],
                    proposal_content_fingerprint=proposal['content_fingerprint'],
                    origin='machine_assisted_human_confirmed' if status=='confirmed' else 'machine_assisted_draft')
            raw=json.dumps(document,ensure_ascii=False,separators=(',',':'))
            stamp=dt.datetime.now(dt.timezone.utc).isoformat();next_version=version+1
            self.store.db.execute('INSERT OR REPLACE INTO annotation_documents VALUES(?,?,?,?)',
                (image_id,raw,next_version,stamp))
            self.store.db.execute('INSERT INTO annotation_history(image_id,version,before_data,after_data,created_at) VALUES(?,?,?,?,?)',
                (image_id,next_version,row['data'] if row else None,raw,stamp))
            if status=='confirmed':
                if proposal_id or regions or (outcome=='boxed' and clean):
                    feedback={'schema_version':1,'kind':'annotation_review','status':'confirmed','source':source,
                        'coordinate_space':'exif_transposed_normalized_xywh','annotation_version':next_version,
                        'label_fingerprint':label_hash,'negative_regions':regions,'whole_image_normal':False,
                        'proposal_id':proposal_id,'events':correction_events(proposal if proposal_id else previous or {},clean,regions)}
                    if proposal_id:feedback.update(model_id=proposal['model_id'],method=proposal['method'])
                    self._record_feedback(image_id,feedback,stamp)
                if proposal_id:
                    self.store.db.execute('UPDATE annotation_proposals SET status=?,reviewed_at=? WHERE image_id=? AND proposal_id=?',
                        ('accepted' if outcome=='boxed' and clean else 'rejected',stamp,image_id,proposal_id))
                self.store.db.execute("UPDATE error_reviews SET status='reviewed',reviewed_at=? WHERE image_id=? AND run_id=?",
                    (stamp,image_id,body.get('review_run')))
        result={'saved':True,'version':next_version,'document':document,'labels_unchanged':not labels_changed,
            'label_fingerprint':label_hash}
        if classification is not None and status=='confirmed':result['label']=classification
        return result

    def _record_feedback(self, image_id, data, stamp):
        raw=json.dumps(data,ensure_ascii=False,sort_keys=True,separators=(',',':'))
        identity=fingerprint(str(image_id)+':'+raw)
        self.store.db.execute('INSERT OR IGNORE INTO annotation_feedback(image_id,identity,data,created_at) VALUES(?,?,?,?)',
            (image_id,identity,raw,stamp))
        return self.store.one('SELECT id FROM annotation_feedback WHERE identity=?',(identity,))['id']

    def save_feedback(self, image_id, body):
        """Append a human region correction without modifying classification or any human frame."""
        if body.get('status')!='confirmed':raise ValueError('独立纠错必须经人工明确确认')
        source=self.source(image_id)
        if body.get('source_sha256')!=source['sha256']:raise ValueError('原图已变化，请重新读取后纠错')
        regions=clean_negative_regions(body.get('negative_regions'))
        if not regions:raise ValueError('请指定局部误报区域，不能将整图视为正常')
        with self.store.lock,self.store.db:
            row=self.store.one('SELECT * FROM annotation_documents WHERE image_id=?',(image_id,))
            version=row['version'] if row else 0
            if 'expected_version' in body and body['expected_version']!=version:
                raise ValueError('补标已被其他页面更新，请重新读取；原结果未覆盖')
            document=json.loads(row['data']) if row else None
            if document:reject_conflicting_regions(document.get('boxes',[]),regions)
            label=self.store.one('SELECT data FROM labels WHERE image_id=?',(image_id,))
            label_hash=fingerprint(label['data'] if label else '')
            if body.get('label_fingerprint')!=label_hash:raise ValueError('人工分类已更新，请重新读取后纠错')
            feedback={'schema_version':1,'kind':'region_correction','status':'confirmed','source':source,
                'coordinate_space':'exif_transposed_normalized_xywh','annotation_version_at_confirmation':version,
                'label_fingerprint':label_hash,'negative_regions':regions,'whole_image_normal':False,
                'proposal_id':body.get('proposal_id'),'model_id':body.get('model_id'),
                'events':[{'kind':'false_positive','source_box_id':r.get('source_box_id'),'reason':r['reason'],
                    'correction_semantics':['normal' if normal_confirmation(r) else 'ambiguous_delete'],
                    'normal_confirmation':normal_confirmation(r),
                    **({'normal_provenance':r['normal_provenance']} if 'normal_provenance' in r else {})} for r in regions]}
            identity=self._record_feedback(image_id,feedback,dt.datetime.now(dt.timezone.utc).isoformat())
        return {'saved':True,'feedback_id':identity,'labels_unchanged':True,'annotations_unchanged':True}

    def feedback_export(self, verify_source=True):
        rows=self.store.query('SELECT f.*,i.relative_path,i.available,i.size,i.mtime_ns,l.data label_data,d.image_id has_draft,'
            'a.data annotation_data,a.version annotation_version FROM annotation_feedback f JOIN images i ON i.id=f.image_id '
            'LEFT JOIN labels l ON l.image_id=i.id LEFT JOIN drafts d ON d.image_id=i.id '
            'LEFT JOIN annotation_documents a ON a.image_id=i.id WHERE i.root_id=? ORDER BY f.id',
            (self.store.current_root()['id'],))
        trained={r['feedback_id'] for r in self.store.query('SELECT DISTINCT feedback_id FROM annotation_feedback_training')}
        result=[];source_cache={}
        for row in rows:
            feedback=json.loads(row['data']);label=json.loads(row['label_data']) if row['label_data'] else None
            document=json.loads(row['annotation_data']) if row['annotation_data'] else None;reasons=[]
            if not label or not label.get('defects') or label.get('image_state','') not in ('','defect'):
                reasons.append('classification_policy_excluded')
            if row['has_draft']:reasons.append('classification_draft')
            if feedback['label_fingerprint']!=fingerprint(row['label_data']):reasons.append('classification_changed')
            if feedback.get('status')!='confirmed':reasons.append('not_human_confirmed')
            if feedback.get('kind')=='annotation_review' and (not document or document.get('status')!='confirmed'
                or row['annotation_version']!=feedback.get('annotation_version') or document.get('outcome')=='unclear'):
                reasons.append('annotation_not_current_confirmed')
            if document and document.get('status')!='confirmed':reasons.append('annotation_draft')
            if document and document.get('outcome')=='unclear':reasons.append('unclear_annotation')
            try:
                reject_conflicting_regions(document.get('boxes',[]) if document else [],feedback.get('negative_regions',[]))
            except ValueError:reasons.append('negative_overlaps_current_positive')
            source=feedback['source'];verified=False
            if verify_source:
                if row['image_id'] not in source_cache:
                    try:source_cache[row['image_id']]=self.source(row['image_id'])
                    except (ValueError,OSError):source_cache[row['image_id']]=None
                current=source_cache[row['image_id']]
                verified=bool(current and all(source[k]==current[k] for k in ('sha256','width','height')))
                if not verified:reasons.append('source_changed_or_unavailable')
            elif not row['available'] or row['size']!=source['size'] or row['mtime_ns']!=source['mtime_ns']:
                reasons.append('source_index_changed_or_unavailable')
            result.append({'feedback_id':row['id'],'image_id':row['image_id'],'relative_path':row['relative_path'],
                'created_at':row['created_at'],'source':source,'feedback':feedback,'annotation':document,
                'original_classification':label,'original_classification_raw':row['label_data'],
                'training_eligible_by_label_policy':not reasons,'exclusion_reasons':reasons,
                'source_sha256_verified':verified,'included_in_training':row['id'] in trained})
        return {'schema_version':1,'coordinate_space':'exif_transposed_normalized_xywh','whole_image_negatives':False,
            'source_hash_must_be_reverified_before_training':not verify_source,'items':result}

    def feedback_state(self):
        items=self.feedback_export(verify_source=False)['items'];eligible=[r for r in items if r['training_eligible_by_label_policy']]
        current_regions={(r['image_id'],g['id']) for r in eligible for g in r['feedback'].get('negative_regions',[])}
        return {'confirmed_corrections':len(items),'confirmed_regions':len(current_regions),
            'eligible_feedback':len(eligible),'pending_training_feedback':sum(not r['included_in_training'] for r in eligible),
            'pending_training_images':len({r['image_id'] for r in eligible if not r['included_in_training']}),
            'trained_feedback':sum(r['included_in_training'] for r in items),
            'weight_update_mode':'reviewed_batch_training','automatic_online_weight_updates':False}

    def record_feedback_training(self, run_id, feedback_ids, model_sha256):
        if not isinstance(run_id,str) or not run_id or len(run_id)>160:raise ValueError('训练批次不正确')
        if not isinstance(model_sha256,str) or len(model_sha256)!=64 or any(c not in '0123456789abcdef' for c in model_sha256):
            raise ValueError('模型校验值不正确')
        if not isinstance(feedback_ids,list) or any(type(v) is not int for v in feedback_ids):raise ValueError('纠错记录不正确')
        with self.store.lock,self.store.db:
            for identity in sorted(set(feedback_ids)):
                if not self.store.one('SELECT id FROM annotation_feedback WHERE id=?',(identity,)):raise ValueError('纠错记录不存在')
                self.store.db.execute('INSERT OR IGNORE INTO annotation_feedback_training(feedback_id,run_id,model_sha256,created_at) VALUES(?,?,?,?)',
                    (identity,run_id,model_sha256,dt.datetime.now(dt.timezone.utc).isoformat()))
        return {'recorded':True,'feedback_count':len(set(feedback_ids)),'production_model_changed':False}

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
