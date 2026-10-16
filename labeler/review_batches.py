"""Track human review batches and freeze JSON receipts without copying NAS images."""
from collections import Counter
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import re
import uuid
from review_comparison import compare_review,summarize_comparisons


def _now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def _raw(value):
    return json.dumps(value,ensure_ascii=False,separators=(',',':'),sort_keys=True,allow_nan=False)


class ReviewBatches:
    def __init__(self,store):
        self.store=store
        with store.lock,store.db:
            store.db.executescript('''
                CREATE TABLE IF NOT EXISTS review_batches(
                  batch_id TEXT PRIMARY KEY,root_id INTEGER NOT NULL REFERENCES roots(id),
                  metadata TEXT NOT NULL,created_at TEXT NOT NULL,completed_at TEXT,
                  archive_data TEXT,archive_sha256 TEXT,archive_written_at TEXT,archive_error TEXT);
                CREATE TABLE IF NOT EXISTS review_batch_items(
                  proposal_id TEXT PRIMARY KEY,batch_id TEXT NOT NULL REFERENCES review_batches(batch_id),
                  image_id INTEGER NOT NULL REFERENCES images(id),proposal_data TEXT NOT NULL,
                  relative_path TEXT NOT NULL,decision TEXT,reviewed_at TEXT,result_data TEXT,
                  blocked_reason TEXT);
            ''')

    @staticmethod
    def _batch_id(value):
        if not isinstance(value,str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,119}',value):
            raise ValueError('审核批次标识格式不正确')
        return value

    def register(self,batch_id,proposal_ids=None,metadata=None):
        """Register only existing proposal identities; a proposal belongs to one batch."""
        batch_id=self._batch_id(batch_id)
        if metadata is not None and not isinstance(metadata,dict):raise ValueError('批次说明格式不正确')
        try:metadata_raw=_raw(metadata or {})
        except (TypeError,ValueError,RecursionError) as error:raise ValueError('批次说明必须是有效JSON') from error
        if len(metadata_raw)>16000:raise ValueError('批次说明过长')
        if proposal_ids is not None and (not isinstance(proposal_ids,list) or len(proposal_ids)>10000 or
            any(not isinstance(v,str) or not v or len(v)>80 for v in proposal_ids)):
            raise ValueError('自动框批次清单不正确')
        root=self.store.current_root()
        if not root:raise ValueError('尚未选择图片目录')
        added=skipped=0
        with self.store.lock,self.store.db:
            if not self.store.db.in_transaction:self.store.db.execute('BEGIN IMMEDIATE')
            old=self.store.one('SELECT * FROM review_batches WHERE batch_id=?',(batch_id,))
            if old and old['root_id']!=root['id']:raise ValueError('审核批次不属于当前目录')
            if proposal_ids is None:
                proposal_ids=[r['proposal_id'] for r in self.store.query(
                    "SELECT p.proposal_id FROM annotation_proposals p JOIN images i ON i.id=p.image_id "
                    "LEFT JOIN labels l ON l.image_id=i.id LEFT JOIN drafts d ON d.image_id=i.id "
                    "LEFT JOIN review_batch_items b ON b.proposal_id=p.proposal_id "
                    "WHERE i.root_id=? AND b.proposal_id IS NULL AND "+self.store.annotations.proposal_match()+
                    " ORDER BY p.created_at,i.id",
                    (root['id'],))]
            prepared=[]
            for identity in dict.fromkeys(proposal_ids):
                assigned=self.store.one('SELECT batch_id FROM review_batch_items WHERE proposal_id=?',(identity,))
                if assigned:skipped+=1;continue
                proposal=self._proposal(identity,root['id'])
                if not proposal:skipped+=1;continue
                if old and old['completed_at']:raise ValueError('已归档批次不能追加新建议')
                prepared.append(proposal)
            if prepared and not old:
                self.store.db.execute('INSERT INTO review_batches(batch_id,root_id,metadata,created_at) VALUES(?,?,?,?)',
                    (batch_id,root['id'],metadata_raw,_now()))
            for proposal in prepared:
                self.store.db.execute('INSERT INTO review_batch_items(proposal_id,batch_id,image_id,proposal_data,relative_path) VALUES(?,?,?,?,?)',
                    (proposal['proposal_id'],batch_id,proposal['image_id'],proposal['data'],proposal['relative_path']))
                added+=1
        return {'batch_id':batch_id,'registered':added,'skipped':skipped,
            'exists':bool(old or added),'human_data_unchanged':True}

    def _proposal(self,identity,root_id):
        for table in ('annotation_proposals','annotation_proposal_history'):
            row=self.store.one('SELECT p.*,i.relative_path FROM '+table+' p JOIN images i ON i.id=p.image_id '
                'WHERE p.proposal_id=? AND i.root_id=? ORDER BY p.created_at DESC LIMIT 1',(identity,root_id))
            if row:return row
        return None

    @staticmethod
    def _timestamp(value):
        try:
            parsed=dt.datetime.fromisoformat(value.replace('Z','+00:00'))
            return parsed if parsed.tzinfo is not None else None
        except (ValueError,AttributeError):return None

    def _source_issue(self,item):
        image=self.store.image(item['image_id']);source=json.loads(item['proposal_data'])['source']
        if not image['available']:return 'source_unavailable'
        if any(image[key]!=source.get(key) for key in ('size','mtime_ns')):return 'source_changed'
        try:stat=self.store.path(image).stat()
        except OSError:return 'source_unavailable'
        if (stat.st_size,stat.st_mtime_ns)!=(source.get('size'),source.get('mtime_ns')):return 'source_changed'
        return None

    def _terminal_state_review(self,item,proposal,root_id):
        """An explicit later human state commit may supersede an older box draft."""
        label=self.store.one('SELECT * FROM labels WHERE image_id=?',(item['image_id'],))
        classification=json.loads(label['data']) if label else {}
        state=classification.get('image_state')
        if state not in ('normal','unrelated','unclear') or classification.get('defects'):
            return None,None
        if self.store.one('SELECT image_id FROM drafts WHERE image_id=?',(item['image_id'],)):
            return None,'classification_draft'
        created=self._timestamp(proposal['created_at']);updated=self._timestamp(label['updated_at'])
        if not created or not updated or updated<=created:return None,'state_confirmation_missing'
        receipt=None
        histories=self.store.query('SELECT * FROM history WHERE root_id=? AND undone=0 AND created_at>=? ORDER BY id DESC',
            (root_id,proposal['created_at'][:19]))
        for history in histories:
            stamp=self._timestamp(history['created_at'])
            if history['kind'] not in ('commit','batch','review-unrelated') or not stamp or stamp<=created or stamp!=updated:
                continue
            if any(change.get('id')==item['image_id'] and change.get('after')==classification
                   for change in json.loads(history['data'])):
                receipt=history;break
        if not receipt:return None,'state_confirmation_missing'
        annotation=self.store.one('SELECT * FROM annotation_documents WHERE image_id=?',(item['image_id'],))
        if annotation and json.loads(annotation['data']).get('status')!='confirmed':
            annotation_time=self._timestamp(annotation['updated_at'])
            if not annotation_time or annotation_time>updated:return None,'annotation_draft'
        problem=self._source_issue(item)
        if problem:return None,problem
        source=json.loads(item['proposal_data'])['source']
        result={'image_id':item['image_id'],'relative_path':item['relative_path'],'proposal_id':item['proposal_id'],
            'decision':'state_'+state,'reviewed_at':receipt['created_at'],'manual_history_id':receipt['id'],
            'classification':classification,'annotation':None,'annotation_version':None,
            'final_boxes':[],'negative_regions':[],'source':source,'non_training':True,
            'disposition_scope':'explicit_human_image_state_not_local_false_positive_truth'}
        result['comparison']=compare_review(json.loads(item['proposal_data']).get('boxes',[]),[],[])
        result['comparison']['disposition_scope']=result['disposition_scope']
        return result,None

    def _reviewed(self,item,root_id):
        proposal=self._proposal(item['proposal_id'],root_id)
        if not proposal:return None,'missing_proposal'
        if proposal['status']=='stale':return None,'stale_proposal'
        if proposal['status'] not in ('accepted','rejected'):
            current=self.store.one('SELECT proposal_id FROM annotation_proposals WHERE image_id=?',(item['image_id'],))
            if current and current['proposal_id']!=item['proposal_id']:return None,'superseded_proposal'
        terminal,terminal_reason=self._terminal_state_review(item,proposal,root_id)
        if terminal or terminal_reason:return terminal,terminal_reason
        if proposal['status'] not in ('accepted','rejected') or not proposal.get('reviewed_at'):
            current=self.store.one('SELECT proposal_id FROM annotation_proposals WHERE image_id=?',(item['image_id'],))
            return None,'superseded_proposal' if current and current['proposal_id']!=item['proposal_id'] else 'pending_review'
        source=json.loads(item['proposal_data'])['source']
        problem=self._source_issue(item)
        if problem:return None,problem
        if self.store.one('SELECT image_id FROM drafts WHERE image_id=?',(item['image_id'],)):
            return None,'classification_draft'
        label=self.store.one('SELECT data FROM labels WHERE image_id=?',(item['image_id'],))
        if not label:return None,'classification_missing'
        annotation=self.store.one('SELECT * FROM annotation_documents WHERE image_id=?',(item['image_id'],))
        document=json.loads(annotation['data']) if annotation else None
        if document and document.get('status')!='confirmed':return None,'annotation_draft'
        if proposal['status']=='accepted':
            if not document or document.get('proposal_id')!=item['proposal_id']:
                return None,'confirmed_annotation_missing'
            if document.get('label_fingerprint')!=hashlib.sha256(label['data'].encode('utf-8')).hexdigest():
                return None,'classification_changed'
            if any(document.get('source',{}).get(k)!=source.get(k) for k in ('sha256','width','height')):
                return None,'annotation_source_changed'
        result={'image_id':item['image_id'],'relative_path':item['relative_path'],'proposal_id':item['proposal_id'],
            'decision':proposal['status'],'reviewed_at':proposal['reviewed_at'],
            'classification':json.loads(label['data']),'annotation':document,
            'annotation_version':annotation['version'] if annotation else None,
            'final_boxes':document.get('boxes',[]) if document else [],
            'negative_regions':document.get('negative_regions',[]) if document else [],
            'source':source}
        result['comparison']=compare_review(json.loads(item['proposal_data']).get('boxes',[]),result['final_boxes'],result['negative_regions'])
        return result,None

    def _summary(self,batch):
        items=self.store.query('SELECT decision,blocked_reason FROM review_batch_items WHERE batch_id=?',(batch['batch_id'],))
        counts=Counter(i['decision'] for i in items if i['decision'])
        reasons=Counter(i['blocked_reason'] for i in items if not i['decision'] and i['blocked_reason'] not in (None,'pending_review'))
        reviewed=sum(counts.values())
        return {'batch_id':batch['batch_id'],'created_at':batch['created_at'],'completed_at':batch['completed_at'],
            'archived_at':batch['completed_at'],
            'metadata':json.loads(batch['metadata']),'total':len(items),'reviewed':reviewed,
            'accepted':counts['accepted'],'rejected':counts['rejected'],
            'state_processed':sum(counts['state_'+s] for s in ('normal','unrelated','unclear')),
            'state_counts':{s:counts['state_'+s] for s in ('normal','unrelated','unclear')},
            'remaining':len(items)-reviewed,'pending':len(items)-reviewed-sum(reasons.values()),
            'blocked':sum(reasons.values()),'blocked_reasons':dict(reasons),
            'status':'archived' if batch['completed_at'] else 'reviewing',
            'archive_available':bool(batch['archive_data']),'archive_written':bool(batch['archive_written_at']),
            'archive_error':batch['archive_error'],'archive_sha256':batch['archive_sha256']}

    def sync(self):
        """Freeze completed batches first in SQLite; failed file writes are retried."""
        root=self.store.current_root()
        if not root:return {'completed':[],'archive_errors':[]}
        completed=[]
        with self.store.lock,self.store.db:
            if not self.store.db.in_transaction:self.store.db.execute('BEGIN IMMEDIATE')
            for batch in self.store.query('SELECT * FROM review_batches WHERE root_id=? AND completed_at IS NULL ORDER BY created_at,batch_id',(root['id'],)):
                items=self.store.query('SELECT * FROM review_batch_items WHERE batch_id=? ORDER BY image_id,proposal_id',(batch['batch_id'],))
                results=[]
                for item in items:
                    result,reason=self._reviewed(item,root['id'])
                    self.store.db.execute('UPDATE review_batch_items SET decision=?,reviewed_at=?,result_data=?,blocked_reason=? WHERE proposal_id=?',
                        (result['decision'] if result else None,result['reviewed_at'] if result else None,
                         _raw(result) if result else None,reason,item['proposal_id']))
                    if result:results.append(result)
                if items and len(results)==len(items):
                    stamp=_now();counts=Counter(r['decision'] for r in results)
                    state_counts={s:counts['state_'+s] for s in ('normal','unrelated','unclear')}
                    comparison=summarize_comparisons([r for r in results if not r['decision'].startswith('state_')])
                    comparison['image_state_dispositions']={'counts':state_counts,
                        'excluded_from_location_comparison':True,
                        'scope':'human_image_state_decisions_not_local_false_positive_truth'}
                    archive={'schema_version':1,'type':'human_review_batch_archive','batch_id':batch['batch_id'],
                        'created_at':batch['created_at'],'completed_at':stamp,'metadata':json.loads(batch['metadata']),
                        'source_images_copied':False,'source_images_modified':False,
                        'summary':{'total':len(results),'reviewed':len(results),'accepted':counts['accepted'],
                            'rejected':counts['rejected'],'state_processed':sum(counts['state_'+s] for s in ('normal','unrelated','unclear')),
                            'state_counts':state_counts,
                            'final_boxes':sum(len(r['final_boxes']) for r in results)},
                        'comparison':comparison,'items':results}
                    raw=_raw(archive);sha=hashlib.sha256(raw.encode('utf-8')).hexdigest()
                    self.store.db.execute('UPDATE review_batches SET completed_at=?,archive_data=?,archive_sha256=? WHERE batch_id=?',
                        (stamp,raw,sha,batch['batch_id']))
                    completed.append(batch['batch_id'])
        errors=[]
        for batch in self.store.query('SELECT * FROM review_batches WHERE root_id=? AND archive_data IS NOT NULL AND archive_written_at IS NULL',(root['id'],)):
            try:
                self._write_archive(batch['batch_id'],batch['archive_data'])
            except OSError as error:
                message=str(error)[:500];errors.append({'batch_id':batch['batch_id'],'error':message})
                with self.store.lock,self.store.db:
                    self.store.db.execute('UPDATE review_batches SET archive_error=? WHERE batch_id=?',(message,batch['batch_id']))
            else:
                with self.store.lock,self.store.db:
                    self.store.db.execute('UPDATE review_batches SET archive_written_at=?,archive_error=NULL WHERE batch_id=?',(_now(),batch['batch_id']))
        return {'completed':completed,'archive_errors':errors,'human_data_unchanged':True}

    def _write_archive(self,batch_id,raw):
        folder=self.store.data_dir/'review-archives';folder.mkdir(parents=True,exist_ok=True)
        target=folder/(self._batch_id(batch_id)+'.json')
        temporary=folder/(batch_id+'.'+uuid.uuid4().hex+'.tmp')
        try:
            with temporary.open('w',encoding='utf-8',newline='') as stream:
                stream.write(raw);stream.flush();os.fsync(stream.fileno())
            temporary.replace(target)
        finally:
            if temporary.exists():temporary.unlink()

    def state(self):
        root=self.store.current_root()
        if not root:return {'current':None,'active_batches':[],'recent_archives':[],'archives_count':0,'needs_next_batch':False}
        batches=self.store.query('SELECT * FROM review_batches WHERE root_id=? ORDER BY created_at,batch_id',(root['id'],))
        active=[self._summary(b) for b in batches if not b['completed_at']]
        recent=[self._summary(b) for b in sorted(batches,key=lambda b:b['completed_at'] or '',reverse=True) if b['completed_at']][:10]
        return {'current':active[0] if active else None,'active_batches':active,'recent_archives':recent,
            'archives_count':sum(bool(b['completed_at']) for b in batches),
            'needs_next_batch':bool(recent and not active)}

    def archive_content(self,batch_id):
        """Return the immutable UTF-8 JSON text, even if the optional file write failed."""
        batch_id=self._batch_id(batch_id);root=self.store.current_root()
        row=self.store.one('SELECT archive_data FROM review_batches WHERE batch_id=? AND root_id=?',
            (batch_id,root['id'] if root else -1))
        if not row or not row['archive_data']:raise ValueError('批次尚未完成审核或不属于当前目录')
        return row['archive_data']
