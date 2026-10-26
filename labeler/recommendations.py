"""Read-only model suggestions. This module NEVER writes human labels or drafts."""
from __future__ import annotations
import hashlib
import io
import json
from pathlib import Path
import queue
import sqlite3
import threading
import time
from PIL import Image, ImageOps


def image_views(content):
    import numpy as np
    with Image.open(io.BytesIO(content)) as original:
        image=ImageOps.exif_transpose(original).convert('RGB')
    full=ImageOps.pad(image,(224,224),color=(124,116,104))
    width,height=image.size;side=max(1,int(min(width,height)*.75))
    positions=[(0,0),(width-side,0),(0,height-side),(width-side,height-side),
               ((width-side)//2,(height-side)//2)]
    crops=[full]+[image.crop((x,y,x+side,y+side)).resize((224,224),Image.Resampling.BILINEAR)
                  for x,y in positions]
    mean=np.array([.485,.456,.406],dtype=np.float32)[:,None,None]
    std=np.array([.229,.224,.225],dtype=np.float32)[:,None,None]
    return np.stack([(np.asarray(crop,dtype=np.float32).transpose(2,0,1)/255.-mean)/std
                     for crop in crops]).astype(np.float32)


class OnnxPredictor:
    def __init__(self, folder):
        import onnxruntime as ort
        self.folder=Path(folder)
        self.meta=json.loads((self.folder/'model.json').read_text(encoding='utf-8'))
        model=self.folder/'problems.onnx'
        if hashlib.sha256(model.read_bytes()).hexdigest()!=self.meta['onnx_sha256']:
            raise ValueError('Recommendation model checksum mismatch')
        options=ort.SessionOptions();options.intra_op_num_threads=2;options.inter_op_num_threads=1
        options.add_session_config_entry('session.intra_op.allow_spinning','0')
        self.session=ort.InferenceSession(str(model),sess_options=options,providers=['CPUExecutionProvider'])

    def predict(self, content):
        auxiliary=self.meta.get('state_filter')
        outputs=self.session.run(['scores','unrelated_score'] if auxiliary else ['scores'],{'views':image_views(content)})
        result={'scores':outputs[0].tolist()}
        if auxiliary:result['unrelated_score']=float(outputs[1][0])
        return result

    def __call__(self, content):
        return self.predict(content)['scores']


class Recommendations:
    def __init__(self, store, image_queue, model_dir=None, *, predictor=None, metadata=None):
        self.store=store;self.image_queue=image_queue;self.model_dir=Path(model_dir) if model_dir else None
        self.predictor=predictor;self.meta=metadata;self.error='';self.lock=threading.RLock()
        self.jobs=queue.PriorityQueue(maxsize=12);self.pending={};self.serial=0
        self.enabled=bool(model_dir or predictor)
        if not self.enabled:return
        try:
            if self.meta is None:
                self.meta=json.loads((self.model_dir/'model.json').read_text(encoding='utf-8'))
            if (self.meta['training_policy']!='problem_points_only_v2' or
                    self.meta.get('responsibility_output') is not False):
                raise ValueError('Incompatible recommendation policy')
            auxiliary=self.meta.get('state_filter')
            if auxiliary and (auxiliary['training_policy']!='unrelated_auxiliary_v1' or
                    auxiliary.get('automatic_skip') is not False or auxiliary.get('suppress_problem_candidates') is not False):
                raise ValueError('Auxiliary filter must only suggest, never skip or suppress problems')
            self.db=sqlite3.connect(store.data_dir/'recommendations.sqlite3',check_same_thread=False)
            self.db.execute('CREATE TABLE IF NOT EXISTS cache(key TEXT PRIMARY KEY,payload TEXT NOT NULL)')
            self.db.commit()
            threading.Thread(target=self.worker,daemon=True,name='problem-recommendations').start()
        except Exception as exc:
            self.enabled=False;self.error=str(exc)

    def state(self):
        return {'enabled':self.enabled,'model_id':self.meta.get('model_id') if self.meta else None,
                'human_confirmation_required':True,'responsibility_output':False,
                'automatic_classification_enabled':self.enabled,'human_confirmation_blocks_automation':False,
                'automatic_counts':self.store.automatic.counts(),
                'unrelated_suggestions_enabled':bool(self.meta and self.meta.get('state_filter'))}

    def start_review_scan(self):
        if not self.enabled:return
        self.store.review_model=self.meta['model_id']
        with self.store.lock,self.store.db:
            self.store.db.execute("INSERT OR REPLACE INTO settings VALUES('review_model',?)",(self.meta['model_id'],))
        threading.Thread(target=self.review_scan,daemon=True,name='unrelated-review-scan').start()

    def review_scan(self):
        # One image at a time, newest first, sharing the low-priority inference queue.
        # Human labels/drafts and every model decision remain in separate tables.
        while True:
            try:
                row=self.store.automatic.next_image(self.meta['model_id'])
                if not row:time.sleep(3);continue
                result=self.get(row['id'],{'prefetch':['1']})
                if result['status'] in ('queued','running'):
                    time.sleep(.25);continue
                if result['status'] in ('error','unavailable','protected','source_changed'):
                    from app import RESPONSIBILITY_RULES
                    self.store.automatic.record(row,{'model_id':self.meta['model_id']},
                        (row['size'],row['mtime_ns']),RESPONSIBILITY_RULES,error=result['status'])
                time.sleep(.15)
            except Exception as exc:
                print(f'Unrelated review scan deferred: {exc}',flush=True);time.sleep(3)

    def eligible(self, image_id):
        with self.store.lock:
            return not self.store.one('SELECT image_id FROM labels WHERE image_id=? UNION ALL '
                "SELECT image_id FROM drafts WHERE image_id=? UNION ALL SELECT image_id FROM annotation_documents "
                "WHERE image_id=? AND json_extract(data,'$.status')='confirmed'",(image_id,image_id,image_id))

    def record_result(self, record, result, stamp):
        from app import RESPONSIBILITY_RULES
        self.store.automatic.record(record,result,stamp,RESPONSIBILITY_RULES)
        self.store.record_model_review(record,result,stamp)

    def get(self, image_id, params=None):
        # Recheck on EVERY request, including cached results and completed jobs.
        if not self.eligible(image_id):return {'status':'protected','image_id':image_id}
        if not self.enabled:return {'status':'disabled','image_id':image_id}
        record=self.store.image(image_id)
        if not record['available']:return {'status':'unavailable','image_id':image_id}
        try:
            path=self.store.path(record);stat=path.stat()
        except OSError:
            return {'status':'unavailable','image_id':image_id}
        stamp=(stat.st_size,stat.st_mtime_ns)
        if stamp != (record['size'],record['mtime_ns']):
            return {'status':'source_changed','image_id':image_id}
        key=hashlib.sha256(f"{self.meta['model_id']}|{self.meta.get('onnx_sha256')}|{record['root_id']}|"
            f"{record['relative_path']}|{stamp}".encode()).hexdigest()
        params=params or {}
        view={'client':params.get('c',[''])[0],'view':params.get('v',[''])[0],
              'focus':params.get('f',[''])[0],'kind':'preview'}
        if not self.image_queue.valid(view):return {'status':'cancelled','image_id':image_id}
        with self.lock:
            cached=self.db.execute('SELECT payload FROM cache WHERE key=?',(key,)).fetchone()
            if cached:
                result=json.loads(cached[0]);self.record_result(record,result,stamp);return result
            previous=self.pending.get(key)
            if previous and previous['status']=='error' and (params.get('retry') or time.time()-previous['at']>30):
                self.pending.pop(key);previous=None
            if previous:return {k:v for k,v in previous.items() if k!='at'}
            self.serial+=1
            job={'record':record,'view':view,'stamp':stamp,'key':key}
            try:self.jobs.put_nowait((10 if params.get('prefetch') else 0,self.serial,job))
            except queue.Full:return {'status':'queued','image_id':image_id}
            result={'status':'queued','image_id':image_id};self.pending[key]=result
            return result.copy()

    def worker(self):
        while True:
            _,_,job=self.jobs.get();record=job['record'];key=job['key'];image_id=record['id']
            try:
                if not self.image_queue.valid(job['view']) or not self.eligible(image_id):continue
                with self.lock:self.pending[key]={'status':'running','image_id':image_id}
                path=self.store.path(record);content=path.read_bytes();stat=path.stat()
                if (stat.st_size,stat.st_mtime_ns)!=job['stamp']:raise ValueError('Source image changed')
                sha=hashlib.sha256(content).hexdigest()
                if record['sha256'] and sha!=record['sha256']:raise ValueError('Source checksum changed')
                if self.predictor is None:self.predictor=OnnxPredictor(self.model_dir)
                started=time.perf_counter()
                prediction=self.predictor.predict(content) if hasattr(self.predictor,'predict') else {'scores':self.predictor(content)}
                scores=prediction['scores']
                if len(scores)!=len(self.meta['problems']):raise ValueError('Unexpected score count')
                # A human may have classified/edited the image while inference ran.
                if not self.eligible(image_id):continue
                candidates=[{'name':name,'score':score} for name,score in zip(self.meta['problems'],scores)
                            if score>=self.meta['threshold']]
                candidates.sort(key=lambda c:(-c['score'],c['name']))
                result={'status':'ready','image_id':image_id,'model_id':self.meta['model_id'],
                    'source_sha256':sha,'candidates':candidates,'threshold':self.meta['threshold'],
                    'requires_human_review':True,'probabilities_calibrated':False,
                    'elapsed_seconds':round(time.perf_counter()-started,3)}
                auxiliary=self.meta.get('state_filter')
                if auxiliary:
                    score=prediction.get('unrelated_score')
                    result['image_state_suggestion']=({'state':'unrelated','name':'疑似无关图片','score':score,
                        'requires_human_confirmation':True} if score is not None and score>=auxiliary['threshold'] else None)
                    result['state_filter_model_id']=auxiliary['model_id']
                self.record_result(record,result,job['stamp'])
                with self.lock:
                    self.db.execute('INSERT OR REPLACE INTO cache VALUES(?,?)',
                        (key,json.dumps(result,ensure_ascii=False)));self.db.commit()
            except Exception as exc:
                from app import RESPONSIBILITY_RULES
                try:
                    self.store.automatic.record(record,{'model_id':self.meta['model_id']},job['stamp'],
                        RESPONSIBILITY_RULES,error=str(exc))
                except Exception as persist_error:
                    print(f'Cannot persist inference failure for {image_id}: {persist_error}',flush=True)
                with self.lock:self.pending[key]={'status':'error','image_id':image_id,'at':time.time()}
                print(f'Recommendations unavailable for {image_id}: {exc}',flush=True)
            finally:
                with self.lock:
                    if self.pending.get(key,{}).get('status')!='error':self.pending.pop(key,None)
                    if len(self.pending)>100:
                        self.pending={k:v for k,v in self.pending.items() if time.time()-v.get('at',time.time())<30}
                self.jobs.task_done()
