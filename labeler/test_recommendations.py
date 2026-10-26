"""Isolated model-integration checks; never touch production labels or NAS files."""
import json
from pathlib import Path
import tempfile
import threading
import time
import unittest
from PIL import Image
from app import Application
from recommendations import Recommendations


class RecommendationTest(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.base=Path(self.temp.name)
        self.source=self.base/'source';self.source.mkdir()
        for i in range(4):Image.new('RGB',(80,100),(i*20,90,100)).save(self.source/f'{i}.jpg')
        self.app=Application(self.base/'data',str(self.source))
        while self.app.scanner.status['running']:time.sleep(.01)
        self.store=self.app.store;self.ids=[r['id'] for r in self.store.query('SELECT id FROM images ORDER BY id')]
        self.meta={'model_id':'isolated-test','onnx_sha256':'test','training_policy':'problem_points_only_v2',
            'responsibility_output':False,'problems':['崩边','划伤'],'threshold':.125}
        self.label={'defects':['崩边'],'responsibilities':['运输'],'image_state':'defect','note':'保留原人工判断'}

    def tearDown(self):
        self.store.db.close();self.temp.cleanup()

    def contents(self):
        return {table:self.store.query('SELECT * FROM '+table+' ORDER BY 1')
                for table in ['labels','drafts','tags','history']}

    def service(self,predictor):
        self.rec=Recommendations(self.store,self.app.images,predictor=predictor,metadata=self.meta)
        return self.rec

    def finish(self):
        self.rec.jobs.join();self.rec.db.close()

    def test_suggestions_and_cache_never_write_labels_or_drafts(self):
        self.store.commit([(self.ids[0],self.label)])
        self.store.save_draft(self.ids[1],self.label)
        before=self.contents();calls=[]
        rec=self.service(lambda content:calls.append(len(content)) or [.8,.01])
        try:
            self.assertEqual(rec.get(self.ids[0])['status'],'protected')
            self.assertEqual(rec.get(self.ids[1])['status'],'protected')
            self.assertEqual(rec.get(self.ids[2])['status'],'queued');rec.jobs.join()
            result=rec.get(self.ids[2]);self.assertEqual(result['status'],'ready')
            self.assertEqual(result['candidates'],[{'name':'崩边','score':.8}])
            self.assertNotIn('responsibilities',result)
            rec.get(self.ids[2]);self.assertEqual(len(calls),1)
            self.assertEqual(self.contents(),before)
            self.assertEqual(rec.db.execute('SELECT count(*) FROM cache').fetchone()[0],1)
        finally:self.finish()

    def test_concurrent_human_classification_is_never_overwritten(self):
        started=threading.Event();release=threading.Event()
        def slow(content):started.set();release.wait(5);return [.8,.2]
        rec=self.service(slow)
        try:
            rec.get(self.ids[0]);self.assertTrue(started.wait(3))
            self.store.commit([(self.ids[0],self.label)]);before=self.contents();release.set();rec.jobs.join()
            self.assertEqual(rec.get(self.ids[0])['status'],'protected')
            for method in [lambda:self.store.save_draft(self.ids[0],{**self.label,'defects':['划伤']},True),
                           lambda:self.store.commit([(self.ids[0],{**self.label,'defects':['划伤']})],only_unclassified=True)]:
                with self.assertRaisesRegex(ValueError,'已有人工分类'):method()
            self.assertEqual(self.contents(),before)
        finally:release.set();self.finish()

    def test_file_and_model_identity_invalidate_cached_suggestions(self):
        calls=[];rec=self.service(lambda content:calls.append(len(content)) or [.2,.8])
        try:
            rec.get(self.ids[0]);rec.jobs.join();first=rec.get(self.ids[0])
            Image.new('RGB',(150,150),'red').save(self.source/'0.jpg')
            self.assertEqual(rec.get(self.ids[0])['status'],'source_changed')
            self.app.scanner.start(self.store.current_root())
            while self.app.scanner.status['running']:time.sleep(.01)
            rec.get(self.ids[0]);rec.jobs.join();second=rec.get(self.ids[0])
            self.assertNotEqual(first['source_sha256'],second['source_sha256'])
            rec.meta={**self.meta,'model_id':'another-version'}
            rec.get(self.ids[0]);rec.jobs.join();third=rec.get(self.ids[0])
            self.assertEqual(third['model_id'],'another-version');self.assertEqual(len(calls),3)
        finally:self.finish()

    def test_missing_model_keeps_manual_labeler_available(self):
        rec=Recommendations(self.store,self.app.images,self.base/'missing-model')
        self.assertFalse(rec.state()['enabled'])
        self.assertEqual(rec.get(self.ids[0])['status'],'disabled')
        self.assertEqual(self.store.counts()['classified'],0)

    def test_concurrent_drafts_are_preserved_and_own_draft_can_be_confirmed(self):
        image_id=self.ids[0]
        self.store.save_draft(image_id,self.label);before=self.contents()
        with self.assertRaisesRegex(ValueError,'已有其他暂存'):
            self.store.save_draft(image_id,{**self.label,'defects':['划伤']},True)
        with self.assertRaisesRegex(ValueError,'已有其他暂存'):
            self.store.commit([(image_id,{**self.label,'note':'do not overwrite'})],only_unclassified=True)
        self.assertEqual(self.contents(),before)
        own=self.ids[1]
        saved=self.store.save_draft(own,self.label,True)
        changed={**self.label,'note':'explicit human edit'}
        updated=self.store.save_draft(own,changed,True,saved['draft_fingerprint'])
        self.store.commit([(own,changed)],only_unclassified=True,expected_draft=updated['draft_fingerprint'])
        self.assertEqual(self.store.detail(own)['label'],changed)

    def test_unrelated_suggestion_never_suppresses_problems_or_writes_labels(self):
        self.meta={**self.meta,'state_filter':{'training_policy':'unrelated_auxiliary_v1','model_id':'aux-test',
            'threshold':.8,'automatic_skip':False,'suppress_problem_candidates':False}}
        class Fake:
            def predict(self,content):return {'scores':[.9,.7],'unrelated_score':.99}
        before=self.contents();rec=self.service(Fake())
        try:
            rec.get(self.ids[0]);rec.jobs.join();result=rec.get(self.ids[0])
            self.assertEqual(result['image_state_suggestion']['state'],'unrelated')
            self.assertEqual([c['name'] for c in result['candidates']],['崩边','划伤'])
            self.assertTrue(result['image_state_suggestion']['requires_human_confirmation'])
            self.assertEqual(self.contents(),before)
        finally:self.finish()

    def prepare_review(self, image_id):
        self.store.review_model='review-test'
        with self.store.lock,self.store.db:
            self.store.db.execute("INSERT OR REPLACE INTO settings VALUES('review_model','review-test')")
        image=self.store.image(image_id);stat=self.store.path(image).stat()
        result={'status':'ready','model_id':'review-test','source_sha256':'isolated',
                'image_state_suggestion':{'state':'unrelated','score':.99}}
        self.store.record_model_review(image,result,(stat.st_size,stat.st_mtime_ns))
        return image,result,(stat.st_size,stat.st_mtime_ns)

    def test_model_queue_separates_predictions_from_human_labels(self):
        before=self.contents();self.prepare_review(self.ids[0])
        self.assertEqual(self.contents(),before)
        self.assertEqual(self.store.listing({'status':['model_unrelated']})['total'],1)
        self.assertEqual(self.store.listing({'status':['unclassified']})['total'],3)
        self.assertEqual(self.store.listing({'status':['all_unclassified']})['total'],4)
        self.store.save_draft(self.ids[0],self.label)
        self.assertEqual(self.store.listing({'status':['model_unrelated']})['total'],0)
        self.assertEqual(self.store.listing({'status':['unclassified']})['total'],4)

    def test_review_batch_rechecks_concurrent_edits_and_can_undo(self):
        for image_id in self.ids[:2]:self.prepare_review(image_id)
        self.store.save_draft(self.ids[1],self.label);before=self.contents()
        with self.assertRaisesRegex(ValueError,'已有其他暂存'):
            self.store.review_unrelated(self.ids[:2],True)
        self.assertEqual(self.contents(),before)
        self.store.review_unrelated([self.ids[0]],True)
        self.assertEqual(self.store.detail(self.ids[0])['label']['image_state'],'unrelated')
        self.store.undo()
        self.assertFalse(self.store.detail(self.ids[0])['classified'])
        self.assertTrue(self.store.detail(self.ids[0])['has_draft'])

    def test_rejected_decision_stays_rejected_and_changed_source_cannot_confirm(self):
        image,result,stamp=self.prepare_review(self.ids[0]);before=self.contents()
        self.store.review_unrelated([self.ids[0]],False)
        self.store.record_model_review(image,result,stamp)
        self.assertEqual(self.contents(),before)
        self.assertEqual(self.store.listing({'status':['model_unrelated']})['total'],0)
        self.prepare_review(self.ids[1]);Image.new('RGB',(300,200),'red').save(self.source/'1.jpg')
        with self.assertRaisesRegex(ValueError,'原图发生变化'):
            self.store.review_unrelated([self.ids[1]],True)
        self.assertEqual(self.contents(),before)


if __name__=='__main__':unittest.main()
