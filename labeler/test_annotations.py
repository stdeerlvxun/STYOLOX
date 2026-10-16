import json
from pathlib import Path
import tempfile
import unittest
from PIL import Image
from app import Store,RESPONSIBILITY_RULES
from annotations import responsibility_links
from deployment.snapshot import digest,annotation_digest
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent/'deployment'))
from recover_error_reviews import recover


class LocalizationTest(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.base=Path(self.temp.name)
        self.source=self.base/'source';self.source.mkdir()
        Image.new('RGB',(400,200),'white').save(self.source/'board.jpg')
        self.store=Store(self.base/'data');root=self.store.set_root(str(self.source))
        stat=(self.source/'board.jpg').stat()
        with self.store.db:
            self.store.db.execute('INSERT INTO images(root_id,relative_path,file_name,case_id,size,mtime_ns) VALUES(?,?,?,?,?,?)',
                (root['id'],'board.jpg','board.jpg','case-1',stat.st_size,stat.st_mtime_ns))
        self.id=self.store.one('SELECT id FROM images')['id']
        self.label={'defects':['其他破损'],'responsibilities':['运输'],'problem_responsibilities':{'其他破损':['运输']},'image_state':'defect','note':'人工原结果'}
        self.store.commit([(self.id,self.label)])

    def tearDown(self):self.store.db.close();self.temp.cleanup()

    def human(self):
        return {name:self.store.query('SELECT * FROM '+name+' ORDER BY 1') for name in ['labels','drafts','tags','history']}

    def payload(self):
        detail=self.store.annotations.detail(self.id)
        return {'source_sha256':detail['source']['sha256'],'label_fingerprint':detail['label_fingerprint'],
            'review_run':detail['review']['run_id'] if detail['review'] else None,
            'expected_version':detail['version'],'status':'confirmed','outcome':'boxed',
            'boxes':[{'id':'box1','problem':'崩边','x':.1,'y':.2,'w':.3,'h':.4}]}

    def test_supplement_preserves_human_classification_and_roundtrips(self):
        before=self.human();saved=self.store.annotations.save(self.id,self.payload())
        self.assertEqual(saved['version'],1);self.assertEqual(self.human(),before)
        self.assertEqual(len(digest(self.store.db)),64);self.assertEqual(len(annotation_digest(self.store.db)),64)
        exported=self.store.annotations.export()['items'][0]
        self.assertTrue(exported['training_eligible_by_label_policy'])
        self.assertEqual(exported['annotation']['source']['width'],400)
        self.assertEqual(self.store.annotations.detail(self.id)['document']['boxes'][0]['problem'],'崩边')
        self.assertEqual(self.store.listing({'status':['annotated']})['total'],1)

    def test_concurrent_annotation_and_classification_cannot_be_overwritten(self):
        first=self.payload();self.store.annotations.save(self.id,first)
        with self.assertRaisesRegex(ValueError,'其他页面更新'):self.store.annotations.save(self.id,first)
        changed=self.payload();self.store.commit([(self.id,{**self.label,'note':'其他页面的确认'})])
        before=self.human()
        with self.assertRaisesRegex(ValueError,'人工分类已更新'):self.store.annotations.save(self.id,changed)
        self.assertEqual(self.human(),before)
        self.assertFalse(self.store.annotations.export()['items'][0]['training_eligible_by_label_policy'])

    def test_invalid_or_changed_source_is_rejected(self):
        for invalid in [float('nan'),True,-.1,1.1]:
            body=self.payload();body['boxes'][0]['x']=invalid
            with self.assertRaises(ValueError):self.store.annotations.save(self.id,body)
        body=self.payload();Image.new('RGB',(200,400),'black').save(self.source/'board.jpg')
        with self.assertRaisesRegex(ValueError,'原图已变化'):self.store.annotations.save(self.id,body)
        self.assertEqual(self.store.query('SELECT * FROM annotation_documents'),[])

    def test_review_completion_and_training_state_exclusion(self):
        with self.store.db:
            self.store.db.execute("INSERT INTO settings VALUES('error_review_run','test-run')")
            self.store.db.execute("INSERT INTO error_reviews(image_id,run_id,data) VALUES(?,'test-run',?)",
                (self.id,json.dumps({'kinds':['problem_miss']})))
        self.assertEqual(self.store.listing({'status':['error_review'],'review_kind':['problem_miss']})['total'],1)
        self.store.annotations.save(self.id,self.payload())
        self.assertEqual(self.store.listing({'status':['error_review']})['total'],0)
        self.assertEqual(self.store.listing({'status':['error_reviewed']})['total'],1)
        for state in ['normal','unclear','unrelated']:
            self.store.commit([(self.id,{'defects':[],'image_state':state,'responsibilities':['待核']})])
            body=self.payload();body['outcome']='unclear'
            self.store.annotations.save(self.id,body)
            self.assertFalse(self.store.annotations.export()['items'][0]['training_eligible_by_label_policy'])

    def test_responsibility_prior_deduplicates_cases_and_preserves_rules(self):
        original=self.store.image(self.id)
        with self.store.db:
            self.store.db.execute('INSERT INTO images(root_id,relative_path,file_name,case_id,size,mtime_ns) VALUES(?,?,?,?,?,?)',
                (original['root_id'],'second.jpg','second.jpg','case-1',original['size'],original['mtime_ns']))
        second=self.store.one('SELECT MAX(id) id FROM images')['id'];self.store.commit([(second,self.label)])
        values=responsibility_links(self.store,RESPONSIBILITY_RULES)
        other=values['其他破损'];self.assertEqual(other['suggestion'],'运输');self.assertEqual(other['resolved_cases'],1)
        self.assertEqual(other['image_count'],2);self.assertEqual(other['responsibility_case_counts']['运输'],1)
        self.assertTrue(other['requires_confirmation']);self.assertEqual(values['崩边']['suggestion'],'生产')
        self.assertEqual(self.store.detail(self.id)['label'],self.label)

    def queue(self):
        with self.store.db:
            self.store.db.execute("INSERT INTO settings VALUES('error_review_run','test-run')")
            self.store.db.execute("INSERT INTO error_reviews(image_id,run_id,data) VALUES(?,'test-run',?)",
                (self.id,json.dumps({'kinds':['problem_miss']})))

    def test_classification_review_commit_is_atomic_and_undo_restores_queue(self):
        self.queue();annotation_before=annotation_digest(self.store.db)
        result=self.store.commit([(self.id,self.label)],review_run='test-run')
        self.assertEqual(result['review_completed'],1)
        self.assertEqual(self.store.listing({'status':['error_review']})['total'],0)
        self.assertEqual(annotation_digest(self.store.db),annotation_before)
        self.assertEqual(self.store.detail(self.id)['error_review']['status'],'reviewed')
        self.store.undo()
        self.assertEqual(self.store.listing({'status':['error_review']})['total'],1)
        before=digest(self.store.db)
        with self.assertRaisesRegex(ValueError,'批次已变化'):
            self.store.commit([(self.id,self.label)],review_run='stale-run')
        self.assertEqual(digest(self.store.db),before)

    def test_history_recovery_preserves_labels_and_never_creates_boxes(self):
        self.queue();boundary=self.store.one('SELECT MAX(id) id FROM history')['id']
        self.store.commit([(self.id,self.label)])
        before=self.human();annotations_before=annotation_digest(self.store.db)
        preview=recover(self.store.data_dir,'test-run',boundary)
        self.assertEqual(preview['recoverable'],1)
        self.assertEqual(self.store.annotations.review(self.id)['status'],'pending')
        result=recover(self.store.data_dir,'test-run',boundary,True,1)
        self.assertTrue(result['applied']);self.assertEqual(self.human(),before)
        self.assertEqual(annotation_digest(self.store.db),annotations_before)
        self.assertEqual(self.store.query('SELECT * FROM annotation_documents'),[])
        self.assertEqual(self.store.annotations.review(self.id)['status'],'reviewed')

    def test_history_recovery_requires_exact_current_confirmed_label(self):
        self.queue();boundary=self.store.one('SELECT MAX(id) id FROM history')['id']
        self.store.commit([(self.id,self.label)])
        self.store.save_draft(self.id,{**self.label,'note':'新编辑尚未确认'})
        self.assertEqual(recover(self.store.data_dir,'test-run',boundary)['recoverable'],0)
        with self.assertRaisesRegex(ValueError,'Recoverable count changed'):
            recover(self.store.data_dir,'test-run',boundary,True,1)
        self.assertEqual(self.store.annotations.review(self.id)['status'],'pending')


if __name__=='__main__':unittest.main()
