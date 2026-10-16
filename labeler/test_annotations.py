import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
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

    def negative(self):
        return {'id':'hole1','x':.65,'y':.1,'w':.1,'h':.1,'reason':'process_hole','source_box_id':'machine-hole'}

    def test_confirmed_negative_region_is_local_and_old_client_preserves_it(self):
        before=self.human();body=self.payload();body['negative_regions']=[self.negative()]
        result=self.store.annotations.save(self.id,body)
        self.assertEqual(result['document']['negative_regions'],[self.negative()])
        old_client=self.payload();self.store.annotations.save(self.id,old_client)
        self.assertEqual(self.store.annotations.detail(self.id)['document']['negative_regions'],[self.negative()])
        exported=self.store.annotations.feedback_export()['items']
        self.assertEqual(len(exported),2)
        self.assertFalse(exported[0]['training_eligible_by_label_policy'])
        self.assertTrue(exported[1]['training_eligible_by_label_policy'])
        self.assertTrue(exported[1]['source_sha256_verified'])
        self.assertFalse(exported[1]['feedback']['whole_image_normal'])
        self.assertEqual(self.human(),before)

    def test_draft_negative_region_never_becomes_training_feedback(self):
        body=self.payload();body.update(status='draft',negative_regions=[self.negative()])
        self.store.annotations.save(self.id,body)
        self.assertEqual(self.store.annotations.feedback_export()['items'],[])
        self.assertFalse(self.store.annotations.export()['items'][0]['training_eligible_by_label_policy'])

    def test_negative_only_confirmation_and_overlap_protection(self):
        body=self.payload();body.update(boxes=[],outcome='needs_correction',negative_regions=[self.negative()])
        result=self.store.annotations.save(self.id,body)
        self.assertEqual(result['document']['outcome'],'needs_correction')
        self.assertTrue(self.store.annotations.feedback_export()['items'][0]['training_eligible_by_label_policy'])
        conflicting=self.payload();conflicting['boxes'][0].update(x=.65,y=.1,w=.1,h=.1)
        with self.assertRaisesRegex(ValueError,'明显重叠'):self.store.annotations.save(self.id,conflicting)
        self.assertEqual(self.store.annotations.detail(self.id)['version'],1)

    def test_standalone_region_feedback_preserves_manual_annotations_and_is_idempotent(self):
        self.store.annotations.save(self.id,self.payload())
        before=self.human();annotation_before=annotation_digest(self.store.db)
        body=self.payload();body['negative_regions']=[self.negative()]
        first=self.store.annotations.save_feedback(self.id,body)
        second=self.store.annotations.save_feedback(self.id,body)
        self.assertEqual(first['feedback_id'],second['feedback_id'])
        self.assertEqual(self.human(),before)
        self.assertEqual(annotation_digest(self.store.db),annotation_before)
        state=self.store.annotations.feedback_state()
        self.assertEqual(state['confirmed_regions'],1)
        self.assertEqual(state['pending_training_images'],1)
        self.assertFalse(state['automatic_online_weight_updates'])
        feedback_ids=[r['feedback_id'] for r in self.store.annotations.feedback_export()['items']]
        self.store.annotations.record_feedback_training('trained-real-run',feedback_ids,'a'*64)
        self.assertEqual(self.store.annotations.feedback_state()['pending_training_feedback'],0)

    def test_negative_feedback_retains_source_label_and_draft_training_exclusions(self):
        body=self.payload();body['negative_regions']=[self.negative()]
        self.store.annotations.save_feedback(self.id,body)
        self.store.save_draft(self.id,{**self.label,'note':'尚未确认'})
        item=self.store.annotations.feedback_export()['items'][0]
        self.assertFalse(item['training_eligible_by_label_policy'])
        self.assertIn('classification_draft',item['exclusion_reasons'])
        with self.store.db:self.store.db.execute('DELETE FROM drafts WHERE image_id=?',(self.id,))
        self.store.commit([(self.id,{**self.label,'defects':[],'image_state':'unclear'})])
        item=self.store.annotations.feedback_export()['items'][0]
        self.assertFalse(item['training_eligible_by_label_policy'])
        self.assertIn('classification_policy_excluded',item['exclusion_reasons'])
        self.assertIn('classification_changed',item['exclusion_reasons'])
        with self.assertRaisesRegex(ValueError,'人工分类已更新'):self.store.annotations.save_feedback(self.id,body)
        Image.new('RGB',(400,200),'black').save(self.source/'board.jpg')
        item=self.store.annotations.feedback_export()['items'][0]
        self.assertFalse(item['source_sha256_verified'])

    def test_invalid_negative_region_never_writes(self):
        for field,value in [('reason','unknown'),('x',float('nan')),('w',0),('h',True),('x',.99)]:
            body=self.payload();region=self.negative();region[field]=value;body['negative_regions']=[region]
            with self.assertRaises(ValueError):self.store.annotations.save(self.id,body)
        self.assertEqual(self.store.annotations.feedback_export()['items'],[])
        self.assertEqual(self.store.query('SELECT * FROM annotation_documents'),[])

    def test_human_feedback_import_helper_preserves_frames_and_original_classification(self):
        from deployment.import_annotation_feedback import install
        self.store.annotations.save(self.id,self.payload());detail=self.store.annotations.detail(self.id)
        manifest={'schema_version':1,'type':'human_reported_local_false_positive','image_id':self.id,
            'whole_image_normal':False,'label_scope':'region_only','source':detail['source'],
            'original_classification_fingerprint':detail['label_fingerprint'],
            'rejected_region':{**self.negative(),'id':'model-hole','problem':'其他破损'}}
        path=self.base/'human-feedback.json';path.write_text(json.dumps(manifest),encoding='utf-8')
        before=self.human(),annotation_digest(self.store.db)
        result=install(self.store.data_dir,path)
        self.assertTrue(result['saved']);self.assertEqual((self.human(),annotation_digest(self.store.db)),before)
        item=self.store.annotations.feedback_export()['items'][-1]
        self.assertEqual(item['feedback']['negative_regions'][0]['reason'],'process_hole')
        self.assertTrue(item['training_eligible_by_label_policy'])

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

    def quick_payload(self, problem='崩边', roles=None):
        body=self.payload();body['boxes'][0]['problem']=problem
        if roles is not None:body['boxes'][0]['responsibilities']=roles
        body.update(classification_updates={'defects':[problem],
            'problem_responsibilities':{problem:roles} if roles is not None else {}},
            expected_classification_draft=None)
        return body

    def test_box_flow_appends_standard_responsibility_without_removing_old_classification(self):
        result=self.store.annotations.save(self.id,self.quick_payload('崩边'))
        self.assertFalse(result['labels_unchanged'])
        self.assertEqual(result['document']['boxes'][0]['responsibilities'],['生产'])
        label=self.store.detail(self.id)['label']
        self.assertEqual(label['defects'],['其他破损','崩边'])
        self.assertEqual(label['responsibilities'],['运输','生产'])
        self.assertEqual(label['problem_responsibilities'],{'其他破损':['运输'],'崩边':['生产']})
        self.assertEqual(label['note'],'人工原结果')
        self.assertEqual(result['label_fingerprint'],self.store.annotations.detail(self.id)['label_fingerprint'])
        self.assertTrue(self.store.annotations.export()['items'][0]['training_eligible_by_label_policy'])
        self.assertTrue(self.store.annotations.feedback_export()['items'][0]['training_eligible_by_label_policy'])
        self.store.undo()
        self.assertEqual(json.loads(self.store.one('SELECT data FROM labels WHERE image_id=?',(self.id,))['data']),self.label)
        self.assertEqual(len(self.store.annotations.detail(self.id)['document']['boxes']),1)
        self.assertFalse(self.store.annotations.export()['items'][0]['training_eligible_by_label_policy'])

    def test_box_flow_common_problem_requires_explicit_responsibility_and_known_roles(self):
        for roles in (None,[],['不存在的责任']):
            before=self.human()
            with self.assertRaisesRegex(ValueError,'责任'):
                self.store.annotations.save(self.id,self.quick_payload('划伤',roles))
            self.assertEqual(self.human(),before)
            self.assertEqual(self.store.query('SELECT * FROM annotation_documents'),[])
        result=self.store.annotations.save(self.id,self.quick_payload('划伤',['运输','安装']))
        self.assertEqual(result['document']['boxes'][0]['responsibilities'],['运输','安装'])
        self.assertEqual(result['label']['problem_responsibilities']['划伤'],['运输','安装'])

    def test_box_flow_material_default_and_pending_manual_responsibility(self):
        self.store.add_tag('defect','五金质量')
        body=self.quick_payload('五金质量')
        body['boxes'].append({'id':'scratch','problem':'划伤','x':.5,'y':.1,'w':.2,'h':.2,'responsibilities':['待核']})
        result=self.store.annotations.save(self.id,body)
        self.assertEqual(result['document']['boxes'][0]['responsibilities'],['原材料'])
        self.assertEqual(result['document']['boxes'][1]['responsibilities'],['待核'])
        self.assertEqual(result['label']['problem_responsibilities']['划伤'],['待核'])

    def test_box_flow_draft_retains_roles_and_updates_but_never_changes_or_trains_labels(self):
        before=self.human();body=self.quick_payload('划伤',['安装']);body['status']='draft'
        result=self.store.annotations.save(self.id,body)
        self.assertTrue(result['labels_unchanged']);self.assertEqual(self.human(),before)
        self.assertEqual(result['document']['boxes'][0]['responsibilities'],['安装'])
        self.assertEqual(result['document']['classification_updates']['defects'],['划伤'])
        self.assertFalse(self.store.annotations.export()['items'][0]['training_eligible_by_label_policy'])
        self.assertEqual(self.store.annotations.feedback_export()['items'],[])
        body=self.quick_payload('划伤',['安装']);body['classification_updates']=result['document']['classification_updates']
        confirmed=self.store.annotations.save(self.id,body)
        self.assertEqual(confirmed['document']['boxes'][0]['responsibilities'],['安装'])
        self.assertTrue(self.store.annotations.export()['items'][0]['training_eligible_by_label_policy'])

    def test_box_flow_concurrency_is_atomic_and_keeps_existing_manual_boxes(self):
        self.store.annotations.save(self.id,self.payload());body=self.quick_payload()
        body['boxes'].append({'id':'new','problem':'划伤','x':.5,'y':.1,'w':.2,'h':.2,'responsibilities':['运输']})
        old_box=dict(self.store.annotations.detail(self.id)['document']['boxes'][0])
        old_history=self.store.query('SELECT * FROM annotation_history')
        for field,value in [('label_fingerprint','stale'),('expected_version',0)]:
            invalid={**body,field:value};before=self.human()
            with self.assertRaises(ValueError):self.store.annotations.save(self.id,invalid)
            self.assertEqual(self.human(),before)
            self.assertEqual(self.store.query('SELECT * FROM annotation_history'),old_history)
        result=self.store.annotations.save(self.id,body)
        saved=result['document']['boxes'][0]
        self.assertTrue(all(saved[key]==value for key,value in old_box.items()))
        self.assertEqual(len(result['document']['boxes']),2)

    def test_box_flow_current_visible_draft_is_merged_only_with_exact_fingerprint(self):
        draft={**self.label,'defects':['划伤'],'responsibilities':['安装'],
            'problem_responsibilities':{'划伤':['安装']},'note':'本次可见备注'}
        saved=self.store.save_draft(self.id,draft);body=self.quick_payload()
        before=self.human()
        with self.assertRaisesRegex(ValueError,'分类暂存已变化'):self.store.annotations.save(self.id,body)
        self.assertEqual(self.human(),before)
        body['expected_classification_draft']=saved['draft_fingerprint']
        result=self.store.annotations.save(self.id,body)
        self.assertEqual(result['label']['defects'],['其他破损','划伤','崩边'])
        self.assertEqual(result['label']['note'],'本次可见备注')
        self.assertFalse(self.store.detail(self.id)['has_draft'])
        self.assertTrue(self.store.annotations.export()['items'][0]['training_eligible_by_label_policy'])
        body=self.quick_payload();body['expected_classification_draft']=saved['draft_fingerprint']
        with self.assertRaisesRegex(ValueError,'分类暂存已变化'):self.store.annotations.save(self.id,body)

    def test_box_flow_image_states_cannot_be_changed_to_bypass_training_exclusion(self):
        for state in ('normal','unclear','unrelated'):
            self.store.commit([(self.id,{'defects':[],'responsibilities':['待核'],'image_state':state,'note':'保留状态'})])
            before=self.human()
            with self.assertRaisesRegex(ValueError,'图片已有状态'):self.store.annotations.save(self.id,self.quick_payload())
            self.assertEqual(self.human(),before)
        self.store.commit([(self.id,self.label)])
        draft=self.store.save_draft(self.id,{'defects':[],'responsibilities':['待核'],'image_state':'unclear'})
        body=self.quick_payload();body['expected_classification_draft']=draft['draft_fingerprint']
        with self.assertRaisesRegex(ValueError,'图片已有状态'):self.store.annotations.save(self.id,body)

    def test_old_client_preserves_box_roles_without_optin_classification_changes(self):
        first=self.store.annotations.save(self.id,self.quick_payload())
        body=self.payload();body['boxes']=first['document']['boxes']
        del body['boxes'][0]['responsibilities']
        before=self.human();result=self.store.annotations.save(self.id,body)
        self.assertEqual(self.human(),before);self.assertTrue(result['labels_unchanged'])
        self.assertEqual(result['document']['boxes'][0]['responsibilities'],['生产'])

    def test_box_flow_failed_feedback_write_rolls_back_labels_frames_and_history_together(self):
        before=self.human();body=self.quick_payload()
        with patch.object(self.store.annotations,'_record_feedback',side_effect=ValueError('反馈写入失败')):
            with self.assertRaisesRegex(ValueError,'反馈写入失败'):self.store.annotations.save(self.id,body)
        self.assertEqual(self.human(),before)
        self.assertEqual(self.store.query('SELECT * FROM annotation_documents'),[])
        self.assertEqual(self.store.query('SELECT * FROM annotation_history'),[])

    def test_manual_responsibility_override_replaces_old_default_and_keeps_other_issue(self):
        original={'defects':['崩边','划伤'],'responsibilities':['生产','安装'],
            'problem_responsibilities':{'崩边':['生产'],'划伤':['安装']},'image_state':'defect','note':'原备注'}
        self.store.commit([(self.id,original)])
        body=self.quick_payload('崩边',['运输'])
        body['boxes'][0]['responsibility_override']=True
        body['classification_updates']['responsibility_overrides']=['崩边']
        result=self.store.annotations.save(self.id,body)
        self.assertEqual(result['document']['boxes'][0]['responsibilities'],['运输'])
        self.assertTrue(result['document']['boxes'][0]['responsibility_override'])
        self.assertEqual(result['label']['defects'],['崩边','划伤'])
        self.assertEqual(result['label']['problem_responsibilities'],{'崩边':['运输'],'划伤':['安装']})
        self.assertEqual(result['label']['responsibilities'],['运输','安装'])
        self.assertEqual(result['label']['note'],'原备注')
        self.assertTrue(self.store.annotations.feedback_export()['items'][0]['training_eligible_by_label_policy'])
        self.store.undo()
        self.assertEqual(json.loads(self.store.one('SELECT data FROM labels WHERE image_id=?',(self.id,))['data']),original)

    def test_nonstandard_explicit_box_responsibility_is_not_forced_back_to_rule(self):
        result=self.store.annotations.save(self.id,self.quick_payload('崩边',['运输']))
        self.assertEqual(result['document']['boxes'][0]['responsibilities'],['运输'])
        self.assertEqual(result['label']['problem_responsibilities']['崩边'],['运输'])

    def test_override_aggregates_current_boxes_without_reintroducing_old_default(self):
        original={**self.label,'defects':['崩边'],'responsibilities':['生产'],
            'problem_responsibilities':{'崩边':['生产']}}
        self.store.commit([(self.id,original)])
        body=self.quick_payload('崩边',['运输']);body['boxes'][0]['responsibility_override']=True
        body['boxes'].append({'id':'second','problem':'崩边','x':.5,'y':.1,'w':.2,'h':.2,
            'responsibilities':['安装'],'responsibility_override':True})
        body['classification_updates']['problem_responsibilities']['崩边']=['运输','安装']
        body['classification_updates']['responsibility_overrides']=['崩边']
        result=self.store.annotations.save(self.id,body)
        self.assertEqual(result['label']['problem_responsibilities']['崩边'],['运输','安装'])
        self.assertEqual(result['label']['responsibilities'],['运输','安装'])

    def test_override_does_not_remove_role_still_used_by_another_issue(self):
        original={**self.label,'defects':['崩边','其他破损'],'responsibilities':['生产'],
            'problem_responsibilities':{'崩边':['生产'],'其他破损':['生产']}}
        self.store.commit([(self.id,original)])
        body=self.quick_payload('崩边',['运输']);body['boxes'][0]['responsibility_override']=True
        result=self.store.annotations.save(self.id,body)
        self.assertEqual(result['label']['problem_responsibilities']['崩边'],['运输'])
        self.assertEqual(result['label']['problem_responsibilities']['其他破损'],['生产'])
        self.assertEqual(result['label']['responsibilities'],['运输','生产'])

    def test_reset_to_default_is_explicit_and_replaces_prior_manual_mapping(self):
        original={**self.label,'defects':['崩边'],'responsibilities':['运输'],
            'problem_responsibilities':{'崩边':['运输']}}
        self.store.commit([(self.id,original)])
        body=self.quick_payload('崩边',['生产']);body['boxes'][0]['responsibility_override']=True
        result=self.store.annotations.save(self.id,body)
        self.assertEqual(result['label']['problem_responsibilities']['崩边'],['生产'])
        self.assertEqual(result['label']['responsibilities'],['生产'])
        self.assertEqual(result['document']['classification_updates']['responsibility_overrides'],['崩边'])

    def test_override_empty_can_be_drafted_but_cannot_confirm_or_train(self):
        body=self.quick_payload('崩边',[]);body['boxes'][0]['responsibility_override']=True
        body['classification_updates']['responsibility_overrides']=['崩边'];body['status']='draft'
        before=self.human();saved=self.store.annotations.save(self.id,body)
        self.assertEqual(saved['document']['boxes'][0]['responsibilities'],[])
        self.assertTrue(saved['document']['boxes'][0]['responsibility_override'])
        self.assertEqual(self.human(),before)
        self.assertFalse(self.store.annotations.export()['items'][0]['training_eligible_by_label_policy'])
        body['expected_version']=saved['version'];body['status']='confirmed'
        with self.assertRaisesRegex(ValueError,'选择责任'):self.store.annotations.save(self.id,body)
        self.assertEqual(self.human(),before)
        self.assertEqual(self.store.annotations.detail(self.id)['version'],saved['version'])
        self.assertEqual(self.store.annotations.feedback_export()['items'],[])

    def test_old_client_omitting_override_flag_preserves_it_and_current_responsibility(self):
        body=self.quick_payload('崩边',['运输']);body['boxes'][0]['responsibility_override']=True
        first=self.store.annotations.save(self.id,body)
        body=self.payload();body['boxes']=first['document']['boxes']
        del body['boxes'][0]['responsibility_override'];del body['boxes'][0]['responsibilities']
        before=self.human();second=self.store.annotations.save(self.id,body)
        self.assertTrue(second['document']['boxes'][0]['responsibility_override'])
        self.assertEqual(second['document']['boxes'][0]['responsibilities'],['运输'])
        self.assertEqual(self.human(),before)

    def test_invalid_override_contract_is_rejected_atomically(self):
        variants=[('box_flag','true'),('list','崩边'),('list',['未知问题']),('list',[True])]
        for field,value in variants:
            body=self.quick_payload('崩边',['运输'])
            if field=='box_flag':body['boxes'][0]['responsibility_override']=value
            else:body['classification_updates']['responsibility_overrides']=value
            before=self.human()
            with self.assertRaisesRegex(ValueError,'纠正'):self.store.annotations.save(self.id,body)
            self.assertEqual(self.human(),before)
            self.assertEqual(self.store.query('SELECT * FROM annotation_documents'),[])

    def test_explicitly_removing_only_conflicting_regions_allows_confirmation(self):
        retained=[{'id':'normal-'+str(i),'x':.65+i*.04,'y':.65,'w':.03,'h':.1,
            'reason':'process_hole' if i%2 else 'false_positive','source_box_id':'removed-'+str(i)}
            for i in range(6)]
        conflicting=[{'id':'conflict-1','x':.1,'y':.2,'w':.3,'h':.4,'reason':'false_positive'},
            {'id':'conflict-2','x':.11,'y':.21,'w':.28,'h':.38,'reason':'false_positive'}]
        body=self.quick_payload('崩边',['运输']);body['boxes'][0]['responsibility_override']=True
        body.update(status='draft',negative_regions=conflicting+retained)
        before=self.human();draft=self.store.annotations.save(self.id,body)
        self.assertEqual(self.human(),before)
        self.assertFalse(self.store.annotations.export()['items'][0]['training_eligible_by_label_policy'])
        body.update(status='confirmed',expected_version=draft['version'])
        with self.assertRaisesRegex(ValueError,'明显重叠'):self.store.annotations.save(self.id,body)
        self.assertEqual(self.human(),before)
        self.assertEqual(self.store.annotations.detail(self.id)['version'],draft['version'])
        # The UI explicitly keeps the positive and removes just these two conflicting ROIs.
        body['negative_regions']=retained
        result=self.store.annotations.save(self.id,body)
        self.assertEqual(result['document']['negative_regions'],retained)
        self.assertEqual(len(result['document']['boxes']),1)
        for field in ('id','problem','x','y','w','h','responsibilities','responsibility_override'):
            self.assertEqual(result['document']['boxes'][0][field],body['boxes'][0][field])
        self.assertEqual(result['label']['problem_responsibilities']['其他破损'],['运输'])
        self.assertEqual(result['label']['problem_responsibilities']['崩边'],['运输'])
        self.assertEqual(result['label']['note'],'人工原结果')
        self.assertTrue(self.store.annotations.export()['items'][0]['training_eligible_by_label_policy'])
        feedback=self.store.annotations.feedback_export()['items']
        self.assertEqual(len(feedback),1)
        self.assertEqual(feedback[0]['feedback']['negative_regions'],retained)
        self.assertTrue(feedback[0]['training_eligible_by_label_policy'])


if __name__=='__main__':unittest.main()
