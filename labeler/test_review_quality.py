import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image
from app import Store
from review_quality import ReviewQuality,wilson_lower


class ReviewQualityTest(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.base=Path(self.temp.name)
        self.source=self.base/'source';self.source.mkdir()
        self.store=Store(self.base/'data');self.root=self.store.set_root(str(self.source))
        self.quality=ReviewQuality(self.store)

    def tearDown(self):self.store.db.close();self.temp.cleanup()

    def image(self,name,model='model-v1',case_id='case-1',unknown=False):
        path=self.source/(name+'.jpg');Image.new('RGB',(400,200),'white').save(path);stat=path.stat()
        with self.store.db:
            cursor=self.store.db.execute('INSERT INTO images(root_id,relative_path,file_name,case_id,size,mtime_ns) VALUES(?,?,?,?,?,?)',
                (self.root['id'],path.name,path.name,case_id,stat.st_size,stat.st_mtime_ns))
        identity=cursor.lastrowid
        if not unknown:self.store.commit([(identity,{'defects':['崩边'],'responsibilities':['生产'],'image_state':'defect'})])
        detail=self.store.annotations.detail(identity)
        manifest={'schema_version':1,'requires_human_review':True,'model_id':model,'method':'location-first','items':[
            {'image_id':identity,'relative_path':path.name,'source':detail['source'],'label_fingerprint':detail['label_fingerprint'],
            'boxes':[{'id':'box-'+name,'problem':'' if unknown else '崩边','x':.1,'y':.2,'w':.3,'h':.4,'origin':'model'}]}]}
        self.assertEqual(self.store.annotations.install_proposals(manifest)['imported'],1)
        return identity

    def body(self,identity):
        detail=self.store.annotations.detail(identity);proposal=detail['proposal']
        return {'source_sha256':detail['source']['sha256'],'label_fingerprint':detail['label_fingerprint'],
            'expected_version':detail['version'],'status':'confirmed','outcome':'boxed',
            'proposal_id':proposal['proposal_id'],'boxes':copy.deepcopy(proposal['boxes'])}

    def accept(self,identity,**changes):
        body=self.body(identity);body.update(changes)
        return self.store.annotations.save(identity,body)

    def reject(self,identity):
        proposal=self.store.annotations.detail(identity)['proposal']
        return self.store.annotations.reject_proposal(identity,proposal['proposal_id'])

    def test_position_acceptance_counts_whole_images_and_all_add_delete_move_errors(self):
        first=self.image('unchanged',case_id='one');self.accept(first)
        second=self.image('moved',case_id='two');body=self.body(second);body['boxes'][0]['x']=.2
        self.accept(second,boxes=body['boxes'])
        third=self.image('added',case_id='three');body=self.body(third)
        body['boxes'].append({'id':'manual','problem':'崩边','x':.6,'y':.6,'w':.1,'h':.1})
        self.accept(third,boxes=body['boxes'])
        fourth=self.image('rejected',case_id='four');self.reject(fourth)
        result=self.quality.state();group=result['current']
        self.assertEqual(group['reviewed_images'],4)
        self.assertEqual(group['whole_image_unchanged'],1)
        self.assertEqual(group['whole_image_unchanged_rate'],.25)
        self.assertEqual(group['reviewed_machine_boxes'],4)
        self.assertEqual(group['unchanged_position_boxes'],2)
        self.assertEqual(group['unchanged_position_rate'],.5)
        self.assertEqual((group['deleted_machine_boxes'],group['moved_machine_boxes'],group['human_added_boxes']),(1,1,1))
        self.assertEqual((group['reviewed_cases'],group['unchanged_cases']),(4,1))
        self.assertFalse(group['auto_enabled'])

    def test_unknown_problem_assignment_is_not_a_location_or_type_prediction_error(self):
        identity=self.image('unknown',unknown=True);body=self.body(identity)
        body['boxes'][0]['problem']='崩边'
        body['classification_updates']={'defects':['崩边'],'problem_responsibilities':{'崩边':['生产']}}
        self.accept(identity,**body)
        group=self.quality.state()['current']
        self.assertEqual((group['whole_image_unchanged'],group['problem_type_assigned'],group['problem_type_changed']),(1,1,0))
        self.assertEqual(group['inherited_problem_labels'],0)
        self.assertEqual(group['type_metric_scope'],'assigned_or_inherited_labels_not_model_type_accuracy')

    def test_type_change_is_reported_separately_without_claiming_type_accuracy(self):
        identity=self.image('retyped');body=self.body(identity);body['boxes'][0]['problem']='缺角'
        self.accept(identity,boxes=body['boxes'])
        group=self.quality.state()['current']
        self.assertEqual((group['whole_image_unchanged'],group['problem_type_changed'],group['inherited_problem_labels']),(1,1,1))

    def test_model_groups_are_separate_and_latest_generated_is_not_mislabeled_reviewed(self):
        first=self.image('old',model='model-old');self.accept(first)
        self.image('new',model='model-new')
        result=self.quality.state()
        self.assertEqual(result['current']['model_id'],'model-old')
        self.assertEqual(len(result['groups']),2)
        self.assertEqual(result['latest_generated_group_key'],json.dumps(['model-new','location-first'],separators=(',',':')))
        self.assertEqual(result['excluded_reasons']['not_reviewed'],1)

    def test_later_review_of_old_model_does_not_replace_the_latest_model_default(self):
        old=self.image('older-model',model='old')
        newer=self.image('newer-model',model='new');self.accept(newer);self.accept(old)
        result=self.quality.state()
        self.assertEqual(result['current']['model_id'],'new')

    def test_drafts_changed_labels_and_sources_and_unclear_are_excluded(self):
        for kind in ('classification_draft','annotation_draft','classification_changed','source_changed','unclear'):
            identity=self.image(kind);self.accept(identity)
            if kind=='classification_draft':self.store.save_draft(identity,{'defects':['崩边'],'image_state':'defect'})
            elif kind=='classification_changed':self.store.commit([(identity,{'defects':['缺角'],'responsibilities':['生产'],'image_state':'defect'})])
            elif kind=='unclear':self.store.commit([(identity,{'defects':[],'image_state':'unclear'})])
            elif kind=='source_changed':
                with self.store.db:self.store.db.execute('UPDATE images SET mtime_ns=mtime_ns+1 WHERE id=?',(identity,))
            else:
                document=self.store.annotations.detail(identity)['document'];document['status']='draft'
                with self.store.db:self.store.db.execute('UPDATE annotation_documents SET data=? WHERE image_id=?',(json.dumps(document),identity))
        result=self.quality.state();self.assertEqual(result['reviewed_images'],0)
        for reason in ('classification_draft','annotation_draft','classification_changed','source_changed','unclear'):
            self.assertEqual(result['excluded_reasons'][reason],1)

    def test_terminal_human_states_count_rejections_only_with_matching_receipt(self):
        for state in ('normal','unrelated'):
            identity=self.image(state)
            with self.store.db:self.store.db.execute("UPDATE annotation_proposals SET created_at='2026-01-01T00:00:00+00:00' WHERE image_id=?",(identity,))
            self.store.commit([(identity,{'defects':[],'image_state':state})])
        identity=self.image('without-receipt')
        with self.store.db:
            self.store.db.execute("UPDATE labels SET data=?,updated_at='2099-01-01T00:00:00+00:00' WHERE image_id=?",
                (json.dumps({'defects':[],'image_state':'normal'}),identity))
        result=self.quality.state();group=result['current']
        self.assertEqual((group['reviewed_images'],group['terminal_state_rejections'],group['deleted_machine_boxes']),(2,2,2))
        self.assertEqual(result['excluded_reasons']['state_confirmation_missing'],1)

    def test_unknown_explicit_rejection_counts_without_inventing_classification(self):
        identity=self.image('unknown-reject',unknown=True);self.reject(identity)
        group=self.quality.state()['current']
        self.assertEqual((group['reviewed_images'],group['whole_image_unchanged'],group['deleted_machine_boxes']),(1,0,1))
        self.assertIsNone(self.store.one('SELECT * FROM labels WHERE image_id=?',(identity,)))

    def test_cases_require_all_images_unchanged_and_missing_ids_are_not_independent_cases(self):
        a=self.image('same-a',case_id='same');self.accept(a)
        b=self.image('same-b',case_id='same');self.reject(b)
        c=self.image('missing-case',case_id=None);self.accept(c)
        group=self.quality.state()['current']
        self.assertEqual((group['reviewed_images'],group['reviewed_cases'],group['unchanged_cases'],group['images_without_case_id']),(3,1,0,1))
        self.assertIn('部分图片缺少售后单分组，不计独立单数',group['shortfalls'])

    def test_current_and_history_duplicates_do_not_inflate_samples(self):
        identity=self.image('duplicate');self.accept(identity)
        with self.store.db:
            self.store.db.execute("INSERT INTO annotation_proposal_history(image_id,proposal_id,data,label_data,status,created_at,reviewed_at,archived_at,replaced_by) SELECT image_id,proposal_id,data,label_data,status,created_at,reviewed_at,'now','new' FROM annotation_proposals WHERE image_id=?",(identity,))
        result=self.quality.state()
        self.assertEqual(result['scanned_proposals'],1);self.assertEqual(result['reviewed_images'],1)

    def test_only_current_root_is_read_and_no_nas_or_data_writes_occur(self):
        identity=self.image('read-only');self.accept(identity)
        before='\n'.join(self.store.db.iterdump())
        with patch.object(self.store,'path',side_effect=AssertionError('Must not access NAS')):
            self.assertEqual(self.quality.state()['reviewed_images'],1)
        self.assertEqual('\n'.join(self.store.db.iterdump()),before)
        other=self.base/'other';other.mkdir();self.store.set_root(str(other))
        self.assertEqual(self.quality.state()['scanned_proposals'],0)

    def test_business_gate_never_enables_auto_without_independent_validation(self):
        group={'reviewed_images':300,'reviewed_machine_boxes':300,'whole_image_unchanged':300,
            'unchanged_position_boxes':300,'images_without_case_id':0}
        result=ReviewQuality._finish(group,{str(n):[True,True,True] for n in range(100)})
        self.assertTrue(result['position_sample_gate_met'])
        self.assertFalse(result['auto_enabled'])
        self.assertGreater(result['case_rate_lower95'],.95)
        self.assertLess(result['case_rate_lower95'],result['whole_image_rate_lower95'])
        self.assertEqual(result['shortfalls'],['独立盲测尚未通过','问题类型及责任准确率尚未验证'])
        self.assertIsNone(wilson_lower(0,0))


if __name__=='__main__':unittest.main()
