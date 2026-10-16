import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image

from app import Store
from review_batches import ReviewBatches


class ReviewBatchTest(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.base=Path(self.temp.name)
        self.source=self.base/'source';self.source.mkdir()
        self.store=Store(self.base/'data');self.root=self.store.set_root(str(self.source))
        self.batches=ReviewBatches(self.store)
        self.label={'defects':['其他破损'],'responsibilities':['运输'],
            'problem_responsibilities':{'其他破损':['运输']},'image_state':'defect','note':'原人工结果'}
        self.first=self.add_image('first.jpg');self.second=self.add_image('second.jpg')
        self.install([self.first,self.second])

    def tearDown(self):
        self.store.db.close();self.temp.cleanup()

    def add_image(self,name):
        path=self.source/name;Image.new('RGB',(400,200),'white').save(path);stat=path.stat()
        with self.store.db:
            cursor=self.store.db.execute('INSERT INTO images(root_id,relative_path,file_name,size,mtime_ns) VALUES(?,?,?,?,?)',
                (self.root['id'],name,name,stat.st_size,stat.st_mtime_ns))
        identity=cursor.lastrowid;self.store.commit([(identity,self.label)])
        return identity

    def manifest(self,ids):
        items=[]
        for identity in ids:
            detail=self.store.annotations.detail(identity)
            items.append({'image_id':identity,'relative_path':self.store.image(identity)['relative_path'],
                'source':detail['source'],'label_fingerprint':detail['label_fingerprint'],
                'boxes':[{'id':'box-'+str(identity),'problem':'其他破损','x':.1,'y':.2,'w':.3,'h':.4,
                    'origin':'model','score':.7}]})
        return {'schema_version':1,'requires_human_review':True,'model_id':'model-v1','method':'test','items':items}

    def install(self,ids):return self.store.annotations.install_proposals(self.manifest(ids))

    def proposal(self,image_id):
        return self.store.one('SELECT * FROM annotation_proposals WHERE image_id=?',(image_id,))

    def accept(self,image_id,status='confirmed'):
        detail=self.store.annotations.detail(image_id);proposal=detail['proposal']
        return self.store.annotations.save(image_id,{'source_sha256':detail['source']['sha256'],
            'label_fingerprint':detail['label_fingerprint'],'expected_version':detail['version'],
            'status':status,'outcome':'boxed','proposal_id':proposal['proposal_id'],
            'boxes':copy.deepcopy(proposal['boxes'])})

    def reject(self,image_id):
        return self.store.annotations.reject_proposal(image_id,self.proposal(image_id)['proposal_id'])

    def protected(self):
        return {table:self.store.query('SELECT * FROM '+table+' ORDER BY 1') for table in
            ('labels','drafts','history','annotation_documents','annotation_history',
             'annotation_proposals','annotation_proposal_history','annotation_feedback')}

    def test_registration_is_idempotent_unique_across_batches_and_never_creates_empty_batch(self):
        before=self.protected()
        ids=[self.proposal(i)['proposal_id'] for i in (self.first,self.second)]
        result=self.batches.register('batch-one',ids,{'source_model':'model-v1'})
        self.assertEqual(result['registered'],2)
        self.assertEqual(self.batches.register('batch-one',ids)['skipped'],2)
        self.assertFalse(self.batches.register('batch-two',ids)['exists'])
        self.assertFalse(self.batches.register('empty',[])['exists'])
        self.batches.sync();state=self.batches.state()
        self.assertEqual(state['current']['total'],2)
        self.assertEqual(state['current']['pending'],2)
        self.assertEqual(state['current']['blocked'],0)
        self.assertEqual(state['archives_count'],0)
        self.assertEqual(self.protected(),before)

    def test_all_human_decisions_create_immutable_downloadable_json_without_copying_images(self):
        self.batches.register('batch-one',metadata={'source_model':'model-v1'})
        self.accept(self.first);self.batches.sync()
        self.assertEqual(self.batches.state()['current']['reviewed'],1)
        self.reject(self.second);before=self.protected()
        result=self.batches.sync();state=self.batches.state()
        self.assertEqual(result['completed'],['batch-one'])
        self.assertIsNone(state['current']);self.assertTrue(state['needs_next_batch'])
        self.assertEqual(state['archives_count'],1)
        raw=self.batches.archive_content('batch-one');archive=json.loads(raw)
        self.assertFalse(archive['source_images_copied'])
        self.assertEqual(archive['summary'],{'total':2,'reviewed':2,'accepted':1,'rejected':1,'final_boxes':1,
            'state_processed':0,'state_counts':{'normal':0,'unrelated':0,'unclear':0}})
        self.assertEqual([i['relative_path'] for i in archive['items']],['first.jpg','second.jpg'])
        self.assertEqual(archive['items'][0]['classification'],self.label)
        self.assertEqual(len(archive['items'][0]['final_boxes']),1)
        self.assertEqual(archive['items'][1]['final_boxes'],[])
        self.assertEqual((self.store.data_dir/'review-archives'/'batch-one.json').read_text(encoding='utf-8'),raw)
        self.assertEqual(state['recent_archives'][0]['archive_sha256'],hashlib.sha256(raw.encode()).hexdigest())
        self.assertEqual(self.protected(),before)
        self.store.commit([(self.first,{**self.label,'note':'后续新修改'})])
        self.batches.sync()
        self.assertEqual(self.batches.archive_content('batch-one'),raw)

    def test_stale_pending_and_unconfirmed_drafts_are_not_human_completion(self):
        self.batches.register('batch-one')
        self.accept(self.first,status='draft');self.reject(self.first)
        with self.store.db:
            self.store.db.execute("UPDATE annotation_proposals SET status='stale' WHERE image_id=?",(self.second,))
        before=self.protected();self.batches.sync();state=self.batches.state()
        self.assertEqual(state['current']['reviewed'],0)
        self.assertEqual(state['current']['blocked'],2)
        self.assertEqual(state['current']['blocked_reasons'],{'annotation_draft':1,'stale_proposal':1})
        self.assertEqual(state['archives_count'],0)
        self.assertEqual(self.protected(),before)

    def test_classification_draft_blocks_accepted_item_until_user_resolves_it(self):
        self.batches.register('batch-one',[self.proposal(self.first)['proposal_id']])
        self.accept(self.first);self.store.save_draft(self.first,{**self.label,'note':'尚未确认'})
        self.batches.sync();self.assertEqual(self.batches.state()['current']['blocked_reasons'],{'classification_draft':1})
        with self.store.db:self.store.db.execute('DELETE FROM drafts WHERE image_id=?',(self.first,))
        self.batches.sync();self.assertEqual(self.batches.state()['archives_count'],1)

    def test_source_unavailable_or_changed_is_blocked_even_after_acceptance(self):
        self.batches.register('batch-one',[self.proposal(self.first)['proposal_id']]);self.accept(self.first)
        with self.store.db:self.store.db.execute('UPDATE images SET available=0 WHERE id=?',(self.first,))
        self.batches.sync();self.assertEqual(self.batches.state()['current']['blocked_reasons'],{'source_unavailable':1})
        with self.store.db:self.store.db.execute('UPDATE images SET available=1 WHERE id=?',(self.first,))
        (self.source/'first.jpg').unlink();self.batches.sync()
        self.assertEqual(self.batches.state()['current']['blocked_reasons'],{'source_unavailable':1})
        Image.new('RGB',(400,200),'black').save(self.source/'first.jpg');self.batches.sync()
        self.assertEqual(self.batches.state()['current']['blocked_reasons'],{'source_changed':1})
        self.assertEqual(self.batches.state()['archives_count'],0)

    def test_superseded_pending_id_is_not_silently_counted_as_reviewed(self):
        old=self.proposal(self.first)['proposal_id'];self.batches.register('batch-one',[old])
        manifest=self.manifest([self.first]);manifest.update(model_id='model-v2',refresh_pending=True)
        manifest['items'][0]['replaces_proposal_id']=old
        self.assertEqual(self.store.annotations.install_proposals(manifest)['refreshed'],1)
        self.batches.sync();state=self.batches.state()
        self.assertEqual(state['current']['blocked_reasons'],{'superseded_proposal':1})
        self.assertEqual(state['current']['reviewed'],0)
        self.assertEqual(state['archives_count'],0)

    def test_archive_write_failure_is_retried_and_database_download_still_works(self):
        self.batches.register('batch-one',[self.proposal(self.first)['proposal_id']]);self.reject(self.first)
        before=self.protected()
        with patch.object(self.batches,'_write_archive',side_effect=OSError('disk unavailable')):
            result=self.batches.sync()
        self.assertEqual(result['completed'],['batch-one']);self.assertEqual(len(result['archive_errors']),1)
        state=self.batches.state()['recent_archives'][0]
        self.assertTrue(state['archive_available']);self.assertFalse(state['archive_written'])
        self.assertIn('disk unavailable',state['archive_error'])
        frozen=self.batches.archive_content('batch-one')
        self.assertEqual(self.protected(),before)
        self.assertEqual(self.batches.sync()['archive_errors'],[])
        self.assertTrue(self.batches.state()['recent_archives'][0]['archive_written'])
        self.assertEqual(self.batches.archive_content('batch-one'),frozen)

    def test_multiple_batches_are_independent_and_root_isolation_is_enforced(self):
        self.batches.register('first-batch',[self.proposal(self.first)['proposal_id']])
        self.batches.register('second-batch',[self.proposal(self.second)['proposal_id']])
        self.reject(self.first);self.batches.sync()
        state=self.batches.state();self.assertEqual(state['current']['batch_id'],'second-batch')
        self.assertEqual(state['archives_count'],1)
        other=self.base/'other';other.mkdir();self.store.set_root(str(other))
        self.assertIsNone(self.batches.state()['current']);self.assertEqual(self.batches.state()['archives_count'],0)
        with self.assertRaises(ValueError):self.batches.archive_content('first-batch')
        with self.assertRaises(ValueError):self.batches.register('first-batch',[])

    def test_invalid_metadata_and_unsafe_archive_names_are_rejected(self):
        for identity in ('../outside','a/b','',None):
            with self.assertRaises(ValueError):self.batches.register(identity)
        for metadata in ([1],{'not-json':object()},{'not-json':float('nan')},{'big':'x'*17000}):
            with self.assertRaises(ValueError):self.batches.register('bad',metadata=metadata)
        self.assertEqual(self.store.query('SELECT * FROM review_batches'),[])

    def terminal_state(self,image_id,state='unrelated'):
        with self.store.db:
            self.store.db.execute("UPDATE annotation_proposals SET created_at='2000-01-01T00:00:00+00:00' WHERE image_id=?",(image_id,))
        label={'defects':[],'responsibilities':['待核'],'problem_responsibilities':{},
            'image_state':state,'note':'人工明确选择图片状态'}
        self.store.commit([(image_id,label)])
        return label

    def test_legacy_registration_skips_raw_pending_with_changed_classification(self):
        self.store.commit([(self.first,{**self.label,'note':'原分类已变化'})])
        before=self.protected();registered=self.batches.register('legacy')
        self.assertEqual(registered['registered'],1)
        self.assertEqual(self.store.one('SELECT image_id FROM review_batch_items')['image_id'],self.second)
        self.batches.sync();self.assertEqual(self.batches.state()['current']['total'],1)
        self.assertEqual(self.protected(),before)

    def test_explicit_human_terminal_states_complete_without_changing_proposals_or_training(self):
        ids=[self.first,self.second,self.add_image('third.jpg')];self.install([ids[-1]])
        for identity,state in zip(ids,('normal','unrelated','unclear')):
            self.batches.register('state-'+state,[self.proposal(identity)['proposal_id']])
            label=self.terminal_state(identity,state);before=self.protected()
            self.batches.sync();archive=json.loads(self.batches.archive_content('state-'+state));item=archive['items'][0]
            self.assertEqual(archive['summary']['state_processed'],1)
            self.assertEqual(archive['summary']['reviewed'],1)
            self.assertEqual(archive['summary']['state_counts'][state],1)
            self.assertEqual(item['decision'],'state_'+state);self.assertEqual(item['classification'],label)
            self.assertEqual(item['final_boxes'],[]);self.assertIsNone(item['annotation'])
            self.assertTrue(item['non_training']);self.assertIn('manual_history_id',item)
            self.assertIn('comparison',item);self.assertEqual(item['comparison']['human_boxes'],0)
            self.assertEqual(archive['comparison']['reviewed_images'],0)
            self.assertEqual(archive['comparison']['image_state_dispositions']['counts'][state],1)
            self.assertEqual(self.protected(),before)
            self.assertEqual(self.proposal(identity)['status'],'pending')

    def test_terminal_state_requires_current_equal_nonundone_human_history_after_proposal(self):
        self.batches.register('state-one',[self.proposal(self.first)['proposal_id']]);self.terminal_state(self.first)
        receipt=self.store.one('SELECT * FROM history ORDER BY id DESC LIMIT 1')
        with self.store.db:self.store.db.execute('UPDATE history SET undone=1 WHERE id=?',(receipt['id'],))
        self.batches.sync();self.assertEqual(self.batches.state()['current']['blocked_reasons'],{'state_confirmation_missing':1})
        altered=json.loads(receipt['data']);altered[0]['after']['note']='与当前分类不同'
        with self.store.db:self.store.db.execute('UPDATE history SET undone=0,data=? WHERE id=?',(json.dumps(altered),receipt['id']))
        self.batches.sync();self.assertEqual(self.batches.state()['current']['reviewed'],0)
        with self.store.db:
            self.store.db.execute('UPDATE history SET data=? WHERE id=?',(receipt['data'],receipt['id']))
            self.store.db.execute('UPDATE annotation_proposals SET created_at=? WHERE image_id=?',(receipt['created_at'],self.first))
        self.batches.sync();self.assertEqual(self.batches.state()['current']['blocked_reasons'],{'state_confirmation_missing':1})

    def test_terminal_state_supersedes_only_older_annotation_draft_and_never_exports_draft_boxes(self):
        self.batches.register('state-one',[self.proposal(self.first)['proposal_id']]);self.accept(self.first,status='draft')
        self.terminal_state(self.first,'unclear')
        with self.store.db:
            self.store.db.execute("UPDATE annotation_documents SET updated_at='2001-01-01T00:00:00+00:00' WHERE image_id=?",(self.first,))
        before=self.protected();self.batches.sync();archive=json.loads(self.batches.archive_content('state-one'))
        self.assertEqual(archive['items'][0]['final_boxes'],[])
        self.assertEqual(archive['items'][0]['negative_regions'],[])
        self.assertTrue(archive['items'][0]['non_training'])
        self.assertEqual(self.protected(),before)
        self.assertEqual(json.loads(self.store.one('SELECT data FROM annotation_documents WHERE image_id=?',(self.first,))['data'])['status'],'draft')
        self.assertFalse(self.store.annotations.export()['items'][0]['training_eligible_by_label_policy'])

    def test_terminal_state_does_not_complete_with_new_annotation_or_classification_draft(self):
        self.batches.register('state-one',[self.proposal(self.first)['proposal_id']]);self.accept(self.first,status='draft')
        label=self.terminal_state(self.first)
        with self.store.db:
            self.store.db.execute("UPDATE annotation_documents SET updated_at='2999-01-01T00:00:00+00:00' WHERE image_id=?",(self.first,))
        self.batches.sync();self.assertEqual(self.batches.state()['current']['blocked_reasons'],{'annotation_draft':1})
        self.store.save_draft(self.first,{**label,'note':'新的未确认编辑'})
        self.batches.sync();self.assertEqual(self.batches.state()['current']['blocked_reasons'],{'classification_draft':1})
        self.assertEqual(self.batches.state()['archives_count'],0)

    def test_terminal_state_does_not_bypass_unavailable_source_or_stale_proposal(self):
        self.batches.register('state-one',[self.proposal(self.first)['proposal_id']]);self.terminal_state(self.first)
        with self.store.db:self.store.db.execute('UPDATE images SET available=0 WHERE id=?',(self.first,))
        self.batches.sync();self.assertEqual(self.batches.state()['current']['blocked_reasons'],{'source_unavailable':1})
        with self.store.db:
            self.store.db.execute('UPDATE images SET available=1 WHERE id=?',(self.first,))
            self.store.db.execute("UPDATE annotation_proposals SET status='stale' WHERE image_id=?",(self.first,))
        self.batches.sync();self.assertEqual(self.batches.state()['current']['blocked_reasons'],{'stale_proposal':1})
        self.assertEqual(self.batches.state()['archives_count'],0)


if __name__=='__main__':unittest.main()
