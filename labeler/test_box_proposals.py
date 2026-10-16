"""Machine location suggestions remain separate until explicit human confirmation."""
import copy
from pathlib import Path
import tempfile
import unittest

from PIL import Image

from app import Store


class BoxProposalTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.base = Path(self.temp.name)
        self.source = self.base / 'source'
        self.source.mkdir()
        self.store = Store(self.base / 'data')
        self.root = self.store.set_root(str(self.source))
        self.label = {'defects': ['其他破损'], 'responsibilities': ['运输'],
                      'problem_responsibilities': {'其他破损': ['运输']},
                      'image_state': 'defect', 'note': '保留人工分类'}
        self.id = self.add_image('board.jpg')
        self.store.commit([(self.id, self.label)])

    def tearDown(self):
        self.store.db.close()
        self.temp.cleanup()

    def add_image(self, name, root=None):
        root = root or self.root
        path = Path(root['path']) / name
        Image.new('RGB', (400, 200), 'white').save(path)
        stat = path.stat()
        with self.store.db:
            cursor = self.store.db.execute(
                'INSERT INTO images(root_id,relative_path,file_name,case_id,size,mtime_ns) '
                'VALUES(?,?,?,?,?,?)',
                (root['id'], name, name, 'case-' + name, stat.st_size, stat.st_mtime_ns))
        return cursor.lastrowid

    def human(self):
        return {name: self.store.query('SELECT * FROM ' + name + ' ORDER BY 1')
                for name in ['labels', 'drafts', 'tags', 'history']}

    def annotations(self):
        return {name: self.store.query('SELECT * FROM ' + name + ' ORDER BY 1')
                for name in ['annotation_documents', 'annotation_history']}

    def manifest(self, image_id=None):
        image_id = image_id or self.id
        detail = self.store.annotations.detail(image_id)
        image = self.store.image(image_id)
        return {'schema_version': 1, 'model_id': 'localization-test-v1',
                'method': 'test-only', 'requires_human_review': True,
                'items': [{'image_id': image_id, 'relative_path': image['relative_path'],
                           'source': detail['source'],
                           'label_fingerprint': detail['label_fingerprint'],
                           'boxes': [{'id': 'model-1', 'problem': '其他破损',
                                      'x': .1, 'y': .2, 'w': .3, 'h': .4,
                                      'origin': 'model', 'score': .76}],
                           'note': '机器位置仅供审核'}]}

    def install(self):
        result = self.store.annotations.install_proposals(self.manifest())
        self.assertEqual(result['imported'], 1)
        self.assertEqual(result['skipped'], 0)
        self.assertEqual(result['imported_image_ids'],[self.id])
        self.assertEqual(result['imported_proposal_ids'],[self.store.one('SELECT proposal_id FROM annotation_proposals WHERE image_id=?',(self.id,))['proposal_id']])
        return self.store.annotations.detail(self.id)['proposal']

    def payload(self, proposal=None, status='confirmed'):
        detail = self.store.annotations.detail(self.id)
        result = {'source_sha256': detail['source']['sha256'],
                  'label_fingerprint': detail['label_fingerprint'],
                  'expected_version': detail['version'], 'status': status,
                  'outcome': 'boxed', 'boxes': [{'id': 'manual-1', 'problem': '其他破损',
                                               'x': .1, 'y': .2, 'w': .3, 'h': .4}]}
        if proposal:
            result['proposal_id'] = proposal['proposal_id']
            result['boxes'] = copy.deepcopy(proposal['boxes'])
        return result

    def assert_no_pending(self):
        self.assertEqual(self.store.listing({'status': ['auto_boxes']})['total'], 0)
        self.assertEqual(self.store.annotations.proposal_state()['pending'], 0)
        self.assertIsNone(self.store.annotations.detail(self.id)['proposal'])

    def test_import_is_separate_and_not_exported_or_training_eligible(self):
        human, annotations = self.human(), self.annotations()
        proposal = self.install()
        self.assertEqual(proposal['status'], 'pending')
        self.assertTrue(proposal['proposal_id'])
        self.assertEqual(self.human(), human)
        self.assertEqual(self.annotations(), annotations)
        self.assertEqual(self.store.annotations.export()['items'], [])
        self.assertEqual(self.store.listing({'status': ['auto_boxes']})['total'], 1)
        self.assertEqual(self.store.annotations.proposal_state()['pending'], 1)

    def test_reimport_is_idempotent_and_rejection_is_not_overwritten(self):
        manifest = self.manifest()
        proposal = self.install()
        before = self.human(), self.annotations()
        repeated = self.store.annotations.install_proposals(manifest)
        self.assertEqual((repeated['imported'], repeated['skipped']), (0, 1))
        self.assertEqual(self.store.annotations.detail(self.id)['proposal']['proposal_id'],
                         proposal['proposal_id'])
        self.store.annotations.reject_proposal(self.id, proposal['proposal_id'])
        self.assert_no_pending()
        repeated = self.store.annotations.install_proposals(manifest)
        self.assertEqual((repeated['imported'], repeated['skipped']), (0, 1))
        self.assertEqual(self.store.annotations.proposal_state()['rejected'], 1)
        self.assertEqual((self.human(), self.annotations()), before)
        with self.assertRaises(ValueError):
            self.store.annotations.save(self.id, self.payload(proposal))
        self.assertEqual((self.human(), self.annotations()), before)

    def test_existing_manual_annotation_is_not_imported_or_overwritten(self):
        manifest = self.manifest()
        self.store.annotations.save(self.id, self.payload(status='draft'))
        before = self.human(), self.annotations()
        result = self.store.annotations.install_proposals(manifest)
        self.assertEqual((result['imported'], result['skipped']), (0, 1))
        self.assertEqual((self.human(), self.annotations()), before)
        self.assert_no_pending()

    def test_explicit_confirmation_retains_machine_provenance(self):
        manifest = self.manifest()
        proposal = self.install()
        before = self.human()
        body = self.payload(proposal)
        # A reviewer can adjust a proposed location before confirming it.
        body['boxes'][0]['x'] = .15
        saved = self.store.annotations.save(self.id, body)
        self.assertEqual(saved['document']['status'], 'confirmed')
        self.assertEqual(saved['document']['origin'], 'machine_assisted_human_confirmed')
        box = saved['document']['boxes'][0]
        self.assertEqual(box['x'], .15)
        self.assertEqual(box['origin'], 'model')
        self.assertEqual(box['proposal_id'], proposal['proposal_id'])
        self.assertEqual(self.human(), before)
        self.assertEqual(self.store.annotations.proposal_state()['accepted'], 1)
        exported = self.store.annotations.export()['items']
        self.assertEqual(len(exported), 1)
        self.assertTrue(exported[0]['training_eligible_by_label_policy'])
        self.assertEqual(exported[0]['annotation']['boxes'][0]['proposal_id'], proposal['proposal_id'])
        result = self.store.annotations.install_proposals(manifest)
        self.assertEqual((result['imported'], result['skipped']), (0, 1))
        self.assertEqual(self.store.annotations.proposal_state()['accepted'], 1)
        self.assertEqual(self.human(), before)

    def test_draft_is_not_acceptance_and_remains_reviewable(self):
        proposal = self.install()
        human = self.human()
        self.store.annotations.save(self.id, self.payload(proposal, status='draft'))
        self.assertEqual(self.store.annotations.proposal_state()['pending'], 1)
        self.assertEqual(self.store.annotations.proposal_state()['accepted'], 0)
        self.assertEqual(self.store.listing({'status': ['auto_boxes']})['total'], 1)
        self.assertFalse(self.store.annotations.export()['items'][0]['training_eligible_by_label_policy'])
        self.store.annotations.save(self.id, self.payload(proposal))
        self.assertEqual(self.store.annotations.proposal_state()['accepted'], 1)
        self.assertEqual(self.human(), human)

    def test_new_human_label_or_label_draft_invalidates_candidate(self):
        proposal = self.install()
        old_body = self.payload(proposal)
        self.store.commit([(self.id, {**self.label, 'note': '人工更新'})])
        self.assert_no_pending()
        before = self.human(), self.annotations()
        with self.assertRaises(ValueError):
            self.store.annotations.save(self.id, old_body)
        self.assertEqual((self.human(), self.annotations()), before)
        # A fresh proposal still cannot bypass a separate classification draft.
        self.store.annotations.install_proposals({**self.manifest(), 'model_id': 'fresh-model'})
        self.store.save_draft(self.id, {**self.label, 'note': '尚未确认的编辑'})
        self.assert_no_pending()
        before = self.human(), self.annotations()
        result = self.store.annotations.install_proposals({**self.manifest(), 'model_id': 'third-model'})
        self.assertEqual(result['imported'], 0)
        self.assertEqual((self.human(), self.annotations()), before)

    def test_source_bytes_change_invalidates_candidate_and_save(self):
        proposal = self.install()
        body = self.payload(proposal)
        Image.new('RGB', (400, 200), 'black').save(self.source / 'board.jpg')
        # Lists use the archive index for speed; opening a suggestion rechecks actual bytes.
        self.assertIsNone(self.store.annotations.detail(self.id)['proposal'])
        self.assert_no_pending()
        before = self.human(), self.annotations()
        with self.assertRaises(ValueError):
            self.store.annotations.save(self.id, body)
        self.assertEqual((self.human(), self.annotations()), before)

    def test_invalid_manifest_items_never_write_annotations_or_human_data(self):
        invalid = []
        for field, value in [('x', float('nan')), ('x', True), ('x', -.1),
                             ('w', 0), ('w', 1.1), ('problem', '未知问题'),
                             ('origin', 'human')]:
            manifest = self.manifest()
            manifest['items'][0]['boxes'][0][field] = value
            invalid.append(manifest)
        for field, value in [('relative_path', '../board.jpg'),
                             ('label_fingerprint', '0' * 64)]:
            manifest = self.manifest()
            manifest['items'][0][field] = value
            invalid.append(manifest)
        for field, value in [('sha256', '0' * 64), ('width', 200), ('height', 400)]:
            manifest = self.manifest()
            manifest['items'][0]['source'][field] = value
            invalid.append(manifest)
        manifest = self.manifest()
        manifest['requires_human_review'] = False
        invalid.append(manifest)
        before = self.human(), self.annotations()
        for manifest in invalid:
            with self.subTest(manifest=manifest):
                try:
                    result = self.store.annotations.install_proposals(manifest)
                except ValueError:
                    pass
                else:
                    self.assertEqual(result['imported'], 0)
                self.assertEqual((self.human(), self.annotations()), before)
                self.assertEqual(self.store.annotations.export()['items'], [])
                self.assert_no_pending()

    def test_image_states_and_missing_labels_are_excluded(self):
        for state in ['normal', 'unclear', 'unrelated']:
            self.store.commit([(self.id, {'defects': [], 'image_state': state})])
            result = self.store.annotations.install_proposals(self.manifest())
            self.assertEqual(result['imported'], 0)
            self.assert_no_pending()
        with self.store.db:
            self.store.db.execute('DELETE FROM labels WHERE image_id=?', (self.id,))
        result = self.store.annotations.install_proposals(self.manifest())
        self.assertEqual(result['imported'], 0)
        self.assert_no_pending()

    def test_proposals_are_scoped_to_current_root(self):
        manifest = self.manifest()
        self.install()
        other = self.base / 'other-root'
        other.mkdir()
        self.store.set_root(str(other))
        before = self.human(), self.annotations()
        result = self.store.annotations.install_proposals(manifest)
        self.assertEqual(result['imported'], 0)
        self.assertEqual(self.store.annotations.proposal_state()['total'], 0)
        self.assertEqual(self.store.listing({'status': ['auto_boxes']})['total'], 0)
        self.assertEqual((self.human(), self.annotations()), before)

    def test_stale_candidate_cannot_overwrite_independent_human_annotation(self):
        proposal = self.install()
        self.store.annotations.save(self.id, self.payload())
        before = self.human(), self.annotations()
        # Even the current annotation version must not make an old proposal authoritative.
        with self.assertRaises(ValueError):
            self.store.annotations.save(self.id, self.payload(proposal))
        self.assertEqual((self.human(), self.annotations()), before)

    def test_accepted_proposal_allows_later_explicit_human_adjustment(self):
        proposal = self.install()
        self.store.annotations.save(self.id, self.payload(proposal))
        before = self.human()
        body = self.payload(proposal)
        body['boxes'][0]['w'] = .45
        saved = self.store.annotations.save(self.id, body)
        self.assertEqual(saved['version'], 2)
        self.assertEqual(saved['document']['boxes'][0]['w'], .45)
        self.assertEqual(saved['document']['origin'], 'machine_assisted_human_confirmed')
        self.assertEqual(self.store.annotations.proposal_state()['accepted'], 1)
        self.assertEqual(self.store.annotations.proposal_state()['pending'], 0)
        self.assertEqual(self.human(), before)

    def refresh_manifest(self, proposal, boxes=None):
        manifest=self.manifest();manifest.update(model_id='next-detector-v2',refresh_pending=True)
        manifest['items'][0]['replaces_proposal_id']=proposal['proposal_id']
        if boxes is not None:manifest['items'][0]['boxes']=boxes
        return manifest

    def test_confirmed_corrections_capture_move_reclassify_delete_and_added_box(self):
        manifest=self.manifest();second=copy.deepcopy(manifest['items'][0]['boxes'][0]);second.update(id='model-2',x=.65,w=.1,h=.1)
        manifest['items'][0]['boxes'].append(second);self.store.annotations.install_proposals(manifest)
        proposal=self.store.annotations.detail(self.id)['proposal'];body=self.payload(proposal)
        body['boxes']=body['boxes'][:1];body['boxes'][0].update(x=.15,problem='崩边')
        body['boxes'].append({'id':'manual-added','problem':'其他破损','x':.05,'y':.8,'w':.1,'h':.1})
        body['negative_regions']=[{'id':'not-damage','source_box_id':'model-2','reason':'process_hole','x':.65,'y':.2,'w':.1,'h':.1}]
        self.store.annotations.save(self.id,body)
        events=self.store.annotations.feedback_export()['items'][0]['feedback']['events']
        self.assertEqual([e['kind'] for e in events],['corrected','deleted','added'])
        self.assertEqual(events[0]['changes'],['reclassified','moved'])
        self.assertEqual(events[1]['reason'],'process_hole')

    def test_all_false_positive_confirmation_rejects_candidate_but_does_not_mark_image_normal(self):
        proposal=self.install();body=self.payload(proposal);old=proposal['boxes'][0]
        body.update(boxes=[],outcome='needs_correction',negative_regions=[{**{k:old[k] for k in ('x','y','w','h')},
            'id':'negative-1','reason':'process_hole','source_box_id':old['id']}])
        before=self.human();self.store.annotations.save(self.id,body)
        self.assertEqual(self.human(),before);self.assertEqual(self.store.annotations.proposal_state()['rejected'],1)
        self.assertFalse(self.store.annotations.export()['items'][0]['training_eligible_by_label_policy'])
        item=self.store.annotations.feedback_export()['items'][0]
        self.assertTrue(item['training_eligible_by_label_policy']);self.assertFalse(item['feedback']['whole_image_normal'])

    def test_refresh_pending_requires_expected_id_and_archives_previous_candidate(self):
        proposal=self.install();before=self.human(),self.annotations()
        manifest=self.refresh_manifest(proposal)
        wrong=copy.deepcopy(manifest);wrong['items'][0]['replaces_proposal_id']='wrong'
        result=self.store.annotations.install_proposals(wrong)
        self.assertEqual((result['refreshed'],result['skipped']),(0,1))
        result=self.store.annotations.install_proposals(manifest)
        self.assertEqual((result['imported'],result['refreshed'],result['withdrawn']),(0,1,0))
        self.assertEqual((self.human(),self.annotations()),before)
        current=self.store.annotations.detail(self.id)['proposal']
        self.assertEqual(current['model_id'],'next-detector-v2')
        history=self.store.query('SELECT * FROM annotation_proposal_history')
        self.assertEqual(history[0]['proposal_id'],proposal['proposal_id'])
        self.assertEqual(history[0]['replaced_by'],current['proposal_id'])
        result=self.store.annotations.install_proposals(manifest)
        self.assertEqual(result['skipped'],1)

    def test_empty_refresh_only_explicitly_withdraws_pending_and_keeps_audit(self):
        proposal=self.install();before=self.human(),self.annotations()
        manifest=self.refresh_manifest(proposal,boxes=[])
        result=self.store.annotations.install_proposals(manifest);self.assertEqual(result['skipped'],1)
        manifest['items'][0]['withdraw_if_empty']=True
        result=self.store.annotations.install_proposals(manifest)
        self.assertEqual((result['withdrawn'],result['refreshed'],result['imported']),(1,0,0))
        self.assertEqual((self.human(),self.annotations()),before)
        old=self.store.one('SELECT * FROM annotation_proposals WHERE image_id=?',(self.id,))
        self.assertEqual(old['status'],'stale');self.assertEqual(old['proposal_id'],proposal['proposal_id'])
        self.assertEqual(len(self.store.query('SELECT * FROM annotation_proposal_history')),1)

    def test_refresh_never_overwrites_human_draft_accepted_rejected_or_stale(self):
        proposal=self.install();manifest=self.refresh_manifest(proposal);before=self.human()
        self.store.annotations.save(self.id,self.payload(proposal,status='draft'))
        human_annotations=self.annotations()
        result=self.store.annotations.install_proposals(manifest)
        self.assertEqual(result['skipped'],1);self.assertEqual(self.annotations(),human_annotations)
        with self.store.db:
            self.store.db.execute('DELETE FROM annotation_documents WHERE image_id=?',(self.id,))
        for status in ['accepted','rejected','stale']:
            with self.store.db:self.store.db.execute('UPDATE annotation_proposals SET status=? WHERE image_id=?',(status,self.id))
            result=self.store.annotations.install_proposals(manifest)
            self.assertEqual(result['skipped'],1)
            self.assertEqual(self.store.one('SELECT status FROM annotation_proposals WHERE image_id=?',(self.id,))['status'],status)
        self.assertEqual(self.human(),before)

    def test_refresh_transaction_rechecks_new_human_work_after_source_validation(self):
        proposal=self.install();manifest=self.refresh_manifest(proposal);source=self.store.annotations.source
        def edit_after_source(image_id):
            result=source(image_id)
            self.store.save_draft(image_id,{**self.label,'note':'编辑期间的草稿'})
            return result
        self.store.annotations.source=edit_after_source
        result=self.store.annotations.install_proposals(manifest)
        self.assertEqual((result['refreshed'],result['skipped']),(0,1))
        self.assertEqual(self.store.one('SELECT proposal_id FROM annotation_proposals WHERE image_id=?',(self.id,))['proposal_id'],proposal['proposal_id'])
        self.assertEqual(self.store.query('SELECT * FROM annotation_proposal_history'),[])


if __name__ == '__main__':
    unittest.main()
