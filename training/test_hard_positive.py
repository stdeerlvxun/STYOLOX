"""Correction provenance and repeat limits without loading torch or a GPU."""
import copy
import json
import sqlite3
import unittest

from hard_positive import (COORDINATES, POLICY, hard_positive_metadata,
                           hard_positive_repeats, original_proposal)


class HardPositiveTests(unittest.TestCase):
    def setUp(self):
        box = {'id': 'm1', 'origin': 'model', 'proposal_id': 'p1', 'problem': '崩边',
               'x': .1, 'y': .1, 'w': .2, 'h': .2}
        self.proposal = {'proposal_id': 'p1', 'model_id': 'model1', 'method': 'detector',
                         'coordinate_space': COORDINATES, 'boxes': [box],
                         'source': {'sha256': 'source-sha', 'width': 400, 'height': 300}}
        self.document = {**copy.deepcopy(self.proposal), 'status': 'confirmed', 'outcome': 'boxed',
                         'origin': 'machine_assisted_human_confirmed'}

    def test_added_and_substantially_moved_boxes_are_both_recorded(self):
        self.document['boxes'][0].update(id='human-edited-id', source_box_id='m1', x=.7)
        # The server omits origin for newly drawn human boxes.
        self.document['boxes'].append({'id': 'h1', 'problem': '划伤', 'x': .2, 'y': .4, 'w': .1, 'h': .2})
        before = copy.deepcopy((self.document, self.proposal))
        result = hard_positive_metadata(self.document, self.proposal)
        self.assertEqual(result['added_box_ids'], ['h1'])
        self.assertEqual(result['moved_boxes'], [{'box_id': 'human-edited-id', 'source_box_id': 'm1', 'original_iou': 0.}])
        self.assertTrue(result['is_hard'])
        self.assertEqual((self.document, self.proposal), before)
        self.assertNotIn('negative_regions', result)

    def test_small_movement_and_reclassification_alone_are_not_hard(self):
        self.document['boxes'][0].update(x=.11, problem='缺角')
        result = hard_positive_metadata(self.document, self.proposal)
        self.assertFalse(result['is_hard'])

    def test_deleted_boxes_alone_do_not_invent_positive_or_normal_regions(self):
        self.document['boxes'] = []
        result = hard_positive_metadata(self.document, self.proposal)
        self.assertFalse(result['is_hard'])
        self.assertEqual(result['moved_boxes'], [])

    def test_missing_or_mismatched_provenance_is_not_hard_evidence(self):
        self.document['boxes'][0]['x'] = .7
        self.assertIsNone(hard_positive_metadata(self.document, None))
        for key, value in [('proposal_id', 'other'), ('model_id', 'other'), ('method', 'other'),
                           ('coordinate_space', 'other'), ('status', 'draft'),
                           ('origin', 'human'), ('outcome', 'unclear')]:
            changed = {**self.document, key: value}
            self.assertIsNone(hard_positive_metadata(changed, self.proposal), key)
        for key, value in [('sha256', 'changed'), ('width', 800), ('height', 600)]:
            changed = {**self.document, 'source': {**self.document['source'], key: value}}
            self.assertIsNone(hard_positive_metadata(changed, self.proposal), key)

    def test_unverified_model_box_is_not_mistaken_for_human_addition(self):
        self.document['boxes'][0].update(id='unknown', x=.7)
        self.assertFalse(hard_positive_metadata(self.document, self.proposal)['is_hard'])
        self.proposal['boxes'][0].pop('origin')
        self.assertIsNone(hard_positive_metadata(self.document, self.proposal))

    def test_original_label_fingerprint_may_change_when_human_adds_issue(self):
        self.document['boxes'].append({'id': 'h1', 'origin': 'human', 'problem': '划伤',
                                      'x': .2, 'y': .4, 'w': .1, 'h': .2})
        self.document['label_fingerprint'] = 'new-human-label'
        self.proposal['label_fingerprint'] = 'original-label'
        self.assertTrue(hard_positive_metadata(self.document, self.proposal)['is_hard'])

    def test_malformed_optional_proposal_does_not_change_training_eligibility(self):
        for proposal in ([], {**self.proposal, 'source': None}, {**self.proposal, 'boxes': [None]}):
            self.assertIsNone(hard_positive_metadata(self.document, proposal))

    def test_current_and_historical_lookup_require_image_and_proposal_identity(self):
        with sqlite3.connect(':memory:') as db:
            self.assertIsNone(original_proposal(db, 1, 'p1', set()))
            tables = {'annotation_proposals', 'annotation_proposal_history'}
            for table in sorted(tables):
                db.execute(f'CREATE TABLE {table}(image_id INTEGER,proposal_id TEXT,data TEXT)')
            db.execute('INSERT INTO annotation_proposals VALUES(?,?,?)', (1, 'new', json.dumps(self.proposal)))
            db.execute('INSERT INTO annotation_proposal_history VALUES(?,?,?)', (1, 'p1', json.dumps(self.proposal)))
            self.assertEqual(original_proposal(db, 1, 'p1', tables), self.proposal)
            self.assertIsNone(original_proposal(db, 2, 'p1', tables))
            self.assertIsNone(original_proposal(db, 1, 'missing', tables))

    def records(self):
        metadata = {'policy': POLICY, 'verified_proposal': True, 'is_hard': True, 'proposal_id': 'p1',
                    'added_box_ids': ['h1'], 'moved_boxes': []}
        return [{'image_id': i, 'split': 'train', 'group': 'order'+str(i), 'defects': ['崩边'],
                 'boxes': [{'id': 'h1'}], 'proposal_id': 'p1', 'annotation_status': 'confirmed',
                 'annotation_outcome': 'boxed', 'hard_positive': copy.deepcopy(metadata)} for i in range(9)]

    def test_repeats_are_train_only_at_most_twice_and_cap_is_half_original_train(self):
        records = self.records()
        records[-2]['split'], records[-1]['split'] = 'validation', 'test'
        records[5]['hard_positive']['moved_boxes'] = [{'box_id': 'm2'}]
        before = copy.deepcopy(records)
        extras, summary = hard_positive_repeats(records)
        self.assertEqual([r['image_id'] for r in extras], [5, 0, 1])
        self.assertEqual(summary['original_train_images'], 7)
        self.assertEqual(summary['extra_images_per_epoch'], 3)
        self.assertTrue(all(r['split'] == 'train' for r in extras))
        self.assertEqual(len({r['image_id'] for r in extras}), len(extras))
        self.assertEqual(records, before)
        self.assertEqual(summary['validation_extra_images'], 0)
        self.assertEqual(summary['test_extra_images'], 0)

    def test_repeat_selection_keeps_all_existing_eligibility_guards(self):
        records = self.records()
        for i, state in enumerate(('normal', 'unclear', 'unrelated')):
            records[i]['image_state'] = state
        records[3]['has_draft'] = True
        records[4]['annotation_status'] = 'draft'
        records[5]['defects'] = []
        records[6]['hard_positive'] = None
        records[7]['proposal_id'] = 'different'
        records[8]['feedback_only'] = True
        extras, summary = hard_positive_repeats(records)
        self.assertEqual(extras, [])
        self.assertEqual(summary['eligible_hard_images'], 0)

    def test_single_training_image_does_not_exceed_half_cap(self):
        self.assertEqual(hard_positive_repeats(self.records()[:1])[0], [])


if __name__ == '__main__':
    unittest.main()
