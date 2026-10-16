import tempfile
import unittest
from pathlib import Path

from retrain_latest import preserve_holdouts,aligned_problem_scores
from unittest.mock import patch
import torch
from vision_classifier import read_json, write_json


class HoldoutTests(unittest.TestCase):
    def test_new_problem_names_are_not_fabricated_as_old_model_predictions(self):
        original=torch.tensor([[.8,.3],[.2,.9]])
        with patch('retrain_latest.ridge_scores',return_value=original):
            aligned=aligned_problem_scores({'ridge':{},'problems':['崩边','划伤']},torch.ones(2,3),
                                            ['划伤','受潮膨胀','崩边'])
        torch.testing.assert_close(aligned,torch.tensor([[.3,0.,.8],[.9,0.,.2]]))

    def prepare(self, root, old, new):
        previous = root / 'previous'
        current = root / 'current'
        previous.mkdir(); current.mkdir()
        write_json(previous / 'dataset.json', {'records': old})
        write_json(current / 'dataset.json', {
            'training_policy': 'unrelated_auxiliary_v1', 'records': new})
        return current, previous

    def row(self, sha, case, order, group, split):
        return dict(sha256=sha, case_id=case, order_keys=[order],
                    group=group, split=split, auxiliary_target=1)

    def test_new_images_of_heldout_order_stay_heldout(self):
        with tempfile.TemporaryDirectory() as directory:
            old = [self.row('old', 'case-old', 'order-a', 'g0', 'test')]
            new = [self.row('new', 'case-new', 'order-a', 'g1', 'train')]
            current, previous = self.prepare(Path(directory), old, new)
            result = preserve_holdouts(current, previous)
            self.assertEqual(result['records'][0]['split'], 'test')
            self.assertEqual(result['splits']['test']['images'], 1)

    def test_current_duplicate_group_stays_together(self):
        with tempfile.TemporaryDirectory() as directory:
            old = [self.row('old', 'case-old', 'order-a', 'g0', 'validation')]
            new = [self.row('old', 'case-old', 'order-a', 'g1', 'train'),
                   self.row('near-copy', 'case-b', 'order-b', 'g1', 'train')]
            current, previous = self.prepare(Path(directory), old, new)
            result = preserve_holdouts(current, previous)
            self.assertEqual([r['split'] for r in result['records']],
                             ['validation', 'validation'])

    def test_new_link_between_old_training_and_test_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            old = [self.row('a', 'case-a', 'order-a', 'g0', 'train'),
                   self.row('b', 'case-b', 'order-b', 'g1', 'test')]
            new = [self.row('a', 'case-a', 'order-a', 'merged', 'train'),
                   self.row('b', 'case-b', 'order-b', 'merged', 'train')]
            current, previous = self.prepare(Path(directory), old, new)
            with self.assertRaisesRegex(ValueError, 'historical training'):
                preserve_holdouts(current, previous)
            self.assertEqual(read_json(current / 'dataset.json')['records'][0]['split'], 'train')

    def test_audited_overlap_can_only_be_quarantined_as_a_whole_group(self):
        with tempfile.TemporaryDirectory() as directory:
            old=[self.row('a','case-a','order-a','g0','train'),
                 self.row('b','case-b','order-b','g1','test')]
            new=[self.row('a','case-a','order-a','merged','train'),
                 self.row('b','case-b','order-b','merged','test')]
            current,previous=self.prepare(Path(directory),old,new)
            result=preserve_holdouts(current,previous,quarantine_historical_conflicts=True)
            self.assertEqual([r['split'] for r in result['records']],['quarantine','quarantine'])
            self.assertEqual(result['quarantined_images'],2)
            self.assertEqual(sum(s['images'] for s in result['splits'].values()),0)


if __name__ == '__main__':
    unittest.main()
