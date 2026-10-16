import unittest
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
from unittest.mock import patch
import torch

from train_detector import adaptable_parameter, compare_validation, evaluate_predictions, location_training_records, match_boxes, partial_objectness_loss, select_threshold, xyxy_iou


class DetectorMetricsTests(unittest.TestCase):
    def test_binary_candidate_comparison_scores_both_models_as_locations(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = root/'data.json'
            dataset.write_text(json.dumps({'categories': [{'id': 3, 'name': '崩边'}], 'records': [
                {'image_id': 1, 'relative_path': 'image', 'full_path': 'image', 'group': 'g',
                 'sha256': 'hash', 'split': 'validation', 'boxes': [
                    {'class_id': 3, 'problem': '崩边', 'xyxy': [0, 0, 10, 10]}]}]}), encoding='utf-8')
            def fake_detector(problem):
                return SimpleNamespace(model=SimpleNamespace(cpu=lambda: None),
                    predict_image=lambda *args: [{'class_id': 0, 'problem': problem,
                        'xyxy': [0, 0, 10, 10], 'score': .1}])
            with patch('train_detector.torch.load', return_value={'config': {'class_agnostic': True, 'fixed_threshold': .05}}), \
                 patch('train_detector.load_detector', side_effect=[fake_detector('崩边'), fake_detector('问题位置')]), \
                 patch('train_detector.sha256', return_value='hash'):
                result = compare_validation(dataset, root, root)
            self.assertEqual(result['comparison_objective'], 'problem_location_only')
            self.assertEqual(result['baseline']['tp'], 1)
            self.assertEqual(result['candidate']['tp'], 1)
            self.assertEqual(result['candidate']['selection']['threshold'], .05)
            self.assertIsNone(result['strict_issue_type_metrics'])
            self.assertFalse(result['test_used'])

    def test_location_only_training_view_preserves_original_human_classes(self):
        box = {'class_id': 3, 'problem': '崩边', 'xyxy': [1, 2, 3, 4]}
        dataset = {'records': [{'boxes': [box], 'split': 'test', 'image_state': ''}],
                   'negative_feedback_records': [{'boxes': [box]}]}
        categories, records, feedback = location_training_records(dataset)
        self.assertEqual(categories, [{'id': 0, 'name': '问题位置'}])
        self.assertEqual(records[0]['boxes'][0]['class_id'], 0)
        self.assertEqual(records[0]['boxes'][0]['original_problem'], '崩边')
        self.assertEqual(feedback[0]['boxes'][0]['original_class_id'], 3)
        self.assertEqual(dataset['records'][0]['boxes'][0]['class_id'], 3)
        self.assertEqual(records[0]['split'], 'test')

    def box(self, kind=0, coordinates=None, score=.9):
        return {'class_id': kind, 'xyxy': coordinates or [0, 0, 10, 10], 'score': score}

    def test_iou_and_touching_boxes(self):
        self.assertEqual(xyxy_iou([0, 0, 10, 10], [0, 0, 10, 10]), 1)
        self.assertEqual(xyxy_iou([0, 0, 10, 10], [10, 0, 20, 10]), 0)

    def test_duplicates_count_as_false_positive(self):
        result = match_boxes([self.box()], [self.box(), self.box(score=.8)])
        self.assertEqual((result['tp'], result['fn'], result['fp']), (1, 0, 1))

    def test_wrong_class_is_miss_but_location_hit(self):
        self.assertEqual(match_boxes([self.box()], [self.box(kind=1)])['tp'], 0)
        self.assertEqual(match_boxes([self.box()], [self.box(kind=1)], False)['tp'], 1)

    def test_one_prediction_cannot_find_two_truth_boxes(self):
        result = match_boxes([self.box(), self.box()], [self.box()])
        self.assertEqual((result['tp'], result['fn']), (1, 1))

    def test_small_overlap_is_not_a_hit(self):
        self.assertEqual(match_boxes([self.box()], [self.box(coordinates=[5, 5, 15, 15])])['tp'], 0)

    def test_unknown_prediction_class_still_counts_as_false_positive(self):
        records = [{'image_id': 1, 'relative_path': 'x', 'group': 'a', 'boxes': [self.box()]}]
        report, _ = evaluate_predictions(records, {'1': [self.box(kind=-1)]}, [{'id': 0, 'name': 'known'}], .1)
        self.assertEqual(report['class_and_location']['fp'], 1)
        self.assertEqual(report['class_and_location']['fn'], 1)
        self.assertEqual(report['unknown_class_false_positives'], 1)

    def test_partial_images_mask_only_negative_objectness_anchors(self):
        targets = torch.tensor([[0.], [1.], [0.], [1.]])
        result = partial_objectness_loss(torch.ones_like(targets), targets, [True, False])
        self.assertEqual(result.flatten().tolist(), [0., 1., 1., 1.])

    def test_partial_images_leave_classification_loss_unchanged(self):
        targets = torch.ones((2, 9))
        result = partial_objectness_loss(torch.ones_like(targets), targets, [True, True])
        self.assertEqual(result.sum().item(), 18.)

    def test_adaptation_trains_only_late_darknet_neck_and_head(self):
        self.assertFalse(adaptable_parameter('backbone.backbone.stem.conv.weight'))
        self.assertFalse(adaptable_parameter('backbone.backbone.dark3.0.conv.weight'))
        self.assertTrue(adaptable_parameter('backbone.backbone.dark4.0.conv.weight'))
        self.assertTrue(adaptable_parameter('backbone.backbone.dark5.0.conv.weight'))
        self.assertTrue(adaptable_parameter('backbone.C3_p3.conv1.conv.weight'))
        self.assertTrue(adaptable_parameter('head.cls_preds.0.weight'))

    def test_validation_selection_prefers_recall_at_acceptable_fp(self):
        categories = [{'id': 0, 'name': 'test'}]
        records = [{'image_id': 1, 'relative_path': 'x', 'group': 'a', 'boxes': [self.box()]}]
        predictions = {'1': [self.box(score=.04), self.box(coordinates=[20, 20, 30, 30], score=.015)]}
        selected, candidates = select_threshold(records, predictions, categories)
        self.assertEqual(selected['threshold'], .025)
        report, _ = evaluate_predictions(records, predictions, categories, .025)
        self.assertEqual(report['class_and_location']['tp'], 1)
        self.assertEqual(report['class_and_location']['fp'], 0)
        self.assertEqual(len(candidates), 6)


if __name__ == '__main__':
    unittest.main()
