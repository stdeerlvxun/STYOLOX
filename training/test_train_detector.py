import unittest
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
from unittest.mock import Mock, patch
import torch

from train_detector import (adaptable_parameter, compare_validation, evaluate_predictions,
    localization_optimizer_groups, localization_parameter_group, localization_selection_key,
    load_detector, location_training_records, match_boxes, partial_objectness_loss, select_threshold,
    training_phase, validate_training_modes, xyxy_iou)


class LocalizationFocusTests(unittest.TestCase):
    def test_requires_typed_retained_adaptation_and_fixed_threshold(self):
        valid = dict(initialize_from='retained', adapt_backbone=True, fixed_threshold=.05,
                     localization_focus=True)
        validate_training_modes(**valid)
        for changes in ({'initialize_from': None}, {'adapt_backbone': False},
                        {'fixed_threshold': None}, {'location_only': True}, {'correction_focus': True}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                validate_training_modes(**{**valid, **changes})

    def test_optimizer_groups_cover_trainable_head_and_late_features_once(self):
        names = ['backbone.backbone.stem.conv.weight', 'backbone.backbone.dark3.0.conv.weight',
                 'backbone.backbone.dark4.0.conv.weight', 'backbone.backbone.dark5.0.conv.weight',
                 'backbone.C3_p3.conv1.conv.weight', 'head.cls_convs.0.0.conv.weight',
                 'head.cls_preds.0.weight', 'head.reg_convs.0.0.conv.weight',
                 'head.reg_preds.0.weight', 'head.obj_preds.0.weight', 'head.stems.0.conv.weight']
        parameters = {name: torch.nn.Parameter(torch.ones(1)) for name in names}
        model = SimpleNamespace(named_parameters=lambda: list(parameters.items()))
        groups = localization_optimizer_groups(model)
        self.assertEqual({g['name']: g['lr'] for g in groups},
                         {'classification': .0001, 'geometry': .00005, 'late_features': .00001})
        actual = [id(p) for g in groups for p in g['params']]
        expected = [id(p) for n, p in parameters.items() if adaptable_parameter(n)]
        self.assertCountEqual(actual, expected)
        self.assertEqual(len(actual), len(set(actual)))
        self.assertEqual(localization_parameter_group('head.stems.0.conv.weight'), ('geometry', .00005))
        parameters['head.cls_preds.0.weight'].requires_grad_(False)
        updated = [id(p) for g in localization_optimizer_groups(model) for p in g['params']]
        self.assertNotIn(id(parameters['head.cls_preds.0.weight']), updated)

    def test_joint_localization_begins_at_first_epoch(self):
        self.assertEqual(training_phase(1, 50, 0., 600., localization_focus=True), 'joint')
        self.assertEqual(training_phase(1, 50, 0., 600., correction_focus=True), 'classification')
        self.assertEqual(training_phase(31, 50, 0., 600., correction_focus=True), 'joint')
        self.assertEqual(training_phase(2, 50, 400., 600., correction_focus=True), 'joint')

    def report(self, recall=.6, f2=.5, fp=10, typed_recall=.3, typed_f2=.2, typed_fp=15):
        return {'location_only': {'recall': recall, 'f2': f2, 'fp': fp},
                'class_and_location': {'recall': typed_recall, 'f2': typed_f2,
                                      'fp': typed_fp, 'precision': .2}}

    def test_initialization_beats_any_location_or_typed_regression(self):
        baseline = self.report()
        initial = localization_selection_key(baseline, baseline, 0)
        for report in (self.report(recall=.59, f2=.7, fp=2),
                       self.report(recall=.8, f2=.7, fp=11),
                       self.report(recall=.8, f2=.7, typed_recall=.29),
                       self.report(recall=.8, f2=.7, typed_f2=.19),
                       self.report(recall=.8, f2=.7, typed_fp=16)):
            with self.subTest(report=report):
                self.assertGreater(initial, localization_selection_key(report, baseline, 1))

    def test_safe_selection_prioritizes_location_f2_before_typed_f2(self):
        baseline = self.report()
        better_location = self.report(f2=.6, typed_f2=.21)
        better_type = self.report(f2=.55, typed_f2=.4)
        self.assertGreater(localization_selection_key(better_location, baseline, 5),
                           localization_selection_key(better_type, baseline, 10))
        self.assertGreater(localization_selection_key(self.report(recall=.7), baseline, 5),
                           localization_selection_key(better_location, baseline, 1))


class DetectorLoadTests(unittest.TestCase):
    def test_tiling_only_wraps_explicitly_enabled_checkpoint(self):
        from tiled_inference import TiledDetector
        for tiled in (None, {'enabled': False}, {'enabled': 'true'}, {'enabled': True}):
            with self.subTest(tiled=tiled):
                config = {'input_size': 640, 'nms': .5, 'max_predictions': 10,
                          'roi_verifier': {'schema': 'preserved'}}
                if tiled is not None:
                    config['tiled_inference'] = tiled
                checkpoint = {'config': config, 'categories': [{'id': 0, 'name': '崩边'}],
                              'model': {'unchanged': 'state'}, 'threshold': .05}
                model, original = Mock(), SimpleNamespace(threshold=.05)
                with patch('train_detector.torch.load', return_value=checkpoint), \
                     patch('train_detector.make_model', return_value=model), \
                     patch('train_detector.Detector', return_value=original):
                    loaded = load_detector('unused', device='cpu')
                model.load_state_dict.assert_called_once_with({'unchanged': 'state'})
                if tiled == {'enabled': True}:
                    self.assertIsInstance(loaded, TiledDetector)
                    self.assertIs(loaded.detector, original)
                    self.assertEqual(loaded.threshold, .05)
                    self.assertEqual(loaded.roi_verifier, {'schema': 'preserved'})
                else:
                    self.assertIs(loaded, original)


class DetectorMetricsTests(unittest.TestCase):
    def test_binary_candidate_comparison_scores_both_models_as_locations(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = root/'data.json'
            dataset.write_text(json.dumps({'categories': [{'id': 3, 'name': '崩边'}], 'records': [
                {'image_id': 1, 'relative_path': 'image', 'full_path': 'image', 'group': 'g',
                 'sha256': 'hash', 'split': 'validation', 'boxes': [
                    {'class_id': 3, 'problem': '崩边', 'xyxy': [0, 0, 10, 10]}]}]}), encoding='utf-8')
            thresholds = []
            def fake_detector(problem):
                return SimpleNamespace(model=SimpleNamespace(cpu=lambda: None),
                    predict_image=lambda image, threshold: thresholds.append(threshold) or
                        [{'class_id': 0, 'problem': problem,
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
            self.assertEqual(thresholds, [.05, .05])

    def test_comparison_without_fixed_threshold_keeps_grid_floor(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = root/'data.json'
            dataset.write_text(json.dumps({'categories': [{'id': 0, 'name': '崩边'}], 'records': [
                {'image_id': 1, 'relative_path': 'image', 'full_path': 'image', 'group': 'g',
                 'sha256': 'hash', 'split': 'val', 'boxes': [
                    {'class_id': 0, 'problem': '崩边', 'xyxy': [0, 0, 10, 10]}]}]}), encoding='utf-8')
            thresholds = []
            def fake_detector(*args):
                return SimpleNamespace(model=SimpleNamespace(cpu=lambda: None),
                    predict_image=lambda image, threshold: thresholds.append(threshold) or
                        [{'class_id': 0, 'problem': '崩边', 'xyxy': [0, 0, 10, 10], 'score': .1}])
            with patch('train_detector.torch.load', return_value={'config': {}}), \
                 patch('train_detector.load_detector', side_effect=fake_detector), \
                 patch('train_detector.sha256', return_value='hash'):
                result = compare_validation(dataset, root, root)
            self.assertEqual(thresholds, [.01, .01])
            self.assertIsNone(result['fixed_threshold_for_both'])

    def test_location_only_training_view_preserves_original_human_classes(self):
        box = {'class_id': 3, 'problem': '崩边', 'xyxy': [1, 2, 3, 4]}
        dataset = {'records': [{'boxes': [box], 'split': 'test', 'image_state': '',
                    'hard_positive': {'policy': 'example', 'moved_boxes': [{'box_id': 'b1'}]}}],
                   'negative_feedback_records': [{'boxes': [box]}]}
        categories, records, feedback = location_training_records(dataset)
        self.assertEqual(categories, [{'id': 0, 'name': '问题位置'}])
        self.assertEqual(records[0]['boxes'][0]['class_id'], 0)
        self.assertEqual(records[0]['boxes'][0]['original_problem'], '崩边')
        self.assertEqual(feedback[0]['boxes'][0]['original_class_id'], 3)
        self.assertEqual(dataset['records'][0]['boxes'][0]['class_id'], 3)
        self.assertEqual(records[0]['split'], 'test')
        self.assertEqual(records[0]['hard_positive'], dataset['records'][0]['hard_positive'])

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
