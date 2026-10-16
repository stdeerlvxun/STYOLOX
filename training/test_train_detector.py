import unittest
import torch

from train_detector import adaptable_parameter, evaluate_predictions, match_boxes, partial_objectness_loss, select_threshold, xyxy_iou


class DetectorMetricsTests(unittest.TestCase):
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
