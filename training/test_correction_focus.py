"""CPU-only regression for bounded train crops and staged parameter freezing."""
import copy
import random
import unittest

import numpy as np

from correction_focus import crop_focus_sample, parameter_phase, select_focus_records


def box(identity, coordinates, class_id=0, problem='崩边'):
    x1, y1, x2, y2 = coordinates
    return {'id': identity, 'problem': problem, 'class_id': class_id, 'origin': 'human',
            'xyxy': list(coordinates), 'xywh': [x1, y1, x2-x1, y2-y1],
            'normalized_xywh': [x1/100, y1/100, (x2-x1)/100, (y2-y1)/100]}


def row(identity=1, boxes=None):
    boxes = boxes if boxes is not None else [box('focus', [40, 40, 50, 50])]
    return {'image_id': identity, 'split': 'train', 'group': 'order-'+str(identity),
            'width': 100, 'height': 100, 'aspect': 1., 'boxes': boxes,
            'defects': list(dict.fromkeys(b['problem'] for b in boxes)), 'image_state': 'defect',
            'has_draft': False, 'annotation_status': 'confirmed', 'annotation_outcome': 'boxed',
            'sha256': 'a'*64, 'label_fingerprint': 'b'*64,
            'classification_problems_without_boxes': [], 'negative_rois': []}


class CenterRng:
    def uniform(self, first, last):
        return (first+last)/2


class CorrectionFocusTests(unittest.TestCase):
    def test_parameter_groups_keep_backbone_neck_stems_frozen(self):
        for phase in ('classification', 'joint'):
            for name in ('backbone.backbone.dark5.0.weight', 'backbone.C3_p4.conv1.weight',
                         'head.stems.0.conv.weight', 'head.stems.0.bn.weight', 'other.weight'):
                self.assertEqual(parameter_phase(name, phase), 'frozen')
            for name in ('head.cls_convs.0.0.conv.weight', 'head.cls_preds.1.bias'):
                self.assertEqual(parameter_phase(name, phase), 'classification')
        for name in ('head.reg_convs.0.0.conv.weight', 'head.reg_preds.2.weight', 'head.obj_preds.0.bias'):
            self.assertEqual(parameter_phase(name, 'classification'), 'frozen')
            self.assertEqual(parameter_phase(name, 'joint'), 'geometry')
        with self.assertRaises(ValueError):
            parameter_phase('head.cls_preds.0.weight', 'unsupported')

    def test_selection_is_bounded_unique_and_does_not_mutate_source(self):
        records = [row(i) for i in range(7)]
        records.append(copy.deepcopy(records[0]))
        original = copy.deepcopy(records)
        selected, report = select_focus_records(records)
        self.assertEqual(len(selected), 3)
        self.assertEqual(len({r['image_id'] for r in selected}), 3)
        self.assertEqual(report['extra_limit'], 3)
        self.assertEqual(report['original_train_images'], 7)
        self.assertTrue(all(r['focus_crop'] for r in selected))
        self.assertEqual(records, original)
        selected[0]['boxes'][0]['problem'] = 'changed in copy'
        self.assertEqual(records, original)

    def test_heldout_classes_cannot_change_ranking_and_rare_small_first(self):
        records = [row(1), row(2), row(3), row(4, [box('rare', [20, 20, 40, 40], 1, '缺角')])]
        records[1]['boxes'] = [box('smaller', [40, 40, 45, 45])]
        selected, _ = select_focus_records(records)
        self.assertEqual([r['image_id'] for r in selected], [4, 2])
        heldout = []
        for i in range(100):
            item = row(i+100, [box('rare', [20, 20, 40, 40], 1, '缺角')])
            item['split'] = 'val' if i % 2 else 'test'
            heldout.append(item)
        again, report = select_focus_records(records+heldout)
        self.assertEqual(again, selected)
        self.assertEqual((report['validation_extra_images'], report['test_extra_images']), (0, 0))

    def test_existing_training_exclusions_stay_excluded(self):
        changes = [{'split': value} for value in ('val', 'validation', 'test')]
        changes += [{'image_state': value} for value in ('normal', 'unrelated', 'unclear', 'pending')]
        changes += [{'has_draft': True}, {'feedback_only': True}, {'defects': []}, {'boxes': []},
                    {'annotation_status': 'draft'}, {'annotation_outcome': 'unclear'}]
        for change in changes:
            bad = {**row(100), **change}
            selected, _ = select_focus_records([row(i) for i in range(4)]+[bad])
            self.assertNotIn(100, [r['image_id'] for r in selected], change)
            with self.assertRaises(ValueError, msg=str(change)):
                crop_focus_sample(bad, np.zeros((100, 100, 3), dtype=np.uint8))

    def test_no_extra_crop_for_one_image_or_large_focus(self):
        self.assertEqual(select_focus_records([row(1)])[0], [])
        large = row(1, [box('large', [0, 0, 90, 95])])
        self.assertEqual(select_focus_records([large, {**copy.deepcopy(large), 'image_id': 2}])[0], [])

    def test_crop_keeps_intersecting_boxes_and_normal_regions_with_partial_mask(self):
        record = row(boxes=[box('focus', [40, 40, 50, 50]),
                           box('cut', [20, 45, 42, 55], 1, '缺角'),
                           box('outside', [0, 0, 10, 10], 2, '划伤')])
        record.update(focus_box_id='focus', negative_rois=[[50, 20, 70, 38], [0, 80, 10, 90]])
        original = copy.deepcopy(record)
        rgb = np.arange(100*100*3, dtype=np.uint16).reshape((100, 100, 3))
        pixels = rgb.copy()
        transformed, cropped = crop_focus_sample(record, rgb, rng=CenterRng())
        self.assertEqual(transformed['focus_crop_window'], [32, 32, 57, 57])
        self.assertEqual(transformed['width'], 25)
        self.assertEqual(transformed['height'], 25)
        self.assertEqual([b['id'] for b in transformed['boxes']], ['focus', 'cut'])
        self.assertEqual(transformed['boxes'][0]['xyxy'], [8, 8, 18, 18])
        self.assertEqual(transformed['boxes'][1]['xyxy'], [0, 13, 10, 23])
        self.assertEqual(transformed['boxes'][1]['xywh'], [0, 13, 10, 10])
        self.assertEqual(transformed['boxes'][1]['normalized_xywh'], [0, 13/25, 10/25, 10/25])
        self.assertEqual(transformed['negative_rois'], [[18, 0, 25, 6]])
        self.assertEqual(transformed['classification_problems_without_boxes'], ['缺角'])
        self.assertTrue(transformed['crop_partial'])
        self.assertEqual(transformed['focus_crop_metadata']['truncated_box_ids'], ['cut'])
        self.assertEqual(transformed['defects'], record['defects'])
        self.assertEqual([b['class_id'] for b in transformed['boxes']], [0, 1])
        self.assertEqual(transformed['sha256'], original['sha256'])
        self.assertEqual(transformed['label_fingerprint'], original['label_fingerprint'])
        np.testing.assert_array_equal(cropped, pixels[32:57, 32:57])
        cropped[:] = 0
        np.testing.assert_array_equal(rgb, pixels)
        self.assertEqual(record, original)

    def test_existing_partial_is_kept_even_without_truncated_boxes(self):
        record = row()
        record['classification_problems_without_boxes'] = ['尺寸不符']
        transformed, _ = crop_focus_sample(record, np.zeros((100, 100, 3)), rng=CenterRng())
        self.assertEqual(transformed['classification_problems_without_boxes'], ['尺寸不符'])
        self.assertFalse(transformed['crop_partial'])
        self.assertEqual(transformed['negative_rois'], [])

    def test_fixed_seed_and_edge_focus_always_keep_full_focus(self):
        for coordinates in ([0, 0, 10, 10], [91.2, 90.1, 99.8, 100], [40.2, 40.3, 50.4, 55.5]):
            record = row(boxes=[box('focus', coordinates)])
            rgb = np.ones((100, 100, 3), dtype=np.uint8)
            first, _ = crop_focus_sample(record, rgb, rng=random.Random(11))
            second, _ = crop_focus_sample(record, rgb, rng=random.Random(11))
            self.assertEqual(first, second)
            left, top, right, bottom = first['focus_crop_window']
            self.assertLessEqual(left, coordinates[0])
            self.assertLessEqual(top, coordinates[1])
            self.assertGreaterEqual(right, coordinates[2])
            self.assertGreaterEqual(bottom, coordinates[3])
            self.assertFalse(first['crop_partial'])

    def test_shape_identity_and_invalid_geometry_fail_closed(self):
        rgb = np.zeros((100, 100, 3))
        for data, image, kwargs in [
                (row(), rgb[:99], {}),
                ({**row(), 'focus_box_id': 'missing'}, rgb, {}),
                ({**row(), 'negative_rois': [[0, 0, 101, 5]]}, rgb, {}),
                (row(), rgb, {'context_scale': .5}),
                (row(), rgb, {'min_crop_fraction': .9, 'max_crop_fraction': .5})]:
            with self.assertRaises(ValueError):
                crop_focus_sample(data, image, **kwargs)


if __name__ == '__main__':
    unittest.main()
