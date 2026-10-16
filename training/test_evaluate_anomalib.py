"""CPU-only protocol checks; no model downloads, source archive or CUDA access."""
import copy
import hashlib
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image

from evaluate_anomalib import (NORMAL_QUANTILE, axis_starts, boxes_from_map,
                              calibration_threshold, eligible_problem, nms_boxes,
                              read_verified, resize_geometry, roi_bounds,
                              split_normal_rois, tile_tensor, tile_windows)


def sample_dataset():
    records, normal = [], []
    for index in range(7):
        row = {'image_id': index, 'case_id': 'case-'+str(index), 'sha256': format(index+1, '064x'),
               'phash': str(0xffff << (16*index)), 'aspect': 1., 'order_keys': ['order:'+str(index)],
               'relative_path': str(index)+'.png', 'full_path': str(index)+'.png',
               'width': 100, 'height': 100, 'image_state': 'defect', 'defects': ['崩边'],
               'has_draft': False, 'annotation_status': 'confirmed', 'annotation_outcome': 'boxed',
               'boxes': [{'class_id': 0, 'problem': '崩边', 'xyxy': [1, 1, 4, 4]}],
               'label_fingerprint': 'label-'+str(index), 'group': index,
               'split': 'train' if index < 6 else 'validation'}
        records.append(row)
        if index < 6:
            negative = copy.deepcopy(row)
            negative.update(feedback_only=True, feedback_provenance_sha256='e'*64,
                            feedback_ids=[index+1], negative_rois=[[10.2, 11.2, 35.8, 39.8]])
            normal.append(negative)
    return {'training_policy': 'human_confirmed_problem_boxes_v1', 'records': records,
            'negative_feedback_records': normal, 'included_feedback_ids': list(range(1, 7)),
            'historical_holdout_records': []}


class EvaluationProtocolTests(unittest.TestCase):
    def test_normal_split_is_deterministic_disjoint_and_preserves_inputs(self):
        dataset = sample_dataset()
        original = copy.deepcopy(dataset)
        bank, calibration = split_normal_rois(dataset)
        self.assertEqual(len(bank)+len(calibration), 6)
        self.assertFalse({r['group'] for r in bank} & {r['group'] for r in calibration})
        self.assertEqual((bank, calibration), split_normal_rois(dataset))
        self.assertEqual(dataset, original)

    def test_linked_normal_cases_stay_in_the_same_partition(self):
        dataset = sample_dataset()
        for rows in (dataset['records'], dataset['negative_feedback_records']):
            rows[1]['case_id'] = rows[0]['case_id']
        bank, calibration = split_normal_rois(dataset)
        partitions = {row['record']['image_id']: name for name, rows in [('bank', bank), ('calibration', calibration)] for row in rows}
        self.assertEqual(partitions[0], partitions[1])

    def test_all_selected_nondefect_states_and_drafts_are_rejected(self):
        for state in ['unclear', 'normal', 'unrelated', 'pending_responsibility']:
            dataset = sample_dataset()
            dataset['records'][0]['image_state'] = state
            with self.assertRaises(ValueError):
                split_normal_rois(dataset)
        dataset = sample_dataset()
        dataset['negative_feedback_records'][0]['has_draft'] = True
        with self.assertRaises(ValueError):
            split_normal_rois(dataset)
        self.assertFalse(eligible_problem({'image_state': 'defect', 'defects': ['崩边']}))

    def test_unconfirmed_problem_boxes_are_rejected(self):
        for key, value in [('annotation_status', 'draft'), ('annotation_outcome', 'unclear'), ('defects', [])]:
            dataset = sample_dataset()
            dataset['records'][0][key] = value
            with self.assertRaises(ValueError):
                split_normal_rois(dataset)

    def test_missing_explicit_normal_provenance_is_rejected(self):
        for key, value in [('feedback_only', False), ('feedback_provenance_sha256', ''),
                           ('feedback_ids', []), ('feedback_ids', [100])]:
            dataset = sample_dataset()
            dataset['negative_feedback_records'][0][key] = value
            with self.assertRaises(ValueError):
                split_normal_rois(dataset)

    def test_holdout_sha_order_and_near_duplicate_leakage_is_rejected(self):
        for key in ['case_id', 'sha256', 'order_keys', 'phash']:
            dataset = sample_dataset()
            dataset['records'][-1][key] = copy.deepcopy(dataset['records'][0][key])
            with self.assertRaises(ValueError):
                split_normal_rois(dataset)
        dataset = sample_dataset()
        historical = copy.deepcopy(dataset['records'][0])
        historical.update(split='test', group='historic')
        dataset['historical_holdout_records'].append(historical)
        with self.assertRaises(ValueError):
            split_normal_rois(dataset)

    def test_explicit_local_feedback_need_not_have_a_positive_location_box(self):
        dataset = sample_dataset()
        dataset['records'] = dataset['records'][1:]
        dataset['negative_feedback_records'][0]['boxes'] = []
        bank, calibration = split_normal_rois(dataset)
        self.assertEqual(len(bank)+len(calibration), 6)

    def test_one_independent_normal_group_is_insufficient(self):
        dataset = sample_dataset()
        dataset['negative_feedback_records'] = dataset['negative_feedback_records'][:1]
        with self.assertRaises(ValueError):
            split_normal_rois(dataset)

    def test_normal_roi_rounds_inward_and_cannot_be_whole_image_or_defect(self):
        self.assertEqual(roi_bounds([1.2, 2.8, 9.7, 10.2], 100, 100), (2, 3, 9, 10))
        for roi in ([0, 0, 100, 100], [-1, 2, 5, 6], [2, 2, 1, 5], [0, 0, float('nan'), 2]):
            with self.assertRaises(ValueError):
                roi_bounds(roi, 100, 100)
        dataset = sample_dataset()
        dataset['negative_feedback_records'][0]['negative_rois'] = [[2, 2, 10, 10]]
        with self.assertRaises(ValueError):
            split_normal_rois(dataset)

    def test_fixed_normal_quantile_uses_only_provided_finite_calibration_values(self):
        self.assertEqual(NORMAL_QUANTILE, .995)
        scores = np.arange(1000, dtype=np.float32)
        self.assertEqual(calibration_threshold(scores), 995.)
        for values in ([], [float('nan')], [float('inf')]):
            with self.assertRaises(ValueError):
                calibration_threshold(values)

    def test_tiles_cover_boundaries_without_gaps_and_keep_small_sources(self):
        self.assertEqual(resize_geometry(2000, 1000), (1024, 512))
        self.assertEqual(resize_geometry(100, 200), (100, 200))
        self.assertEqual(axis_starts(100), [0])
        self.assertEqual(axis_starts(600)[-1], 344)
        coverage = np.zeros((300, 600), dtype=np.int32)
        for x1, y1, x2, y2 in tile_windows(600, 300):
            coverage[y1:y2, x1:x2] += 1
        self.assertTrue((coverage > 0).all())

    def test_map_coordinates_are_mapped_to_original_image(self):
        heat = np.zeros((50, 100), dtype=np.float32)
        heat[10:20, 30:40] = 5.
        boxes = boxes_from_map(heat, 2., (1000, 500))
        self.assertEqual(len(boxes), 1)
        self.assertEqual(boxes[0]['xyxy'], [300., 100., 400., 200.])
        self.assertEqual(boxes[0]['normalized_xywh'], [.3, .2, .1, .2])
        self.assertEqual(boxes[0]['score'], 5.)
        self.assertEqual(boxes_from_map(heat, 5., (1000, 500)), [])

    def test_nms_removes_duplicate_candidates_and_caps_at_ten(self):
        boxes = [{'score': 3., 'xyxy': [0., 0., 10., 10.]},
                 {'score': 2., 'xyxy': [1., 1., 11., 11.]}]
        self.assertEqual(len(nms_boxes(boxes)), 1)
        separated = [{'score': float(i), 'xyxy': [20.*i, 0., 20.*i+10., 10.]} for i in range(15)]
        self.assertEqual(len(nms_boxes(separated)), 10)

    def test_tile_rgb_normalization_and_model_size(self):
        image = Image.fromarray(np.full((10, 20, 3), 255, dtype=np.uint8))
        tensor = tile_tensor(image, [(0, 0, 20, 10)], 'cpu')
        self.assertEqual(tuple(tensor.shape), (1, 3, 224, 224))
        np.testing.assert_allclose(tensor[0, :, 0, 0].numpy(), (1-np.array([.485, .456, .406]))/np.array([.229, .224, .225]), rtol=1e-6)

    def test_verified_reader_checks_root_hash_and_exif_geometry(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'source.png'
            Image.new('RGB', (20, 10)).save(path)
            row = {'relative_path': 'source.png', 'full_path': str(path), 'width': 20, 'height': 10,
                   'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
            self.assertEqual(read_verified(row, directory).size, (20, 10))
            for changes in ({'sha256': 'a'*64}, {'width': 21}, {'relative_path': '../source.png'}):
                with self.assertRaises(ValueError):
                    read_verified({**row, **changes}, directory)


if __name__ == '__main__':
    unittest.main()
