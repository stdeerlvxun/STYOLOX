"""CPU-only geometric checks; no model weights, CUDA, NAS or database required."""
import copy
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image

from tiled_inference import (TiledDetector, TiledInferenceConfig, merge_predictions,
                              read_oriented_rgb, remap_tile_prediction, tile_windows)


def box(xyxy, score=.5, class_id=0):
    return {'xyxy': list(xyxy), 'score': score, 'class_id': class_id,
            'problem': '崩边' if class_id == 0 else '划伤',
            'provenance': {'model': 'retained-yolox'}, 'class_agnostic': False}


class FakeDetector:
    threshold = .05
    max_predictions = 10
    categories = {0: '崩边', 1: '划伤'}

    def __init__(self, output=None):
        self.calls = []
        self.output = output or (lambda image, index: [])

    def predict_image(self, image, threshold=None):
        self.calls.append((image.shape, threshold))
        return copy.deepcopy(self.output(image, len(self.calls) - 1))


class TiledInferenceTests(unittest.TestCase):
    def test_default_disabled_is_exact_original_output_and_threshold(self):
        prediction = box([10, 20, 30, 40])
        detector = FakeDetector(lambda image, index: [prediction])
        result, diagnostics = TiledDetector(detector).predict_image_with_diagnostics(
            np.zeros((800, 1000, 3), dtype=np.uint8))
        self.assertEqual(result, [prediction])
        self.assertEqual(detector.calls, [((800, 1000, 3), .05)])
        self.assertEqual(diagnostics['tile_count'], 0)
        self.assertEqual(diagnostics['interference_filter']['status'], 'not_applied')

    def test_grid_covers_last_pixels_and_never_silently_truncates(self):
        config = TiledInferenceConfig(enabled=True)
        windows = tile_windows(1920, 1920, config)
        self.assertEqual(len(windows), 16)
        covered = np.zeros((1920, 1920), dtype=np.uint8)
        for x1, y1, x2, y2 in windows:
            covered[y1:y2, x1:x2] = 1
        self.assertTrue(covered.all())
        with self.assertRaisesRegex(ValueError, 'no tiles were silently omitted'):
            tile_windows(1920, 1920, TiledInferenceConfig(enabled=True, max_tiles=15))
        with self.assertRaises(ValueError):
            tile_windows(0, 30, config)

    def test_coordinate_mapping_and_normalized_boxes_preserve_metadata(self):
        source = box([10, 20, 50, 60], .321)
        before = copy.deepcopy(source)
        mapped = remap_tile_prediction(source, (480, 0, 1120, 640),
            (1920, 960), (3840, 1920), 1)
        self.assertEqual(mapped['xyxy'], [980, 40, 1060, 120])
        np.testing.assert_allclose(mapped['normalized_xywh'],
            [980/3840, 40/1920, 80/3840, 80/1920])
        self.assertEqual(mapped['xywh'], mapped['normalized_xywh'])
        for key in ('score', 'class_id', 'problem', 'provenance', 'class_agnostic'):
            self.assertEqual(mapped[key], source[key])
        self.assertEqual(source, before)

    def test_internal_boundary_is_only_a_diagnostic_no_score_rewriting(self):
        source = box([-3, 20, 641, 50], .1234)
        result = remap_tile_prediction(source, (480, 0, 1120, 640),
            (1920, 1000), (1920, 1000), 1)
        self.assertTrue(result['tiled_inference']['touches_internal_boundary'])
        self.assertEqual(result['xyxy'], [480, 20, 1120, 50])
        self.assertEqual(result['score'], .1234)
        exterior = remap_tile_prediction(box([0, 0, 20, 20]), (0, 0, 640, 640),
            (1000, 1000), (1000, 1000), 0)
        self.assertFalse(exterior['tiled_inference']['touches_internal_boundary'])
        for coordinates in ([10, 10, 2, 2], [0, 0, float('nan'), 20]):
            with self.assertRaises(ValueError):
                remap_tile_prediction(box(coordinates), (0, 0, 640, 640),
                    (1000, 1000), (1000, 1000), 0)

    def test_duplicate_suppression_keeps_full_view_but_not_adjacent_defects(self):
        full = [box([10, 10, 50, 50], .1)]
        tiled = [box([11, 11, 51, 51], .9), box([60, 10, 100, 50], .8),
                 box([61, 11, 101, 51], .7), box([10, 10, 50, 50], .6, 1)]
        before = copy.deepcopy((full, tiled))
        result, diagnostics = merge_predictions(full, tiled, TiledInferenceConfig(enabled=True))
        self.assertEqual(len(result), 3)
        self.assertEqual(result[0], full[0])
        self.assertEqual(result[1]['xyxy'], [60, 10, 100, 50])
        self.assertEqual(result[2]['class_id'], 1)
        self.assertEqual(diagnostics['same_class_duplicates_suppressed'], 2)
        self.assertEqual((full, tiled), before)

    def test_final_cap_ten_does_not_displace_old_full_image_candidates(self):
        full = [box([i*20, 0, i*20+10, 10], .1) for i in range(9)]
        tiled = [box([i*20, 30, i*20+10, 40], .99-i*.01) for i in range(20)]
        result, diagnostics = merge_predictions(full, tiled, TiledInferenceConfig(enabled=True))
        self.assertEqual(len(result), 10)
        self.assertEqual(result[:9], full)
        self.assertEqual(diagnostics['predictions_dropped_by_final_limit'], 19)

    def test_exif_rotation_once_for_pil_path_and_full_pass(self):
        image = Image.fromarray(np.zeros((20, 40, 3), dtype=np.uint8))
        image.getexif()[274] = 6
        np.testing.assert_equal(read_oriented_rgb(image).shape, (40, 20, 3))
        detector = FakeDetector()
        TiledDetector(detector, TiledInferenceConfig(enabled=True)).predict_image(image)
        self.assertEqual(detector.calls, [((40, 20, 3), .05)])
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'oriented.jpg'
            image.save(path, exif=image.getexif())
            self.assertEqual(read_oriented_rgb(path).shape, (40, 20, 3))

    def test_large_original_keeps_full_pass_then_controlled_downsize(self):
        detector = FakeDetector(lambda image, index:
            [] if index == 0 else [box([10, 20, 30, 40], .07)])
        original = np.zeros((2000, 4000, 3), dtype=np.uint8)
        result, diagnostics = TiledDetector(detector,
            TiledInferenceConfig(enabled=True)).predict_image_with_diagnostics(original)
        self.assertEqual(detector.calls[0], ((2000, 4000, 3), .05))
        self.assertEqual(diagnostics['working_size'], [1920, 960])
        self.assertEqual(diagnostics['tile_count'], 8)
        self.assertTrue(all(shape == (640, 640, 3) for shape, _ in detector.calls[1:]))
        np.testing.assert_allclose(result[0]['xyxy'], [10*4000/1920,
            20*2000/960, 30*4000/1920, 40*2000/960])
        self.assertTrue((original == 0).all())
        self.assertTrue(all(threshold == .05 for _, threshold in detector.calls))

    def test_lower_threshold_is_rejected_before_inference(self):
        detector = FakeDetector()
        wrapper = TiledDetector(detector, TiledInferenceConfig(enabled=True))
        for value in (.01, float('nan')):
            with self.assertRaises(ValueError):
                wrapper.predict_image(np.zeros((10, 10, 3), dtype=np.uint8), value)
        self.assertEqual(detector.calls, [])

    def test_config_limits(self):
        for kwargs in ({'max_tiles': 17}, {'max_predictions': 11}, {'overlap': 1},
                       {'overlap': -.1}, {'nms_iou': 0}, {'max_side': 0},
                       {'enabled': 'yes'}, {'tile_size': True}):
            with self.assertRaises(ValueError):
                TiledInferenceConfig(**kwargs)


if __name__ == '__main__':
    unittest.main()
