"""Small CPU checks; no source images, production state or CUDA are touched."""
import copy
import json
from types import SimpleNamespace
import unittest

import numpy as np
import torch

from roi_verifier import (SUPPRESS_THRESHOLD, _roi_input, extract_roi_features,
                          fit_verifier, keep_probabilities)


class FakePan(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.bn = torch.nn.BatchNorm2d(3)
        self.inputs = []

    def forward(self, image):
        self.inputs.append(image.detach().clone())
        output = self.bn(image)
        return output, output[:, :2, ::2, ::2], output[:, :1, ::4, ::4]


class RoiVerifierTests(unittest.TestCase):
    def test_roi_context_bgr_and_padding_are_fixed(self):
        rgb = np.zeros((100, 100, 3), dtype=np.uint8)
        rgb[:] = [10, 20, 30]
        image = _roi_input(rgb, [40, 40, 60, 50])
        self.assertEqual(image.shape, (3, 224, 224))
        np.testing.assert_array_equal(image[:, 0, 0], [30, 20, 10])
        np.testing.assert_array_equal(image[:, -1, -1], [114, 114, 114])
        self.assertEqual(image.dtype, np.float32)

    def test_explicit_unit_context_does_not_include_surrounding_pixels(self):
        rgb = np.zeros((100, 100, 3), dtype=np.uint8)
        rgb[40:60, 40:60] = 255
        image = _roi_input(rgb, [40, 40, 60, 60], context_scale=1.)
        self.assertTrue((image == 255).all())
        meta = fit_verifier(np.ones((2, 3)), np.zeros((2, 3)), context_scale=1.)
        self.assertEqual(meta['context_scale'], 1.)

    def test_extraction_uses_three_levels_and_preserves_backbone(self):
        backbone = FakePan().train()
        backbone.bn.eval()
        before = copy.deepcopy(backbone.state_dict())
        flags = [module.training for module in backbone.modules()]
        detector = SimpleNamespace(model=SimpleNamespace(backbone=backbone), device='cpu')
        rgb = np.full((100, 100, 3), 20, dtype=np.uint8)
        original = rgb.copy()
        result = extract_roi_features(detector, rgb, [[20, 20, 40, 40], {'xyxy': [10, 30, 40, 40]}])
        self.assertEqual(result.shape, (2, 12))
        self.assertIsInstance(result, np.ndarray)
        self.assertTrue(np.isfinite(result).all())
        self.assertEqual(flags, [module.training for module in backbone.modules()])
        for key in before:
            torch.testing.assert_close(backbone.state_dict()[key], before[key], rtol=0, atol=0)
        self.assertTrue(all(parameter.grad is None for parameter in backbone.parameters()))
        np.testing.assert_array_equal(rgb, original)
        self.assertEqual(extract_roi_features(detector, rgb, []).shape, (0, 0))

    def test_invalid_roi_and_state_restoration_on_failure(self):
        backbone = FakePan().train()
        detector = SimpleNamespace(model=SimpleNamespace(backbone=backbone), device='cpu')
        for roi in ([0, 0, 101, 2], [5, 5, 2, 2], [0, 0, float('nan'), 5]):
            with self.assertRaises(ValueError):
                extract_roi_features(detector, np.zeros((100, 100, 3), dtype=np.uint8), [roi])
            self.assertTrue(backbone.training)

    def test_fit_balances_classes_and_serializes_without_changing_features(self):
        positive = np.full((100, 4), 2.)
        negative = np.full((5, 4), -2.)
        before_positive, before_negative = positive.copy(), negative.copy()
        hashes = {'positive_image': 'a'*64, 'negative_image': 'b'*64, 'backbone_model': 'c'*64}
        threads = torch.get_num_threads()
        meta = fit_verifier(positive, negative, source_hashes=hashes)
        scores = keep_probabilities(np.concatenate((positive[:1], negative[:1])), meta)
        self.assertGreater(scores[0], .95)
        self.assertLessEqual(scores[1], SUPPRESS_THRESHOLD)
        self.assertEqual(meta['positive_regions'], 100)
        self.assertEqual(meta['negative_regions'], 5)
        self.assertFalse(meta['threshold_searched'])
        self.assertEqual(meta['suppress_if_problem_probability_at_most'], .05)
        self.assertTrue(meta['requires_validation_no_true_positive_loss'])
        self.assertEqual(meta['source_hashes'], hashes)
        self.assertEqual(torch.get_num_threads(), threads)
        np.testing.assert_array_equal(positive, before_positive)
        np.testing.assert_array_equal(negative, before_negative)
        restored = json.loads(json.dumps(meta, allow_nan=False))
        self.assertEqual(scores, keep_probabilities(np.concatenate((positive[:1], negative[:1])), restored))

    def test_identical_evidence_gives_balanced_half_instead_of_majority_bias(self):
        meta = fit_verifier(np.ones((40, 3)), np.ones((2, 3)))
        self.assertAlmostEqual(keep_probabilities([[1., 1., 1.]], meta)[0], .5, places=5)
        self.assertTrue(all(value > 0 for value in meta['scale']))
        self.assertEqual(keep_probabilities([], meta), [])

    def test_no_missing_or_implicit_negative_class_and_bad_metadata(self):
        for positive, negative in [(np.ones((2, 3)), np.empty((0, 3))),
                                   (np.ones((2, 3)), np.ones((2, 4))),
                                   (np.full((2, 3), np.nan), np.ones((2, 3)))]:
            with self.assertRaises(ValueError):
                fit_verifier(positive, negative)
        with self.assertRaises(ValueError):
            fit_verifier(np.ones((2, 3)), np.zeros((2, 3)), source_hashes=['invalid'])
        meta = fit_verifier(np.ones((2, 3)), np.zeros((2, 3)))
        with self.assertRaises(ValueError):
            keep_probabilities([[1., 2.]], meta)
        with self.assertRaises(ValueError):
            keep_probabilities([[1., 2., 3.]], {**meta, 'scale': [0., 1., 1.]})


if __name__ == '__main__':
    unittest.main()
