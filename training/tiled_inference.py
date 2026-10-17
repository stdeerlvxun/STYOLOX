"""Optional full-image plus overlapping-tile inference for the existing detector.

No model weights, image files, labels, confidence scores or detector thresholds are
changed. Coordinates always refer to the EXIF-oriented original image. This is a
proposal generator, not a substitute for one-to-one ground-truth evaluation.
"""
from __future__ import annotations

from dataclasses import dataclass
import copy
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps


@dataclass(frozen=True)
class TiledInferenceConfig:
    enabled: bool = False
    tile_size: int = 640
    overlap: float = .25
    max_tiles: int = 16
    max_side: int = 1920
    nms_iou: float = .5
    max_predictions: int = 10

    def __post_init__(self):
        if not isinstance(self.enabled, bool):
            raise ValueError('enabled must be boolean')
        for name in ('tile_size', 'max_tiles', 'max_side', 'max_predictions'):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f'{name} must be a positive integer')
        if self.max_tiles > 16 or self.max_predictions > 10:
            raise ValueError('At most 16 tiles and 10 final predictions are supported')
        if not 0 <= self.overlap < 1 or not 0 < self.nms_iou <= 1:
            raise ValueError('Invalid overlap or NMS IoU')


def read_oriented_rgb(image):
    if isinstance(image, (str, Path)):
        with Image.open(image) as source:
            return np.asarray(ImageOps.exif_transpose(source).convert('RGB')).copy()
    if isinstance(image, Image.Image):
        return np.asarray(ImageOps.exif_transpose(image).convert('RGB')).copy()
    array = np.asarray(image)
    if array.ndim != 3 or array.shape[2] != 3 or min(array.shape[:2]) <= 0:
        raise ValueError('Expected a nonempty RGB HWC image, PIL image or path')
    return array.astype(np.uint8, copy=True)


def tile_windows(width, height, config):
    """Cover the complete working image, including its last row and column."""
    if min(width, height) <= 0:
        raise ValueError('Image dimensions must be positive')
    stride = max(1, int(config.tile_size * (1 - config.overlap)))

    def starts(length):
        if length <= config.tile_size:
            return [0]
        values = list(range(0, length - config.tile_size + 1, stride))
        if values[-1] != length - config.tile_size:
            values.append(length - config.tile_size)
        return values

    windows = [(x, y, min(width, x + config.tile_size),
                min(height, y + config.tile_size))
               for y in starts(height) for x in starts(width)]
    if len(windows) > config.max_tiles:
        raise ValueError(f'Tile budget exceeded: {len(windows)} > {config.max_tiles}; '
                         'no tiles were silently omitted')
    return windows


def xyxy_iou(left, right):
    intersection = max(0., min(left[2], right[2]) - max(left[0], right[0])) * max(
        0., min(left[3], right[3]) - max(left[1], right[1]))
    left_area = max(0., left[2] - left[0]) * max(0., left[3] - left[1])
    right_area = max(0., right[2] - right[0]) * max(0., right[3] - right[1])
    return intersection / max(left_area + right_area - intersection, 1e-12)


def remap_tile_prediction(prediction, window, working_size, original_size,
                          tile_index):
    """Copy a tile prediction without recalibrating or inventing its score."""
    tile_x, tile_y, tile_right, tile_bottom = window
    working_width, working_height = working_size
    original_width, original_height = original_size
    local = [float(v) for v in prediction['xyxy']]
    if len(local) != 4 or not all(math.isfinite(v) for v in local):
        raise ValueError('Prediction xyxy must contain four finite coordinates')
    tile_width, tile_height = tile_right - tile_x, tile_bottom - tile_y
    x1, x2 = np.clip([local[0], local[2]], 0, tile_width)
    y1, y2 = np.clip([local[1], local[3]], 0, tile_height)
    if x2 <= x1 or y2 <= y1:
        raise ValueError('Tile prediction has no visible area')
    coordinates = [float(v) for v in (
        (tile_x + x1) * original_width / working_width,
        (tile_y + y1) * original_height / working_height,
        (tile_x + x2) * original_width / working_width,
        (tile_y + y2) * original_height / working_height)]
    normalized = [coordinates[0] / original_width, coordinates[1] / original_height,
                  (coordinates[2] - coordinates[0]) / original_width,
                  (coordinates[3] - coordinates[1]) / original_height]
    result = copy.deepcopy(prediction)
    result.update(xyxy=coordinates, xywh=normalized, normalized_xywh=list(normalized))
    result['tiled_inference'] = {
        'source': 'tile', 'tile_index': tile_index,
        'working_window_xyxy': list(window),
        'working_size': list(working_size), 'original_size': list(original_size),
        # Diagnostic only: boundary fragments are never assigned a new score.
        'touches_internal_boundary': bool(
            (tile_x > 0 and x1 <= 1) or (tile_y > 0 and y1 <= 1) or
            (tile_right < working_width and x2 >= tile_width - 1) or
            (tile_bottom < working_height and y2 >= tile_height - 1))}
    return result


def merge_predictions(full_predictions, tile_predictions, config):
    """Keep original proposals first; suppress tile duplicates of the same type.

    Boxes are not unioned or moved. Adjacent non-overlapping defects remain
    separate. Original full-image proposals cannot be displaced by tile scores.
    """
    full = [copy.deepcopy(p) for p in full_predictions]
    tile = sorted(tile_predictions, key=lambda p: -p['score'])
    kept = list(full)
    suppressed = 0
    for candidate in tile:
        if any(candidate['class_id'] == previous['class_id'] and
               xyxy_iou(candidate['xyxy'], previous['xyxy']) >= config.nms_iou
               for previous in kept):
            suppressed += 1
            continue
        kept.append(copy.deepcopy(candidate))
    limit_dropped = max(0, len(kept) - config.max_predictions)
    return kept[:config.max_predictions], {
        'same_class_duplicates_suppressed': suppressed,
        'predictions_dropped_by_final_limit': limit_dropped}


class TiledDetector:
    """Duck-typed Detector wrapper; default configuration retains old behavior."""
    def __init__(self, detector, config=None):
        self.detector = detector
        self.config = TiledInferenceConfig(**config) if isinstance(config, dict) else (
            config or TiledInferenceConfig())

    def __getattr__(self, name):
        return getattr(self.detector, name)

    def predict_image(self, image, threshold=None):
        return self.predict_image_with_diagnostics(image, threshold)[0]

    def predict_image_with_diagnostics(self, image, threshold=None):
        threshold = self.detector.threshold if threshold is None else threshold
        if not math.isfinite(threshold) or threshold < self.detector.threshold:
            raise ValueError('Tiled inference must not lower the retained detector threshold')
        rgb = read_oriented_rgb(image)
        height, width = rgb.shape[:2]
        # Always run the unmodified whole image, not a resized tile canvas.
        full = self.detector.predict_image(rgb, threshold)
        diagnostics = {
            'enabled': self.config.enabled, 'threshold': threshold,
            'original_size': [width, height], 'full_image_predictions': len(full),
            'tile_count': 0, 'tile_predictions': 0,
            'interference_filter': {'status': 'not_applied',
                'reason': 'No verified text/watermark mask source; no pixels or proposals removed'}}
        if not self.config.enabled:
            return full, diagnostics
        scale = min(1., self.config.max_side / max(width, height))
        working_size = (max(1, round(width * scale)), max(1, round(height * scale)))
        if working_size == (width, height):
            working = rgb
        else:
            working = np.asarray(Image.fromarray(rgb).resize(
                working_size, Image.Resampling.BICUBIC))
        windows = tile_windows(*working_size, self.config)
        # A single identical view cannot add information to the whole-image pass.
        if len(windows) == 1 and working_size == (width, height):
            windows = []
        predictions = []
        for tile_index, window in enumerate(windows):
            x1, y1, x2, y2 = window
            tile = np.ascontiguousarray(working[y1:y2, x1:x2])
            for prediction in self.detector.predict_image(tile, threshold):
                predictions.append(remap_tile_prediction(prediction, window,
                    working_size, (width, height), tile_index))
        full = [dict(copy.deepcopy(p), tiled_inference={'source': 'full_image'}) for p in full]
        merged, merge_diagnostics = merge_predictions(full, predictions, self.config)
        diagnostics.update(working_size=list(working_size), tile_count=len(windows),
            tile_predictions=len(predictions), **merge_diagnostics)
        return merged, diagnostics
