"""Bounded, supervised crops for confirmed training boxes; no GPU or source writes."""
from __future__ import annotations

import copy
from collections import Counter
import math
import random

import numpy as np


POLICY = 'confirmed_train_rare_small_context_crop_v1'


def parameter_phase(name: str, phase: str) -> str:
    """Return a training group; stems/backbone/neck remain frozen in both phases.

    The caller freezes BN running statistics separately, including trainable heads.
    """
    if phase not in ('classification', 'joint'):
        raise ValueError('Unknown correction training phase: '+str(phase))
    if name.startswith(('head.cls_convs.', 'head.cls_preds.')):
        return 'classification'
    if phase == 'joint' and name.startswith(('head.reg_convs.', 'head.reg_preds.', 'head.obj_preds.')):
        return 'geometry'
    return 'frozen'


def _valid_xyxy(values, width, height):
    return (isinstance(values, (list, tuple)) and len(values) == 4
            and all(type(v) in (int, float) and math.isfinite(v) for v in values)
            and 0 <= values[0] < values[2] <= width+1e-6
            and 0 <= values[1] < values[3] <= height+1e-6)


def _eligible(record):
    if (record.get('split') != 'train' or record.get('has_draft') or record.get('feedback_only')
            or record.get('image_state') not in (None, '', 'defect')
            or not record.get('defects') or not record.get('boxes')
            or record.get('annotation_status') != 'confirmed'
            or record.get('annotation_outcome') != 'boxed'):
        return False
    width, height = record.get('width'), record.get('height')
    if type(width) is not int or type(height) is not int or min(width, height) < 2:
        return False
    return all(isinstance(box, dict) and box.get('id') and box.get('problem')
               and type(box.get('class_id')) is int and box['class_id'] >= 0
               and _valid_xyxy(box.get('xyxy'), width, height) for box in record['boxes'])


def _area(box, record):
    x1, y1, x2, y2 = box['xyxy']
    return (x2-x1)*(y2-y1)/(record['width']*record['height'])


def _zoomable(box, record):
    x1, y1, x2, y2 = box['xyxy']
    return max((x2-x1)/record['width'], (y2-y1)/record['height']) <= .6


def select_focus_records(records):
    """One extra view per selected train image, at most half the original train set.

    Held-out rows never affect class counts or ranking. The copy is only an
    in-memory augmentation marker; human annotations and input records are intact.
    """
    train = {}
    for record in records:
        if record.get('split') == 'train' and not record.get('feedback_only'):
            train.setdefault(record['image_id'], record)
    eligible = [record for record in train.values() if _eligible(record)]
    counts = Counter(box['class_id'] for record in eligible for box in record['boxes'])
    ranked = []
    for record in eligible:
        boxes = [box for box in record['boxes'] if _zoomable(box, record)]
        if not boxes:
            continue
        focus = min(boxes, key=lambda box: (counts[box['class_id']], _area(box, record), str(box['id'])))
        key = (counts[focus['class_id']], _area(focus, record), str(record['image_id']))
        ranked.append((key, record, focus['id']))
    ranked.sort(key=lambda item: item[0])
    limit = len(train)//2
    selected = []
    for _, record, focus_id in ranked[:limit]:
        augmented = copy.deepcopy(record)
        augmented.update(focus_crop=True, focus_box_id=focus_id)
        augmented.setdefault('negative_rois', [])
        selected.append(augmented)
    return selected, {'policy': POLICY, 'original_train_images': len(train),
        'eligible_focus_images': len(ranked), 'extra_images_per_epoch': len(selected),
        'extra_image_ids': [record['image_id'] for record in selected], 'extra_limit': limit,
        'max_extra_per_image': 1, 'validation_extra_images': 0, 'test_extra_images': 0,
        'ranking': 'rarer confirmed class, then smaller normalized box; stable image ID tie-break',
        'changes_source_or_human_annotations': False}


def _crop_start(low, high, length, total, rng):
    # Pick a valid context position, always retaining the complete focus box.
    first = max(0, math.ceil(high)-length)
    last = min(math.floor(low), total-length)
    if first > last:
        raise ValueError('Focus box does not fit crop')
    center = (low+high-length)/2
    jitter = max(0., length-(high-low))*.2
    return min(last, max(first, int(round(center+rng.uniform(-jitter, jitter)))))


def _intersect(values, window):
    left, top, right, bottom = window
    x1, y1, x2, y2 = values
    clipped = [max(left, x1), max(top, y1), min(right, x2), min(bottom, y2)]
    if clipped[0] >= clipped[2] or clipped[1] >= clipped[3]:
        return None
    return [clipped[0]-left, clipped[1]-top, clipped[2]-left, clipped[3]-top]


def crop_focus_sample(record, rgb, rng=None, context_scale=2.5,
                      min_crop_fraction=.25, max_crop_fraction=.85):
    """Crop RGB and every intersecting GT/normal ROI together, without new labels.

    A clipped positive is retained and marks partial coverage so the caller can
    suppress unknown-background negatives using its existing partial-label loss.
    Source identity/provenance stays on the original record; crop geometry is
    explicitly marked as an in-memory training view.
    """
    if not _eligible(record):
        raise ValueError('Focus crop requires eligible confirmed train positives')
    if (not math.isfinite(context_scale) or context_scale < 1
            or not 0 < min_crop_fraction <= max_crop_fraction <= 1):
        raise ValueError('Invalid focus context crop limits')
    width, height = record['width'], record['height']
    if not isinstance(rgb, np.ndarray) or rgb.shape != (height, width, 3):
        raise ValueError('RGB geometry differs from the source record')
    rng = rng if rng is not None else random
    focus_id = record.get('focus_box_id')
    if focus_id is not None:
        matches = [box for box in record['boxes'] if box['id'] == focus_id]
        if len(matches) != 1:
            raise ValueError('Focus box identity is missing or ambiguous')
        focus = matches[0]
    else:
        focus = min(record['boxes'], key=lambda box: (_area(box, record), str(box['id'])))
    x1, y1, x2, y2 = focus['xyxy']

    def length(low, high, total):
        context = min(total*max_crop_fraction, max(total*min_crop_fraction, (high-low)*context_scale))
        return min(total, max(math.ceil(context), math.ceil(high)-math.floor(low)))

    crop_width, crop_height = length(x1, x2, width), length(y1, y2, height)
    left = _crop_start(x1, x2, crop_width, width, rng)
    top = _crop_start(y1, y2, crop_height, height, rng)
    window = [left, top, left+crop_width, top+crop_height]
    result = copy.deepcopy(record)
    result['boxes'] = []
    truncated = []
    partial_problems = list(record.get('classification_problems_without_boxes') or [])
    for box in record['boxes']:
        coordinates = _intersect(box['xyxy'], window)
        if coordinates is None:
            continue
        clipped = copy.deepcopy(box)
        bx1, by1, bx2, by2 = coordinates
        clipped['xyxy'] = coordinates
        clipped['xywh'] = [bx1, by1, bx2-bx1, by2-by1]
        clipped['normalized_xywh'] = [bx1/crop_width, by1/crop_height,
                                      (bx2-bx1)/crop_width, (by2-by1)/crop_height]
        result['boxes'].append(clipped)
        original = box['xyxy']
        if (original[0] < window[0] or original[1] < window[1]
                or original[2] > window[2] or original[3] > window[3]):
            truncated.append(box['id'])
            if box['problem'] not in partial_problems:
                partial_problems.append(box['problem'])
    result['negative_rois'] = []
    for roi in record.get('negative_rois', []):
        if not _valid_xyxy(roi, width, height):
            raise ValueError('Invalid confirmed normal ROI geometry')
        clipped = _intersect(roi, window)
        if clipped is not None:
            result['negative_rois'].append(clipped)
    result.update(width=crop_width, height=crop_height, aspect=crop_width/crop_height,
        classification_problems_without_boxes=partial_problems,
        crop_partial=bool(record.get('crop_partial') or truncated),
        focus_crop_window=window, focus_crop=True, focus_box_id=focus['id'],
        focus_crop_metadata={'policy': POLICY, 'original_width': width, 'original_height': height,
            'truncated_box_ids': truncated, 'scope': 'in_memory_training_view'})
    return result, np.ascontiguousarray(rgb[top:top+crop_height, left:left+crop_width]).copy()
