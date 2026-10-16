"""Optional ROI verifier using frozen PAN features and explicit human region labels.

The caller must supply only eligible, confirmed train positive boxes and accepted
train normal ROIs. This module neither mines negatives nor changes issue types.
"""
from __future__ import annotations

import copy
import math
from pathlib import Path
import re

import cv2
import numpy as np
from PIL import Image, ImageOps
import torch
from torch.nn import functional as F


SCHEMA = 'frozen_pan_roi_linear_v1'
SUPPRESS_THRESHOLD = .05
CONTEXT_SCALE = 1.5
INPUT_SIZE = 224
L2 = .01
MAX_ITERATIONS = 80


def _rgb(image):
    if isinstance(image, (str, Path)):
        with Image.open(image) as opened:
            return np.asarray(ImageOps.exif_transpose(opened).convert('RGB')).copy()
    if isinstance(image, Image.Image):
        return np.asarray(ImageOps.exif_transpose(image).convert('RGB')).copy()
    if not isinstance(image, np.ndarray) or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError('ROI image must be an RGB array, PIL image or path')
    if image.dtype != np.uint8:
        raise ValueError('ROI RGB arrays must use uint8 pixels')
    return image


def _roi_input(rgb, box, context_scale=CONTEXT_SCALE):
    if not math.isfinite(context_scale) or context_scale < 1:
        raise ValueError('ROI context scale must be a finite number at least 1')
    values = box.get('xyxy') if isinstance(box, dict) else box
    if not isinstance(values, (list, tuple, np.ndarray)) or len(values) != 4:
        raise ValueError('ROI requires absolute xyxy coordinates')
    xyxy = np.asarray(values, dtype=np.float64)
    height, width = rgb.shape[:2]
    if (not np.isfinite(xyxy).all() or not 0 <= xyxy[0] < xyxy[2] <= width+1e-6
            or not 0 <= xyxy[1] < xyxy[3] <= height+1e-6):
        raise ValueError('ROI coordinates outside source image')
    cx, cy = (xyxy[:2]+xyxy[2:])/2
    half_width, half_height = (xyxy[2:]-xyxy[:2])*context_scale/2
    left, top = max(0, math.floor(cx-half_width)), max(0, math.floor(cy-half_height))
    right, bottom = min(width, math.ceil(cx+half_width)), min(height, math.ceil(cy+half_height))
    crop = np.ascontiguousarray(rgb[top:bottom, left:right, ::-1])
    ratio = min(INPUT_SIZE/crop.shape[0], INPUT_SIZE/crop.shape[1])
    resized = cv2.resize(crop, (max(1, int(crop.shape[1]*ratio)),
                                max(1, int(crop.shape[0]*ratio))), interpolation=cv2.INTER_LINEAR)
    padded = np.full((INPUT_SIZE, INPUT_SIZE, 3), 114, dtype=np.float32)
    padded[:resized.shape[0], :resized.shape[1]] = resized
    return np.ascontiguousarray(padded.transpose(2, 0, 1))


@torch.inference_mode()
def extract_roi_features(detector, image, boxes, *, context_scale=CONTEXT_SCALE):
    """Return CPU N x D features; use all three PAN outputs, avg then max per level.

    Coordinates are absolute on the EXIF-transposed source. Backbone parameters,
    gradients, BN buffers and prior training flags are preserved.
    """
    boxes = list(boxes)
    if not boxes:
        return np.empty((0, 0), dtype=np.float32)
    rgb = _rgb(image)
    backbone = detector.model.backbone
    modes = [(module, module.training) for module in backbone.modules()]
    parameter = next(backbone.parameters(), None)
    device = parameter.device if parameter is not None else torch.device(detector.device)
    dtype = parameter.dtype if parameter is not None else torch.float32
    output = []
    try:
        backbone.eval()
        for offset in range(0, len(boxes), 16):
            batch = np.stack([_roi_input(rgb, box, context_scale) for box in boxes[offset:offset+16]])
            features = backbone(torch.from_numpy(batch).to(device=device, dtype=dtype))
            if not isinstance(features, (list, tuple)) or len(features) != 3:
                raise ValueError('Verifier requires the three YOLOX PAN feature levels')
            if any(value.ndim != 4 or value.shape[0] != len(batch) for value in features):
                raise ValueError('Unexpected PAN feature shape')
            pooled = [part for value in features
                      for part in (value.mean(dim=(2, 3)), value.amax(dim=(2, 3)))]
            output.append(torch.cat(pooled, dim=1).float().cpu().numpy().copy())
    finally:
        for module, mode in modes:
            module.training = mode
    result = np.concatenate(output, axis=0)
    if not np.isfinite(result).all():
        raise ValueError('Non-finite ROI backbone features')
    return result


def _matrix(features):
    matrix = np.asarray(features, dtype=np.float64)
    if matrix.ndim != 2 or not matrix.shape[1] or not np.isfinite(matrix).all():
        raise ValueError('ROI features must be a finite N x D matrix')
    return matrix


def _source_hashes(value):
    if value is None:
        return []
    hashes = list(value.values()) if isinstance(value, dict) else value
    if not isinstance(hashes, (list, tuple)) or any(
            not isinstance(digest, str) or not re.fullmatch('[0-9a-fA-F]{64}', digest) for digest in hashes):
        raise ValueError('Source hashes must contain SHA-256 hex digests')
    return copy.deepcopy(value)


def fit_verifier(pos_features, negative_features, *, source_hashes=None, context_scale=CONTEXT_SCALE):
    """CPU-only balanced logistic regression with fixed L2 and no threshold search.

    Caller-provided source hashes are copied into serializable metadata. Positive
    and negative arrays must come from explicit confirmed train regions only.
    """
    if not math.isfinite(context_scale) or context_scale < 1:
        raise ValueError('ROI context scale must be a finite number at least 1')
    positive, negative = _matrix(pos_features), _matrix(negative_features)
    if not len(positive) or not len(negative) or positive.shape[1] != negative.shape[1]:
        raise ValueError('Verifier requires both classes with matching feature dimensions')
    hashes = _source_hashes(source_hashes)
    matrix = np.concatenate((positive, negative))
    mean, scale = matrix.mean(axis=0), matrix.std(axis=0)
    scale = np.maximum(scale, 1e-6)
    normalized = (matrix-mean)/scale
    inputs = torch.from_numpy(normalized)
    truth = torch.cat((torch.ones(len(positive), dtype=torch.float64),
                       torch.zeros(len(negative), dtype=torch.float64)))
    balanced = torch.cat((torch.full((len(positive),), .5/len(positive), dtype=torch.float64),
                          torch.full((len(negative),), .5/len(negative), dtype=torch.float64)))
    coefficients = torch.zeros(matrix.shape[1], dtype=torch.float64, requires_grad=True)
    intercept = torch.zeros((), dtype=torch.float64, requires_grad=True)
    optimizer = torch.optim.LBFGS([coefficients, intercept], lr=1., max_iter=MAX_ITERATIONS,
        history_size=10, tolerance_grad=1e-7, tolerance_change=1e-10, line_search_fn='strong_wolfe')

    def objective():
        logits = inputs@coefficients+intercept
        return (F.binary_cross_entropy_with_logits(logits, truth, reduction='none')*balanced).sum() + .5*L2*coefficients.square().sum()

    def closure():
        optimizer.zero_grad(set_to_none=True)
        loss = objective()
        loss.backward()
        return loss

    prior_threads = torch.get_num_threads()
    try:
        torch.set_num_threads(min(prior_threads, 2))
        optimizer.step(closure)
        final_loss = float(objective().detach())
    finally:
        torch.set_num_threads(prior_threads)
    if not math.isfinite(final_loss) or not torch.isfinite(coefficients).all() or not torch.isfinite(intercept):
        raise ValueError('Non-finite ROI verifier fit')
    return {'schema': SCHEMA, 'feature_dimension': matrix.shape[1],
        'mean': mean.tolist(), 'scale': scale.tolist(),
        'coefficients': coefficients.detach().tolist(), 'intercept': float(intercept.detach()),
        'source_hashes': hashes, 'positive_regions': len(positive), 'negative_regions': len(negative),
        'context_scale': context_scale, 'input_size': INPUT_SIZE, 'input_color': 'BGR',
        'padding_value': 114, 'padding_alignment': 'top_left',
        'feature_pooling': 'PAN levels in order; global average then global maximum per level',
        'positive_class': 'confirmed_problem_region', 'negative_class': 'explicit_confirmed_normal_region',
        'loss': 'class_balanced_BCE_plus_L2', 'l2': L2, 'max_iterations': MAX_ITERATIONS,
        'iterations': int(optimizer.state[coefficients].get('n_iter', 0)), 'fit_loss': final_loss,
        'suppress_if_problem_probability_at_most': SUPPRESS_THRESHOLD,
        'threshold_searched': False, 'requires_validation_no_true_positive_loss': True,
        'issue_type_changed': False}


def keep_probabilities(features, meta):
    """Probability of keeping each candidate; <= .05 is the fixed suppression rule."""
    if meta.get('schema') != SCHEMA:
        raise ValueError('Unsupported ROI verifier metadata')
    if len(features) == 0:
        return []
    matrix = _matrix(features)
    dimension = meta['feature_dimension']
    arrays = [np.asarray(meta[key], dtype=np.float64) for key in ('mean', 'scale', 'coefficients')]
    if (matrix.shape[1] != dimension or any(value.shape != (dimension,) for value in arrays)
            or any(not np.isfinite(value).all() for value in arrays)
            or (arrays[1] <= 0).any() or not math.isfinite(meta['intercept'])):
        raise ValueError('Invalid ROI verifier dimensions or coefficients')
    logits = ((matrix-arrays[0])/arrays[1])@arrays[2]+meta['intercept']
    return (1/(1+np.exp(-np.clip(logits, -60, 60)))).tolist()
