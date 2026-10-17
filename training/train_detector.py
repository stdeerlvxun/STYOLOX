"""Small, reproducible YOLOX-s detector experiment on confirmed human boxes.

Original images are read only. No application/database writes or online deployment.
Threshold and checkpoint selection use validation groups only, never test groups.
"""
from __future__ import annotations

import argparse
from collections import Counter
import copy
import hashlib
import io
import json
import math
from pathlib import Path
import random
import sys
import time

import numpy as np
from PIL import Image, ImageOps
import torch

from roi_feedback import anchor_centers, region_objectness_loss, transform_rois
from hard_positive import hard_positive_repeats

YOLOX_SOURCE = Path(r'D:\视觉检测工作区\runtime\YOLOX-6ddff4824372')
COCO_WEIGHTS = Path(r'D:\视觉检测工作区\models\YOLOX-s\yolox_s.pth')
THRESHOLDS = [.01, .025, .05, .1, .2, .3]


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def initialize_yolox(source=YOLOX_SOURCE):
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))


def make_model(num_classes, config=None):
    config = config or {}
    initialize_yolox(config.get('source', YOLOX_SOURCE))
    from yolox.exp import Exp
    exp = Exp()
    exp.depth = config.get('depth', .33)
    exp.width = config.get('width', .50)
    exp.num_classes = num_classes
    return exp.get_model()


def xyxy_iou(left, right):
    x1, y1 = max(left[0], right[0]), max(left[1], right[1])
    x2, y2 = min(left[2], right[2]), min(left[3], right[3])
    intersection = max(0., x2-x1) * max(0., y2-y1)
    a = max(0., left[2]-left[0]) * max(0., left[3]-left[1])
    b = max(0., right[2]-right[0]) * max(0., right[3]-right[1])
    return intersection / max(a+b-intersection, 1e-12)


def match_boxes(truth, predictions, class_aware=True, iou_threshold=.5):
    """Score-ordered one-to-one matching. A duplicate is an FP, not another TP."""
    remaining = set(range(len(truth)))
    matches = []
    false_positives = []
    for prediction_index in sorted(range(len(predictions)),
                                   key=lambda i: -predictions[i]['score']):
        prediction = predictions[prediction_index]
        candidates = [i for i in remaining if not class_aware or
                      truth[i]['class_id'] == prediction['class_id']]
        best = max(candidates, key=lambda i: xyxy_iou(truth[i]['xyxy'], prediction['xyxy']),
                   default=None)
        if best is not None and xyxy_iou(truth[best]['xyxy'], prediction['xyxy']) >= iou_threshold:
            remaining.remove(best)
            matches.append({'truth': best, 'prediction': prediction_index,
                            'iou': xyxy_iou(truth[best]['xyxy'], prediction['xyxy'])})
        else:
            false_positives.append(prediction_index)
    return {'tp': len(matches), 'fn': len(remaining), 'fp': len(false_positives),
            'matches': matches, 'missed_indices': sorted(remaining),
            'false_positive_indices': false_positives}


def rates(tp, fn, fp):
    precision = tp / (tp+fp) if tp+fp else 0.
    recall = tp / (tp+fn) if tp+fn else 0.
    return {'tp': tp, 'fn': fn, 'fp': fp, 'support': tp+fn,
            'precision': precision, 'recall': recall,
            'f2': 5*tp / (5*tp+4*fn+fp) if 5*tp+4*fn+fp else 0.}


def partial_objectness_loss(loss, targets, partial_images):
    """Ignore unmatched anchors on incompletely boxed images, keep positives."""
    if not partial_images or targets.ndim != 2 or targets.shape[-1] != 1:
        return loss
    batch = len(partial_images)
    if targets.shape[0] % batch:
        raise ValueError('Unexpected flattened objectness anchor count')
    partial = torch.as_tensor(partial_images, dtype=torch.bool, device=targets.device).view(batch, 1, 1)
    keep = (~partial) | (targets.reshape(batch, -1, 1) > 0)
    return loss * keep.reshape_as(loss)


def evaluate_predictions(records, predictions, categories, threshold):
    totals = Counter()
    localization = Counter()
    per_class = {int(c['id']): Counter() for c in categories}
    detailed = []
    for record in records:
        selected = [p for p in predictions[str(record['image_id'])] if p['score'] >= threshold]
        result = match_boxes(record['boxes'], selected)
        region_result = match_boxes(record['boxes'], selected, class_aware=False)
        for key in ('tp', 'fn', 'fp'):
            totals[key] += result[key]
            localization[key] += region_result[key]
        for match in result['matches']:
            per_class[record['boxes'][match['truth']]['class_id']]['tp'] += 1
        for index in result['missed_indices']:
            per_class[record['boxes'][index]['class_id']]['fn'] += 1
        for index in result['false_positive_indices']:
            if selected[index]['class_id'] in per_class:
                per_class[selected[index]['class_id']]['fp'] += 1
            else:
                totals['unknown_class_fp'] += 1
        detailed.append({'image_id': record['image_id'], 'relative_path': record['relative_path'],
                         'group': record['group'], 'truth': record['boxes'],
                         'predictions': selected, 'class_and_location': result,
                         'location_only': region_result})
    image_pass = {}
    for objective in ('class_and_location', 'location_only'):
        passed = sum(row[objective]['fn'] == 0 and row[objective]['fp'] == 0 for row in detailed)
        image_pass[objective] = {'passed': passed, 'total': len(detailed),
                                 'rate': passed / len(detailed) if detailed else None}
    return {'class_and_location': rates(totals['tp'], totals['fn'], totals['fp']),
            'location_only': rates(localization['tp'], localization['fn'], localization['fp']),
            'whole_image_pass': image_pass,
            'per_class': {c['name']: rates(per_class[c['id']]['tp'], per_class[c['id']]['fn'],
                                           per_class[c['id']]['fp']) for c in categories},
            'unknown_class_false_positives': totals['unknown_class_fp'],
            'images': len(records), 'threshold': threshold}, detailed


def select_threshold(records, predictions, categories):
    candidates = []
    for threshold in THRESHOLDS:
        report, _ = evaluate_predictions(records, predictions, categories, threshold)
        candidates.append({'threshold': threshold, **report['class_and_location']})
    selected = max(candidates, key=lambda x: (x['f2'], x['recall'], -x['fp'], x['threshold']))
    return selected, candidates


def correction_selection_key(report, baseline, epoch):
    """Issue learning cannot win checkpoint selection by forgetting existing locations."""
    location, original = report['location_only'], baseline['location_only']
    typed, old_typed = report['class_and_location'], baseline['class_and_location']
    safe = (location['recall'] >= original['recall'] and location['f2'] >= original['f2']
            and location['fp'] <= original['fp'] and typed['recall'] >= old_typed['recall']
            and typed['f2'] >= old_typed['f2'] and typed['fp'] <= old_typed['fp'])
    return (safe, location['recall'], typed['f2'], -location['fp'], typed['precision'], -epoch)


def validate_training_modes(*, initialize_from, adapt_backbone, fixed_threshold,
                            location_only=False, correction_focus=False, localization_focus=False):
    if correction_focus and (location_only or not initialize_from or fixed_threshold is None):
        raise ValueError('Correction focus requires a retained typed model and a fixed threshold')
    if localization_focus and (location_only or correction_focus or not initialize_from
                               or not adapt_backbone or fixed_threshold is None):
        raise ValueError('Localization focus requires a retained typed model, backbone adaptation, '
                         'and a fixed threshold; other focus/location-only modes are incompatible')


def localization_parameter_group(name):
    """Late feature adaptation and full head geometry learn together from epoch one."""
    if not adaptable_parameter(name):
        return 'frozen', 0.
    if name.startswith(('head.cls_convs.', 'head.cls_preds.')):
        return 'classification', .0001
    if name.startswith('head.'):
        return 'geometry', .00005
    return 'late_features', .00001


def localization_optimizer_groups(model):
    groups = {}
    for name, parameter in model.named_parameters():
        group, learning_rate = localization_parameter_group(name)
        if group == 'frozen' or not parameter.requires_grad:
            continue
        groups.setdefault(group, {'name': group, 'lr': learning_rate, 'params': []})['params'].append(parameter)
    return list(groups.values())


def training_phase(epoch, epochs, elapsed, seconds, correction_focus=False, localization_focus=False):
    if localization_focus or not correction_focus:
        return 'joint'
    return 'joint' if epoch > max(1, epochs*3//5) or elapsed >= seconds*.65 else 'classification'


def localization_selection_key(report, baseline, epoch):
    """Keep typed and location safety, then prioritize finding all problem locations."""
    safe = correction_selection_key(report, baseline, epoch)[0]
    location, typed = report['location_only'], report['class_and_location']
    return (safe, location['recall'], location['f2'], -location['fp'], typed['f2'], -epoch)


def read_rgb(image):
    if isinstance(image, (str, Path)):
        with Image.open(image) as source:
            return np.asarray(ImageOps.exif_transpose(source).convert('RGB')).copy()
    if isinstance(image, Image.Image):
        return np.asarray(ImageOps.exif_transpose(image).convert('RGB')).copy()
    array = np.asarray(image)
    if array.ndim != 3 or array.shape[2] != 3:
        raise ValueError('Expected RGB HWC image, PIL image, or source path')
    return array.astype(np.uint8, copy=True)


class Detector:
    def __init__(self, model, categories, input_size, device='cuda', threshold=.1,
                 nms=.5, max_predictions=10):
        self.device = torch.device(device)
        self.model = model.to(self.device).eval()
        self.categories = {int(c['id']): c['name'] for c in categories}
        self.input_size = input_size
        self.threshold = threshold
        self.nms = nms
        self.max_predictions = max_predictions
        self.class_agnostic = len(categories) == 1 and categories[0]['name'] == '问题位置'
        self.roi_verifier = None

    @torch.inference_mode()
    def predict_image(self, image, threshold=None):
        initialize_yolox()
        from yolox.data.data_augment import ValTransform
        from yolox.utils import postprocess
        threshold = self.threshold if threshold is None else threshold
        rgb = read_rgb(image)
        height, width = rgb.shape[:2]
        bgr = np.ascontiguousarray(rgb[..., ::-1])
        inputs, _ = ValTransform()(bgr, None, (self.input_size, self.input_size))
        tensor = torch.from_numpy(inputs).unsqueeze(0).to(self.device)
        outputs = self.model(tensor)
        processed = postprocess(outputs, len(self.categories), threshold, self.nms)[0]
        if processed is None:
            return []
        ratio = min(self.input_size/width, self.input_size/height)
        values = processed.detach().cpu().numpy()
        values = sorted(values, key=lambda value: -float(value[4]*value[5]))[:self.max_predictions]
        predictions = []
        for value in values:
            x1, y1, x2, y2 = value[:4] / ratio
            x1, x2 = np.clip([x1, x2], 0, width)
            y1, y2 = np.clip([y1, y2], 0, height)
            if x2 <= x1 or y2 <= y1:
                continue
            class_id = int(value[6])
            xywh = [float(x1/width), float(y1/height), float((x2-x1)/width), float((y2-y1)/height)]
            predictions.append({'class_id': class_id, 'problem': self.categories[class_id],
                                'class_agnostic': self.class_agnostic,
                                'score': float(value[4]*value[5]),
                                'xyxy': [float(x1), float(y1), float(x2), float(y2)],
                                'xywh': xywh, 'normalized_xywh': xywh})
        if self.roi_verifier and predictions:
            from roi_verifier import extract_roi_features, keep_probabilities
            vectors = extract_roi_features(self, rgb, predictions,
                context_scale=self.roi_verifier['context_scale'])
            probabilities = keep_probabilities(vectors, self.roi_verifier)
            predictions = [dict(prediction, roi_keep_probability=probability)
                for prediction, probability in zip(predictions, probabilities) if probability > .05]
        return predictions


def load_detector(run_dir, device='cuda'):
    run_dir = Path(run_dir)
    checkpoint = torch.load(run_dir/'detector.pt', map_location='cpu', weights_only=False)
    config = checkpoint['config']
    model = make_model(len(checkpoint['categories']), config)
    model.load_state_dict(checkpoint['model'])
    detector = Detector(model, checkpoint['categories'], config['input_size'], device,
                        checkpoint['threshold'], config['nms'], config['max_predictions'])
    detector.roi_verifier = config.get('roi_verifier')
    return detector


def predict_image(detector, image, threshold=None):
    return detector.predict_image(image, threshold)


def predict_records(detector, records, cache):
    return {str(r['image_id']): detector.predict_image(cache[r['image_id']], min(THRESHOLDS))
            for r in records}


def adaptable_parameter(name):
    return not name.startswith('backbone.backbone.') or name.startswith(
        ('backbone.backbone.dark4.', 'backbone.backbone.dark5.'))


def transferred_state(target_state, checkpoint, categories):
    """Transfer architecture tensors and issue output channels by name, never class index."""
    source = checkpoint['model']
    result = {key: value for key, value in source.items() if key in target_state and
              value.shape == target_state[key].shape and '.cls_preds.' not in key}
    previous = {c['name']: int(c['id']) for c in checkpoint['categories']}
    for key, target in target_state.items():
        if '.cls_preds.' not in key or key not in source:
            continue
        value = target.clone()
        for category in categories:
            if category['name'] in previous:
                value[int(category['id'])] = source[key][previous[category['name']]]
            elif key.endswith('.weight'):
                # Old learned class features can amplify random new output rows.
                # A truly new class starts at a constant 1% prior and can learn
                # from its own positives; previously learned classes are copied.
                value[int(category['id'])].zero_()
            elif key.endswith('.bias'):
                value[int(category['id'])].fill_(math.log(.01/.99))
        result[key] = value
    return result


def location_training_records(dataset):
    """Make an in-memory one-class view without rewriting human issue labels."""
    records = copy.deepcopy(dataset['records'])
    feedback = copy.deepcopy(dataset.get('negative_feedback_records', []))
    for record in records + feedback:
        for box in record['boxes']:
            box['original_class_id'] = box['class_id']
            box['original_problem'] = box['problem']
            box['class_id'] = 0
            box['problem'] = '问题位置'
    return [{'id': 0, 'name': '问题位置'}], records, feedback


def feedback_sample(record, rgb, input_size, max_labels):
    """Same resize/flip for the source, positive boxes and human normal regions."""
    from yolox.data.data_augment import preproc, augment_hsv
    bgr = np.ascontiguousarray(rgb[..., ::-1])
    height, width = bgr.shape[:2]
    flipped = random.random() < .5
    if flipped:
        bgr = np.ascontiguousarray(bgr[:, ::-1])
    if random.random() < .5:
        augment_hsv(bgr)
    image, ratio = preproc(bgr, (input_size, input_size))
    target = np.zeros((max_labels, 5), dtype=np.float32)
    for i, box in enumerate(record['boxes']):
        x1, y1, x2, y2 = transform_rois([box['xyxy']], width, ratio, flipped)[0]
        target[i] = [box['class_id'], (x1+x2)/2, (y1+y2)/2, x2-x1, y2-y1]
    return image, target, transform_rois(record.get('negative_rois', []), width, ratio, flipped)


def train(dataset_path, run_dir, epochs=50, seconds=450, batch_size=2, input_size=512,
          adapt_backbone=False, defer_test=False, initialize_from=None,
          location_only=False, fixed_threshold=None, correction_focus=False, localization_focus=False):
    validate_training_modes(initialize_from=initialize_from, adapt_backbone=adapt_backbone,
        fixed_threshold=fixed_threshold, location_only=location_only,
        correction_focus=correction_focus, localization_focus=localization_focus)
    started = time.monotonic()
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    config = {'architecture': 'YOLOX-s', 'depth': .33, 'width': .5, 'input_size': input_size,
              'batch_size': batch_size, 'max_epochs': epochs, 'training_seconds_budget': seconds,
              'dtype': 'float32', 'seed': 17610, 'optimizer': 'AdamW',
              'learning_rate': .0001 if adapt_backbone else .0002,
              'weight_decay': .0001, 'backbone_frozen': not adapt_backbone,
              'train_head_only': not adapt_backbone, 'adapt_backbone': adapt_backbone,
              'frozen_early_layers': ['stem', 'dark2', 'dark3'] if adapt_backbone else ['all_backbone'],
              'batch_norm_running_statistics_frozen': adapt_backbone,
              'augmentation': 'official TrainTransform horizontal flip .5 HSV .5',
              'mosaic': False, 'mixup': False, 'nms': .5, 'max_predictions': 10,
              'threshold_grid': THRESHOLDS, 'selection': 'validation class-aware F2 at IoU .5',
              'validation_interval_epochs': 5, 'source': str(YOLOX_SOURCE),
              'source_commit': '6ddff4824372906469a7fae2dc3206c7aa4bbaee',
              'pretrained': str(COCO_WEIGHTS), 'pretrained_sha256': sha256(COCO_WEIGHTS),
              'dataset_sha256': sha256(dataset_path), 'test_used_for_selection': False,
              'evaluation_iou': .5, 'source_images_copied': False,
              'test_deferred_until_validation_report': defer_test}
    config['location_only'] = location_only
    config['class_agnostic'] = location_only
    config['fixed_threshold'] = fixed_threshold
    config['correction_focus'] = bool(correction_focus)
    config['localization_focus'] = bool(localization_focus)
    if correction_focus:
        config.update(backbone_frozen=True, train_head_only=True,
                      frozen_early_layers=['all_backbone_and_neck', 'head.stems'],
                      batch_norm_running_statistics_frozen=True, learning_rate=.00003)
    if fixed_threshold is not None:
        config['selection'] = 'Fixed threshold; validation recall with FP <= 2*GT, then F2; over-budget fallback F2'
    if localization_focus:
        config['selection'] = 'Initialization non-regression for typed and location metrics; prioritize location recall, F2, then fewer FP'
        config['learning_rate'] = None
        config['optimizer_learning_rates'] = {'classification': .0001, 'geometry': .00005, 'late_features': .00001}
    config['partial_annotations'] = 'For images with classified problems lacking boxes, ignore unmatched objectness anchors; retain positives and all class/regression losses'
    write_json(run_dir/'planned-config.json', config)
    dataset = json.loads(Path(dataset_path).read_text(encoding='utf-8-sig'))
    categories = dataset['categories']
    records = dataset['records']
    feedback_records = dataset.get('negative_feedback_records', [])
    if location_only:
        categories, records, feedback_records = location_training_records(dataset)
        config['issue_type_prediction'] = False
        config['original_issue_labels_preserved'] = True
    config['human_normal_roi_feedback'] = {'images': len(feedback_records),
        'regions': sum(len(r['negative_rois']) for r in feedback_records), 'weight': .25,
        'repeats_per_epoch': 1, 'normalization': 'bounded total auxiliary mean BCE after YOLOX foreground normalization',
        'scope': 'normal ROI negative objectness only; all other unboxed anchors ignored',
        'whole_image_normal': False, 'positive_truth_changed': False}
    if correction_focus:
        config['human_normal_roi_feedback'].update(weight=.5, repeats_per_epoch=2,
            scope='confirmed local false-positive ROIs only; joint phase only, unknown anchors ignored')
    splits = {split: [r for r in records if r['split'] == split] for split in ('train', 'val', 'test')}
    if not splits['val'] and any(r['split'] == 'validation' for r in records):
        splits['val'] = [r for r in records if r['split'] == 'validation']
    groups = {split: {r['group'] for r in rows} for split, rows in splits.items()}
    if any(not value for value in groups.values()):
        raise ValueError('Independent train/val/test groups required')
    if groups['train'] & groups['val'] or groups['train'] & groups['test'] or groups['val'] & groups['test']:
        raise ValueError('Group leakage')
    for record in records:
        if record.get('has_draft') or record.get('image_state') not in (None, '', 'defect') or not record.get('defects'):
            raise ValueError('Training exclusion policy violation')
        if not record['boxes']:
            raise ValueError('Only confirmed human boxed images may enter this experiment')
    for record in feedback_records:
        if not record.get('feedback_only') or record.get('split') != 'train' or not record.get('negative_rois'):
            raise ValueError('Normal ROI feedback must be explicitly sourced train-only auxiliary supervision')
        if record.get('has_draft') or record.get('image_state') not in (None, '', 'defect') or not record.get('defects'):
            raise ValueError('Normal ROI feedback image violates unchanged classification exclusions')
    hard_positive_records, config['human_correction_extra_practice'] = hard_positive_repeats(splits['train'])
    focus_records = []
    if correction_focus or localization_focus:
        from correction_focus import select_focus_records
        focus_records, focus_report = select_focus_records(splits['train'])
        hard_positive_records = []  # Use real local views instead of repeating the same full image.
    if correction_focus:
        config['correction_focus_policy'] = {**focus_report,
            'classification_epochs': max(1, epochs*3//5), 'classification_lr': .00003,
            'geometry_lr': .00001, 'geometry_scope': 'head only; retained backbone and neck frozen',
            'minimum_joint_epochs': min(5, max(1, epochs//4)),
            'selection': 'same fixed validation threshold; class and location must not regress from initialization'}
    if localization_focus:
        config['localization_focus_policy'] = {**focus_report,
            'phase': 'joint from epoch one', 'l1_enabled_from_epoch': 1,
            'geometry_scope': 'head, neck and dark4/dark5; early features and all BN statistics frozen',
            'selection': config['selection'], 'human_normal_roi_weight': .25,
            'extra_full_image_repeats': 0}
    torch.set_num_threads(2)
    random.seed(config['seed'])
    np.random.seed(config['seed'])
    torch.manual_seed(config['seed'])
    torch.cuda.manual_seed_all(config['seed'])
    torch.backends.cudnn.benchmark = False
    initialize_yolox()
    from yolox.data.data_augment import TrainTransform
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device != 'cuda':
        raise RuntimeError('This bounded training experiment requires the configured CUDA GPU')
    cache = {}
    for record in records + feedback_records:
        raw = Path(record['full_path']).read_bytes()
        if hashlib.sha256(raw).hexdigest() != record['sha256']:
            raise ValueError(f"Source hash changed: {record['image_id']}")
        with Image.open(io.BytesIO(raw)) as source:
            rgb = np.asarray(ImageOps.exif_transpose(source).convert('RGB')).copy()
        if rgb.shape[:2] != (record['height'], record['width']):
            raise ValueError('EXIF-transposed source dimensions differ from annotations')
        cache[record['image_id']] = rgb
    model = make_model(len(categories), config)
    if initialize_from:
        checkpoint_path = Path(initialize_from)/'detector.pt'
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        if localization_focus and (checkpoint['config'].get('class_agnostic')
                or checkpoint['config'].get('location_only')
                or any(c['name'] == '问题位置' for c in checkpoint['categories'])):
            raise ValueError('Localization focus requires a typed initialization checkpoint')
        matching = transferred_state(model.state_dict(), checkpoint, categories)
        config['initialize_from'] = str(checkpoint_path)
        config['initialize_from_sha256'] = sha256(checkpoint_path)
        config['category_transfer'] = 'class output channels copied by issue name'
        config['new_class_initialization'] = 'only unseen issue rows: zero weight and 1 percent sigmoid prior'
        if correction_focus and checkpoint['config'].get('roi_verifier'):
            config['roi_verifier'] = checkpoint['config']['roi_verifier']
    else:
        state = torch.load(COCO_WEIGHTS, map_location='cpu', weights_only=False)
        state = state.get('model', state)
        matching = {key: value for key, value in state.items() if key in model.state_dict() and
                    value.shape == model.state_dict()[key].shape and '.cls_preds.' not in key}
    missing, unexpected = model.load_state_dict(matching, strict=False)
    config['loaded_parameters'] = len(matching)
    config['new_parameters'] = list(missing)
    write_json(run_dir/'initialization.json', {'loaded': len(matching), 'missing': list(missing),
                                            'unexpected': list(unexpected)})
    baseline_state = copy.deepcopy(model.state_dict())
    if correction_focus:
        from correction_focus import parameter_phase
        for name, parameter in model.named_parameters():
            parameter.requires_grad_(parameter_phase(name, 'joint') != 'frozen')
    elif adapt_backbone:
        for name, parameter in model.named_parameters():
            parameter.requires_grad_(adaptable_parameter(name))
    else:
        for parameter in model.backbone.parameters():
            parameter.requires_grad_(False)
    write_json(run_dir/'trainable-parameters.json',
               {'names': [name for name, p in model.named_parameters() if p.requires_grad],
                'count': sum(p.numel() for p in model.parameters() if p.requires_grad)})
    model = model.to(device)
    if localization_focus:
        optimizer = torch.optim.AdamW(localization_optimizer_groups(model), weight_decay=config['weight_decay'])
    elif correction_focus:
        optimizer = torch.optim.AdamW([
            {'params': [p for n,p in model.named_parameters() if parameter_phase(n,'joint') == 'classification'], 'lr': .00003},
            {'params': [p for n,p in model.named_parameters() if parameter_phase(n,'joint') == 'geometry'], 'lr': .00001}],
            weight_decay=config['weight_decay'])
    else:
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                     lr=config['learning_rate'], weight_decay=config['weight_decay'])
    if localization_focus:
        config['actual_optimizer_groups'] = [{'name': group['name'], 'lr': group['lr'],
            'parameter_tensors': len(group['params']),
            'parameter_count': sum(p.numel() for p in group['params'])} for group in optimizer.param_groups]
    partial_batch = []
    roi_batch = []
    centers = anchor_centers(input_size, device)
    def mask_partial_negatives(module, inputs, output):
        return region_objectness_loss(output, inputs[1], partial_batch, roi_batch, centers,
                                       weight=.5 if correction_focus else .25, bounded=True)
    objectness_hook = model.head.bcewithlog_loss.register_forward_hook(mask_partial_negatives)
    transform = TrainTransform(max_labels=max(20, max(len(r['boxes']) for r in records)),
                               flip_prob=.5, hsv_prob=.5)
    write_json(run_dir/'planned-config.json', config)
    detector = Detector(model, categories, input_size, device, nms=config['nms'], max_predictions=10)
    detector.roi_verifier = config.get('roi_verifier')
    best_key = None
    best_state = None
    best_selection = None
    history = []
    initial_report = None
    if correction_focus or localization_focus:
        initial_predictions = predict_records(detector, splits['val'], cache)
        initial_report, _ = evaluate_predictions(splits['val'], initial_predictions, categories, fixed_threshold)
        write_json(run_dir/'initial-validation.json', initial_report)
        best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
        selection_key = localization_selection_key if localization_focus else correction_selection_key
        best_key = selection_key(initial_report, initial_report, 0)
        best_selection = {'epoch': 0, 'threshold': fixed_threshold,
                          'validation': {'threshold': fixed_threshold, **initial_report['class_and_location']},
                          'threshold_candidates': [], 'rule': config['selection'] if localization_focus else config['correction_focus_policy']['selection'],
                          'test_used': False}
        torch.save({'model': best_state, 'categories': categories, 'config': config,
                    'threshold': fixed_threshold, 'epoch': 0}, run_dir/'detector.pt')
        write_json(run_dir/'selection.json', best_selection)
    train_started = time.monotonic()
    joint_epochs = 0
    torch.cuda.reset_peak_memory_stats()
    for epoch in range(1, epochs+1):
        epoch_started = time.monotonic()
        model.train()
        phase = training_phase(epoch, epochs, time.monotonic()-train_started, seconds,
                               correction_focus, localization_focus)
        joint_epochs += phase == 'joint'
        if correction_focus:
            for name, parameter in model.named_parameters():
                parameter.requires_grad_(parameter_phase(name, phase) != 'frozen')
        if adapt_backbone or correction_focus:
            # A batch of one or two is too small to re-estimate pretrained BN statistics.
            for module in model.modules():
                if isinstance(module, torch.nn.BatchNorm2d):
                    module.eval()
        else:
            model.backbone.eval()
        model.head.use_l1 = True if localization_focus else phase == 'joint' if correction_focus else epoch >= 20
        roi_order = (feedback_records*2 if phase == 'joint' else []) if correction_focus else feedback_records
        order = list(splits['train']) + hard_positive_records + roi_order + focus_records
        random.shuffle(order)
        losses = []
        for offset in range(0, len(order), batch_size):
            inputs, targets = [], []
            partial_batch[:] = []
            roi_batch[:] = []
            for record in order[offset:offset+batch_size]:
                rgb = cache[record['image_id']]
                if record.get('focus_crop'):
                    from correction_focus import crop_focus_sample
                    record, rgb = crop_focus_sample(record, rgb, rng=random)
                partial_batch.append(bool(record.get('classification_problems_without_boxes')) or bool(record.get('crop_partial')))
                if record.get('negative_rois') or correction_focus or localization_focus:
                    image, target, rois = feedback_sample(record, rgb, input_size, transform.max_labels)
                else:
                    bgr = np.ascontiguousarray(rgb[..., ::-1])
                    labels = np.array([b['xyxy'] + [b['class_id']] for b in record['boxes']], dtype=np.float32)
                    image, target = transform(bgr, labels, (input_size, input_size))
                    rois = []
                roi_batch.append(rois)
                inputs.append(image)
                targets.append(target)
            x = torch.from_numpy(np.stack(inputs)).to(device)
            y = torch.from_numpy(np.stack(targets)).to(device)
            optimizer.zero_grad(set_to_none=True)
            result = model(x, y)
            loss = result['total_loss']
            if not torch.isfinite(loss):
                raise RuntimeError(f'Non-finite loss at epoch {epoch}')
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], max_norm=10.)
            optimizer.step()
            losses.append(float(loss.detach()))
        epoch_log = {'epoch': epoch, 'loss': sum(losses)/len(losses),
                     'seconds': time.monotonic()-epoch_started,
                     'hard_positive_extra_images': len(hard_positive_records)}
        if correction_focus:
            epoch_log.update(phase=phase, focus_crop_images=len(focus_records),
                             negative_roi_extra_images=len(roi_order), l1_enabled=model.head.use_l1)
        if localization_focus:
            epoch_log.update(phase=phase, focus_crop_images=len(focus_records),
                negative_roi_extra_images=len(roi_order), l1_enabled=model.head.use_l1,
                optimizer_learning_rates={group['name']: group['lr'] for group in optimizer.param_groups})
        reached_budget = time.monotonic()-train_started >= seconds
        if epoch == 1 or epoch % 5 == 0 or epoch == epochs or reached_budget:
            model.eval()
            predictions = predict_records(detector, splits['val'], cache)
            if fixed_threshold is None:
                selected, candidates = select_threshold(splits['val'], predictions, categories)
                key = (selected['f2'], selected['recall'], -selected['fp'], -epoch)
            else:
                report, _ = evaluate_predictions(splits['val'], predictions, categories, fixed_threshold)
                selected = {'threshold': fixed_threshold, **report['class_and_location']}
                candidates = [selected]
                within_budget = selected['fp'] <= 2 * selected['support']
                key = (within_budget, selected['recall'] if within_budget else selected['f2'],
                       selected['f2'], -selected['fp'], -epoch)
                if correction_focus or localization_focus:
                    selection_key = localization_selection_key if localization_focus else correction_selection_key
                    key = selection_key(report, initial_report, epoch)
                    epoch_log['location_validation'] = report['location_only']
                    epoch_log['baseline_non_regression_met'] = bool(key[0])
            epoch_log['validation'] = selected
            if best_key is None or key > best_key:
                best_key = key
                best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
                best_selection = {'epoch': epoch, 'threshold': selected['threshold'],
                                  'validation': selected, 'threshold_candidates': candidates,
                                  'rule': config['selection'], 'test_used': False}
                torch.save({'model': best_state, 'categories': categories, 'config': config,
                            'threshold': selected['threshold'], 'epoch': epoch}, run_dir/'detector.pt')
                write_json(run_dir/'selection.json', best_selection)
        history.append(epoch_log)
        write_json(run_dir/'training-history.json', history)
        print(json.dumps(epoch_log, ensure_ascii=False), flush=True)
        if reached_budget and (not correction_focus or joint_epochs >= config['correction_focus_policy']['minimum_joint_epochs']):
            break
    training_seconds = time.monotonic()-train_started
    objectness_hook.remove()
    model.load_state_dict(best_state)
    detector.model.eval()
    detector.threshold = best_selection['threshold']
    reports = {}
    for split in (('val',) if defer_test else ('val', 'test')):
        predictions = predict_records(detector, splits[split], cache)
        report, details = evaluate_predictions(splits[split], predictions, categories, detector.threshold)
        reports[split] = report
        write_json(run_dir/('validation-predictions.json' if split == 'val' else 'test-predictions.json'),
                   {'model_sha256': sha256(run_dir/'detector.pt'), 'threshold': detector.threshold,
                    'raw_predictions_at_min_threshold': predictions, 'evaluated_images': details})
    baseline = None
    if not defer_test:
        # Baseline is computed only after checkpoint/threshold are irrevocably selected.
        model.load_state_dict(baseline_state)
        model.eval()
        predictions = predict_records(detector, splits['test'], cache)
        baseline, details = evaluate_predictions(splits['test'], predictions, categories, detector.threshold)
        write_json(run_dir/'baseline-test-predictions.json', {'description': 'Previous detector initialization' if initialize_from else 'COCO initialization with fresh untrained issue class outputs',
                                                           'threshold': detector.threshold, 'evaluated_images': details})
        model.load_state_dict(best_state)
    support = {split: dict(Counter(b['problem'] for r in rows for b in r['boxes']))
               for split, rows in splits.items()}
    metrics = {'experiment': 'human-confirmed YOLOX-s detector with explicit local normal ROI corrections' if feedback_records else 'first human-confirmed YOLOX-s bounding-box detector',
               'exploratory_small_holdout': True, 'image_counts': {s: len(v) for s, v in splits.items()},
               'class_agnostic': location_only,
               'metric_semantics': 'single-class issue localization only; original issue types are not predicted' if location_only else 'class_and_location requires original issue type and IoU >= .5; location_only ignores issue type',
               'group_counts': {s: len(v) for s, v in groups.items()}, 'class_support': support,
               'selected_epoch': best_selection['epoch'], 'selected_threshold': detector.threshold,
               'completed_epochs': epoch, 'training_seconds': training_seconds,
               'total_seconds': time.monotonic()-started,
               'peak_cuda_allocated_bytes': torch.cuda.max_memory_allocated(),
               'peak_cuda_reserved_bytes': torch.cuda.max_memory_reserved(),
               'model_sha256': sha256(run_dir/'detector.pt'), 'validation': reports['val'],
               'test': reports.get('test'), 'test_evaluation_pending': defer_test,
               'untrained_issue_head_baseline_test': baseline,
               'initialization_baseline_test': baseline,
               'human_normal_roi_feedback': config['human_normal_roi_feedback'],
               'human_correction_extra_practice': {**config['human_correction_extra_practice'],
                   'completed_extra_positive_exposures': len(hard_positive_records)*epoch},
               'limitations': ['Small order-grouped test set: exploratory only',
                              'No image-state negatives or unclear images enter training; explicit local normal ROI feedback is auxiliary and never an entire-image label',
                              'False positives are relative to reviewed boxes; unmarked real issues may exist',
                              'Unsupported or absent test classes have no measured recall',
                              'Location-only metric ignores issue type and is not issue recognition accuracy',
                              'Initialization baseline is a prior detector or fresh issue head, not the deployed image classifier'],
               'online_model_replaced': False, 'human_classifications_modified': False}
    write_json(run_dir/'metrics.json', metrics)
    print(json.dumps({'finished': True, **metrics}, ensure_ascii=False), flush=True)
    return metrics


def finalize_test(dataset_path, run_dir):
    """One final held-out score after the validation-only report is reviewed."""
    run_dir = Path(run_dir)
    metrics = json.loads((run_dir/'metrics.json').read_text(encoding='utf-8'))
    if metrics.get('test') is not None or not metrics.get('test_evaluation_pending'):
        raise ValueError('Final test has already been evaluated or is not pending')
    config = json.loads((run_dir/'planned-config.json').read_text(encoding='utf-8'))
    if sha256(dataset_path) != config['dataset_sha256']:
        raise ValueError('Frozen dataset changed')
    dataset = json.loads(Path(dataset_path).read_text(encoding='utf-8-sig'))
    categories = dataset['categories']
    records = dataset['records']
    if config.get('location_only'):
        categories, records, _ = location_training_records(dataset)
    records = [r for r in records if r['split'] == 'test']
    torch.set_num_threads(2)
    detector = load_detector(run_dir)
    predictions = {}
    for record in records:
        raw = Path(record['full_path']).read_bytes()
        if hashlib.sha256(raw).hexdigest() != record['sha256']:
            raise ValueError('Source test image hash changed')
        with Image.open(io.BytesIO(raw)) as source:
            image = ImageOps.exif_transpose(source).convert('RGB')
            if image.size != (record['width'], record['height']):
                raise ValueError('Source test image geometry changed')
            predictions[str(record['image_id'])] = detector.predict_image(image, min(THRESHOLDS))
    report, details = evaluate_predictions(records, predictions, categories, detector.threshold)
    write_json(run_dir/'test-predictions.json', {'model_sha256': sha256(run_dir/'detector.pt'),
              'threshold': detector.threshold, 'raw_predictions_at_min_threshold': predictions,
              'evaluated_images': details})
    metrics['test'] = report
    metrics['test_evaluation_pending'] = False
    write_json(run_dir/'metrics.json', metrics)
    print(json.dumps({'final_test': report}, ensure_ascii=False), flush=True)
    return report


def diagnose_fit(dataset_path, run_dir):
    """Read-only train/validation fit report; never select or inspect test data."""
    run_dir = Path(run_dir)
    checkpoint = torch.load(run_dir/'detector.pt', map_location='cpu', weights_only=False)
    if sha256(dataset_path) != checkpoint['config']['dataset_sha256']:
        raise ValueError('Frozen dataset changed')
    dataset = json.loads(Path(dataset_path).read_text(encoding='utf-8-sig'))
    categories, records = dataset['categories'], dataset['records']
    if checkpoint['config'].get('location_only'):
        categories, records, _ = location_training_records(dataset)
    torch.set_num_threads(2)
    detector = load_detector(run_dir)
    result = {'model_sha256': sha256(run_dir/'detector.pt'),
              'class_agnostic': detector.class_agnostic, 'test_inspected': False}
    for split, names in [('train', ('train',)), ('validation', ('val', 'validation'))]:
        rows = [record for record in records if record['split'] in names]
        predictions = {}
        for record in rows:
            raw = Path(record['full_path']).read_bytes()
            if hashlib.sha256(raw).hexdigest() != record['sha256']:
                raise ValueError('Fit image changed')
            with Image.open(io.BytesIO(raw)) as image:
                image = ImageOps.exif_transpose(image).convert('RGB')
                if image.size != (record['width'], record['height']):
                    raise ValueError('Fit image geometry changed')
                predictions[str(record['image_id'])] = detector.predict_image(image)
        result[split], _ = evaluate_predictions(rows, predictions, categories, detector.threshold)
    write_json(run_dir/'train-validation-fit.json', result)
    print(json.dumps(result, ensure_ascii=False), flush=True)
    return result


def compare_validation(dataset_path, run_dir, base_run):
    """Compare on the same current validation truth by class name; never touch test."""
    torch.set_num_threads(2)
    dataset = json.loads(Path(dataset_path).read_text(encoding='utf-8-sig'))
    records = [r for r in dataset['records'] if r['split'] in ('val', 'validation')]
    categories = dataset['categories']
    names = {c['name']: int(c['id']) for c in categories}
    candidate_checkpoint = torch.load(Path(run_dir)/'detector.pt', map_location='cpu', weights_only=False)
    location_objective = bool(candidate_checkpoint['config'].get('class_agnostic') or
                              candidate_checkpoint['config'].get('location_only'))
    fixed_threshold = candidate_checkpoint['config'].get('fixed_threshold')
    if location_objective:
        categories, converted, _ = location_training_records(dataset)
        records = [r for r in converted if r['split'] in ('val', 'validation')]
    reports = {}
    for name, directory in [('base', base_run), ('candidate', run_dir)]:
        detector = load_detector(directory)
        predictions = {}
        for record in records:
            if sha256(record['full_path']) != record['sha256']:
                raise ValueError('Validation source changed during comparison')
            values = detector.predict_image(record['full_path'], min(THRESHOLDS))
            predictions[str(record['image_id'])] = [dict(v, class_id=0 if location_objective else names.get(v['problem'], -1))
                                                     for v in values]
        if fixed_threshold is None:
            selected, candidates = select_threshold(records, predictions, categories)
        else:
            fixed_report, _ = evaluate_predictions(records, predictions, categories, fixed_threshold)
            selected = {'threshold': fixed_threshold, **fixed_report['class_and_location']}
            candidates = [selected]
        report, _ = evaluate_predictions(records, predictions, categories, selected['threshold'])
        reports[name] = {'model_sha256': sha256(Path(directory)/'detector.pt'),
                         'selection': selected, 'threshold_candidates': candidates, **report}
        detector.model.cpu()
        del detector
        torch.cuda.empty_cache()
    base, candidate = [reports[name]['class_and_location'] for name in ('base', 'candidate')]
    non_regression = candidate['recall'] >= base['recall'] and candidate['f2'] >= base['f2'] and candidate['fp'] <= base['fp']
    strict_improvement = candidate['recall'] > base['recall'] or candidate['f2'] > base['f2'] or candidate['fp'] < base['fp']
    result = {'validation_only': True, 'test_used': False, 'dataset_sha256': sha256(dataset_path),
              'class_mapping': 'all issue classes mapped to problem-location class 0' if location_objective else 'issue name',
              'comparison_objective': 'problem_location_only' if location_objective else 'issue_type_and_location',
              'fixed_threshold_for_both': fixed_threshold,
              'strict_issue_type_metrics': None if location_objective else {'base': base, 'candidate': candidate},
              'base': reports['base'], 'candidate': reports['candidate'],
              'baseline': {**reports['base'], **base},
              'same_validation_images': True, 'problem_name_mapping': True,
              'promotion_recommended': non_regression and strict_improvement,
              'gate': 'candidate recall and F2 not lower, FP not higher, and at least one strict improvement',
              'exploratory_small_holdout': True}
    result['candidate'].update(candidate)
    write_json(Path(run_dir)/'validation-comparison.json', result)
    return result


def region_fit_diagnostic(dataset_path, run_dir, base_run):
    """Training-fit check on reviewed local regions, not held-out generalization accuracy."""
    torch.set_num_threads(2)
    from yolox.data.data_augment import ValTransform
    dataset = json.loads(Path(dataset_path).read_text(encoding='utf-8-sig'))
    rows = dataset.get('negative_feedback_records', [])
    reports = []
    for record in rows:
        item = {'image_id': record['image_id'], 'regions_xyxy': record['negative_rois']}
        for name, directory in [('before', base_run), ('after', run_dir)]:
            detector = load_detector(directory)
            rgb = read_rgb(record['full_path'])
            bgr = np.ascontiguousarray(rgb[..., ::-1])
            array, _ = ValTransform()(bgr, None, (detector.input_size, detector.input_size))
            with torch.inference_mode():
                output = detector.model(torch.from_numpy(array).unsqueeze(0).to(detector.device))[0]
            centers = anchor_centers(detector.input_size, detector.device)
            ratio = min(detector.input_size/record['width'], detector.input_size/record['height'])
            values = []
            for region in transform_rois(record['negative_rois'], record['width'], ratio):
                x1, y1, x2, y2 = region
                mask = (centers[:, 0] >= x1) & (centers[:, 0] < x2) & (centers[:, 1] >= y1) & (centers[:, 1] < y2)
                scores = output[mask, 4:5]*output[mask, 5:]
                values.append({'anchor_count': int(mask.sum()),
                    'max_issue_score': float(scores.max()) if scores.numel() else None,
                    'max_objectness': float(output[mask, 4].max()) if mask.any() else None})
            item[name] = values
            detector.model.cpu()
            del detector
            torch.cuda.empty_cache()
        reports.append(item)
    result = {'scope': 'reviewed training-region fit only, not unseen-hole accuracy',
              'whole_image_normal': False, 'records': reports}
    write_json(Path(run_dir)/'normal-region-fit.json', result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--run', required=True)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--seconds', type=int, default=450)
    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument('--input-size', type=int, default=512)
    parser.add_argument('--adapt-backbone', action='store_true')
    parser.add_argument('--defer-test', action='store_true')
    parser.add_argument('--evaluate-test-only', action='store_true')
    parser.add_argument('--diagnose-fit-only', action='store_true')
    parser.add_argument('--initialize-from', type=Path)
    parser.add_argument('--location-only', '--class-agnostic', dest='location_only', action='store_true', help='Train one generic issue-location class while preserving original human labels')
    parser.add_argument('--fixed-threshold', type=float, help='Keep the pre-registered threshold fixed across checkpoints and comparisons')
    parser.add_argument('--correction-focus', action='store_true', help='Retain localization while learning reviewed issue types, then gently adapt geometry with local views')
    parser.add_argument('--localization-focus', action='store_true', help='Jointly adapt typed localization from epoch one with bounded local views and fixed validation safety')
    parser.add_argument('--compare-base', type=Path, help='Validation-only comparison after training or with --compare-only')
    parser.add_argument('--compare-only', action='store_true')
    parser.add_argument('--compare-validation-to', type=Path,
                        help='Run only same-dataset validation comparison to this baseline; no training or test')
    args = parser.parse_args()
    if args.diagnose_fit_only:
        diagnose_fit(args.dataset, args.run)
    elif args.compare_only or args.compare_validation_to:
        base_run = args.compare_validation_to or args.compare_base
        if not base_run:
            parser.error('--compare-only requires --compare-base')
        print(json.dumps(compare_validation(args.dataset, args.run, base_run), ensure_ascii=False), flush=True)
    elif args.evaluate_test_only:
        finalize_test(args.dataset, args.run)
    else:
        train(args.dataset, args.run, args.epochs, args.seconds, args.batch_size, args.input_size,
              args.adapt_backbone, args.defer_test, args.initialize_from, args.location_only,
              args.fixed_threshold, args.correction_focus, args.localization_focus)
        if args.compare_base:
            print(json.dumps(compare_validation(args.dataset, args.run, args.compare_base), ensure_ascii=False), flush=True)
