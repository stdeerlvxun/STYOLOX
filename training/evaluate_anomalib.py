"""Isolated official PatchCore experiment; no label writes or automatic deployment.

Only explicitly confirmed normal ROIs from otherwise eligible problem photographs
build/calibrate the memory bank. Validation is location-only; test is never read.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import random
import time

import numpy as np
from PIL import Image, ImageDraw, ImageOps

NORMAL_QUANTILE = .995  # Fixed before looking at validation predictions.
MAX_SIDE = 1024
TILE_SIZE = 256
TILE_STRIDE = 128
MODEL_SIZE = 224
MAX_FEATURES = 20000
SEED = 42
RESNET18_SHA256 = 'f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec'


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def eligible_problem(row):
    return (row.get('has_draft') is False and row.get('image_state') in (None, '', 'defect')
            and isinstance(row.get('defects'), list) and bool(row['defects']))


def roi_bounds(roi, width, height):
    """Round inward: never introduce unconfirmed neighbouring pixels into normal."""
    if len(roi) != 4 or not all(math.isfinite(float(v)) for v in roi):
        raise ValueError('Invalid explicit normal ROI')
    x1, y1, x2, y2 = map(float, roi)
    if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
        raise ValueError('Normal ROI outside EXIF-transposed source')
    bounds = math.ceil(x1), math.ceil(y1), math.floor(x2), math.floor(y2)
    if bounds[2] <= bounds[0] or bounds[3] <= bounds[1]:
        raise ValueError('Normal ROI has no completely confirmed pixels')
    if (x2-x1)*(y2-y1) >= .98*width*height:
        raise ValueError('Whole-image normal supervision is not allowed')
    return bounds


def split_normal_rois(dataset):
    """Recheck prepared provenance/holdouts, then split by independent source group."""
    from vision_classifier import assign_groups

    if dataset.get('training_policy') != 'human_confirmed_problem_boxes_v1':
        raise ValueError('Unsupported or missing unchanged training policy')
    records = dataset['records']
    for row in records:
        if (not eligible_problem(row) or row.get('annotation_status') != 'confirmed'
                or row.get('annotation_outcome') != 'boxed' or not row.get('boxes')
                or row.get('split') not in ('train', 'val', 'validation', 'test')):
            raise ValueError('Problem records violate confirmed-box/image-state policy')
    originals = {row['image_id']: row for row in records}
    negatives = dataset.get('negative_feedback_records', [])
    if not negatives:
        raise ValueError('Explicit confirmed local normal feedback is required')
    included = set(dataset.get('included_feedback_ids', []))
    combined = copy.deepcopy(records + dataset.get('historical_holdout_records', []) + negatives)
    for row in combined:
        row['phash'] = int(row['phash'])
    assign_groups(combined)  # Includes case/order/SHA and conservative perceptual duplicates.
    group_splits = {}
    for row in combined:
        split = 'validation' if row.get('split') == 'val' else row.get('split')
        group_splits.setdefault(row['group'], set()).add(split)
    if any(len(splits) > 1 for splits in group_splits.values()):
        raise ValueError('Current or historical source groups cross train/validation/test')
    held_groups = {row['group'] for row in combined if row.get('split') in ('val', 'validation', 'test')}
    normal_rows = combined[-len(negatives):]
    rois = []
    for row in normal_rows:
        provenance = row.get('feedback_provenance_sha256', '')
        if (not eligible_problem(row) or row.get('feedback_only') is not True
                or row.get('split') != 'train' or not row.get('negative_rois')
                or len(provenance) != 64 or any(c not in '0123456789abcdef' for c in provenance)
                or not row.get('feedback_ids') or not set(row['feedback_ids']) <= included):
            raise ValueError('Normal ROIs lack accepted explicit human feedback provenance')
        if row['group'] in held_groups:
            raise ValueError('Normal ROI overlaps current or historical holdout')
        original = originals.get(row['image_id'])
        if original and (original['split'] != 'train' or original['sha256'] != row['sha256']
                         or original.get('label_fingerprint') != row.get('label_fingerprint')):
            raise ValueError('Normal feedback differs from current confirmed training source')
        for index, roi in enumerate(row['negative_rois']):
            roi_bounds(roi, row['width'], row['height'])
            for box in row.get('boxes', []):
                other = box['xyxy']
                if min(roi[2], other[2]) > max(roi[0], other[0]) and min(roi[3], other[3]) > max(roi[1], other[1]):
                    raise ValueError('Normal ROI overlaps confirmed defect')
            rois.append({'record': row, 'roi': list(roi), 'roi_index': index,
                         'group': row['group']})
    groups = sorted({item['group'] for item in rois}, key=str)
    if len(groups) < 2:
        raise ValueError('At least two independent normal ROI groups are required')
    random.Random(SEED).shuffle(groups)
    calibration_groups = set(groups[:max(1, math.ceil(len(groups)*.25))])
    bank = [item for item in rois if item['group'] not in calibration_groups]
    calibration = [item for item in rois if item['group'] in calibration_groups]
    return bank, calibration


def resize_geometry(width, height, max_side=MAX_SIDE):
    if width <= 0 or height <= 0:
        raise ValueError('Invalid image dimensions')
    scale = min(1., max_side/max(width, height))
    return max(1, round(width*scale)), max(1, round(height*scale))


def axis_starts(length, tile=TILE_SIZE, stride=TILE_STRIDE):
    if length <= 0 or tile <= 0 or not 0 < stride <= tile:
        raise ValueError('Invalid tiling geometry')
    if length <= tile:
        return [0]
    positions = list(range(0, length-tile+1, stride))
    if positions[-1] != length-tile:
        positions.append(length-tile)
    return positions


def tile_windows(width, height):
    return [(x, y, min(width, x+TILE_SIZE), min(height, y+TILE_SIZE))
            for y in axis_starts(height) for x in axis_starts(width)]


def calibration_threshold(values):
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if not len(values) or not np.isfinite(values).all():
        raise ValueError('Finite held-out normal calibration pixels are required')
    return float(np.quantile(values, NORMAL_QUANTILE, method='higher'))


def nms_boxes(boxes, threshold=.5, limit=10):
    from train_detector import xyxy_iou
    remaining = sorted(boxes, key=lambda item: (-item['score'], item['xyxy']))
    kept = []
    while remaining and len(kept) < limit:
        selected = remaining.pop(0)
        kept.append(selected)
        remaining = [other for other in remaining if xyxy_iou(selected['xyxy'], other['xyxy']) <= threshold]
    return kept


def boxes_from_map(anomaly_map, threshold, original_size):
    """Fixed connected-component extraction; no validation-label-driven box tuning."""
    import cv2
    heat = np.asarray(anomaly_map, dtype=np.float32)
    if heat.ndim != 2 or not np.isfinite(heat).all() or not math.isfinite(threshold):
        raise ValueError('Invalid anomaly map or normal calibration threshold')
    height, width = heat.shape
    original_width, original_height = original_size
    count, labels, stats, _ = cv2.connectedComponentsWithStats((heat > threshold).astype(np.uint8), 8)
    values = []
    for label in range(1, count):
        x, y, box_width, box_height, area = stats[label]
        if area < 16:
            continue
        xyxy = [float(x*original_width/width), float(y*original_height/height),
                float((x+box_width)*original_width/width), float((y+box_height)*original_height/height)]
        normalized = [xyxy[0]/original_width, xyxy[1]/original_height,
                      (xyxy[2]-xyxy[0])/original_width, (xyxy[3]-xyxy[1])/original_height]
        values.append({'class_id': 0, 'problem': '问题位置', 'class_agnostic': True,
                       'score': float(heat[labels == label].max()), 'xyxy': xyxy,
                       'normalized_xywh': normalized})
    return nms_boxes(values)


def read_verified(record, source_root):
    root = Path(source_root).resolve()
    relative = str(record['relative_path']).replace('\\', '/')
    if relative.startswith('/') or '..' in relative.split('/') or ':' in relative:
        raise ValueError('Unsafe source relative path')
    expected = (root / Path(*relative.split('/'))).resolve()
    if not expected.is_relative_to(root) or Path(record['full_path']).resolve() != expected:
        raise ValueError('Source escaped declared original archive')
    if file_sha256(expected) != record['sha256']:
        raise ValueError('Source image SHA changed')
    with Image.open(expected) as image:
        image = ImageOps.exif_transpose(image).convert('RGB')
        if image.size != (record['width'], record['height']):
            raise ValueError('Source EXIF-transposed geometry changed')
        return image.copy()


def resized_source(image):
    size = resize_geometry(*image.size)
    return image if image.size == size else image.resize(size, Image.Resampling.BILINEAR)


def tile_tensor(image, windows, device):
    import torch
    arrays = [np.asarray(image.crop(window).resize((MODEL_SIZE, MODEL_SIZE), Image.Resampling.BILINEAR),
                         dtype=np.float32)/255. for window in windows]
    tensor = torch.from_numpy(np.stack(arrays)).permute(0, 3, 1, 2).to(device)
    mean = tensor.new_tensor([.485, .456, .406])[None, :, None, None]
    std = tensor.new_tensor([.229, .224, .225])[None, :, None, None]
    return (tensor-mean)/std


def anomaly_heatmap(model, image, device):
    import cv2
    import torch
    image = resized_source(image)
    windows = tile_windows(*image.size)
    merged = np.zeros((image.height, image.width), dtype=np.float32)
    weights = np.zeros_like(merged)
    model.eval()
    with torch.inference_mode():
        for offset in range(0, len(windows), 8):
            batch_windows = windows[offset:offset+8]
            output = model(tile_tensor(image, batch_windows, device))
            maps = output.anomaly_map.detach().cpu().numpy()
            for window, heat in zip(batch_windows, maps):
                x1, y1, x2, y2 = window
                heat = cv2.resize(heat.squeeze(), (x2-x1, y2-y1), interpolation=cv2.INTER_LINEAR)
                merged[y1:y2, x1:x2] += heat
                weights[y1:y2, x1:x2] += 1
    if not (weights > 0).all():
        raise ValueError('Uncovered image pixels')
    return merged/weights


def train_memory_bank(model, bank, source_root, device):
    """Fixed reservoir cap before official coreset keeps this pilot bounded on 8 GB."""
    import torch
    rng = np.random.default_rng(SEED)
    features, priorities, total = None, np.empty(0), 0
    model.train()
    model.feature_extractor.eval()  # Frozen pretrained statistics, even in embedding-return mode.
    with torch.inference_mode():
        for index, item in enumerate(bank):
            record = item['record']
            image = read_verified(record, source_root)
            roi = image.crop(roi_bounds(item['roi'], *image.size))
            roi = resized_source(roi)
            windows = tile_windows(*roi.size)
            for offset in range(0, len(windows), 8):
                embedding = model(tile_tensor(roi, windows[offset:offset+8], device)).detach().cpu()
                model.embedding_store.clear()  # Official forward appends; bounded reservoir owns these tensors.
                keys = rng.random(len(embedding))
                features = embedding if features is None else torch.cat((features, embedding))
                priorities = np.concatenate((priorities, keys))
                total += len(embedding)
                if len(priorities) > MAX_FEATURES:
                    selected = np.argpartition(priorities, -MAX_FEATURES)[-MAX_FEATURES:]
                    features, priorities = features[selected], priorities[selected]
            print(json.dumps({'phase': 'normal_memory', 'roi': index+1, 'total_rois': len(bank)}, ensure_ascii=False), flush=True)
    if features is None or len(features) < 180:
        raise ValueError('Too few local normal patch embeddings for 9-neighbour coreset')
    model.embedding_store = [features.to(device)]
    model.subsample_embedding(sampling_ratio=.05)
    model.eval()
    return {'patch_embeddings_seen': total, 'pre_coreset_embeddings': len(features),
            'reservoir_cap': MAX_FEATURES, 'coreset_ratio': .05,
            'memory_bank_embeddings': len(model.memory_bank)}


def save_overlay(path, source, heat, predictions, truth):
    import cv2
    image = resized_source(source).convert('RGB')
    low, high = float(heat.min()), float(heat.max())
    normalized = np.zeros_like(heat) if high <= low else (heat-low)/(high-low)
    colors = cv2.applyColorMap(np.uint8(normalized*255), cv2.COLORMAP_TURBO)[..., ::-1]
    overlay = Image.blend(image, Image.fromarray(colors), .35)
    draw = ImageDraw.Draw(overlay)
    sx, sy = image.width/source.width, image.height/source.height
    for items, color in ((truth, '#30e070'), (predictions, '#ff5252')):
        for item in items:
            box = item['xyxy']
            draw.rectangle([box[0]*sx, box[1]*sy, box[2]*sx, box[3]*sy], outline=color, width=3)
    overlay.save(path)


def evaluate(dataset_path, run, device='cuda', baseline_path=None, backbone_weights=None):
    import torch
    import anomalib
    from anomalib.models.image.patchcore.torch_model import PatchcoreModel
    from train_detector import evaluate_predictions, location_training_records

    if anomalib.__version__ != '2.2.0':
        raise ValueError('This isolated protocol is pinned to official anomalib 2.2.0')
    started = time.monotonic()
    dataset_path, run = Path(dataset_path), Path(run)
    digest = file_sha256(dataset_path)
    dataset = json.loads(dataset_path.read_text(encoding='utf-8-sig'))
    bank, calibration = split_normal_rois(dataset)
    categories, records, _ = location_training_records(dataset)
    validation = [row for row in records if row['split'] in ('val', 'validation')]
    if not validation:
        raise ValueError('No fixed validation records')
    run.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    weight_info = {'pretrained': True, 'source': 'anomalib/timm default pretrained retrieval'}
    if backbone_weights:
        backbone_weights = Path(backbone_weights)
        weight_digest = file_sha256(backbone_weights)
        if not weight_digest.startswith('f37072fd') or weight_digest != RESNET18_SHA256:
            raise ValueError('Local ResNet18 weights do not match the pinned official torchvision SHA256')
        model = PatchcoreModel(backbone='resnet18', layers=['layer2', 'layer3'],
                              pre_trained=False, num_neighbors=9)
        state = torch.load(backbone_weights, map_location='cpu', weights_only=True)
        missing, unexpected = model.feature_extractor.feature_extractor.load_state_dict(state, strict=False)
        if missing or any(not key.startswith(('layer4.', 'fc.')) for key in unexpected):
            raise ValueError('Official ResNet18 weights do not match the layer2/layer3 feature extractor')
        if file_sha256(backbone_weights) != weight_digest:
            raise ValueError('Local pretrained weights changed while loading')
        weight_info = {'pretrained': True, 'local_path': str(backbone_weights), 'sha256': weight_digest,
                       'source': 'official torchvision ResNet18 ImageNet weights',
                       'source_url': 'https://download.pytorch.org/models/resnet18-f37072fd.pth',
                       'missing_keys': list(missing), 'excluded_deeper_layer_keys': sorted(unexpected)}
    else:
        model = PatchcoreModel(backbone='resnet18', layers=['layer2', 'layer3'],
                              pre_trained=True, num_neighbors=9)
    model = model.to(device)
    bank_info = train_memory_bank(model, bank, dataset['source_root'], device)
    normal_pixels = []
    for index, item in enumerate(calibration):
        image = read_verified(item['record'], dataset['source_root'])
        normal = image.crop(roi_bounds(item['roi'], *image.size))
        normal_pixels.append(anomaly_heatmap(model, normal, device).ravel())
        print(json.dumps({'phase': 'normal_calibration', 'roi': index+1, 'total_rois': len(calibration)}), flush=True)
    threshold = calibration_threshold(np.concatenate(normal_pixels))
    predictions, derived = {}, []
    for index, row in enumerate(validation):
        image = read_verified(row, dataset['source_root'])
        heat = anomaly_heatmap(model, image, device)
        values = boxes_from_map(heat, threshold, image.size)
        predictions[str(row['image_id'])] = values
        if index < 3:
            name = 'validation-overlay-'+str(row['image_id'])+'.png'
            save_overlay(run/name, image, heat, values, row['boxes'])
            derived.append({'image_id': row['image_id'], 'path': name,
                            'green': 'confirmed truth', 'red': 'PatchCore location proposal'})
        print(json.dumps({'phase': 'validation', 'image': index+1, 'total_images': len(validation)}), flush=True)
    metrics, details = evaluate_predictions(validation, predictions, categories, threshold)
    baseline = None
    if baseline_path:
        baseline = json.loads(Path(baseline_path).read_text(encoding='utf-8-sig'))
        if baseline.get('images') != len(validation) or baseline['location_only']['support'] != metrics['location_only']['support']:
            raise ValueError('Baseline does not match fixed validation image/box counts')
    if file_sha256(dataset_path) != digest:
        raise ValueError('Frozen dataset changed during isolated evaluation')
    torch.save({'model': model.state_dict(), 'backbone': 'resnet18', 'layers': ['layer2', 'layer3'],
                'normal_threshold': threshold, 'dataset_sha256': digest,
                'backbone_weights': weight_info}, run/'patchcore-experiment.pt')
    report = {'experiment': 'official anomalib 2.2.0 PatchCore local-normal-ROI pilot',
              'exploratory_only': True, 'automatic_deployment': False, 'online_model_changed': False,
              'test_used': False, 'test': None, 'training_policy': dataset['training_policy'],
              'dataset_sha256': digest, 'normal_quantile': NORMAL_QUANTILE,
              'backbone_weights': weight_info,
              'threshold_selected_from': 'held-out normal ROI pixels only; no validation truth',
              'normal_threshold': threshold, 'normal_bank_rois': len(bank), 'normal_calibration_rois': len(calibration),
              'normal_bank_groups': len({v['group'] for v in bank}),
              'normal_calibration_groups': len({v['group'] for v in calibration}),
              'normal_training_scope': 'only explicit human-confirmed local normal ROIs on eligible problem photos',
              'normal_whole_images_used': False, 'unboxed_background_used_as_normal': False,
              'source_photos_copied': False, 'derived_overlays': derived,
              'original_human_labels_changed': False, 'original_human_boxes_changed': False,
              'protocol': {'max_side': MAX_SIDE, 'tile_size': TILE_SIZE, 'stride': TILE_STRIDE,
                           'network_size': MODEL_SIZE, 'normalization': 'ImageNet RGB',
                           'map_fusion': 'overlap mean', 'min_component_area': 16,
                           'max_predictions': 10, 'nms_iou': .5, 'evaluation_iou': .5,
                           'num_neighbors': 9, 'seed': SEED},
              'memory_bank': bank_info, 'validation': metrics, 'baseline': baseline,
              'elapsed_seconds': time.monotonic()-started}
    if baseline:
        current, old = metrics['location_only'], baseline['location_only']
        report['comparison'] = {key+'_delta': current[key]-old[key] for key in ('tp', 'fn', 'fp', 'recall', 'f2')}
        report['comparison']['same_validation_counts'] = True
        report['comparison']['baseline_json_sha256'] = file_sha256(baseline_path)
        report['comparison']['promotion_requires_separate_review'] = True
    write_json(run/'metrics.json', report)
    write_json(run/'predictions.json', predictions)
    write_json(run/'validation-details.json', details)
    write_json(run/'normal-roi-split.json', {name: [{'image_id': r['record']['image_id'], 'group': r['group'],
               'sha256': r['record']['sha256'], 'roi': r['roi']} for r in values]
               for name, values in [('bank', bank), ('calibration', calibration)]})
    print(json.dumps(report, ensure_ascii=False), flush=True)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--device', choices=('cuda', 'cpu'), default='cuda')
    parser.add_argument('--baseline-json', '--baseline', dest='baseline', type=Path)
    parser.add_argument('--backbone-weights', type=Path)
    args = parser.parse_args()
    evaluate(args.dataset, args.run, args.device, args.baseline, args.backbone_weights)
