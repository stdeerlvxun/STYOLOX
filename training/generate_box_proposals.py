"""Read-only Grad-CAM location suggestions from the deployed image classifier.

These are coarse, unverified proposals, never detector annotations or training labels.
No database writes, source copies, training, model replacement or responsibility output.
"""
from __future__ import annotations

import argparse
from collections import Counter
import datetime as dt
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import sqlite3
import time
import uuid

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageOps
import torch
from torch.nn import functional as F
from torchvision.transforms import functional as TF

from improve_classifier import classification_encoder, digest, pack_views
from vision_classifier import NAS_ROOT, TRAINING_POLICY, training_exclusion_reason, write_json

METHOD = 'convnext_6view_gradcam_consensus_ablation_v1'
COORDINATES = 'exif_transposed_normalized_xywh'
POLICY = {'activation_fraction': .60, 'min_component_area_fraction': .0015,
          'max_component_area_fraction': .45, 'max_box_area_fraction': .60,
          'max_active_area_fraction': .45, 'max_boxes_per_problem': 2,
          'reject_boundary_components': True, 'boundary_margin_fraction': .01,
          'reject_corner_centers_fraction': .18,
          'require_two_layer_agreement': True, 'minimum_ablation_score_drop': .01,
          'minimum_relative_ablation_score_drop': .05,
          'proposal_score_is_localization_confidence': False}


def fingerprint(raw):
    return hashlib.sha256(raw.encode('utf-8')).hexdigest()


def safe_source_path(root, relative):
    relative = str(relative).replace('\\', '/')
    path = PurePosixPath(relative)
    if path.is_absolute() or '..' in path.parts or not path.parts or ':' in relative:
        raise ValueError('Unsafe relative image path')
    destination = (Path(root) / Path(*path.parts)).resolve()
    if not destination.is_relative_to(Path(root).resolve()):
        raise ValueError('Image path escapes source directory')
    return destination


def select_images(snapshot):
    """Read one consistent offline SQLite backup; never initialize the application store."""
    connection = sqlite3.connect(Path(snapshot).resolve().as_uri() + '?mode=ro', uri=True)
    connection.row_factory = sqlite3.Row
    try:
        root = connection.execute("SELECT value FROM settings WHERE key='root'").fetchone()
        if not root:
            raise ValueError('Snapshot has no current root')
        root_id = int(root['value'])
        records = connection.execute('SELECT i.id image_id,i.relative_path,i.sha256,i.reported_at,'
            'i.available,l.data,d.image_id IS NOT NULL has_draft,a.image_id IS NOT NULL has_annotation '
            'FROM images i JOIN labels l ON l.image_id=i.id '
            'LEFT JOIN drafts d ON d.image_id=i.id '
            'LEFT JOIN annotation_documents a ON a.image_id=i.id WHERE i.root_id=? '
            'ORDER BY i.reported_at DESC,i.relative_path DESC,i.id DESC', (root_id,)).fetchall()
        selected = []; excluded = Counter()
        for record in records:
            row = dict(record)
            label = json.loads(row.pop('data'))
            # Fingerprint the exact original JSON string, not reserialized business values.
            raw = record['data']
            reason = training_exclusion_reason(label, bool(row['has_draft']))
            if not reason and row['has_annotation']:
                reason = 'existing_human_annotation'
            if not reason and not row['available']:
                reason = 'source_unavailable'
            if reason:
                excluded[reason] += 1
                continue
            row['problems'] = list(dict.fromkeys(label['defects']))
            row['label_fingerprint'] = fingerprint(raw)
            selected.append(row)
        return selected, {'root_id': root_id, 'confirmed_images_in_root': len(records),
                          'eligible_images': len(selected), 'excluded': dict(excluded)}
    finally:
        connection.close()


def image_views(image):
    """Exactly the frozen classifier crops, with inverse coordinate transforms."""
    width, height = image.size
    fitted = ImageOps.contain(image, (224, 224))
    pad_x, pad_y = round((224-fitted.width)*.5), round((224-fitted.height)*.5)
    views = [ImageOps.pad(image, (224, 224), color=(124, 116, 104))]
    geometry = [{'kind': 'pad', 'content': [pad_x, pad_y, fitted.width, fitted.height]}]
    side = max(1, int(min(width, height) * .75))
    origins = [(0, 0), (width-side, 0), (0, height-side), (width-side, height-side),
               ((width-side)//2, (height-side)//2)]
    for x, y in origins:
        views.append(image.crop((x, y, x+side, y+side)).resize((224, 224), Image.Resampling.BILINEAR))
        geometry.append({'kind': 'crop', 'region': [x, y, side, side]})
    tensors = torch.stack([TF.normalize(TF.to_tensor(view), [.485, .456, .406], [.229, .224, .225])
                           for view in views])
    return tensors, geometry


def project_cams(cams, geometry, image_size, max_side=512):
    """Fuse class activation maps in EXIF-corrected source-image coordinates."""
    width, height = image_size
    scale = min(1., max_side/max(width, height))
    target_w, target_h = max(1, round(width*scale)), max(1, round(height*scale))
    fused = np.zeros((target_h, target_w), np.float32)
    peak = float(np.max(cams))
    if not np.isfinite(peak) or peak <= 1e-12:
        return fused
    # A single common scale retains relative evidence strength across all six views.
    for cam, transform in zip(cams, geometry):
        cam = cv2.resize(cam/peak, (224, 224), interpolation=cv2.INTER_LINEAR)
        if transform['kind'] == 'pad':
            x, y, w, h = transform['content']
            mapped = cv2.resize(cam[y:y+h, x:x+w], (target_w, target_h), interpolation=cv2.INTER_LINEAR)
            np.maximum(fused, mapped, out=fused)
        else:
            x, y, w, h = transform['region']
            x0, y0 = round(x/width*target_w), round(y/height*target_h)
            x1, y1 = round((x+w)/width*target_w), round((y+h)/height*target_h)
            if x1 <= x0 or y1 <= y0:
                continue
            mapped = cv2.resize(cam, (x1-x0, y1-y0), interpolation=cv2.INTER_LINEAR)
            np.maximum(fused[y0:y1, x0:x1], mapped, out=fused[y0:y1, x0:x1])
    return fused


def heatmap_boxes(heatmap, problem, score):
    """Conservative components; diffuse/no class evidence leads to explicit abstention."""
    peak = float(heatmap.max(initial=0))
    if not np.isfinite(heatmap).all() or peak <= 1e-8:
        return [], 'no_positive_class_activation'
    normalized = heatmap / peak
    mask = (normalized >= POLICY['activation_fraction']).astype(np.uint8)
    if float(mask.mean()) > POLICY['max_active_area_fraction']:
        return [], 'diffuse_activation'
    count, _, components, _ = cv2.connectedComponentsWithStats(mask, 8)
    height, width = mask.shape
    regions = []
    for x, y, w, h, area in components[1:count]:
        area_fraction = float(area)/(height*width)
        if (area_fraction < POLICY['min_component_area_fraction'] or
                area_fraction > POLICY['max_component_area_fraction']):
            continue
        # Border-attached activations often respond to camera overlays or cut-off
        # context. Abstain instead of turning them into defect boxes. This can
        # also withhold genuine defects at the photo edge; reviewers must inspect.
        margin_x, margin_y = max(1, round(width*.01)), max(1, round(height*.01))
        if x <= margin_x or y <= margin_y or x+w >= width-margin_x or y+h >= height-margin_y:
            continue
        corner = POLICY['reject_corner_centers_fraction']
        cx, cy = (x+w*.5)/width, (y+h*.5)/height
        if (cx < corner or cx > 1-corner) and (cy < corner or cy > 1-corner):
            continue
        # Small fixed padding is only for the coarse 14x14 activation resolution.
        padding = max(1, round(.015*max(width, height)))
        x0, y0 = max(0, x-padding), max(0, y-padding)
        x1, y1 = min(width, x+w+padding), min(height, y+h+padding)
        if (x1-x0)*(y1-y0)/(height*width) > POLICY['max_box_area_fraction']:
            continue
        regions.append((int(area), dict(id='model-'+uuid.uuid4().hex, problem=problem,
            x=float(x0/width), y=float(y0/height), w=float((x1-x0)/width), h=float((y1-y0)/height),
            score=float(score), origin='model')))
    regions.sort(key=lambda item: item[0], reverse=True)
    boxes = [box for _, box in regions[:POLICY['max_boxes_per_problem']]]
    return boxes, '' if boxes else 'no_reliable_interior_component_or_corner_overlay_risk'


class GradCamProposer:
    def __init__(self, run, device):
        self.checkpoint = torch.load(Path(run)/'classifier.pt', map_location='cpu', weights_only=True)
        if (self.checkpoint.get('training_policy') != TRAINING_POLICY or
                self.checkpoint.get('feature_kind') != 'convnext_local_ridge' or
                not self.checkpoint.get('weights_retrained')):
            raise ValueError('Requires the deployed reviewed ConvNeXt local-ridge problem checkpoint')
        self.encoder, meta = classification_encoder(device)
        if self.checkpoint['encoder_sha256'] != meta['sha256']:
            raise ValueError('Encoder hash differs from the problem checkpoint')
        self.device = device
        for parameter in self.encoder.parameters():
            parameter.requires_grad_(False)
        self.activation = self.final_activation = None
        self.capture_enabled = True

        def capture(module, inputs, output):
            # The earlier frozen stages require no backward graph. Backpropagate
            # through the final stages only, retaining the actual classifier head.
            if not self.capture_enabled:
                return output
            self.activation = output.detach().requires_grad_(True)
            return self.activation
        self.hook = self.encoder.features[5].register_forward_hook(capture)
        def capture_final(module, inputs, output):
            if self.capture_enabled:
                self.final_activation = output
        self.final_hook = self.encoder.features[7].register_forward_hook(capture_final)
        ridge = self.checkpoint['ridge']
        self.center = ridge['center'].to(device)
        self.coefficients = (ridge['training_vectors'].T@ridge['alpha']).to(device)
        self.prior = ridge['prior'].to(device)
        self.names = self.checkpoint['problems']
        self.minimum_score = float(self.checkpoint['threshold'])

    def propose(self, image, problems):
        views, geometry = image_views(image)
        vectors = self.encoder(views.to(self.device))
        _, raw = pack_views(vectors)
        scores = F.normalize(raw-self.center, dim=0)@self.coefficients+self.prior
        boxes = []; abstentions = []; maps = {}
        for problem in problems:
            if problem not in self.names:
                abstentions.append({'problem': problem, 'reason': 'class_not_in_deployed_model'})
                continue
            index = self.names.index(problem)
            score = float(scores[index].detach().clamp(0, 1))
            if score < self.minimum_score:
                abstentions.append({'problem': problem, 'reason': 'insufficient_classifier_evidence', 'score': score})
                continue
            gradients = torch.autograd.grad(scores[index], (self.activation, self.final_activation), retain_graph=True)
            projected = []
            for gradient, activation in zip(gradients, (self.activation, self.final_activation)):
                weights = gradient.mean(dim=(2, 3), keepdim=True)
                cams = (weights*activation).sum(dim=1).relu().detach().cpu().numpy()
                layer_map = project_cams(cams, geometry, image.size)
                layer_peak = float(layer_map.max(initial=0))
                projected.append(layer_map/max(layer_peak, 1e-12))
            heatmap = np.minimum(*projected)
            candidates, reason = heatmap_boxes(heatmap, problem, score)
            boxes.extend(candidates)
            if candidates:
                maps[problem] = heatmap
            else:
                abstentions.append({'problem': problem, 'reason': reason, 'score': score})
        self.activation = self.final_activation = None
        supported = []
        self.capture_enabled = False
        try:
            with torch.no_grad():
                for box in boxes:
                    index = self.names.index(box['problem'])
                    perturbed = image.copy()
                    x0, y0 = round(box['x']*image.width), round(box['y']*image.height)
                    x1, y1 = round((box['x']+box['w'])*image.width), round((box['y']+box['h'])*image.height)
                    # Destroy local details while retaining average color. This is
                    # a model-evidence check, not proof that the region is damage.
                    region = image.crop((x0, y0, x1, y1))
                    blurred = region.resize((1, 1), Image.Resampling.BOX).resize(region.size)
                    perturbed.paste(blurred, (x0, y0))
                    changed_views, _ = image_views(perturbed)
                    _, changed_raw = pack_views(self.encoder(changed_views.to(self.device)))
                    changed_score = F.normalize(changed_raw-self.center, dim=0)@self.coefficients+self.prior
                    drop = box['score']-float(changed_score[index].clamp(0, 1))
                    minimum = max(POLICY['minimum_ablation_score_drop'],
                                  box['score']*POLICY['minimum_relative_ablation_score_drop'])
                    if drop >= minimum:
                        box['ablation_score_drop'] = float(drop)
                        supported.append(box)
                    else:
                        abstentions.append({'problem': box['problem'], 'reason': 'local_ablation_not_supporting',
                                            'score': box['score'], 'ablation_score_drop': float(drop)})
        finally:
            self.capture_enabled = True
        return supported, abstentions, maps

    def close(self):
        self.hook.remove()
        self.final_hook.remove()


def preview(image, boxes, maps, output):
    display = image.copy()
    display.thumbnail((1000, 800))
    if maps:
        heatmap = np.maximum.reduce(list(maps.values()))
        heatmap = cv2.resize(heatmap, display.size)
        colors = cv2.cvtColor(cv2.applyColorMap((heatmap.clip(0, 1)*255).astype('uint8'), cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB)
        display = Image.blend(display, Image.fromarray(colors), .25)
    draw = ImageDraw.Draw(display)
    for number, box in enumerate(boxes, 1):
        x, y, w, h = (box[k] for k in ('x', 'y', 'w', 'h'))
        draw.rectangle([x*display.width, y*display.height, (x+w)*display.width, (y+h)*display.height], outline='red', width=3)
        draw.text((x*display.width+3, y*display.height+3), str(number), fill='red')
    output.parent.mkdir(parents=True, exist_ok=True)
    display.save(output)


def generate(snapshot, run, output, limit=None, source_root=NAS_ROOT, preview_dir=None):
    snapshot, run, output = Path(snapshot), Path(run), Path(output)
    if output.exists():
        raise ValueError('Choose a new output file; existing proposals are never overwritten')
    snapshot_hash = digest(snapshot)
    rows, selection = select_images(snapshot)
    if limit is not None:
        rows = rows[:limit]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.set_num_threads(4)
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats()
    proposer = GradCamProposer(run, device)
    checkpoint_hash = digest(run/'classifier.pt')
    result = {'schema_version': 1, 'method': METHOD, 'model_id': 'recall-trained-v3-'+checkpoint_hash[:12],
        'source_checkpoint_sha256': checkpoint_hash, 'source_snapshot_sha256': snapshot_hash,
        'created_at': dt.datetime.now(dt.timezone.utc).isoformat(), 'requires_human_review': True,
        'coordinate_space': COORDINATES, 'is_trained_detector': False,
        'quality_warning': '弱定位仍可能框到背景、包装字样、水印，局部遮挡仅证明模型依赖此处，不证明这里是缺陷；未测定位准确率。',
        'unreviewed_proposals_training_eligible': False, 'classification_results_modified': False,
        'probabilities_calibrated': False, 'policy': POLICY, 'selection': selection,
        'items': [], 'errors': []}
    start = time.perf_counter()
    try:
        for index, row in enumerate(rows):
            try:
                path = safe_source_path(source_root, row['relative_path'])
                before = path.stat(); raw = path.read_bytes(); after = path.stat()
                if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                    raise ValueError('Source changed during read')
                source_hash = hashlib.sha256(raw).hexdigest()
                if row['sha256'] and row['sha256'] != source_hash:
                    raise ValueError('Source image SHA256 differs from snapshot')
                with Image.open(io.BytesIO(raw)) as original:
                    image = ImageOps.exif_transpose(original).convert('RGB')
                boxes, abstentions, maps = proposer.propose(image, row['problems'])
                result['items'].append({'image_id': row['image_id'], 'relative_path': row['relative_path'],
                    'label_fingerprint': row['label_fingerprint'], 'source': {'sha256': source_hash,
                        'width': image.width, 'height': image.height}, 'boxes': boxes, 'abstentions': abstentions,
                    'note': '图级分类激活产生的粗定位候选；可能框错、框大或漏框，请调整并明确确认后再作为训练标注。'})
                if preview_dir:
                    preview(image, boxes, maps, Path(preview_dir)/f"{row['image_id']}.jpg")
            except (OSError, ValueError, RuntimeError) as error:
                result['errors'].append({'image_id': row['image_id'], 'error': str(error)})
            if index % 10 == 0 or index+1 == len(rows):
                progress = {'stage': 'weak_localization', 'done': index+1, 'total': len(rows),
                            'seconds': round(time.perf_counter()-start, 1)}
                write_json(output.with_suffix('.progress.json'), progress)
                print(json.dumps(progress), flush=True)
    finally:
        proposer.close()
    if digest(snapshot) != snapshot_hash:
        raise ValueError('Input snapshot changed during generation; discard this run')
    boxes = [box for item in result['items'] for box in item['boxes']]
    result['summary'] = {'attempted_images': len(rows), 'processed_images': len(result['items']),
        'images_with_proposals': sum(bool(item['boxes']) for item in result['items']), 'proposed_boxes': len(boxes),
        'images_without_proposals': sum(not item['boxes'] for item in result['items']),
        'abstention_events': sum(len(item['abstentions']) for item in result['items']),
        'fully_abstained_problems': sum(len({entry['problem'] for entry in item['abstentions']}
            - {box['problem'] for box in item['boxes']}) for item in result['items']),
        'abstention_reasons': dict(Counter(reason['reason'] for item in result['items'] for reason in item['abstentions'])),
        'errors': len(result['errors']), 'seconds': round(time.perf_counter()-start, 3),
        'device': str(device), 'peak_allocated_mib': torch.cuda.max_memory_allocated()/2**20 if device.type == 'cuda' else 0,
        'localization_accuracy_measured': False, 'source_images_modified': False}
    write_json(output, result)
    print(json.dumps(result['summary']), flush=True)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--snapshot', type=Path, required=True)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--limit', type=int)
    parser.add_argument('--source-root', type=Path, default=NAS_ROOT)
    parser.add_argument('--preview-dir', type=Path)
    args = parser.parse_args()
    if args.limit is not None and args.limit <= 0:
        parser.error('--limit must be positive')
    generate(args.snapshot, args.run, args.output, args.limit, args.source_root, args.preview_dir)
