"""Explicit human normal regions: local auxiliary supervision, never whole-image labels."""
from __future__ import annotations

import copy
from collections import Counter
from contextlib import closing
import hashlib
import io
import json
from pathlib import Path
import sqlite3

import cv2
import numpy as np
from PIL import Image, ImageOps
import torch

from vision_classifier import assign_groups, training_exclusion_reason


def normal_region_exclusion(region):
    """A removed/moved/retyped box is not evidence of a normal structure.

    Only an explicit local-normal action can supply negative supervision. The
    two legacy structure actions were explicit even before provenance existed;
    generic legacy ``false_positive`` was also produced by ordinary deletion.
    """
    if not isinstance(region, dict):
        return 'invalid_normal_region'
    reason = region.get('reason')
    provenance = region.get('normal_provenance')
    if provenance is not None:
        actions = {'process_hole': 'process_hole', 'normal_structure': 'normal_structure',
                   'false_positive': 'confirmed_false_positive'}
        if (not isinstance(provenance, dict) or provenance.get('source') != 'human_explicit_normal'
                or provenance.get('confirmed') is not True
                or provenance.get('action') != actions.get(reason)
                or reason not in actions):
            return 'invalid_explicit_normal_provenance'
        return ''
    if reason in ('process_hole', 'normal_structure'):
        return ''
    if reason == 'false_positive':
        return 'legacy_false_positive_without_explicit_normal_confirmation'
    return 'normal_region_not_explicitly_confirmed'


def correction_event_semantics(event):
    """Audit the correction without converting any edit event into a negative."""
    explicit = event.get('correction_semantics')
    allowed = {'geometry', 'type', 'normal', 'replacement', 'ambiguous_delete', 'positive_addition'}
    if isinstance(explicit, list) and all(value in allowed for value in explicit):
        return sorted(set(explicit))
    changes = event.get('changes', [])
    result = []
    if any(value in changes for value in ('moved', 'resized', 'geometry')):
        result.append('geometry')
    if any(value in changes for value in ('reclassified', 'type')):
        result.append('type')
    if event.get('kind') == 'added':
        result.append('positive_addition')
    if event.get('kind') in ('deleted', 'false_positive'):
        # Even a named legacy normal action is independently checked against
        # negative-region provenance. Audit labels never authorize supervision.
        result.append('normal' if event.get('reason') in ('process_hole', 'normal_structure')
                      else 'ambiguous_delete')
    return sorted(set(result))


def feedback_policy(feedback, label_raw, has_draft=False):
    if feedback.get('type') != 'human_reported_local_false_positive' or feedback.get('label_scope') != 'region_only':
        return 'not_explicit_human_region_feedback'
    if feedback.get('whole_image_normal') is not False or not feedback.get('human_statement'):
        return 'ambiguous_normal_region_scope'
    if not label_raw:
        return 'missing_human_classification'
    reason = training_exclusion_reason(json.loads(label_raw), has_draft)
    if reason:
        return reason
    if hashlib.sha256(label_raw.encode()).hexdigest() != feedback.get('original_classification_fingerprint'):
        return 'classification_changed_since_feedback'
    return ''


def heldout_overlap(record, positives, historical):
    """Same case/order/SHA or conservative near duplicate must not train across holdouts."""
    rows = copy.deepcopy(positives + historical + [record])
    for row in rows:
        row['phash'] = int(row['phash'])
    assign_groups(rows)
    group = rows[-1]['group']
    return [r for r in rows[:-1] if r['group'] == group and r.get('split') in ('val', 'validation', 'test')]


def snapshot_annotation_exclusion(feedback, document, version):
    """An HTTP export cannot authorize a draft or a later annotation version."""
    if document:
        if document.get('status') != 'confirmed' or document.get('outcome') == 'unclear':
            return 'annotation_not_current_confirmed_or_unclear'
        if document.get('label_fingerprint') != feedback.get('original_classification_fingerprint'):
            return 'annotation_classification_changed'
        if document.get('source', {}).get('sha256') != feedback.get('source', {}).get('sha256'):
            return 'annotation_source_changed'
    if feedback.get('feedback_kind') == 'annotation_review':
        if not document or version != feedback.get('annotation_version'):
            return 'annotation_changed_since_feedback_export'
    return ''


def prepare_feedback(paths, snapshot, source_root, positives, historical=None):
    from prepare_detector import normalize_boxes, safe_path
    accepted, excluded, included_feedback_ids, region_audit, event_audit = [], [], [], [], []
    entries = []
    for feedback_path in paths:
        payload = json.loads(Path(feedback_path).read_text(encoding='utf-8-sig'))
        if isinstance(payload.get('items'), list):
            for item in payload['items']:
                feedback = item.get('feedback', {})
                entry = {'type': 'human_reported_local_false_positive', 'label_scope': 'region_only',
                    'image_id': item.get('image_id'), 'relative_path': item.get('relative_path'),
                    'source': item.get('source'), 'whole_image_normal': feedback.get('whole_image_normal'),
                    'human_statement': 'Explicitly confirmed annotation correction',
                    'original_classification_fingerprint': feedback.get('label_fingerprint'),
                    'negative_regions': feedback.get('negative_regions', []),
                    'events': feedback.get('events', []),
                    'feedback_id': item.get('feedback_id'),
                    'export_eligible': item.get('training_eligible_by_label_policy') is True,
                    'human_confirmed': feedback.get('status') == 'confirmed',
                    'feedback_kind': feedback.get('kind', 'region_correction'),
                    'annotation_version': feedback.get('annotation_version'),
                    'annotation_version_at_confirmation': feedback.get('annotation_version_at_confirmation'),
                    'coordinate_space': feedback.get('coordinate_space')}
                entries.append((feedback_path, entry))
        else:
            entries.append((feedback_path, payload))
    with closing(sqlite3.connect(Path(snapshot).resolve().as_uri()+'?mode=ro', uri=True)) as db:
        db.row_factory = sqlite3.Row
        root_id = int(db.execute("SELECT value FROM settings WHERE key='root'").fetchone()[0])
        for feedback_path, feedback in entries:
            image_id = feedback.get('image_id')
            try:
                row = db.execute('SELECT i.*,l.data label_data,d.image_id IS NOT NULL has_draft FROM images i '
                    'LEFT JOIN labels l ON l.image_id=i.id LEFT JOIN drafts d ON d.image_id=i.id '
                    'WHERE i.id=? AND i.root_id=?', (image_id, root_id)).fetchone()
                if not row:
                    raise ValueError('image_not_in_current_root')
                if feedback.get('export_eligible') is False or feedback.get('human_confirmed') is False:
                    raise ValueError('feedback_export_not_current_human_confirmed_or_eligible')
                if feedback.get('coordinate_space') not in (None, 'exif_transposed_normalized_xywh'):
                    raise ValueError('feedback_coordinate_space_unsupported')
                reason = feedback_policy(feedback, row['label_data'], bool(row['has_draft']))
                if reason:
                    raise ValueError(reason)
                annotation_row = db.execute('SELECT data,version FROM annotation_documents WHERE image_id=?', (image_id,)).fetchone()
                current_document = json.loads(annotation_row['data']) if annotation_row else None
                reason = snapshot_annotation_exclusion(feedback, current_document,
                    annotation_row['version'] if annotation_row else 0)
                if reason:
                    raise ValueError(reason)
                for event in feedback.get('events', []):
                    if isinstance(event, dict):
                        event_audit.append({'image_id': image_id, 'feedback_id': feedback.get('feedback_id'),
                            'source_box_id': event.get('source_box_id'),
                            'semantics': correction_event_semantics(event)})
                candidate_regions = feedback.get('negative_regions')
                if candidate_regions is None:
                    candidate_regions = [feedback['rejected_region']]
                if not isinstance(candidate_regions, list):
                    raise ValueError('invalid_normal_region_list')
                explicit_regions = []
                for region in candidate_regions:
                    exclusion = normal_region_exclusion(region)
                    region_audit.append({'image_id': image_id, 'feedback_id': feedback.get('feedback_id'),
                        'region_id': region.get('id') if isinstance(region, dict) else None,
                        'reason': region.get('reason') if isinstance(region, dict) else None,
                        'normal_supervision_allowed': not bool(exclusion), 'exclusion': exclusion,
                        'provenance': region.get('normal_provenance') if isinstance(region, dict) else None})
                    if not exclusion:
                        explicit_regions.append(region)
                matching_positive = next((r for r in positives if r['image_id'] == image_id), None)
                # Do not open pixels for ambiguous-only feedback that supplies
                # no current confirmed positive record either.
                if not explicit_regions and not matching_positive:
                    raise ValueError('feedback_has_no_current_confirmed_train_positive_or_explicit_normal_region')
                if not row['available'] or feedback.get('relative_path') != row['relative_path']:
                    raise ValueError('feedback_source_unavailable_or_path_changed')
                path = safe_path(source_root, row['relative_path'])
                raw = path.read_bytes()
                digest = hashlib.sha256(raw).hexdigest()
                if digest != feedback.get('source', {}).get('sha256'):
                    raise ValueError('feedback_source_hash_changed')
                with Image.open(io.BytesIO(raw)) as image:
                    image = ImageOps.exif_transpose(image).convert('RGB')
                    width, height = image.size
                    if [width, height] != [feedback['source'].get('width'), feedback['source'].get('height')]:
                        raise ValueError('feedback_source_geometry_changed')
                    gray = np.asarray(image.convert('L').resize((32, 32)), dtype=np.float32)
                dct = cv2.dct(gray)[:8, :8].flatten()[1:]
                phash = sum(int(value > np.median(dct)) << bit for bit, value in enumerate(dct))
                regions = normalize_boxes([dict(r, problem=r.get('problem', 'explicit_normal_region'))
                                           for r in explicit_regions], width, height)
                case = json.loads((path.parent.parent.parent/'case.json').read_text(encoding='utf-8'))
                orders = [field+':'+str(order[field]).strip() for order in case.get('related_orders', [])
                    for field in ('h3_object_id', 'production_no', 'crm_no', 'contract_no') if order.get(field)]
                orders += ['h3_object_id:'+str(v) for v in case.get('order_link', {}).get('ids', []) if v]
                label = json.loads(row['label_data'])
                record = {'image_id': image_id, 'relative_path': row['relative_path'], 'full_path': str(path),
                    'case_id': row['case_id'], 'sha256': digest, 'width': width, 'height': height,
                    'aspect': width/height, 'phash': str(phash), 'order_keys': sorted(set(orders)),
                    'label_fingerprint': feedback['original_classification_fingerprint'],
                    'defects': label['defects'], 'image_state': label.get('image_state', ''), 'has_draft': False,
                    'boxes': [], 'negative_rois': [region['xyxy'] for region in regions], 'feedback_only': True,
                    'feedback_provenance_sha256': hashlib.sha256(Path(feedback_path).read_bytes()).hexdigest(),
                    'split': 'train', 'group': 'human-roi:'+digest,
                    'classification_problems_without_boxes': list(label['defects'])}
                overlaps = heldout_overlap(record, positives, (historical or {}).get('records', []))
                if overlaps:
                    raise ValueError('human_roi_overlaps_historical_or_current_holdout')
                if matching_positive:
                    record['boxes'] = copy.deepcopy(matching_positive['boxes'])
                    for roi in record['negative_rois']:
                        for box in record['boxes']:
                            positive = box['xyxy']
                            if min(roi[2], positive[2]) > max(roi[0], positive[0]) and min(roi[3], positive[3]) > max(roi[1], positive[1]):
                                raise ValueError('normal_region_overlaps_confirmed_positive_box')
                if not regions and not matching_positive:
                    raise ValueError('feedback_has_no_current_confirmed_train_positive_or_normal_region')
                if type(feedback.get('feedback_id')) is int:
                    included_feedback_ids.append(feedback['feedback_id'])
                    record['feedback_ids'] = [feedback['feedback_id']]
                if not regions:
                    continue  # Positive corrections are already present in the normal detector dataset.
                existing = next((r for r in accepted if r['image_id'] == image_id), None)
                if existing:
                    existing['negative_rois'] = [list(region) for region in sorted({tuple(region)
                        for region in existing['negative_rois']+record['negative_rois']})]
                    existing.setdefault('feedback_ids', []).extend(record.get('feedback_ids', []))
                    continue
                accepted.append(record)
            except (OSError, ValueError, KeyError, TypeError) as error:
                excluded.append({'image_id': image_id, 'reason': str(error)})
    return {'records': accepted, 'excluded': excluded,
            'included_feedback_ids': sorted(set(included_feedback_ids)),
            'normal_region_semantics': {'schema': 'explicit_human_normal_roi_v2',
                'region_audit': region_audit,
                'correction_events': event_audit,
                'event_semantic_counts': dict(Counter(value for row in event_audit for value in row['semantics'])),
                'input_regions_after_item_policy': len(region_audit),
                'explicit_or_legacy_structure_regions': sum(r['normal_supervision_allowed'] for r in region_audit),
                'excluded_reasons': dict(Counter(r['exclusion'] for r in region_audit if r['exclusion'])),
                'note': 'Semantic permission alone does not bypass source, holdout, geometry or positive-overlap exclusions.'},
            'policy': 'Explicit human normal ROI only; generic legacy deletions never negative supervision; never positive truth or whole-image normal; original image-state/draft/source/fingerprint exclusions unchanged',
            'holdout_policy': 'Reject any feedback whose case/order/SHA/conservative near duplicate overlaps historical or current validation/test',
            'images': len(accepted), 'regions': sum(len(r['negative_rois']) for r in accepted)}


def transform_rois(rois, width, ratio, flipped=False):
    result = []
    for x1, y1, x2, y2 in rois:
        if flipped:
            x1, x2 = width-x2, width-x1
        result.append([x1*ratio, y1*ratio, x2*ratio, y2*ratio])
    return result


def anchor_centers(input_size, device, strides=(8, 16, 32)):
    centers = []
    for stride in strides:
        count = input_size//stride
        y, x = torch.meshgrid(torch.arange(count, device=device), torch.arange(count, device=device), indexing='ij')
        centers.append(torch.stack(((x.flatten()+.5)*stride, (y.flatten()+.5)*stride), dim=1))
    return torch.cat(centers)


def region_objectness_loss(loss, targets, partial_images, roi_batches=None, centers=None, weight=2.,
                           bounded=False):
    """Restore local negatives; optionally bound their post-YOLOX-normalization mass.

    YOLOX divides the summed objectness BCE by max(num_foreground, 1). An ROI-only
    batch therefore must not sum dozens of negative anchors at full weight. In
    bounded mode the auxiliary ROI contribution, after that division, is at most
    ``weight * mean(ROI BCE)``. Positive and ordinary fully boxed supervision is
    unchanged. Unknown regions on partial images remain ignored.
    """
    if not partial_images or targets.ndim != 2 or targets.shape[-1] != 1:
        return loss
    batch = len(partial_images)
    # YOLOX reuses the BCE module for classification. A one-class head has the
    # same last dimension but only foreground rows, not the full anchor grid.
    if centers is not None and targets.shape[0] != batch * len(centers):
        return loss
    if targets.shape[0] % batch:
        raise ValueError('Unexpected flattened objectness anchor count')
    truth = targets.reshape(batch, -1, 1)
    partial = torch.as_tensor(partial_images, dtype=torch.bool, device=targets.device).view(batch, 1, 1)
    scale = ((~partial) | (truth > 0)).to(loss.dtype)
    if roi_batches and any(roi_batches):
        if centers is None or len(centers) != truth.shape[1]:
            raise ValueError('ROI anchor geometry mismatch')
        restored = torch.zeros_like(truth, dtype=torch.bool)
        for index, rois in enumerate(roi_batches):
            normal = torch.zeros(len(centers), dtype=torch.bool, device=targets.device)
            for x1, y1, x2, y2 in rois:
                normal |= (centers[:, 0] >= x1) & (centers[:, 0] < x2) & (centers[:, 1] >= y1) & (centers[:, 1] < y2)
            # Assigned positive anchors are never suppressed or reweighted by normal feedback.
            negative = normal & (truth[index, :, 0] == 0)
            if bounded:
                # Complete images already provide ordinary background loss. Only
                # restore masked unknown anchors inside explicit normal regions.
                restored[index, negative, 0] = partial_images[index]
            else:
                scale[index, negative, 0] = weight
        if bounded and restored.any():
            foreground_denominator = max(float((truth > 0).sum()), 1.)
            scale[restored] = weight * foreground_denominator / int(restored.sum())
    return loss * scale.reshape_as(loss)
