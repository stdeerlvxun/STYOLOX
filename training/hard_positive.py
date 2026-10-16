"""Bounded extra practice for verified human corrections; no training dependencies."""
import json
import math


POLICY = 'verified_human_added_or_moved_iou_lt_0.5_v1'
COORDINATES = 'exif_transposed_normalized_xywh'


def original_proposal(db, image_id, proposal_id, tables):
    """Read the exact linked proposal from the frozen snapshot, including history."""
    if not isinstance(proposal_id, str) or not proposal_id:
        return None
    for table in ('annotation_proposals', 'annotation_proposal_history'):
        if table not in tables:
            continue
        row = db.execute(f'SELECT data FROM {table} WHERE image_id=? AND proposal_id=?',
                         (image_id, proposal_id)).fetchone()
        if row:
            try:
                data = json.loads(row[0])
                return dict(data, proposal_id=proposal_id) if isinstance(data, dict) else None
            except (TypeError, ValueError):
                return None
    return None


def _xywh(box):
    values = [box.get(k) for k in ('x', 'y', 'w', 'h')]
    if any(type(v) not in (int, float) or not math.isfinite(v) for v in values):
        raise ValueError('Invalid correction coordinates')
    x, y, w, h = values
    if min(x, y) < 0 or min(w, h) <= 0 or x+w > 1.000001 or y+h > 1.000001:
        raise ValueError('Correction outside image')
    return values


def _iou(left, right):
    x, y, w, h = _xywh(left)
    xx, yy, ww, hh = _xywh(right)
    overlap = max(0, min(x+w, xx+ww)-max(x, xx)) * max(0, min(y+h, yy+hh)-max(y, yy))
    return overlap / (w*h + ww*hh - overlap)


def hard_positive_metadata(document, proposal):
    """Missing/ambiguous provenance produces no hardness claim, not a new exclusion."""
    if not isinstance(proposal, dict) or document.get('status') != 'confirmed' or document.get('outcome') != 'boxed':
        return None
    if document.get('origin') != 'machine_assisted_human_confirmed':
        return None
    for key in ('proposal_id', 'model_id', 'method'):
        if not proposal.get(key) or document.get(key) != proposal[key]:
            return None
    if any(obj.get('coordinate_space') != COORDINATES for obj in (document, proposal)):
        return None
    source, original_source = document.get('source', {}), proposal.get('source', {})
    if not isinstance(source, dict) or not isinstance(original_source, dict):
        return None
    if any(not source.get(key) or source.get(key) != original_source.get(key)
           for key in ('sha256', 'width', 'height')):
        return None
    originals = proposal.get('boxes', [])
    if not isinstance(originals, list) or not originals or any(not isinstance(box, dict) or box.get('origin') != 'model' or
        box.get('proposal_id') != proposal['proposal_id'] or not box.get('id') for box in originals):
        return None
    by_id = {box['id']: box for box in originals}
    if len(by_id) != len(originals):
        return None
    added, moved = [], []
    try:
        for box in originals:
            _xywh(box)
        for box in document.get('boxes', []):
            _xywh(box)
            identity = box.get('source_box_id', box.get('id'))
            if identity in by_id:
                if box.get('origin') == 'model' and box.get('proposal_id') == proposal['proposal_id']:
                    iou = _iou(box, by_id[identity])
                    if iou < .5:
                        moved.append({'box_id': box['id'], 'source_box_id': identity, 'original_iou': iou})
            elif (box.get('origin') in (None, 'human') and box.get('id') and
                  not box.get('source_box_id') and not box.get('proposal_id')):
                added.append(box['id'])
    except (KeyError, TypeError, ValueError):
        return None
    return {'policy': POLICY, 'verified_proposal': True, 'proposal_id': proposal['proposal_id'],
            'model_id': proposal['model_id'], 'is_hard': bool(added or moved),
            'added_box_ids': added, 'moved_boxes': moved}


def hard_positive_repeats(records):
    """Return at most one extra row per eligible train image, capped at half of train."""
    train = [r for r in records if r.get('split') == 'train']
    candidates = {}
    for row in train:
        metadata = row.get('hard_positive') or {}
        if (row.get('has_draft') or row.get('image_state') not in (None, '', 'defect') or
            not row.get('defects') or not row.get('boxes') or row.get('feedback_only') or
            row.get('annotation_status') != 'confirmed' or row.get('annotation_outcome') != 'boxed' or
            metadata.get('policy') != POLICY or metadata.get('verified_proposal') is not True or
            metadata.get('is_hard') is not True or not metadata.get('proposal_id') or
            metadata.get('proposal_id') != row.get('proposal_id')):
            continue
        if metadata.get('added_box_ids') or metadata.get('moved_boxes'):
            candidates[row['image_id']] = row
    ranked = sorted(candidates.values(), key=lambda r: (
        -(len(r['hard_positive']['added_box_ids']) + len(r['hard_positive']['moved_boxes'])),
        str(r['image_id'])))
    selected = ranked[:len(train)//2]
    return selected, {'policy': POLICY, 'original_train_images': len(train),
        'eligible_hard_images': len(ranked), 'extra_images_per_epoch': len(selected),
        'extra_image_ids': [r['image_id'] for r in selected], 'extra_limit': len(train)//2,
        'max_positive_appearances_per_image_per_epoch': 2,
        'validation_extra_images': 0, 'test_extra_images': 0,
        'ranking': 'more human added or substantially moved boxes first; stable image ID tie-break',
        'old_moved_regions_used_as_normal': False}
