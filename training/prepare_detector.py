"""Freeze human-confirmed location annotations without changing the NAS or label store."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from contextlib import closing
import datetime as dt
import hashlib
import io
import json
import math
from pathlib import Path, PurePosixPath
import random
import sqlite3

import cv2
import numpy as np
from PIL import Image, ImageOps

from vision_classifier import NAS_ROOT, assign_groups, training_exclusion_reason, write_json

POLICY = 'human_confirmed_problem_boxes_v1'
COORDINATES = 'exif_transposed_normalized_xywh'
PARTS = ('train', 'validation', 'test')
DEFAULT_HISTORY = Path(r'D:\视觉检测工作区\runs\20261016-retrain-141737\problem\dataset.json')


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def safe_path(root, relative):
    relative = str(relative).replace('\\', '/')
    path = PurePosixPath(relative)
    if path.is_absolute() or '..' in path.parts or ':' in relative or not path.parts:
        raise ValueError('unsafe_source_path')
    target = (Path(root) / Path(*path.parts)).resolve()
    if not target.is_relative_to(Path(root).resolve()):
        raise ValueError('unsafe_source_path')
    return target


def annotation_exclusion(label_raw, doc, has_draft=False):
    if doc.get('status') != 'confirmed':
        return 'annotation_not_confirmed'
    if doc.get('outcome') != 'boxed':
        return 'annotation_not_boxed'
    if not doc.get('boxes'):
        return 'annotation_no_boxes'
    if not label_raw:
        return 'missing_human_classification'
    reason = training_exclusion_reason(json.loads(label_raw), has_draft)
    if reason:
        return reason
    if doc.get('label_fingerprint') != hashlib.sha256(label_raw.encode('utf-8')).hexdigest():
        return 'classification_changed_since_confirmation'
    if doc.get('coordinate_space') != COORDINATES:
        return 'unsupported_coordinate_space'
    return ''


def normalize_boxes(boxes, width, height):
    result, identities = [], set()
    for box in boxes:
        identity, name = box.get('id'), box.get('problem')
        if not isinstance(identity, str) or not identity or identity in identities or not isinstance(name, str) or not name:
            raise ValueError('invalid_box_identity_or_class')
        values = [box.get(k) for k in ('x', 'y', 'w', 'h')]
        if any(type(v) not in (float, int) or not math.isfinite(v) for v in values):
            raise ValueError('invalid_box_coordinates')
        x, y, w, h = values
        if min(x, y) < 0 or min(w, h) <= 0 or x + w > 1.000001 or y + h > 1.000001:
            raise ValueError('box_outside_image')
        w, h = min(w, 1-x), min(h, 1-y)
        if w * width < 1 or h * height < 1:
            raise ValueError('subpixel_box')
        result.append({'id': identity, 'problem': name, 'origin': box.get('origin', 'human'),
                       'proposal_id': box.get('proposal_id'), 'normalized_xywh': [x, y, w, h],
                       'xywh': [x*width, y*height, w*width, h*height],
                       'xyxy': [x*width, y*height, (x+w)*width, (y+h)*height]})
        identities.add(identity)
    return result


def identity_keys(row):
    return {'sha:' + row['sha256'], 'case:' + row['case_id']} | {'order:' + v for v in row['order_keys']}


def split_records(records, historical=None, seed=42):
    """New detector split; preserve image classifier holdouts where identity overlaps."""
    grouped = defaultdict(list)
    for row in records:
        grouped[row['group']].append(row)
    if len(grouped) < 8:
        raise ValueError('Too few independent order groups for train/validation/test (minimum 8)')
    history = defaultdict(set)
    for row in (historical or {}).get('records', []):
        for key in identity_keys(row):
            history[key].add(row['split'])
    fixed, overlap = {}, []
    for group, rows in grouped.items():
        membership = set().union(*(history[key] for row in rows for key in identity_keys(row)))
        if membership:
            # Do not recycle a historic classifier test or validation order into detector training.
            fixed[group] = 'test' if 'test' in membership else 'validation' if 'validation' in membership else 'train'
            overlap.append({'group': group, 'image_ids': [r['image_id'] for r in rows],
                            'classifier_splits': sorted(membership), 'detector_split': fixed[group]})
        for row in rows:
            row['historical_classifier_splits'] = sorted(membership)
    targets = {'train': .7, 'validation': .15, 'test': .15}
    rng = random.Random(seed)
    all_classes = {box['problem'] for row in records for box in row['boxes']}
    best = None
    for _ in range(1000):
        assignments = {group: fixed.get(group) or rng.choices(PARTS, weights=targets.values())[0] for group in grouped}
        counts = Counter(assignments[r['group']] for r in records)
        group_counts = Counter(assignments.values())
        if any(counts[p] == 0 for p in PARTS):
            continue
        # A single example can still be evaluated, but require at least two held-out orders.
        if group_counts['validation'] < 2 or group_counts['test'] < 2 or group_counts['train'] < 3:
            continue
        score = sum(((counts[p]/len(records)-targets[p])/targets[p])**2 for p in PARTS)
        train_classes = {b['problem'] for r in records if assignments[r['group']] == 'train' for b in r['boxes']}
        score += 100 * len(all_classes-train_classes)
        if best is None or score < best[0]:
            best = (score, assignments)
    if best is None:
        raise ValueError('No independent train/validation/test split under historical holdout constraints')
    for row in records:
        row['split'] = best[1][row['group']]
    return {'seed': seed, 'historical_overlap_groups': overlap,
            'historical_holdouts_preserved': True, 'test_used_for_model_selection': False}


def verify_no_leakage(records):
    seen = {}
    for row in records:
        for key in identity_keys(row) | {'group:' + str(row['group'])}:
            if key in seen and seen[key] != row['split']:
                raise ValueError('Cross-split identity leakage: ' + key)
            seen[key] = row['split']


def prepare(run, snapshot=None, source_root=NAS_ROOT, historical_path=DEFAULT_HISTORY, roi_feedback_paths=None):
    run, source_root = Path(run), Path(source_root)
    target = run/'dataset.json'
    if target.exists():
        raise ValueError('Dataset already frozen; use a new run directory')
    snapshot = Path(snapshot) if snapshot else run/'source/labels.sqlite3'
    snapshot_hash = sha256(snapshot)
    with closing(sqlite3.connect(snapshot.resolve().as_uri()+'?mode=ro', uri=True)) as db:
        db.row_factory = sqlite3.Row
        root_row = db.execute("SELECT value FROM settings WHERE key='root'").fetchone()
        if not root_row:
            raise ValueError('Snapshot has no current source root')
        root_id = int(root_row['value'])
        rows = [dict(r) for r in db.execute('SELECT i.id image_id,i.relative_path,i.case_id,i.available,'
            'a.data annotation_data,a.version annotation_version,a.updated_at annotation_updated_at,'
            'l.data label_data,d.image_id IS NOT NULL has_draft FROM annotation_documents a '
            'JOIN images i ON i.id=a.image_id LEFT JOIN labels l ON l.image_id=i.id '
            'LEFT JOIN drafts d ON d.image_id=i.id WHERE i.root_id=? ORDER BY i.id', (root_id,))]
        label_counts = db.execute('SELECT COUNT(*) FROM labels l JOIN images i ON i.id=l.image_id WHERE i.root_id=?', (root_id,)).fetchone()[0]
        draft_counts = db.execute('SELECT COUNT(*) FROM drafts d JOIN images i ON i.id=d.image_id WHERE i.root_id=?', (root_id,)).fetchone()[0]
    records, excluded, cases = [], [], {}
    for row in rows:
        try:
            document = json.loads(row['annotation_data'])
            reason = annotation_exclusion(row['label_data'], document, bool(row['has_draft']))
            if reason:
                excluded.append({'image_id': row['image_id'], 'reason': reason})
                continue
            if not row['available']:
                raise ValueError('source_unavailable')
            if not row['case_id']:
                raise ValueError('missing_case_identity')
            source = document.get('source', {})
            path = safe_path(source_root, row['relative_path'])
            before = path.stat()
            raw = path.read_bytes()
            after = path.stat()
            if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                raise ValueError('source_changed_during_read')
            actual_hash = hashlib.sha256(raw).hexdigest()
            if source.get('sha256') != actual_hash:
                raise ValueError('source_hash_mismatch')
            with Image.open(io.BytesIO(raw)) as image:
                image = ImageOps.exif_transpose(image).convert('RGB')
                width, height = image.size
                if (source.get('width'), source.get('height')) != (width, height):
                    raise ValueError('source_dimensions_mismatch')
                gray = np.asarray(image.convert('L').resize((32, 32)), dtype=np.float32)
            dct = cv2.dct(gray)[:8, :8].flatten()[1:]
            phash = sum(int(value > np.median(dct)) << bit for bit, value in enumerate(dct))
            boxes = normalize_boxes(document['boxes'], width, height)
            case_path = path.parent.parent.parent/'case.json'
            if case_path not in cases:
                cases[case_path] = json.loads(case_path.read_text(encoding='utf-8'))
            case = cases[case_path]
            orders = [field+':'+str(order[field]).strip() for order in case.get('related_orders', [])
                      for field in ('h3_object_id', 'production_no', 'crm_no', 'contract_no') if order.get(field)]
            orders += ['h3_object_id:'+str(v) for v in case.get('order_link', {}).get('ids', []) if v]
            label = json.loads(row['label_data'])
            records.append({k: row[k] for k in ('image_id', 'relative_path', 'case_id', 'annotation_version', 'annotation_updated_at')} |
                {'full_path': str(path), 'sha256': actual_hash, 'width': width, 'height': height,
                 'aspect': width/height, 'phash': phash, 'order_keys': sorted(set(orders)),
                 'label_fingerprint': document['label_fingerprint'], 'annotation_status': 'confirmed',
                 'annotation_outcome': 'boxed', 'coordinate_space': COORDINATES,
                 'image_state': label.get('image_state', ''), 'defects': label['defects'], 'has_draft': False,
                 'boxes': boxes, 'proposal_id': document.get('proposal_id'),
                 'classification_problems_without_boxes': sorted(set(label['defects'])-{b['problem'] for b in boxes})})
        except (OSError, ValueError, KeyError, TypeError) as exc:
            excluded.append({'image_id': row['image_id'], 'reason': str(exc), 'exception_type': type(exc).__name__})
    if sha256(snapshot) != snapshot_hash:
        raise ValueError('Source snapshot changed during preparation')
    assign_groups(records)
    historical = json.loads(Path(historical_path).read_text(encoding='utf-8')) if historical_path and Path(historical_path).exists() else None
    split_audit = split_records(records, historical)
    verify_no_leakage(records)
    names = sorted({box['problem'] for row in records for box in row['boxes']})
    categories = [{'id': i, 'name': name} for i, name in enumerate(names)]
    for row in records:
        row['phash'] = str(row['phash'])  # Keep precision when consumed by JS or other JSON tools.
        for box in row['boxes']:
            box['class_id'] = names.index(box['problem'])
    splits = {part: {'images': sum(r['split'] == part for r in records),
                     'groups': len({r['group'] for r in records if r['split'] == part}),
                     'boxes': sum(len(r['boxes']) for r in records if r['split'] == part),
                     'classes': dict(Counter(b['problem'] for r in records if r['split'] == part for b in r['boxes']))}
              for part in PARTS}
    summary = {'human_classifications': label_counts, 'classification_drafts': draft_counts,
               'annotation_documents': len(rows), 'confirmed_boxed_documents': sum(json.loads(r['annotation_data']).get('status') == 'confirmed' and json.loads(r['annotation_data']).get('outcome') == 'boxed' for r in rows),
               'usable_unique_images': len({r['sha256'] for r in records}), 'eligible_images': len(records),
               'boxes': sum(len(r['boxes']) for r in records), 'independent_groups': len({r['group'] for r in records}),
               'problem_classes': names, 'all_problem_counts': dict(Counter(b['problem'] for r in records for b in r['boxes'])),
               'splits': splits, 'excluded_reasons': dict(Counter(r['reason'] for r in excluded)),
               'classes_without_training_boxes': [n for n in names if not splits['train']['classes'].get(n)],
               'classes_without_validation_boxes': [n for n in names if not splits['validation']['classes'].get(n)],
               'classes_without_test_boxes': [n for n in names if not splits['test']['classes'].get(n)],
               'classifications_missing_boxed_problem': [{'image_id': r['image_id'], 'problems': r['classification_problems_without_boxes']} for r in records if r['classification_problems_without_boxes']],
               'annotation_completeness_note': 'Only confirmed boxes are ground truth; unlabeled visible problems remain possible and require human audit.',
               'source_policy': 'NAS source images read only; no source copies or moves',
               'training_policy': POLICY, 'state_policy': 'Any selected image state is excluded; classification drafts and annotation drafts excluded',
               'responsibility_policy': 'No pixel-based responsibility training',
               'split_policy': 'Linked case/order/exact SHA/conservative near duplicates stay in one split; preserve historical classifier holdouts',
               'historical_reference': str(historical_path) if historical else None,
               'historical_reference_sha256': sha256(historical_path) if historical else None,
               'first_detector_training': True,
               'annotation_selection_bias': 'Some human boxes originated from classifier-generated proposals and error review; this is not a random production test set.',
               'training_ready': True, 'evaluation_scope': 'Small-sample held-out human-confirmed boxes only; no general production accuracy claim'}
    feedback = None
    if roi_feedback_paths:
        from roi_feedback import prepare_feedback
        feedback = prepare_feedback(roi_feedback_paths, snapshot, source_root, records, historical)
        summary['human_normal_roi_feedback'] = {k: v for k, v in feedback.items() if k != 'records'}
    dataset = {'schema_version': 1, 'training_policy': POLICY, 'source_root': str(source_root),
               'snapshot_path': str(snapshot), 'snapshot_sha256': snapshot_hash,
               'created_at': dt.datetime.now(dt.timezone.utc).isoformat(), 'categories': categories,
               'summary': summary, 'split_audit': split_audit, 'records': records, 'excluded': excluded}
    if feedback is not None:
        dataset['negative_feedback_records'] = feedback['records']
        dataset['included_feedback_ids'] = feedback['included_feedback_ids']
    write_json(target, dataset)
    write_json(run/'data-summary.json', summary)
    return dataset


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--snapshot', type=Path)
    parser.add_argument('--source-root', type=Path, default=NAS_ROOT)
    parser.add_argument('--historical', type=Path, default=DEFAULT_HISTORY)
    parser.add_argument('--roi-feedback', type=Path, action='append', default=[])
    parser.add_argument('--feedback-json', type=Path, action='append', default=[],
                        help='Confirmed /api/annotation-feedback/export JSON; revalidated against this snapshot and NAS')
    args = parser.parse_args()
    result = prepare(args.run, args.snapshot, args.source_root, args.historical, args.roi_feedback+args.feedback_json)
    print(json.dumps(result['summary'], ensure_ascii=True), flush=True)
