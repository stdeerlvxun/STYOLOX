"""Read-only first gate for human localization labels; never edits the source DB."""
from __future__ import annotations
import argparse
from collections import Counter
from contextlib import closing
import datetime as dt
import hashlib
import json
import math
from pathlib import Path
import sqlite3


def fingerprint(raw):
    return hashlib.sha256((raw or '').encode('utf-8')).hexdigest()


def exclusion(label_raw, has_draft, document):
    label = json.loads(label_raw) if label_raw else {}
    if label.get('image_state', '') not in ('', 'defect'):
        return 'image_state_selected'
    if not label.get('defects'):
        return 'missing_problem_labels'
    if has_draft:
        return 'unconfirmed_classification_edit'
    if document.get('status') != 'confirmed':
        return 'annotation_not_confirmed'
    if document.get('outcome') != 'boxed':
        return 'annotation_not_boxed'
    if document.get('label_fingerprint') != fingerprint(label_raw):
        return 'classification_changed'
    if document.get('coordinate_space') != 'exif_transposed_normalized_xywh':
        return 'unsupported_coordinate_space'
    boxes = document.get('boxes')
    if not isinstance(boxes, list) or not boxes:
        return 'missing_boxes'
    for box in boxes:
        if not isinstance(box, dict) or not isinstance(box.get('problem'), str) or not box['problem']:
            return 'invalid_box_class'
        values = [box.get(k) for k in ('x', 'y', 'w', 'h')]
        if any(type(v) not in (int, float) or not math.isfinite(v) for v in values):
            return 'invalid_coordinates'
        x, y, w, h = values
        if min(x, y) < 0 or min(w, h) <= 0 or x + w > 1.000001 or y + h > 1.000001:
            return 'invalid_coordinates'
    return ''


def audit(snapshot, root=None):
    snapshot = Path(snapshot).resolve(strict=True)
    # mode=ro, query_only: the live DB is never opened by the labeling application here.
    with closing(sqlite3.connect(snapshot.as_uri() + '?mode=ro', uri=True)) as db:
        db.row_factory = sqlite3.Row
        db.execute('PRAGMA query_only=ON')
        root_id = db.execute("SELECT value FROM settings WHERE key='root'").fetchone()[0]
        tables = {r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        counts = {t: db.execute('SELECT COUNT(*) FROM ' + t).fetchone()[0] if t in tables else 0
                  for t in ('labels', 'drafts', 'annotation_documents', 'annotation_history')}
        rows = list(db.execute('''SELECT a.data,i.relative_path,i.available,l.data label_data,d.image_id has_draft
            FROM annotation_documents a JOIN images i ON i.id=a.image_id
            LEFT JOIN labels l ON l.image_id=i.id LEFT JOIN drafts d ON d.image_id=i.id
            WHERE i.root_id=?''', (root_id,))) if 'annotation_documents' in tables else []
    reasons, classes = Counter(), Counter()
    states = Counter()
    verified = candidates = 0
    for row in rows:
        try:
            document = json.loads(row['data'])
            states[str(document.get('status'))] += 1
            reason = exclusion(row['label_data'], row['has_draft'] is not None, document)
        except (ValueError, TypeError, AttributeError):
            reasons['invalid_document'] += 1
            continue
        if not reason and not row['available']:
            reason = 'source_unavailable'
        if not reason and root is not None:
            try:
                from PIL import Image, ImageOps
                base = Path(root).resolve(strict=True)
                relative = row['relative_path'].replace('\\', '/')
                path = (base / relative).resolve(strict=True)
                if not path.is_relative_to(base):
                    raise ValueError('Source escapes the configured root')
                raw = path.read_bytes()
                if hashlib.sha256(raw).hexdigest() != document['source']['sha256']:
                    reason = 'source_hash_changed'
                else:
                    import io
                    with Image.open(io.BytesIO(raw)) as image:
                        size = ImageOps.exif_transpose(image).size
                    if size != (document['source']['width'], document['source']['height']):
                        reason = 'source_dimensions_changed'
                    else:
                        verified += 1
            except (OSError, ValueError, KeyError):
                reason = 'source_verification_failed'
        if reason:
            reasons[reason] += 1
        else:
            candidates += 1
            classes.update(b['problem'] for b in document['boxes'])
    return {
        'schema_version': 1, 'checked_at': dt.datetime.now(dt.timezone.utc).isoformat(),
        'source_snapshot_sha256': hashlib.sha256(snapshot.read_bytes()).hexdigest(),
        'counts': counts, 'current_root_annotation_documents': len(rows),
        'annotation_statuses': dict(states), 'excluded': dict(reasons),
        'policy_candidate_images': candidates, 'candidate_boxes_by_class': dict(classes),
        'source_verified_images': verified, 'source_verification_requested': root is not None,
        'readiness': 'not_ready_no_eligible_boxes' if not candidates else 'needs_group_split_and_class_coverage_audit',
        'ready_for_training': False,
        'note': 'This is the annotation eligibility gate, not a complete detector dataset preparation or training pipeline.',
        'training_started': False, 'live_service_modified': False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--snapshot', type=Path, required=True)
    parser.add_argument('--root', type=Path, help='Original image root; enables hash and EXIF dimensions checks')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.resolve() == args.snapshot.resolve():
        parser.error('Output must not overwrite the snapshot')
    result = audit(args.snapshot, args.root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__':
    main()
