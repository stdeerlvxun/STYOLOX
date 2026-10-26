"""Bounded, independently scheduled review suggestions from the validated detector.

This module never trains, edits classifications, or refreshes an existing proposal.
GPU inference runs in the configured environment, through the silent worker launcher.
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
import urllib.request
import uuid

import correction_cycle as worker

SELECTION_POLICY = 'all_unreviewed_typed_images_v3'


def settings(config):
    result = {'enabled': config.get('review_refill_enabled', False),
              'low_water': config.get('review_refill_low_water', 10),
              'batch_size': config.get('review_refill_batch_size', 50),
              'scan_limit': config.get('review_refill_scan_limit', 200),
              'location_first': config.get('review_refill_location_first', True)}
    if type(result['enabled']) is not bool or type(result['location_first']) is not bool:
        raise ValueError('Review-refill switches must be booleans')
    for key, upper in (('low_water', 50), ('batch_size', 50), ('scan_limit', 200)):
        if type(result[key]) is not int or not 1 <= result[key] <= upper:
            raise ValueError('Invalid bounded review-refill setting: '+key)
    if result['scan_limit'] < result['batch_size']:
        raise ValueError('Review-refill scan limit cannot be smaller than batch size')
    return result


def file_hash(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for part in iter(lambda: stream.read(1024*1024), b''):
            value.update(part)
    return value.hexdigest()


def api(config, path, body=None):
    headers = {}
    if body is not None:
        headers = {'Content-Type': 'application/json',
                   'X-Local-Token': api(config, '/api/state')['token']}
    request = urllib.request.Request(config['api_url'].rstrip('/')+path,
        data=None if body is None else json.dumps(body, ensure_ascii=False).encode(), headers=headers)
    with urllib.request.urlopen(request, timeout=60) as response:
        raw = response.read(32*1024*1024+1)
    if len(raw) > 32*1024*1024:
        raise ValueError('Review-refill API response exceeds limit')
    return json.loads(raw)


def select_refill_rows(snapshot):
    """Suggest locations on all unreviewed images without changing training eligibility."""
    db = sqlite3.connect(Path(snapshot).resolve().as_uri()+'?mode=ro', uri=True)
    db.row_factory = sqlite3.Row
    try:
        root = int(db.execute("SELECT value FROM settings WHERE key='root'").fetchone()[0])
        rows = db.execute('SELECT i.id image_id,i.relative_path,i.sha256,i.reported_at,i.size,i.mtime_ns,'
            'i.available,l.data,d.image_id IS NOT NULL has_draft,a.image_id IS NOT NULL has_annotation,'
            'p.image_id IS NOT NULL has_proposal,'
            'EXISTS(SELECT 1 FROM annotation_history h WHERE h.image_id=i.id) has_annotation_history,'
            'EXISTS(SELECT 1 FROM annotation_feedback f WHERE f.image_id=i.id) has_feedback,'
            'EXISTS(SELECT 1 FROM annotation_proposal_history h WHERE h.image_id=i.id) has_proposal_history,'
            "EXISTS(SELECT 1 FROM error_reviews e WHERE e.image_id=i.id AND e.status='reviewed') has_error_review,"
            "EXISTS(SELECT 1 FROM model_reviews m WHERE m.image_id=i.id AND m.decision='unrelated' AND m.status='accepted') has_unrelated_review "
            'FROM images i LEFT JOIN labels l ON l.image_id=i.id '
            'LEFT JOIN drafts d ON d.image_id=i.id LEFT JOIN annotation_documents a ON a.image_id=i.id '
            'LEFT JOIN annotation_proposals p ON p.image_id=i.id WHERE i.root_id=? '
            'ORDER BY i.reported_at DESC,i.relative_path DESC,i.id DESC', (root,)).fetchall()
        selected, excluded = [], Counter()
        for record in rows:
            row = dict(record); raw = row.pop('data') or ''; label = json.loads(raw) if raw else {}
            reason = ('image_state_selected' if label.get('image_state', '') not in ('', 'defect')
                else 'confirmed_unrelated_review' if row['has_unrelated_review']
                else 'unconfirmed_edit' if row['has_draft']
                else 'existing_human_annotation' if row['has_annotation'] or row['has_annotation_history'] or row['has_feedback']
                else 'existing_human_error_review' if row['has_error_review']
                else 'existing_machine_proposal' if row['has_proposal'] or row['has_proposal_history']
                else 'source_unavailable' if not row['available'] else '')
            if reason:
                excluded[reason] += 1; continue
            row.update(problems=list(dict.fromkeys(label.get('defects', []))),
                       label_fingerprint=hashlib.sha256(raw.encode()).hexdigest(), root_id=root)
            selected.append(row)
        return selected, {'images_in_root': len(rows), 'eligible_images': len(selected),
                          'selection_policy': SELECTION_POLICY,
                          'excluded': dict(excluded)}
    finally:
        db.close()


def attempt_key(model_sha, row, source_root):
    identity = [SELECTION_POLICY, model_sha, row['image_id'], row['label_fingerprint'], str(source_root),
                row.get('root_id'), row['relative_path'], row.get('sha256'),
                row.get('size'), row.get('mtime_ns')]
    return hashlib.sha256(json.dumps(identity, ensure_ascii=False).encode()).hexdigest()


def normalize_location_predictions(predictions, problems, max_boxes=10):
    """Unknown issue names remain blank until a human chooses them; geometry is reusable."""
    from generate_detector_proposals import normalize_predictions
    names = set(problems)
    context = next(iter(names)) if len(names) == 1 else ''
    return normalize_predictions(predictions, {context}, max_boxes=max_boxes, class_agnostic=True)


def safe_source_path(root, relative):
    relative = str(relative).replace('\\', '/')
    parts = PurePosixPath(relative)
    if parts.is_absolute() or '..' in parts.parts or not parts.parts or ':' in relative:
        raise ValueError('Unsafe relative image path')
    destination = (Path(root)/Path(*parts.parts)).resolve()
    if not destination.is_relative_to(Path(root).resolve()):
        raise ValueError('Image path escapes source directory')
    return destination


def generate(config, snapshot, output, attempts_path, model=None):
    from PIL import Image, ImageOps
    from typed_proposals import normalize_typed_predictions
    policy = settings(config); run = Path(config['current_detector_run']); output = Path(output)
    if output.exists():
        raise ValueError('Refill manifest already exists')
    model_sha = file_hash(run/'detector.pt')
    metrics = worker.read_json(run/'metrics.json') if (run/'metrics.json').exists() else {}
    if metrics.get('model_sha256', model_sha) != model_sha:
        raise ValueError('Current detector checkpoint differs from validated model metadata')
    state = worker.read_json(attempts_path) if Path(attempts_path).exists() else {}
    attempts = state.get('attempts', {}) if state.get('model_sha256') == model_sha else {}
    rows, selection = select_refill_rows(snapshot)
    now = time.time()
    def untried(row):
        entry = attempts.get(attempt_key(model_sha, row, config['source_root']))
        # Imported positives are excluded by the snapshot. A generated but not
        # imported manifest may safely be retried after an interrupted process.
        return (not entry or entry.get('status') == 'proposed'
                or entry.get('status') == 'error' and entry.get('retry_after', 0) <= now)
    candidates = [row for row in rows if untried(row)]
    result = {'schema_version': 1, 'model_id': 'yolox-human-boxes-'+model_sha[:12],
        'method': 'yolox_typed_problem_suggestions_v3',
        'source_checkpoint_sha256': model_sha, 'source_snapshot_sha256': file_hash(snapshot),
        'requires_human_review': True, 'unreviewed_proposals_training_eligible': False,
        'refresh_pending': False, 'classification_results_modified': False, 'source_images_modified': False,
        'coordinate_space': 'exif_transposed_normalized_xywh', 'location_first_suggestions': False,
        'selection_policy': SELECTION_POLICY, 'problem_name_source': 'model_prediction_with_per_box_provenance',
        'created_at': dt.datetime.now(dt.timezone.utc).isoformat(), 'selection': selection, 'items': [], 'errors': []}
    if candidates and model is None:
        from train_detector import load_detector
        model = load_detector(run, device='cuda')
    if candidates and getattr(model,'class_agnostic',False):
        raise ValueError('A typed detector is required; retain the current issue-class model before refill')
    processed = 0; empty = 0
    for row in candidates[:policy['scan_limit']]:
        if len(result['items']) >= policy['batch_size']:
            break
        key = attempt_key(model_sha, row, config['source_root']); processed += 1
        entry = {'image_id': row['image_id'], 'label_fingerprint': row['label_fingerprint'], 'checked_at': now}
        try:
            path = safe_source_path(config['source_root'], row['relative_path'])
            before = path.stat(); raw = path.read_bytes(); after = path.stat()
            if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                raise ValueError('Source changed during read')
            sha = hashlib.sha256(raw).hexdigest()
            if row.get('sha256') and row['sha256'] != sha:
                raise ValueError('Source hash changed from snapshot')
            with Image.open(io.BytesIO(raw)) as opened:
                image = ImageOps.exif_transpose(opened).convert('RGB')
            cutoff = config.get('fixed_threshold')
            predictions = model.predict_image(image, **({'threshold': cutoff} if cutoff is not None else {}))
            boxes = normalize_typed_predictions(predictions,model_sha,
                class_agnostic=bool(getattr(model,'class_agnostic',False)))
            entry.update(status='proposed' if boxes else 'no_boxes', source_sha256=sha)
            if boxes:
                result['items'].append({'image_id': row['image_id'], 'relative_path': row['relative_path'],
                    'label_fingerprint': row['label_fingerprint'],
                    'source': {'sha256': sha, 'width': image.width, 'height': image.height}, 'boxes': boxes,
                    'note': '问题点与位置来自检测模型的真实预测，请审核纠正。模型分数不等于准确率；已有人工分类保留。'})
            else:
                empty += 1
        except (OSError, ValueError, KeyError) as error:
            entry.update(status='error', retry_after=now+3600, error=str(error)[:500])
            result['errors'].append({'image_id': row['image_id'], 'error': str(error)[:500]})
        attempts[key] = entry
    if file_hash(run/'detector.pt') != model_sha:
        raise ValueError('Detector checkpoint changed during inference')
    result['summary'] = {'eligible_images': len(rows), 'eligible_for_current_model': len(rows),
        'already_attempted_images': len(rows)-len(candidates), 'attempted_images': processed,
        'remaining_eligible_images': max(0, len(candidates)-processed), 'generated_images': len(result['items']),
        'generated_boxes': sum(len(item['boxes']) for item in result['items']), 'images_without_boxes': empty,
        'typed_boxes':sum(bool(box['problem']) for item in result['items'] for box in item['boxes']),
        'unassigned_boxes':sum(not box['problem'] for item in result['items'] for box in item['boxes']),
        'errors': len(result['errors']), 'scan_limit': policy['scan_limit'], 'batch_size': policy['batch_size']}
    worker.atomic_json(output, result)
    worker.atomic_json(attempts_path, {'model_sha256': model_sha, 'selection_policy': SELECTION_POLICY, 'attempts': attempts})
    return result


def publish_status(config, report):
    from runtime_report import publish_inference
    publish_inference(config, report)
    policy = settings(config)
    body = {'enabled': policy['enabled'], 'low_water': policy['low_water'], 'batch_size': policy['batch_size'],
            'status': report['status'], 'detail': str(report.get('detail', report.get('error', '')))[:1000]}
    for key in ('pending', 'generated_images', 'generated_boxes', 'eligible_images', 'attempted_images', 'batch_id'):
        if key in report:
            body[key] = report[key]
    try:
        api(config, '/api/review-batches/worker', body)
    except Exception as error:
        report['status_report_error'] = str(error)


def finish_import(config, state, state_path, report):
    pending = state.get('pending_import')
    if pending:
        remote = PurePosixPath(config['remote_base'])/'reviews'/pending['batch_id']
        worker.ssh(config, ['mkdir', '-p', remote])
        worker.transfer(config, pending['manifest'], config['ssh_alias']+':'+str(remote/'proposals.json'))
        # Reuse the original server-side report after an uncertain SSH outcome.
        code = "from pathlib import Path\nimport subprocess,sys\np=Path(sys.argv[1])\nif p.exists(): print(p.read_text(encoding='utf-8'))\nelse: subprocess.run(sys.argv[2:],check=True)"
        raw = worker.ssh(config, [config['remote_python'], '-c', code, remote/'import-report.json',
            config['remote_python'], PurePosixPath(config['remote_base'])/'current'/'deployment'/'import_box_proposals.py',
            '--data', config['remote_data'], '--manifest', remote/'proposals.json', '--output', remote/'import-report.json'],
            timeout=300, log=Path(pending['run'])/'import.log')
        imported = json.loads(raw)
        if imported.get('human_data_unchanged') is not True or imported.get('annotation_data_unchanged') is not True:
            raise ValueError('Import did not confirm preservation of human classifications and boxes')
        ids = imported.get('imported_proposal_ids', [])
        if len(ids) != imported.get('imported', 0):
            raise ValueError('Import report lacks the actual inserted proposal IDs')
        state.pop('pending_import')
        state['pending_registration'] = {**pending, 'proposal_ids': ids, 'import': imported}
        worker.atomic_json(state_path, state)
    pending = state.get('pending_registration')
    if pending:
        if pending['proposal_ids']:
            api(config, '/api/review-batches/register', {'batch_id': pending['batch_id'],
                'proposal_ids': pending['proposal_ids'], 'metadata': pending['metadata']})
        report.update(pending['summary'], batch_id=pending['batch_id'], import_result=pending['import'],
                      status='refilled' if pending['proposal_ids'] else 'no_candidates_imported')
        report['imported_images'] = len(pending['proposal_ids'])
        state.pop('pending_registration'); worker.atomic_json(state_path, state)
    return report


def refill_cycle(config_path, check_only=False):
    config = worker.validate_config(worker.read_json(config_path)); policy = settings(config)
    report = {'status': 'disabled', 'enabled': policy['enabled'], 'check_only': check_only,
              'generated_images': 0, 'generated_boxes': 0, 'attempted_images': 0}
    if not policy['enabled']:
        return report
    root = Path(config['runs_root']); state_path = root/'review-refill-state.json'
    if check_only:
        report.update(pending=api(config, '/api/annotation-proposals/state')['pending'], status='checked')
        return report
    try:
        with worker.ExclusiveLock(root/'.review-refill.lock'), worker.ExclusiveLock(root/'.correction-cycle.lock'):
            state = worker.read_json(state_path) if state_path.exists() else {}
            if state.get('pending_import') or state.get('pending_registration'):
                finish_import(config, state, state_path, report)
            else:
                pending = api(config, '/api/annotation-proposals/state')['pending']; report['pending'] = pending
                if type(pending) is not int or pending < 0:
                    raise ValueError('Invalid pending review count')
                if pending >= policy['low_water'] and not config.get('continuous_inference_enabled', False):
                    report['status'] = 'queue_sufficient'
                else:
                    available, reason = worker.gpu_available(config, inference=True)
                    if not available:
                        report.update(status='waiting_for_gpu', detail=reason)
                    else:
                        batch_id = 'refill-'+dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'-'+uuid.uuid4().hex[:6]
                        run = root/batch_id; run.mkdir(parents=True, exist_ok=False)
                        report.update(status='generating', batch_id=batch_id, detail='正在独立识别新图片，人工审核进度不影响执行。')
                        publish_status(config, report)
                        snapshot = run/'labels.sqlite3'; manifest = run/'proposals.json'
                        worker.fetch_snapshot(config, batch_id, snapshot, log=run/'snapshot.log')
                        worker.run_process([config['python_executable'], Path(config['training_root'])/'review_refill.py',
                            '--config', config_path, '--infer', '--snapshot', snapshot, '--output', manifest,
                            '--attempts', root/'review-refill-attempts.json'], timeout=600,
                            cwd=config['training_root'], log=run/'inference.log')
                        generated = worker.read_json(manifest); summary = generated['summary']
                        report.update(summary, eligible_images=summary['remaining_eligible_images'], batch_id=batch_id)
                        if generated['items']:
                            state['pending_import'] = {'batch_id': batch_id, 'manifest': str(manifest), 'run': str(run),
                                'summary': {**summary, 'eligible_images': summary['remaining_eligible_images']},
                                'metadata': {'source_model': generated['source_checkpoint_sha256'],
                                             'model_id': generated['model_id'], 'generated_at': generated['created_at']}}
                            worker.atomic_json(state_path, state)
                            finish_import(config, state, state_path, report)
                        else:
                            report['status'] = 'no_more_eligible_images' if not summary['remaining_eligible_images'] else 'no_boxes_this_round'
                        worker.atomic_json(run/'report.json', report)
            report['pending'] = api(config, '/api/annotation-proposals/state')['pending']
            report['detail'] = ('本轮尝试 '+str(report.get('attempted_images', 0))+' 张，生成 '+str(report.get('generated_images', 0))+
                ' 张；当前模型尚未尝试的合格图片 '+str(report.get('eligible_images', '待统计'))+' 张。') if 'eligible_images' in report else report.get('detail', '')
            worker.atomic_json(root/'review-refill-latest.json', report)
    except worker.AlreadyRunning:
        report['status'] = 'waiting_for_worker'
    except Exception as error:
        report.update(status='error', error=str(error))
        worker.atomic_json(root/'review-refill-latest.json', report)
    publish_status(config, report)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--check-only', action='store_true')
    parser.add_argument('--infer', action='store_true')
    parser.add_argument('--snapshot', type=Path); parser.add_argument('--output', type=Path)
    parser.add_argument('--attempts', type=Path)
    arguments = parser.parse_args()
    if arguments.infer:
        result = generate(worker.validate_config(worker.read_json(arguments.config)), arguments.snapshot,
                          arguments.output, arguments.attempts)['summary']
    else:
        result = refill_cycle(arguments.config, arguments.check_only)
    print(json.dumps(result, ensure_ascii=False), flush=True)
