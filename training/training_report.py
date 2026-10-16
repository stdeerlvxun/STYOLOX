"""Publish compact, reproducible validation comparisons, never predicted human rates."""
import hashlib
import json
from pathlib import Path
import urllib.request


def round_summary(report):
    comparison = report.get('validation_comparison')
    if not report.get('training_completed') or not comparison:
        return None
    if comparison.get('same_validation_images') is not True or comparison.get('test_used') is not False:
        raise ValueError('Cannot publish incomparable validation results')
    sides = {}
    for target, key in [('before', 'base'), ('after', 'candidate')]:
        row = comparison[key]
        metrics = row['class_and_location']
        whole = row.get('whole_image_pass', {}).get('class_and_location', {})
        sides[target] = {'found': metrics['tp'], 'missed': metrics['fn'],
            'false_positive': metrics['fp'], 'total_boxes': metrics['tp'] + metrics['fn'],
            'images': row['images'], 'whole_image_passed': whole.get('passed'),
            'whole_image_pass_rate': whole.get('rate')}
    objective = comparison.get('comparison_objective', 'issue_type_and_location')
    return {'round_id': report['run_id'], 'status': report['status'],
            'created_at': report['checked_at'],
            'reviewed_images': report.get('trainable_confirmed_images_after_snapshot', 0),
            'promoted': report.get('promoted', False), 'promotion_reason': report.get('promotion_reason', ''),
            'baseline_model_sha256': comparison['base']['model_sha256'],
            'candidate_model_sha256': comparison['candidate']['model_sha256'],
            'same_validation_images': True, 'test_used': False,
            'human_pass_rate_after': None,
            'comparison': {'objective': objective, **sides}}


def publish_reports(config):
    """Idempotent retry on subsequent worker ticks, independent of training success."""
    from correction_cycle import atomic_json
    root = Path(config['runs_root'])
    receipt_path = root/'training-report-receipts.json'
    receipts = json.loads(receipt_path.read_text(encoding='utf-8')) if receipt_path.exists() else {}
    published = 0
    token = None
    # Bounded latest rounds; the server keeps the durable 200-round history.
    for path in sorted(root.glob('auto-*/report.json'), reverse=True)[:20]:
        report = json.loads(path.read_text(encoding='utf-8-sig'))
        summary = round_summary(report)
        if summary is None:
            continue
        body = json.dumps(summary, ensure_ascii=False, sort_keys=True).encode('utf-8')
        digest = hashlib.sha256(body).hexdigest()
        if receipts.get(summary['round_id']) == digest:
            continue
        if token is None:
            with urllib.request.urlopen(config['api_url'].rstrip('/')+'/api/state', timeout=15) as response:
                token = json.load(response)['token']
        request = urllib.request.Request(config['api_url'].rstrip('/')+'/api/training-rounds/record', body,
            headers={'Content-Type': 'application/json', 'X-Local-Token': token})
        with urllib.request.urlopen(request, timeout=15) as response:
            result = json.load(response)
        if not result.get('ok') or result.get('round_id') != summary['round_id']:
            raise ValueError('Training summary was not acknowledged')
        receipts[summary['round_id']] = digest
        atomic_json(receipt_path, receipts)
        published += 1
    return {'published': published, 'status': 'synced'}
