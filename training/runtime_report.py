"""Best-effort operational telemetry; never participates in model promotion."""
import datetime as dt
import json
from pathlib import Path
import threading
import urllib.request
from contextlib import contextmanager


def timestamp():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def send(config, kind, value):
    if not config.get('api_url'):
        return False
    try:
        base = config['api_url'].rstrip('/')
        with urllib.request.urlopen(base+'/api/state', timeout=3) as response:
            state = json.load(response)
        body = json.dumps({'kind': kind, 'root_id': state['root']['id'], 'value': value}).encode('utf-8')
        request = urllib.request.Request(base+'/api/automation/report', body,
            headers={'Content-Type': 'application/json', 'X-Local-Token': state['token']})
        with urllib.request.urlopen(request, timeout=3) as response:
            return json.load(response).get('ok') is True
    except (OSError, ValueError, KeyError):
        # An older/offline service must not stop inference or training.
        return False


def training_value(report, run=None):
    keys = ('status','run_id','pending_confirmed_images','minimum_confirmed_images',
            'training_completed','promoted','promotion_reason','epoch','max_epochs',
            'validation_images','test_images','started_at')
    value = {key: report[key] for key in keys if key in report}
    value['updated_at'] = timestamp()
    reason = report.get('error') or report.get('reason')
    if reason:value['reason'] = str(reason)[:1000]
    if 'trainable_confirmed_images_after_snapshot' in report:
        value['trainable_confirmed_images'] = report['trainable_confirmed_images_after_snapshot']
    if run and report.get('status') == 'training':
        try:
            history = json.loads((Path(run)/'detector'/'training-history.json').read_text(encoding='utf-8'))
            if history:value['epoch'] = int(history[-1]['epoch'])
        except (OSError, ValueError, KeyError, TypeError):
            pass
    return value


def publish_training(config, report, run=None):
    from correction_cycle import atomic_json
    if report.get('status') == 'skipped_another_worker_running':
        return
    value = training_value(report, run)
    try:
        atomic_json(Path(config['runs_root'])/'training-runtime.json', value)
    except OSError:
        pass
    send(config, 'training', value)


@contextmanager
def stage(config, report, status, run=None):
    report.update(status=status, started_at=timestamp())
    snapshot = dict(report)
    publish_training(config, snapshot, run)
    stopped = threading.Event()
    def heartbeat():
        while not stopped.wait(20):
            try:publish_training(config, snapshot, run)
            except (OSError, ValueError):pass
    thread = threading.Thread(target=heartbeat, daemon=True, name='training-progress')
    thread.start()
    try:
        yield
    finally:
        stopped.set()
        thread.join(timeout=7)


COUNTERS = ('attempted_images','generated_images','generated_boxes','images_without_boxes','errors','imported_images')


def inference_value(config, report):
    """Each completed batch contributes once; retries replace its receipt."""
    from correction_cycle import atomic_json
    root = Path(config['runs_root'])
    path = root/'refill-statistics.json'
    ledger = json.loads(path.read_text(encoding='utf-8')) if path.exists() else {}
    bootstrap = ledger.get('schema_version') != 2
    if bootstrap:ledger = {'schema_version': 2, 'batches': {}}
    batches = ledger['batches']
    # Backfill existing immutable batch reports on first use.
    records = []
    if bootstrap:
        for file in sorted(root.glob('refill-*/report.json')):
            try:
                item = json.loads(file.read_text(encoding='utf-8'))
                item['_reported_at'] = dt.datetime.fromtimestamp(file.stat().st_mtime, dt.timezone.utc).isoformat()
                records.append(item)
            except (OSError, ValueError):continue
    records.append(report)
    for item in records:
        identity = item.get('batch_id')
        if not identity or not item.get('attempted_images') or item.get('status') == 'generating':
            continue
        values = {key:item[key] for key in COUNTERS if type(item.get(key)) is int}
        values['reported_at'] = batches.get(identity, {}).get('reported_at') or item.get('_reported_at') or timestamp()
        batches[identity] = {**batches.get(identity, {}), **values}
    ledger['statistics_since'] = min((row['reported_at'] for row in batches.values()), default=timestamp())
    atomic_json(path, ledger)
    value = {key: report[key] for key in (*COUNTERS,'status','batch_id','already_attempted_images','remaining_eligible_images')
             if key in report}
    value.update(updated_at=timestamp(), statistics_since=ledger['statistics_since'], batches=len(batches))
    for key in COUNTERS:
        # Unknown historical values stay unknown rather than becoming false zeros.
        if batches and all(key in row for row in batches.values()):
            value['total_'+key] = sum(row[key] for row in batches.values())
    successes = [row['reported_at'] for row in batches.values() if row.get('imported_images', 0)>0]
    if successes:value['last_success_at'] = max(successes)
    if report.get('error'):value['reason'] = str(report['error'])[:1000]
    try:
        meta = json.loads((Path(config['current_detector_run'])/'metrics.json').read_text(encoding='utf-8'))
        if meta.get('model_sha256'):value['model_id'] = 'yolox-'+meta['model_sha256'][:12]
    except (OSError, ValueError, KeyError):pass
    return value


def publish_inference(config, report):
    if report.get('status') in ('waiting_for_worker','disabled'):
        return
    try:
        value = inference_value(config, report)
        send(config, 'inference', value)
    except (OSError, ValueError, KeyError) as error:
        report['statistics_error'] = str(error)[:500]
