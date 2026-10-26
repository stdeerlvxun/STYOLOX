"""Bounded, reviewed-batch YOLOX updates; no unreviewed labels or automatic test reuse.

The private config is supplied with --config. --check-only performs HTTP/file reads
only: it does not acquire a lock, create files, query CUDA, train, or publish.
"""
from __future__ import annotations

import argparse
from contextlib import closing
import datetime as dt
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import shlex
import sqlite3
import subprocess
import sys
import urllib.parse
import urllib.request
import uuid


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name+'.'+uuid.uuid4().hex+'.tmp')
    try:
        temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def validate_config(config):
    required = ('python_executable', 'training_root', 'api_url', 'ssh_alias', 'remote_python',
                'remote_base', 'remote_data', 'runs_root', 'source_root',
                'current_detector_run', 'historical_dataset')
    if any(not isinstance(config.get(key), str) or not config[key] for key in required):
        raise ValueError('Missing private worker configuration fields')
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,80}', config['ssh_alias']):
        raise ValueError('ssh_alias must be a saved SSH alias, without shell syntax')
    for key in ('remote_python', 'remote_base', 'remote_data'):
        value = config[key]
        if not value.startswith('/') or '..' in PurePosixPath(value).parts or any(ord(c)<32 for c in value):
            raise ValueError('Remote paths must be absolute without traversal/control characters')
    parsed = urllib.parse.urlsplit(config['api_url'])
    if parsed.scheme not in ('http', 'https') or not parsed.hostname or parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ValueError('api_url must be an HTTP service URL without embedded credentials')
    minimum = config.get('minimum_confirmed_images', 20)
    if type(minimum) is not int or minimum < 20:
        raise ValueError('minimum_confirmed_images must be at least 20 distinct reviewed images')
    recall = config.get('minimum_validation_recall', .5)
    support = config.get('minimum_validation_boxes', 10)
    if type(recall) not in (int, float) or not math.isfinite(recall) or not 0<=recall<=1:
        raise ValueError('minimum_validation_recall must be a finite fraction')
    if type(support) is not int or support<10:
        raise ValueError('minimum_validation_boxes must be at least 10')
    if type(config.get('class_agnostic',False)) is not bool:
        raise ValueError('class_agnostic must be a boolean')
    if type(config.get('correction_focus_enabled',False)) is not bool:
        raise ValueError('correction_focus_enabled must be a boolean')
    if type(config.get('localization_focus_enabled',False)) is not bool:
        raise ValueError('localization_focus_enabled must be a boolean')
    if type(config.get('roi_verifier_enabled',False)) is not bool:
        raise ValueError('roi_verifier_enabled must be a boolean')
    if config.get('correction_focus_enabled') and config.get('class_agnostic'):
        raise ValueError('Correction focus must retain real issue classes')
    if config.get('localization_focus_enabled') and (config.get('class_agnostic') or config.get('correction_focus_enabled') or config.get('fixed_threshold') is None):
        raise ValueError('Localization focus requires typed classes, a fixed threshold, and no classification-first mode')
    cutoff=config.get('fixed_threshold')
    if cutoff is not None and (type(cutoff) not in (int,float) or not math.isfinite(cutoff) or not 0<cutoff<1):
        raise ValueError('fixed_threshold must be a finite fraction')
    for key, default, upper in (('minimum_gpu_free_mib', 2048, None), ('maximum_gpu_utilization_percent', 30, 100)):
        value = config.get(key, default)
        if type(value) not in (int, float) or not math.isfinite(value) or value<0 or (upper is not None and value>upper):
            raise ValueError('Invalid GPU availability limit: '+key)
    return config


def export_feedback(config):
    url = config['api_url'].rstrip('/')+'/api/annotation-feedback/export'
    with urllib.request.urlopen(url, timeout=60) as response:
        raw = response.read(32*1024*1024+1)
    if len(raw)>32*1024*1024:
        raise ValueError('Feedback export exceeds bounded response size')
    payload = json.loads(raw.decode('utf-8-sig'))
    if not isinstance(payload.get('items'), list):
        raise ValueError('Invalid annotation feedback export')
    return payload


def valid_region(region):
    if not isinstance(region, dict):
        return False
    values = [region.get(key) for key in ('x', 'y', 'w', 'h')]
    if any(type(v) not in (int, float) or not math.isfinite(v) for v in values):
        return False
    x, y, w, h = values
    return (x>=0 and y>=0 and w>0 and h>0 and x+w<=1.000001 and y+h<=1.000001
            and region.get('reason') in ('process_hole', 'normal_structure', 'false_positive'))


def eligible_pending(payload):
    """Count images, not button presses; keep the existing image-state policy."""
    from normal_semantics import normal_region_exclusion

    result = []
    for item in payload['items']:
        feedback = item.get('feedback') or {}
        annotation = item.get('annotation') or {}
        label = item.get('original_classification') or {}
        raw_label = item.get('original_classification_raw')
        if (type(item.get('feedback_id')) is not int or type(item.get('image_id')) is not int
            or item.get('training_eligible_by_label_policy') is not True
            or item.get('source_sha256_verified') is not True or item.get('included_in_training') is not False
            or feedback.get('status')!='confirmed' or feedback.get('whole_image_normal') is not False
            or not label.get('defects') or label.get('image_state', '') not in ('', 'defect')
            or not isinstance(raw_label, str)
            or hashlib.sha256(raw_label.encode()).hexdigest()!=feedback.get('label_fingerprint')):
            continue
        if annotation and (annotation.get('status')!='confirmed' or annotation.get('outcome')=='unclear'):
            continue
        positive = (annotation.get('status')=='confirmed' and annotation.get('outcome')=='boxed'
                    and bool(annotation.get('boxes'))
                    and annotation.get('label_fingerprint')==feedback.get('label_fingerprint'))
        negative = any(valid_region(region) and not normal_region_exclusion(region)
                       for region in feedback.get('negative_regions', []))
        if positive or negative:
            result.append(item)
    return sorted(result, key=lambda row: row['feedback_id'])


def batch_signature(items):
    pairs = sorted({(row['image_id'], row['feedback_id']) for row in items})
    return hashlib.sha256(json.dumps(pairs).encode()).hexdigest()


def actual_feedback_ids(dataset, items):
    """Only feedback revalidated into actual train rows can be marked consumed."""
    declared = set(dataset.get('included_feedback_ids', []))
    positives = {r['image_id']:r for r in dataset.get('records', []) if r.get('split')=='train'}
    negatives = dataset.get('negative_feedback_records', [])
    included = []
    for item in items:
        feedback_id = item['feedback_id']
        if feedback_id not in declared:
            continue
        feedback, source = item['feedback'], item['source']
        def matches(record):
            return (record.get('image_id')==item['image_id'] and record.get('split')=='train'
                    and record.get('sha256')==source.get('sha256')
                    and record.get('label_fingerprint')==feedback.get('label_fingerprint'))
        positive = positives.get(item['image_id'], {})
        positive_used = (matches(positive) and bool(positive.get('boxes'))
                         and positive.get('annotation_version')==feedback.get('annotation_version'))
        negative_used = any(matches(record) and bool(record.get('negative_rois'))
                            and feedback_id in record.get('feedback_ids', []) for record in negatives)
        if positive_used or negative_used:
            included.append(feedback_id)
    return sorted(set(included))


def round_seen_feedback_ids(dataset, items):
    """Locally consume a reviewed round, without calling held-out feedback trained."""
    seen = set(actual_feedback_ids(dataset, items))
    records = {row['image_id']:row for row in dataset.get('records', [])
               if row.get('split') in ('train', 'val', 'validation', 'test')}
    for item in items:
        row = records.get(item['image_id'], {})
        feedback = item['feedback']
        if (row.get('sha256')==item['source'].get('sha256') and bool(row.get('boxes'))
            and row.get('label_fingerprint')==feedback.get('label_fingerprint')
            and row.get('annotation_version')==feedback.get('annotation_version')):
            seen.add(item['feedback_id'])
    return sorted(seen)


def promotion_decision(comparison, minimum_recall=.5, minimum_support=10):
    if (comparison.get('same_validation_images') is not True
        or comparison.get('problem_name_mapping') is not True or comparison.get('test_used') is not False):
        return False, 'comparison_not_same_validation_or_not_aligned_by_problem_name'
    base, candidate = comparison.get('baseline', {}), comparison.get('candidate', {})
    values = [report.get(key) for report in (base, candidate) for key in ('f2', 'recall')]
    if any(type(v) not in (int, float) or not math.isfinite(v) or not 0<=v<=1 for v in values):
        return False, 'invalid_validation_metrics'
    if candidate.get('tp', 0)+candidate.get('fn', 0)<minimum_support:
        return False, 'validation_support_below_minimum'
    # The registered minimum belongs to finding locations. Type labeling is an
    # assistive second step; it must improve on the retained model, not borrow
    # an absolute location cutoff. Human automatic-finalization stays disabled.
    recall_for_minimum = candidate['recall']
    if comparison.get('comparison_objective') == 'issue_type_and_location':
        old_location = comparison.get('base', {}).get('location_only', {})
        new_location = candidate.get('location_only', {})
        location_values = [row.get(key) for row in (old_location, new_location) for key in ('recall','f2')]
        if any(type(v) not in (int,float) or not math.isfinite(v) or not 0<=v<=1 for v in location_values):
            return False, 'missing_or_invalid_location_comparison'
        if (new_location['recall'] < old_location['recall'] or new_location['f2'] < old_location['f2']
            or type(new_location.get('fp')) is not int or type(old_location.get('fp')) is not int
            or new_location['fp'] > old_location['fp']):
            return False, 'location_metrics_regressed'
        recall_for_minimum = new_location['recall']
    if recall_for_minimum<minimum_recall:
        return False, 'validation_recall_below_minimum'
    if candidate['recall']+1e-12<base['recall']:
        return False, 'validation_recall_regressed'
    if candidate['f2']<=base['f2']+1e-9:
        return False, 'validation_f2_has_no_strict_improvement'
    return True, 'same_validation_f2_improved_and_recall_did_not_regress'


def filter_manifest(manifest):
    manifest = dict(manifest)
    manifest['items'] = [item for item in manifest['items'] if item.get('boxes') or
                         (item.get('replaces_proposal_id') and item.get('withdraw_if_empty') is True)]
    return manifest


def update_config_reference(config_path, original, dataset_path, detector=None):
    fresh = validate_config(read_json(config_path))
    if (fresh['current_detector_run']!=original['current_detector_run']
        or fresh['historical_dataset'] not in (original['historical_dataset'], str(dataset_path))):
        raise ValueError('Worker baseline configuration changed concurrently; configuration preserved')
    # Every used holdout stays held out, including runs that fail promotion.
    fresh['historical_dataset'] = str(dataset_path)
    if detector is not None:
        fresh['current_detector_run'] = str(detector)
    atomic_json(config_path, fresh)


class AlreadyRunning(Exception):
    pass


class ExclusiveLock:
    """OS-held lock releases even after a crashed worker; no stale PID deletion."""
    def __init__(self, path):
        self.path, self.file = Path(path), None

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.file = self.path.open('a+b')
        self.file.seek(0, os.SEEK_END)
        if self.file.tell()==0:
            self.file.write(b'0');self.file.flush()
        self.file.seek(0)
        try:
            if os.name=='nt':
                import msvcrt
                msvcrt.locking(self.file.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(self.file.fileno(), fcntl.LOCK_EX|fcntl.LOCK_NB)
        except OSError as error:
            self.file.close()
            raise AlreadyRunning() from error
        return self

    def __exit__(self, *_):
        self.file.close()


def run_process(arguments, *, timeout, cwd=None, log=None):
    # A GUI/pythonw parent does not stop console executables (Python, ssh,
    # scp, nvidia-smi) from creating their own terminal on Windows.
    window_options = {}
    if os.name=='nt':
        startup = subprocess.STARTUPINFO()
        startup.dwFlags |= subprocess.STARTF_USESHOWWINDOW
        startup.wShowWindow = subprocess.SW_HIDE
        window_options = {'creationflags':subprocess.CREATE_NO_WINDOW, 'startupinfo':startup}
    result = subprocess.run([str(v) for v in arguments], cwd=cwd, timeout=timeout,
                            encoding='utf-8', errors='replace', capture_output=True,
                            env={**os.environ, 'PYTHONIOENCODING':'utf-8'}, shell=False,
                            **window_options)
    if log:
        Path(log).write_text(result.stdout+'\n'+result.stderr, encoding='utf-8')
    if result.returncode:
        raise RuntimeError('Command failed: '+str(arguments[0])+'; '+result.stderr[-1200:])
    return result.stdout


def gpu_available(config=None, *, inference=False):
    config = config or {}
    try:
        output = run_process(['nvidia-smi', '--query-compute-apps=pid,process_name',
                              '--format=csv,noheader,nounits'], timeout=15)
        resources = run_process(['nvidia-smi', '--query-gpu=memory.free,utilization.gpu',
                                 '--format=csv,noheader,nounits'], timeout=15)
    except (OSError, subprocess.SubprocessError, RuntimeError):
        return False, 'gpu_process_check_unavailable'
    # Under WDDM this query also lists ordinary graphics clients. Unknown or
    # permission-hidden names are not evidence of an active training process.
    compute_names = {'python', 'pythonw', 'torchrun', 'llama-server', 'llama-cli', 'ollama', 'vllm'}
    for line in output.splitlines():
        fields = line.split(',', 1)
        if len(fields)!=2:
            continue
        try:
            pid = int(fields[0].strip())
        except ValueError:
            continue
        name = fields[1].strip().strip('"').replace('\\', '/').rsplit('/', 1)[-1].lower()
        if name.endswith('.exe'):
            name = name[:-4]
        if not inference and pid!=os.getpid() and name in compute_names:
            return False, 'another_gpu_compute_process_is_running'
    try:
        first_gpu = next(line for line in resources.splitlines() if line.strip())
        free_mib, utilization = [float(value.strip()) for value in first_gpu.split(',')]
        if not all(math.isfinite(value) and value>=0 for value in (free_mib, utilization)):
            raise ValueError('Unknown GPU resource counters')
    except (StopIteration, ValueError):
        return False, 'gpu_resource_check_unavailable'
    if free_mib<config.get('minimum_gpu_free_mib', 2048):
        return False, 'insufficient_free_gpu_memory'
    if utilization>config.get('maximum_gpu_utilization_percent', 30):
        return False, 'gpu_utilization_above_limit'
    return True, 'gpu_available'


def ssh(config, arguments, timeout=120, log=None):
    command = ' '.join(shlex.quote(str(v)) for v in arguments)
    return run_process(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15',
                        config['ssh_alias'], command], timeout=timeout, log=log)


def transfer(config, source, destination, timeout=180):
    return run_process(['scp', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15',
                        source, destination], timeout=timeout)


SNAPSHOT_RUN = re.compile(r'(?:auto|refill)-[0-9]{8}T[0-9]{6}Z-[a-f0-9]{6}')
# Sent only by the published worker. The ownership token and directory lock also
# cover SSH timeouts: a still-running snapshot child inherits the lock descriptor.
REMOTE_SNAPSHOT = r'''
import contextlib, hashlib, json, os, pathlib, re, shutil, stat, subprocess, sys

def targets(data, run_id):
    if not re.fullmatch(r'(?:auto|refill)-[0-9]{8}T[0-9]{6}Z-[a-f0-9]{6}', run_id):
        raise ValueError('SNAPSHOT_RUN_INVALID')
    root = pathlib.Path(data)
    if not root.is_absolute() or root.resolve() != root or not root.is_dir():
        raise ValueError('SNAPSHOT_ROOT_INVALID')
    folder = root / 'training-snapshots'
    if folder.exists() and (folder.resolve() != folder or not folder.is_dir()):
        raise ValueError('SNAPSHOT_PARENT_INVALID')
    target = folder / (run_id + '.sqlite3')
    paths = [target, target.with_suffix('.report.json'), target.with_suffix('.labels.json')]
    paths.append(pathlib.Path(str(target) + '-journal'))
    return root, folder, target, folder / ('.' + run_id + '.owner'), paths

def regular(path):
    info = path.lstat()
    if path.resolve() != path or not stat.S_ISREG(info.st_mode):
        raise ValueError('SNAPSHOT_FILE_INVALID')
    return info

@contextlib.contextmanager
def folder_lock(folder):
    import fcntl
    fd = os.open(folder, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        try: fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError: raise ValueError('SNAPSHOT_ACTIVE') from None
        yield fd
    finally:
        os.close(fd)

def require_closed(paths):
    if not paths: return
    identities = {(p.stat().st_dev, p.stat().st_ino) for p in paths}
    proc = pathlib.Path('/proc')
    if not proc.is_dir(): raise ValueError('SNAPSHOT_OPEN_FILES_UNVERIFIED')
    for process in proc.iterdir():
        if not process.name.isdigit() or process.name == str(os.getpid()): continue
        try:
            descriptors = list((process / 'fd').iterdir())
            for descriptor in descriptors:
                try: info = descriptor.stat()
                except FileNotFoundError: continue
                if (info.st_dev, info.st_ino) in identities: raise ValueError('SNAPSHOT_ACTIVE')
        except FileNotFoundError: continue
        except PermissionError: raise ValueError('SNAPSHOT_OPEN_FILES_UNVERIFIED') from None

def digest(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''): value.update(block)
    return value.hexdigest()

def operation(mode, data, run_id, token, python, snapshot_program):
    if not re.fullmatch('[a-f0-9]{32}', token): raise ValueError('SNAPSHOT_OWNER_INVALID')
    root, folder, target, owner, paths = targets(data, run_id)
    if mode not in ('create', 'cleanup'): raise ValueError('SNAPSHOT_MODE_INVALID')
    if not folder.exists():
        if mode == 'cleanup': return {'status': 'absent'}
        folder.mkdir(mode=0o700)
    with folder_lock(folder) as lock_fd:
        if mode == 'create':
            if any(p.exists() or p.is_symlink() for p in [owner, *paths]):
                raise ValueError('SNAPSHOT_ALREADY_EXISTS')
            source = root / 'labels.sqlite3'; size = regular(source).st_size
            disk = os.statvfs(folder)
            if shutil.disk_usage(folder).free < size + 512 * 1024 * 1024:
                raise ValueError('SNAPSHOT_SPACE_LOW')
            if disk.f_files and disk.f_favail < 16: raise ValueError('SNAPSHOT_INODES_LOW')
            with owner.open('x', encoding='ascii') as marker: marker.write(token)
            result = subprocess.run([python, snapshot_program, str(source), str(target)],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=120,
                pass_fds=(lock_fd,), check=False)
            if result.returncode: raise ValueError('SNAPSHOT_CREATE_FAILED')
            regular(target)
            return {'status': 'created', 'sha256': digest(target), 'bytes': target.stat().st_size}
        if not owner.exists():
            if any(p.exists() or p.is_symlink() for p in paths):
                raise ValueError('SNAPSHOT_OWNERSHIP_UNVERIFIED')
            return {'status': 'absent'}
        regular(owner)
        if owner.stat().st_size != 32 or owner.read_text(encoding='ascii') != token:
            raise ValueError('SNAPSHOT_OWNERSHIP_UNVERIFIED')
        present = [p for p in paths if p.exists() or p.is_symlink()]
        for path in present: regular(path)
        require_closed(present)
        for path in present: path.unlink()
        owner.unlink()
        return {'status': 'cleaned'}

if __name__ == '__main__':
    print(json.dumps(operation(*sys.argv[1:])))
'''


def fetch_snapshot(config, run_id, destination, log=None):
    """Download one verified snapshot and remove only this invocation's remote files."""
    if not isinstance(run_id, str) or not SNAPSHOT_RUN.fullmatch(run_id):
        raise ValueError('SNAPSHOT_RUN_INVALID')
    for key in ('remote_data', 'remote_base', 'remote_python'):
        value = config[key]
        if (not isinstance(value, str) or not value.startswith('/')
                or '..' in PurePosixPath(value).parts or any(ord(c) < 32 for c in value)):
            raise ValueError('SNAPSHOT_REMOTE_PATH_INVALID')
    destination = Path(destination)
    if destination.exists() or destination.is_symlink():
        raise ValueError('SNAPSHOT_LOCAL_EXISTS')
    remote = PurePosixPath(config['remote_data']) / 'training-snapshots' / (run_id + '.sqlite3')
    program = PurePosixPath(config['remote_base']) / 'current' / 'deployment' / 'snapshot.py'
    token, original = uuid.uuid4().hex, None
    arguments = [config['remote_python'], '-c', REMOTE_SNAPSHOT]
    tail = [config['remote_data'], run_id, token, config['remote_python'], program]
    try:
        result = json.loads(ssh(config, [*arguments, 'create', *tail], timeout=180, log=log))
        if (result.get('status') != 'created' or not re.fullmatch('[a-f0-9]{64}', result.get('sha256', ''))
                or type(result.get('bytes')) is not int or result['bytes'] <= 0):
            raise ValueError('SNAPSHOT_REMOTE_PROOF_INVALID')
        transfer(config, config['ssh_alias'] + ':' + str(remote), str(destination))
        if destination.is_symlink() or not destination.is_file() or destination.stat().st_size != result['bytes']:
            raise ValueError('SNAPSHOT_DOWNLOAD_INVALID')
        value = hashlib.sha256()
        with destination.open('rb') as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b''): value.update(block)
        if value.hexdigest() != result['sha256']:
            raise ValueError('SNAPSHOT_HASH_MISMATCH')
        with closing(sqlite3.connect(destination.resolve().as_uri() + '?mode=ro&immutable=1', uri=True)) as db:
            if db.execute('PRAGMA quick_check').fetchall() != [('ok',)]:
                raise ValueError('SNAPSHOT_SQLITE_INVALID')
        return destination
    except BaseException as error:
        original = error
        raise
    finally:
        try:
            cleanup = json.loads(ssh(config, [*arguments, 'cleanup', *tail], timeout=60))
            if cleanup.get('status') not in ('absent', 'cleaned'):
                raise ValueError('SNAPSHOT_CLEANUP_UNVERIFIED')
        except Exception as cleanup_error:
            if original is None:
                raise
            if hasattr(original, 'add_note'):
                original.add_note('SNAPSHOT_CLEANUP_FAILED: ' + type(cleanup_error).__name__)


def execute_cycle(config, config_path, payload, items, run, report):
    from runtime_report import stage, publish_training, timestamp
    run.mkdir(parents=True, exist_ok=False)
    report.update(status='preparing', started_at=timestamp())
    publish_training(config, report, run)
    (run/'source').mkdir()
    atomic_json(run/'feedback.json', payload)
    atomic_json(run/'report.json', report)
    run_id = run.name
    deployment = PurePosixPath(config['remote_base'])/'current'/'deployment'
    snapshot = run/'source'/'labels.sqlite3'
    fetch_snapshot(config, run_id, snapshot, log=run/'snapshot.log')
    python, root = config['python_executable'], Path(config['training_root'])
    dataset_path, detector = run/'dataset.json', run/'detector'
    run_process([python, root/'prepare_detector.py', '--run', run, '--snapshot', snapshot,
                 '--source-root', config['source_root'], '--historical', config['historical_dataset'],
                 '--feedback-json', run/'feedback.json'], timeout=300, cwd=root, log=run/'prepare.log')
    dataset = read_json(dataset_path)
    included = actual_feedback_ids(dataset, items)
    trainable_images = len({r['image_id'] for r in items if r['feedback_id'] in included})
    report['trainable_confirmed_images_after_snapshot'] = trainable_images
    report['validation_images'] = sum(row.get('split') in ('val','validation') for row in dataset.get('records', []))
    report['test_images'] = sum(row.get('split') == 'test' for row in dataset.get('records', []))
    if not included:
        report.update(status='waiting_for_trainable_confirmed_images', feedback_ids_actually_trained=[])
        return report
    focused_training = config.get('correction_focus_enabled') or config.get('localization_focus_enabled')
    training_arguments=[python, root/'train_detector.py', '--dataset', dataset_path, '--run', detector,
                 '--adapt-backbone', '--input-size', '640', '--batch-size',
                 '4' if focused_training else '2', '--epochs', '50',
                 '--seconds', '900' if focused_training else '240',
                 '--initialize-from', config['current_detector_run'], '--defer-test']
    if config.get('correction_focus_enabled'):training_arguments.append('--correction-focus')
    if config.get('localization_focus_enabled'):training_arguments.append('--localization-focus')
    if config.get('class_agnostic',False):training_arguments.append('--class-agnostic')
    if config.get('fixed_threshold') is not None:
        training_arguments.extend(['--fixed-threshold',str(config['fixed_threshold'])])
    report['max_epochs'] = 50
    with stage(config, report, 'training', run):
        run_process(training_arguments,timeout=1200 if focused_training else 600,
                    cwd=root,log=run/'train.log')
    if config.get('roi_verifier_enabled') and sum(len(row.get('negative_rois',[]))
            for row in dataset.get('negative_feedback_records',[])) >= 5:
        verified_detector=run/'roi-detector'
        run_process([python,root/'fit_roi_verifier.py','--dataset',dataset_path,
                     '--base-run',detector,'--run',verified_detector],timeout=300,cwd=root,log=run/'roi-verifier.log')
        detector=verified_detector
    metrics = read_json(detector/'metrics.json')
    if type(metrics.get('completed_epochs')) is int:
        report['epoch'] = metrics['completed_epochs']
    if metrics.get('test') is not None or metrics.get('test_evaluation_pending') is not True:
        raise ValueError('Automatic rounds must not reuse the held-out test set')
    model_sha = hashlib.sha256((detector/'detector.pt').read_bytes()).hexdigest()
    if metrics.get('model_sha256')!=model_sha:
        raise ValueError('Trained checkpoint hash does not match metrics')
    report.update(training_completed=True, feedback_ids_actually_trained=included,
                  round_seen_feedback_ids=round_seen_feedback_ids(dataset, items),
                  model_sha256=model_sha, validation=metrics['validation'], test_used=False)
    atomic_json(run/'report.json', report)
    update_config_reference(config_path, config, dataset_path)
    # Mark actual supervised inputs even if validation will reject promotion.
    marker_remote = str(PurePosixPath(config['remote_base'])/'reviews'/run_id/'feedback-training.json')
    ssh(config, [config['remote_python'], deployment/'record_feedback_training.py',
                 '--data', config['remote_data'], '--run-id', run_id,
                 '--feedback-ids', ','.join(map(str, included)), '--model-sha256', model_sha,
                 '--output', marker_remote], log=run/'feedback-training.log')
    report['feedback_training_recorded'] = True
    with stage(config, report, 'validating', run):
        run_process([python, root/'train_detector.py', '--dataset', dataset_path, '--run', detector,
                     '--compare-validation-to', config['current_detector_run']], timeout=300,
                    cwd=root, log=run/'compare.log')
    comparison = read_json(detector/'validation-comparison.json')
    if (comparison.get('dataset_sha256')!=hashlib.sha256(dataset_path.read_bytes()).hexdigest()
        or comparison.get('candidate', {}).get('model_sha256')!=model_sha):
        raise ValueError('Validation comparison dataset/checkpoint hash changed')
    promote, reason = promotion_decision(comparison, config.get('minimum_validation_recall', .5),
                                        config.get('minimum_validation_boxes', 10))
    report.update(validation_comparison=comparison, promotion_reason=reason, promoted=False)
    # A joint issue/location model must not trade away localization recall.
    if promote and comparison.get('comparison_objective') == 'issue_type_and_location':
        old_location = comparison['base']['location_only']
        new_location = comparison['candidate']['location_only']
        if (new_location['recall'] < old_location['recall'] or
            new_location['f2'] < old_location['f2'] or new_location['fp'] > old_location['fp']):
            promote = False
            report['promotion_reason'] = 'Location recall/F2 or false positives regressed; retain current model'
    if not promote:
        report['status'] = 'trained_not_promoted'
        return report
    if config.get('review_refill_enabled', False):
        # The independent refill owns review batches. Existing pending IDs must
        # remain stable until human review completes, including after promotion.
        update_config_reference(config_path, config, dataset_path, detector)
        report.update(status='promoted_awaiting_review_refill', promoted=True,
                      original_classifications_modified=False, human_boxes_modified=False,
                      existing_online_classifier_replaced=False)
        return report
    output = run/'proposals.json'
    run_process([python, root/'generate_detector_proposals.py', '--snapshot', snapshot,
                 '--run', detector, '--output', output, '--source-root', config['source_root'],
                 '--refresh-pending'], timeout=600, cwd=root, log=run/'generate.log')
    manifest = filter_manifest(read_json(output))
    frozen = run/'review-proposals.json'
    atomic_json(frozen, manifest)
    report['proposal_images'] = sum(bool(row.get('boxes')) for row in manifest['items'])
    report['proposal_boxes'] = sum(len(row.get('boxes', [])) for row in manifest['items'])
    if manifest['items']:
        remote_directory = PurePosixPath(config['remote_base'])/'reviews'/run_id
        ssh(config, ['mkdir', '-p', remote_directory])
        remote_manifest = str(remote_directory/'proposals.json')
        transfer(config, str(frozen), config['ssh_alias']+':'+remote_manifest)
        raw = ssh(config, [config['remote_python'], deployment/'import_box_proposals.py',
                           '--data', config['remote_data'], '--manifest', remote_manifest,
                           '--output', remote_directory/'import-report.json'], timeout=300,
                  log=run/'import.log')
        imported = json.loads(raw)
        if imported.get('human_data_unchanged') is not True or imported.get('annotation_data_unchanged') is not True:
            raise ValueError('Pending proposal import did not confirm human data preservation')
        report['import'] = imported
    update_config_reference(config_path, config, dataset_path, detector)
    report.update(status='promoted_pending_human_review', promoted=True,
                  original_classifications_modified=False, human_boxes_modified=False,
                  existing_online_classifier_replaced=False)
    return report


def training_cycle(config_path, check_only=False):
    config_path = Path(config_path)
    config = validate_config(read_json(config_path))
    payload = export_feedback(config)
    root = Path(config['runs_root'])
    state_path = root/'correction-cycle-state.json'
    state = read_json(state_path) if state_path.exists() else {}
    previously_seen = set(state.get('round_seen_feedback_ids', []))
    items = [item for item in eligible_pending(payload) if item['feedback_id'] not in previously_seen]
    count = len({item['image_id'] for item in items})
    signature = batch_signature(items)
    paused = state.get('failed_batch')==signature and state.get('consecutive_failures', 0)>=3
    ready = count>=config.get('minimum_confirmed_images', 20) and not paused
    report = {'checked_at':dt.datetime.now(dt.timezone.utc).isoformat(), 'check_only':bool(check_only),
              'pending_confirmed_images':count, 'pending_feedback_records':len(items),
              'minimum_confirmed_images':config.get('minimum_confirmed_images', 20),
              'ready':ready, 'paused_same_batch_after_three_failures':paused,
              'batch_signature':signature, 'gpu_queried':False, 'training_completed':False,
              'promoted':False, 'test_used':False}
    if check_only:
        report['status'] = 'ready' if ready else 'paused' if paused else 'waiting_for_confirmed_images'
        return report
    try:
        with ExclusiveLock(root/'.correction-cycle.lock'):
            if not ready:
                report['status'] = 'paused' if paused else 'waiting_for_confirmed_images'
            else:
                available, reason = gpu_available(config);report['gpu_queried'] = True
                if not available:
                    report.update(status='skipped_gpu_busy_or_unavailable', reason=reason)
                else:
                    run = root/('auto-'+dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'-'+uuid.uuid4().hex[:6])
                    report['run_id'] = run.name
                    try:
                        report = execute_cycle(config, config_path, payload, items, run, report)
                        state = {**state, 'failed_batch':None, 'consecutive_failures':0,
                                 'round_seen_feedback_ids':sorted(previously_seen|set(report.get('round_seen_feedback_ids', [])))}
                        atomic_json(state_path, state)
                    except Exception as error:
                        failures = (state.get('consecutive_failures', 0)+1 if state.get('failed_batch')==signature else 1)
                        report.update(status='error', error=str(error), consecutive_failures=failures,
                                      paused_same_batch_after_three_failures=failures>=3)
                        state = {**state, 'failed_batch':signature, 'consecutive_failures':failures,
                                 'last_run':run.name, 'last_error':str(error),
                                 'round_seen_feedback_ids':sorted(previously_seen|set(report.get('round_seen_feedback_ids', [])))}
                        atomic_json(state_path, state)
                    if run.exists():
                        atomic_json(run/'report.json', report)
            atomic_json(root/'latest-state.json', report)
    except AlreadyRunning:
        report['status'] = 'skipped_another_worker_running'
    return report


def cycle(config_path, check_only=False, refill_only=False, training_only=False):
    config = validate_config(read_json(config_path))
    refill = None
    if not training_only and (config.get('review_refill_enabled', False) or refill_only):
        from review_refill import refill_cycle
        try:
            refill = refill_cycle(config_path, check_only=check_only)
        except Exception as error:
            refill = {'status': 'error', 'error': str(error)}
    if refill_only:
        return {'status': refill['status'], 'review_refill': refill, 'training_status': 'not_requested'}
    try:
        report = training_cycle(config_path, check_only)
    except Exception as error:
        report = {'status': 'error', 'error': str(error), 'training_completed': False}
    if refill is not None:
        report['training_status'] = report['status']
        report['review_refill'] = refill
        if report['status'] == 'error' and refill['status'] == 'refilled':
            report['status'] = 'review_refilled_training_error'
        if not check_only:
            atomic_json(Path(config['runs_root'])/'latest-state.json', report)
    if not check_only:
        from training_report import publish_reports
        from runtime_report import publish_training
        try:
            publish_training(config, report)
        except (OSError, ValueError) as error:
            report['runtime_report_error'] = str(error)
        try:
            report['training_reports'] = publish_reports(config)
        except Exception as error:
            report['training_reports'] = {'status': 'retry_next_tick', 'error': str(error)}
        atomic_json(Path(config['runs_root'])/'latest-state.json', report)
    return report


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--check-only', action='store_true')
    parser.add_argument('--refill-only', action='store_true', help='Only refill review proposals; never start training')
    parser.add_argument('--training-only', action='store_true', help='Only process reviewed training; inference runs independently')
    args = parser.parse_args()
    if args.training_only and args.refill_only:parser.error('Choose only one worker mode')
    try:
        result = cycle(args.config, args.check_only, args.refill_only, args.training_only)
        print(json.dumps(result, ensure_ascii=False), flush=True)
        sys.exit(1 if result['status']=='error' else 0)
    except Exception as error:
        print(json.dumps({'status':'error', 'error':str(error)}, ensure_ascii=False), flush=True)
        sys.exit(1)
