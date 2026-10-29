"""Bounded local refill inputs, never the authoritative annotation database.

Call retention only while both worker locks are held. Legacy snapshots without
our ownership receipt and historical ZIP archives are deliberately untouched.
"""
import hashlib
import json
from contextlib import contextmanager
from pathlib import Path
import re
import shutil
import stat
import time

GIB = 1024 ** 3
MAX_SNAPSHOT = 2 * GIB
MAX_RETAINED = 3 * GIB
KEEP = 3
MIN_AGE = 600
RUN = re.compile(r'refill-\d{8}T\d{6}Z-[a-f0-9]{6}')
RECEIPT = 'snapshot-input-retention.json'


class DiskSpaceLow(ValueError):
    pass


def space_check(directory, incoming=0):
    if type(incoming) is not int or not 0 <= incoming <= MAX_SNAPSHOT:
        raise DiskSpaceLow('SNAPSHOT_SIZE_LIMIT')
    directory = Path(directory)
    while not directory.exists():
        directory = directory.parent
    usage = shutil.disk_usage(directory)
    reserve = max(10 * GIB, usage.total // 20)
    if usage.free < reserve + 2 * incoming:
        raise DiskSpaceLow('SNAPSHOT_LOCAL_SPACE_LOW')
    return {'free_bytes': usage.free, 'reserve_bytes': reserve,
            'max_snapshot_bytes': MAX_SNAPSHOT}


def regular(path):
    path = Path(path).absolute()
    for part in (path, *path.parents):
        info = part.lstat()
        if stat.S_ISLNK(info.st_mode) or getattr(info, 'st_file_attributes', 0) & 1024:
            raise ValueError('SNAPSHOT_REPARSE_REJECTED')
    info = path.stat()
    if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
        raise ValueError('SNAPSHOT_FILE_IDENTITY')
    return info


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def run_path(root, run):
    root, run = Path(root).absolute(), Path(run).absolute()
    if run.parent != root or not RUN.fullmatch(run.name) or root.resolve() != root or run.resolve() != run:
        raise ValueError('SNAPSHOT_SCOPE_INVALID')
    # Reject junctions in ancestors even when resolve normalizes them.
    for path in (run, *run.parents):
        info = path.lstat()
        if getattr(info, 'st_file_attributes', 0) & 1024 or stat.S_ISLNK(info.st_mode):
            raise ValueError('SNAPSHOT_REPARSE_REJECTED')
    return run


def record_input(root, run, atomic_json):
    run = run_path(root, run)
    source = run / 'labels.sqlite3'
    info = regular(source)
    if not 0 < info.st_size <= MAX_SNAPSHOT or (run / RECEIPT).exists():
        raise ValueError('SNAPSHOT_INPUT_IDENTITY')
    atomic_json(run / RECEIPT, {'kind': 'refill-input-v1', 'run': run.name,
                'bytes': info.st_size, 'sha256': digest(source), 'finished_at': None})


def finish_input(root, run, state, atomic_json):
    run = run_path(root, run)
    if any(state.get(k, {}).get('batch_id') == run.name for k in ('pending_import', 'pending_registration')):
        return
    receipt = run / RECEIPT
    if not receipt.exists():
        return
    regular(receipt)
    value = json.loads(receipt.read_text(encoding='utf-8'))
    if value.get('kind') != 'refill-input-v1' or value.get('run') != run.name:
        raise ValueError('SNAPSHOT_RECEIPT_INVALID')
    value['finished_at'] = time.time()
    atomic_json(receipt, value)


@contextmanager
def tracked_input(root, run, state, atomic_json):
    record_input(root, run, atomic_json)
    try:
        yield
    finally:
        # subprocess.run has reaped the inference child before returning/raising.
        # Pending imports/registrations keep their input until a later successful retry.
        finish_input(root, run, state, atomic_json)


def maintain(root, state, atomic_json, now=None):
    root = Path(root).absolute(); now = time.time() if now is None else now
    protected = {state.get(k, {}).get('batch_id') for k in ('pending_import', 'pending_registration')}
    candidates = []; retained = 0; removed = 0
    if not root.exists():
        return {'removed_inputs': 0, 'retained_bytes': 0}
    for run in root.iterdir():
        if not RUN.fullmatch(run.name) or not run.is_dir():
            continue
        receipt = run / RECEIPT
        if not receipt.exists():
            continue
        run_path(root, run); regular(receipt)
        value = json.loads(receipt.read_text(encoding='utf-8'))
        if value.get('kind') != 'refill-input-v1' or value.get('run') != run.name:
            raise ValueError('SNAPSHOT_RECEIPT_INVALID')
        source = run / 'labels.sqlite3'
        if not source.exists():
            continue
        info = regular(source); retained += info.st_size
        finished = value.get('finished_at')
        if run.name in protected or type(finished) not in (int, float) or not 0 <= finished <= now:
            continue
        candidates.append((finished, source, value, info))
    for finished, source, value, before in sorted(candidates, key=lambda x: x[0], reverse=True)[KEEP:]:
        if now - finished < MIN_AGE:
            continue
        # An unexpected journal or changed input is preserved, never guessed away.
        if any(Path(str(source) + suffix).exists() for suffix in ('-wal', '-shm', '-journal')):
            continue
        if before.st_size != value.get('bytes') or digest(source) != value.get('sha256'):
            raise ValueError('SNAPSHOT_INPUT_CHANGED')
        after = regular(source)
        if (before.st_size, before.st_mtime_ns, before.st_ino) != (after.st_size, after.st_mtime_ns, after.st_ino):
            raise ValueError('SNAPSHOT_INPUT_CHANGED')
        # Only this exact proven temporary input; no recursive deletion.
        source.unlink()
        value['removed_at'] = now
        atomic_json(source.parent / RECEIPT, value)
        retained -= before.st_size; removed += 1
    if retained > MAX_RETAINED:
        raise DiskSpaceLow('SNAPSHOT_RETAINED_BUDGET')
    return {'removed_inputs': removed, 'retained_bytes': retained}
