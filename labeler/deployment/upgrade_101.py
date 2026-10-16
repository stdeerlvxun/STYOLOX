"""Switch an existing 101 installation to a tested release, preserving its live store."""
import argparse
import datetime as dt
import json
import os
from pathlib import Path
import subprocess
import time
from urllib.request import urlopen
from snapshot import snapshot

BASE = Path('/opt/after-sales-image-labeler')
DATA = Path('/var/lib/after-sales-image-labeler')
SERVICE = 'after-sales-image-labeler.service'


def service(action):
    subprocess.run(['systemctl', action, SERVICE], check=True)


def switch(release, temporary):
    temporary.symlink_to(release, target_is_directory=True)
    os.replace(temporary, BASE / 'current')


def upgrade(release):
    release = release.resolve(strict=True)
    if release.parent != (BASE / 'releases').resolve() or not (release / 'app.py').is_file():
        raise ValueError('Release must be a complete directory directly under releases')
    current = BASE / 'current'
    if not current.is_symlink():
        raise ValueError('Existing deployment must use the current symlink')
    previous = current.resolve(strict=True)
    stamp = dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    folder = DATA / 'upgrade-backups' / stamp
    temporary = BASE / ('current.next-' + stamp)
    service('stop')
    try:
        before = snapshot(DATA / 'labels.sqlite3', folder / 'before.sqlite3')
        switch(release, temporary)
        service('start')
        state = None
        for attempt in range(12):
            try:
                with urlopen('http://192.168.0.101:8878/api/state', timeout=3) as response:
                    state = json.load(response)
                break
            except OSError:
                time.sleep(.5)
        if not state or not state.get('responsibility_rules'):
            raise RuntimeError('New service did not become ready with responsibility rules')
        after = snapshot(DATA / 'labels.sqlite3', folder / 'after.sqlite3')
        report = {'release': str(release), 'previous_release': str(previous), 'backup_directory': str(folder),
                  'before': before, 'after': after,
                  'data_unchanged': before['data_digest'] == after['data_digest'],
                  'rules': state['responsibility_rules'], 'counts': state['counts']}
        (folder / 'upgrade-report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
        print(json.dumps(report, ensure_ascii=False))
    except Exception:
        if temporary.is_symlink():
            temporary.unlink()
        switch(previous, temporary)
        service('start')
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('release', type=Path)
    upgrade(parser.parse_args().release)
