"""Atomically upgrade 101 with a reversible inference-only systemd override."""
import argparse
import datetime as dt
import json
import os
from pathlib import Path
import subprocess
import time
from urllib.request import urlopen
from snapshot import snapshot

BASE=Path('/opt/after-sales-image-labeler');DATA=Path('/var/lib/after-sales-image-labeler')
UNIT='after-sales-image-labeler.service'
OVERRIDE=Path('/etc/systemd/system/after-sales-image-labeler.service.d/recommendations.conf')


def command(*args):subprocess.run(args,check=True)


def enable(release,model,unrelated_review=False):
    release=release.resolve(strict=True);model=model.resolve(strict=True)
    if release.parent!=BASE/'releases':raise ValueError('Invalid release directory')
    meta=json.loads((model/'model.json').read_text(encoding='utf-8'))
    verified=json.loads((model/'verification-101.json').read_text(encoding='utf-8'))
    if verified['status']!='passed' or verified['model_id']!=meta['model_id']:raise ValueError('Model not verified')
    previous=(BASE/'current').resolve(strict=True)
    stamp=dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    backup=DATA/'upgrade-backups'/('recommendations-'+stamp);backup.mkdir(parents=True)
    old_override=OVERRIDE.read_bytes() if OVERRIDE.exists() else None
    if old_override is not None:(backup/'previous-override.conf').write_bytes(old_override)
    temporary=BASE/('current.next-'+stamp)
    command('systemctl','stop',UNIT)
    try:
        before=snapshot(DATA/'labels.sqlite3',backup/'before.sqlite3')
        OVERRIDE.parent.mkdir(parents=True,exist_ok=True)
        OVERRIDE.write_text('[Service]\nExecStart=\nExecStart='+str(BASE/'inference-venv/bin/python')+
            ' '+str(BASE/'current/app.py')+' --port 8878 --host 192.168.0.101 --allow-host 192.168.0.101:8878'+
            ' --data-dir '+str(DATA)+' --root /mnt/stdeer_analysis/本地分析图片/cases'+
            ' --nas-root /mnt/stdeer_analysis --recommendation-model '+str(model)+
            (' --unrelated-review' if unrelated_review else '')+'\n',encoding='utf-8')
        temporary.symlink_to(release,target_is_directory=True);os.replace(temporary,BASE/'current')
        command('systemctl','daemon-reload');command('systemctl','start',UNIT)
        state=None
        for attempt in range(20):
            try:
                with urlopen('http://192.168.0.101:8878/api/state',timeout=2) as response:state=json.load(response)
                if state.get('recommendations',{}).get('model_id')==meta['model_id'] and state['recommendations']['enabled']:break
            except OSError:pass
            time.sleep(.5)
        if (not state or not state.get('recommendations',{}).get('enabled') or
                state['recommendations']['model_id']!=meta['model_id']):raise RuntimeError('New service not ready')
        after=snapshot(DATA/'labels.sqlite3',backup/'after.sqlite3')
        report={'status':'deployed','release':str(release),'previous_release':str(previous),'backup':str(backup),
            'before':before,'after':after,'human_data_unchanged':before['data_digest']==after['data_digest'],
            'annotation_data_unchanged':before['annotation_data_digest']==after['annotation_data_digest'],
            'model':meta['model_id'],'suggestions_only':True,'nas_mount_reused':True}
        (backup/'upgrade-report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
        print(json.dumps(report,ensure_ascii=False))
    except Exception:
        command('systemctl','stop',UNIT)
        if old_override is None:OVERRIDE.unlink(missing_ok=True)
        else:OVERRIDE.write_bytes(old_override)
        temporary.unlink(missing_ok=True);temporary.symlink_to(previous,target_is_directory=True)
        os.replace(temporary,BASE/'current')
        command('systemctl','daemon-reload');command('systemctl','start',UNIT)
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--release',type=Path,required=True)
    parser.add_argument('--model',type=Path,required=True);parser.add_argument('--unrelated-review',action='store_true')
    args=parser.parse_args();enable(args.release,args.model,args.unrelated_review)
