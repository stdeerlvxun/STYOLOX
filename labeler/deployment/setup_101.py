"""Prepare an immutable release and isolated staging service on server 101."""
import argparse
import json
import os
from pathlib import Path
import shutil
import sqlite3
import subprocess
import sys
import tarfile

BASE = Path("/opt/after-sales-image-labeler")
DATA = Path("/var/lib/after-sales-image-labeler")
NAS = Path("/mnt/stdeer_analysis")
ARCHIVE = NAS / "\u672c\u5730\u5206\u6790\u56fe\u7247" / "cases"


def prepare(package, initial_db, release):
    if not ARCHIVE.is_dir():
        raise RuntimeError("The existing NAS mount is unavailable")
    if not release.replace("-", "").replace("_", "").isalnum():
        raise ValueError("Invalid release identifier")
    target = BASE / "releases" / release
    target.mkdir(parents=True, exist_ok=False)
    with tarfile.open(package) as archive:
        archive.extractall(target, filter="data")
    staging = DATA / "staging"
    staging.mkdir(parents=True, exist_ok=False)
    shutil.copy2(initial_db, staging / "labels.sqlite3")
    sys.path.insert(0, str(target))
    from app import PathMapper
    mapper = PathMapper(NAS)
    with sqlite3.connect(staging / "labels.sqlite3") as db:
        for root_id, path in db.execute("SELECT id,path FROM roots").fetchall():
            db.execute("UPDATE roots SET path=? WHERE id=?", (mapper.server(path), root_id))
    current = BASE / "current"
    if current.exists() or current.is_symlink():
        raise RuntimeError("An existing release needs an explicit upgrade path")
    current.symlink_to(target, target_is_directory=True)
    command = [sys.executable, str(current / "app.py"), "--port", "8878",
               "--host", "192.168.0.101", "--allow-host", "192.168.0.101:8878",
               "--data-dir", str(DATA), "--root", str(ARCHIVE), "--nas-root", str(NAS)]
    unit = """[Unit]
Description=After-sales image labeling
After=network-online.target
Wants=network-online.target
RequiresMountsFor=/mnt/stdeer_analysis

[Service]
Type=simple
WorkingDirectory=/opt/after-sales-image-labeler/current
Environment=PYTHONUNBUFFERED=1
Environment=PYTHONIOENCODING=utf-8
Environment=PYTHONDONTWRITEBYTECODE=1
ExecStart=""" + " ".join(command) + """
Restart=on-failure
RestartSec=3
ProtectSystem=strict
ProtectHome=true
PrivateTmp=true
NoNewPrivileges=true
ReadOnlyPaths=/mnt/stdeer_analysis
ReadWritePaths=/var/lib/after-sales-image-labeler

[Install]
WantedBy=multi-user.target
"""
    Path("/etc/systemd/system/after-sales-image-labeler.service").write_text(unit, encoding="utf-8")
    subprocess.run(["systemctl", "daemon-reload"], check=True)
    staging_command = [sys.executable, str(target / "app.py"), "--port", "8879",
                       "--host", "192.168.0.101", "--allow-host", "192.168.0.101:8879",
                       "--data-dir", str(staging), "--root", str(ARCHIVE), "--nas-root", str(NAS)]
    subprocess.run(["systemd-run", "--unit=after-sales-image-labeler-staging",
                    "--property=WorkingDirectory=" + str(target),
                    "--setenv=PYTHONDONTWRITEBYTECODE=1", *staging_command], check=True)
    return {"release": str(target), "python": sys.executable, "staging_url": "http://192.168.0.101:8879",
            "mount_reused": str(NAS), "final_service_started": False}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--initial-db", type=Path, required=True)
    parser.add_argument("--release", required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.package, args.initial_db, args.release), ensure_ascii=True))
