"""Package only published Git source; keep previous fixed-name releases intact."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import subprocess
import tempfile
import uuid
import zipfile

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = Path("D:/项目开发/STYOLOX-release-artifacts")
PACKAGE = "styolox-latest.zip"
RECORD = "release.json"
SUFFIXES = {".py", ".md", ".txt", ".html", ".css", ".js", ".cjs", ".mjs",
            ".cmd", ".ps1", ".sh", ".toml", ".ini", ".yml", ".yaml", ".service", ".timer", ".example"}
PRIVATE_DIRS = {"data", "private", "evidence", "runs", "envs", ".venv", "node_modules", ".git"}


def validate_archive(path: Path) -> int:
    with zipfile.ZipFile(path) as archive:
        bad = archive.testzip()
        if bad:
            raise ValueError("Archive CRC failed: " + bad)
        count = 0
        for entry in archive.infolist():
            name = PurePosixPath(entry.filename)
            if name.is_absolute() or ".." in name.parts or "\\" in entry.filename:
                raise ValueError("Unsafe archive path")
            if any(part.lower() in PRIVATE_DIRS for part in name.parts):
                raise ValueError("Private directory rejected: " + entry.filename)
            if stat.S_ISLNK(entry.external_attr >> 16):
                raise ValueError("Symlink rejected: " + entry.filename)
            if entry.is_dir():
                continue
            if name.name.startswith(".env") or not (
                name.suffix.lower() in SUFFIXES or name.name == ".gitignore"
                or entry.filename in {"package.json", "source-provenance.json"}
            ):
                raise ValueError("Non-source file rejected: " + entry.filename)
            content = archive.read(entry)
            if b"\x00" in content:
                raise ValueError("Binary content rejected: " + entry.filename)
            content.decode("utf-8-sig")
            if re.search(rb"-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----|gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{50,}", content):
                raise ValueError("Credential marker rejected: " + entry.filename)
            count += 1
        if count == 0:
            raise ValueError("Empty source archive")
        return count


def publish(output: Path, candidate: Path, record: Path) -> None:
    """Archive both old files first; roll back any failed replacement."""
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + "-" + uuid.uuid4().hex[:8]
    history = output / "history" / stamp
    moved, installed = [], []
    try:
        for name in (PACKAGE, RECORD):
            old = output / name
            if old.is_symlink() or (old.exists() and not old.is_file()):
                raise ValueError("Unexpected release target: " + str(old))
            if old.exists():
                history.mkdir(parents=True, exist_ok=True)
                saved = history / (stamp + "-" + name)
                os.replace(old, saved)
                moved.append((saved, old))
        for source, name in ((candidate, PACKAGE), (record, RECORD)):
            target = output / name
            os.replace(source, target)
            installed.append(target)
    except BaseException:
        for target in reversed(installed):
            target.unlink()
        for saved, old in reversed(moved):
            os.replace(saved, old)
        raise


def package(output: Path) -> dict:
    output = output.resolve()
    if output == ROOT or ROOT in output.parents:
        raise ValueError("Release output must be outside the source repository")
    # This deliberately fails until the complete source commit is pushed.
    subprocess.run(["node", str(ROOT / "tools/verify-github-before-deploy.mjs")], cwd=ROOT, check=True)
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    output.mkdir(parents=True, exist_ok=True)
    lock = output / ".package.lock"
    lock.mkdir()  # Never race another publisher or silently break its lock.
    try:
        with tempfile.TemporaryDirectory(prefix=".source-", dir=output) as temporary:
            staging = Path(temporary)
            candidate, record = staging / PACKAGE, staging / RECORD
            subprocess.run(["git", "archive", "--format=zip", "--output", str(candidate), head], cwd=ROOT, check=True)
            count = validate_archive(candidate)
            info = {"repository": "stdeerlvxun/STYOLOX", "commit": head,
                    "generated_at": datetime.now(timezone.utc).isoformat(), "file": PACKAGE,
                    "size": candidate.stat().st_size, "sha256": hashlib.sha256(candidate.read_bytes()).hexdigest(),
                    "source_files": count, "scope": "public-source-only"}
            record.write_text(json.dumps(info, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            publish(output, candidate, record)
            return info
    finally:
        lock.rmdir()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    print(json.dumps(package(args.output), ensure_ascii=False, indent=2))
