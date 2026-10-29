import contextlib
import hashlib
import json
from pathlib import Path
import shutil
import sqlite3
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
import uuid
from unittest.mock import MagicMock, patch

import correction_cycle as worker


RUN = 'refill-20261026T100403Z-548363'
TOKEN = uuid.uuid4().hex


def database(path):
    with contextlib.closing(sqlite3.connect(path)) as db, db:
        db.execute('CREATE TABLE proof(value TEXT)')
        db.execute("INSERT INTO proof VALUES('synthetic')")


class DownloadTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name); self.source = self.root / 'source.sqlite3'
        database(self.source); self.destination = self.root / 'download.sqlite3'
        self.config = {'remote_python': '/usr/bin/python3', 'remote_base': '/opt/labeler',
            'remote_data': '/var/lib/labeler', 'ssh_alias': 'local-vm'}
        self.proof = {'status': 'created', 'bytes': self.source.stat().st_size,
            'sha256': hashlib.sha256(self.source.read_bytes()).hexdigest()}
        self.calls = []

    def remote(self, _config, arguments, **_kwargs):
        self.calls.append(arguments)
        return json.dumps(self.proof if arguments[3] == 'create' else {'status': 'cleaned'})

    def transfer(self, _config, source, destination):
        self.assertEqual(source, 'local-vm:/var/lib/labeler/training-snapshots/' + RUN + '.sqlite3')
        shutil.copyfile(self.source, destination)

    def test_verified_download_cleans_remote_and_preserves_local_database(self):
        with patch.object(worker, 'ssh', side_effect=self.remote), patch.object(worker, 'transfer', side_effect=self.transfer):
            self.assertEqual(worker.fetch_snapshot(self.config, RUN, self.destination), self.destination)
        self.assertEqual([args[3] for args in self.calls], ['create', 'cleanup'])
        self.assertEqual(self.calls[0][2], worker.REMOTE_SNAPSHOT)
        self.assertEqual(self.calls[0][6], self.calls[1][6])
        self.assertEqual(self.destination.read_bytes(), self.source.read_bytes())

    def test_hash_mismatch_and_sqlite_corruption_always_cleanup(self):
        for kind in ('hash', 'sqlite'):
            with self.subTest(kind=kind):
                self.calls.clear(); self.destination.unlink(missing_ok=True)
                if kind == 'hash':
                    self.proof['sha256'] = '0' * 64
                    expected = 'SNAPSHOT_HASH_MISMATCH'
                else:
                    self.source.write_bytes(b'not a SQLite database')
                    self.proof.update(bytes=self.source.stat().st_size,
                        sha256=hashlib.sha256(self.source.read_bytes()).hexdigest())
                    expected = 'file is not a database'
                with patch.object(worker, 'ssh', side_effect=self.remote), patch.object(worker, 'transfer', side_effect=self.transfer):
                    with self.assertRaisesRegex((ValueError, sqlite3.DatabaseError), expected):
                        worker.fetch_snapshot(self.config, RUN, self.destination)
                self.assertEqual([args[3] for args in self.calls], ['create', 'cleanup'])

    def test_create_and_transfer_errors_keep_original_even_when_cleanup_fails(self):
        for phase in ('create', 'transfer'):
            with self.subTest(phase=phase):
                self.calls.clear(); primary = RuntimeError('original bounded failure')
                def remote(config, arguments, **kwargs):
                    self.calls.append(arguments)
                    if arguments[3] == 'cleanup': raise ValueError('cleanup unavailable')
                    if phase == 'create': raise primary
                    return json.dumps(self.proof)
                with patch.object(worker, 'ssh', side_effect=remote), patch.object(worker, 'transfer', side_effect=primary):
                    with self.assertRaises(RuntimeError) as caught:
                        worker.fetch_snapshot(self.config, RUN, self.destination)
                self.assertIs(caught.exception, primary)
                self.assertEqual([args[3] for args in self.calls], ['create', 'cleanup'])
                self.assertTrue(any('SNAPSHOT_CLEANUP_FAILED' in x for x in primary.__notes__))

    def test_cleanup_failure_after_valid_download_does_not_report_success(self):
        def remote(config, arguments, **kwargs):
            if arguments[3] == 'cleanup': raise ValueError('SNAPSHOT_ACTIVE')
            return self.remote(config, arguments, **kwargs)
        with patch.object(worker, 'ssh', side_effect=remote), patch.object(worker, 'transfer', side_effect=self.transfer):
            with self.assertRaisesRegex(ValueError, 'SNAPSHOT_ACTIVE'):
                worker.fetch_snapshot(self.config, RUN, self.destination)
        self.assertTrue(self.destination.is_file())

    def test_unsafe_names_paths_and_existing_destination_never_call_ssh(self):
        with patch.object(worker, 'ssh', side_effect=AssertionError('remote execution')):
            for value in ('../labels', 'refill-test', '/tmp/' + RUN, RUN + '/child', RUN + '.sqlite3'):
                with self.subTest(run=value), self.assertRaisesRegex(ValueError, 'SNAPSHOT_RUN_INVALID'):
                    worker.fetch_snapshot(self.config, value, self.destination)
            with self.assertRaisesRegex(ValueError, 'SNAPSHOT_REMOTE_PATH_INVALID'):
                worker.fetch_snapshot({**self.config, 'remote_data': '/var/lib/../other'}, RUN, self.destination)
            self.destination.write_bytes(b'preserve')
            with self.assertRaisesRegex(ValueError, 'SNAPSHOT_LOCAL_EXISTS'):
                worker.fetch_snapshot(self.config, RUN, self.destination)
        self.assertEqual(self.destination.read_bytes(), b'preserve')


class RemoteLifecycleTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve(); self.source = self.root / 'labels.sqlite3'
        database(self.source)
        self.namespace = {'__name__': 'snapshot_remote_test'}
        exec(compile(worker.REMOTE_SNAPSHOT, '<published-snapshot-helper>', 'exec'), self.namespace)
        self.namespace['folder_lock'] = lambda folder: contextlib.nullcontext(123)
        self.namespace['require_closed'] = MagicMock()
        self.folder = self.root / 'training-snapshots'; self.target = self.folder / (RUN + '.sqlite3')
        self.owner = self.folder / ('.' + RUN + '.owner')
        self.disk = patch.object(self.namespace['shutil'], 'disk_usage', return_value=SimpleNamespace(free=2 * 1024**3))
        self.inodes = patch.object(self.namespace['os'], 'statvfs', create=True,
            return_value=SimpleNamespace(f_files=1000, f_favail=1000))
        self.disk.start(); self.addCleanup(self.disk.stop)
        self.inodes.start(); self.addCleanup(self.inodes.stop)

    def operation(self, mode, token=TOKEN, run=RUN):
        return self.namespace['operation'](mode, str(self.root), run, token,
            '/usr/bin/python3', '/opt/labeler/current/deployment/snapshot.py')

    def create(self, arguments, **kwargs):
        self.assertEqual(arguments[-2], str(self.source))
        self.assertEqual(kwargs['pass_fds'], (123,))
        self.assertEqual(kwargs['timeout'], 120)
        shutil.copyfile(self.source, arguments[-1])
        self.target.with_suffix('.report.json').write_text('{}')
        self.target.with_suffix('.labels.json').write_text('{}')
        return SimpleNamespace(returncode=0)

    def test_create_proof_and_owned_cleanup_are_exact_and_keep_unrelated_files(self):
        with patch.object(self.namespace['subprocess'], 'run', side_effect=self.create):
            result = self.operation('create')
        self.assertEqual(result['sha256'], hashlib.sha256(self.source.read_bytes()).hexdigest())
        other = self.folder / 'refill-unrelated.sqlite3'; other.write_bytes(b'preserve')
        journal = Path(str(self.target) + '-journal'); journal.write_bytes(b'failed transaction')
        self.assertEqual(self.operation('cleanup'), {'status': 'cleaned'})
        self.namespace['require_closed'].assert_called_once()
        self.assertEqual(other.read_bytes(), b'preserve')
        self.assertTrue(self.source.is_file()); self.assertFalse(self.owner.exists())
        self.assertFalse(self.target.exists()); self.assertFalse(journal.exists())
        self.assertFalse(self.target.with_suffix('.report.json').exists())
        self.assertFalse(self.target.with_suffix('.labels.json').exists())

    def test_failed_creation_keeps_ownership_for_finally_cleanup(self):
        def partial(arguments, **kwargs):
            self.target.write_bytes(b'partial'); Path(str(self.target) + '-journal').write_bytes(b'journal')
            return SimpleNamespace(returncode=1)
        with patch.object(self.namespace['subprocess'], 'run', side_effect=partial):
            with self.assertRaisesRegex(ValueError, 'SNAPSHOT_CREATE_FAILED'):
                self.operation('create')
        self.assertTrue(self.owner.exists())
        self.assertEqual(self.operation('cleanup'), {'status': 'cleaned'})
        self.assertFalse(self.target.exists()); self.assertTrue(self.source.is_file())

    def test_insufficient_bytes_or_inodes_do_not_create_any_snapshot(self):
        for kind in ('bytes', 'inodes'):
            with self.subTest(kind=kind):
                with patch.object(self.namespace['shutil'], 'disk_usage', return_value=SimpleNamespace(
                        free=self.source.stat().st_size + 512 * 1024**2 - 1 if kind == 'bytes' else 2 * 1024**3)), \
                        patch.object(self.namespace['os'], 'statvfs', return_value=SimpleNamespace(f_files=1000,
                            f_favail=1 if kind == 'inodes' else 1000)), \
                        patch.object(self.namespace['subprocess'], 'run', side_effect=AssertionError('snapshot started')):
                    with self.assertRaisesRegex(ValueError, 'SNAPSHOT_SPACE_LOW|SNAPSHOT_INODES_LOW'):
                        self.operation('create')
                self.assertFalse(self.owner.exists()); self.assertFalse(self.target.exists())
                self.assertEqual(self.operation('cleanup'), {'status': 'absent'})

    def test_wrong_owner_existing_or_active_snapshot_is_never_deleted(self):
        with patch.object(self.namespace['subprocess'], 'run', side_effect=self.create):
            self.operation('create')
        with self.assertRaisesRegex(ValueError, 'SNAPSHOT_ALREADY_EXISTS'):
            self.operation('create')
        with self.assertRaisesRegex(ValueError, 'SNAPSHOT_OWNERSHIP_UNVERIFIED'):
            self.operation('cleanup', token='d' * 32)
        self.namespace['require_closed'].side_effect = ValueError('SNAPSHOT_ACTIVE')
        with self.assertRaisesRegex(ValueError, 'SNAPSHOT_ACTIVE'):
            self.operation('cleanup')
        self.assertTrue(self.target.exists()); self.assertTrue(self.owner.exists())

    def test_preexisting_snapshot_without_owner_and_unsafe_run_are_preserved(self):
        self.folder.mkdir(); self.target.write_bytes(b'not ours')
        with self.assertRaisesRegex(ValueError, 'SNAPSHOT_OWNERSHIP_UNVERIFIED'):
            self.operation('cleanup')
        for name in ('../labels', RUN + '/child', 'auto-anything'):
            with self.subTest(run=name), self.assertRaisesRegex(ValueError, 'SNAPSHOT_RUN_INVALID'):
                self.operation('cleanup', run=name)
        self.assertEqual(self.target.read_bytes(), b'not ours')

    def test_open_descriptor_detection_refuses_active_artifact(self):
        namespace = {'__name__': 'snapshot_open_files_test'}
        exec(compile(worker.REMOTE_SNAPSHOT, '<published-snapshot-helper>', 'exec'), namespace)
        artifact = self.root / 'artifact'; artifact.write_bytes(b'open')
        descriptor = MagicMock(); descriptor.stat.return_value = artifact.stat()
        process = MagicMock(); process.name = '999999'; (process / 'fd').iterdir.return_value = [descriptor]
        proc = MagicMock(); proc.is_dir.return_value = True; proc.iterdir.return_value = [process]
        with patch.object(namespace['pathlib'], 'Path', return_value=proc):
            with self.assertRaisesRegex(ValueError, 'SNAPSHOT_ACTIVE'):
                namespace['require_closed']([artifact])


if __name__ == '__main__':
    unittest.main()
