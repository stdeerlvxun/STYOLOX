import json
import os
from pathlib import Path
import shutil
import tempfile
import time
import unittest
from unittest.mock import patch

import correction_cycle as worker
import review_refill as refill
import snapshot_storage as storage
from test_review_refill import fixture, FakeDetector


class StorageTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def run_input(self, index, finished=1):
        run = self.root / ('refill-20261029T010000Z-' + format(index, '06x'))
        run.mkdir()
        (run / 'labels.sqlite3').write_bytes(b'known read-only snapshot')
        storage.record_input(self.root, run, worker.atomic_json)
        value = worker.read_json(run / storage.RECEIPT)
        value['finished_at'] = finished
        worker.atomic_json(run / storage.RECEIPT, value)
        return run

    def test_retains_three_completed_inputs_and_all_annotation_artifacts(self):
        runs = [self.run_input(i, i + 1) for i in range(6)]
        legacy = self.root / 'refill-20261029T020000Z-123abc'; legacy.mkdir()
        for path in [legacy/'labels.sqlite3', self.root/'labels.sqlite3',
                     runs[0]/'proposals.json', runs[0]/'snapshot-inputs-verified.zip',
                     runs[0]/'image.jpg', runs[0]/'detector.pt']:
            path.write_bytes(b'must preserve')
        result = storage.maintain(self.root, {}, worker.atomic_json, now=1000)
        self.assertEqual(result['removed_inputs'], 3)
        self.assertEqual([r.joinpath('labels.sqlite3').exists() for r in runs], [False]*3+[True]*3)
        self.assertEqual((legacy/'labels.sqlite3').read_bytes(), b'must preserve')
        self.assertEqual((self.root/'labels.sqlite3').read_bytes(), b'must preserve')
        for name in ('proposals.json', 'snapshot-inputs-verified.zip', 'image.jpg', 'detector.pt'):
            self.assertEqual((runs[0]/name).read_bytes(), b'must preserve')

    def test_pending_unfinished_young_and_journal_inputs_are_preserved(self):
        pending = self.run_input(1); unfinished = self.run_input(2, None)
        journal = self.run_input(3); Path(str(journal/'labels.sqlite3')+'-wal').write_bytes(b'journal')
        for i in range(4, 9): self.run_input(i, 990+i)
        state = {'pending_registration': {'batch_id': pending.name}}
        storage.maintain(self.root, state, worker.atomic_json, now=1000)
        self.assertEqual(len(list(self.root.glob('*/labels.sqlite3'))), 8)
        self.assertTrue((unfinished/'labels.sqlite3').exists())

    def test_changed_snapshot_fails_closed(self):
        oldest = self.run_input(1)
        for i in range(2, 5): self.run_input(i, i)
        (oldest/'labels.sqlite3').write_bytes(b'changed content')
        with self.assertRaisesRegex(ValueError, 'INPUT_CHANGED'):
            storage.maintain(self.root, {}, worker.atomic_json, now=1000)
        self.assertEqual(len(list(self.root.glob('*/labels.sqlite3'))), 4)

    def test_hardlink_and_out_of_scope_rejected(self):
        run = self.run_input(1)
        os.link(run/'labels.sqlite3', self.root/'second-name')
        with self.assertRaisesRegex(ValueError, 'FILE_IDENTITY'):
            storage.maintain(self.root, {}, worker.atomic_json, now=1000)
        with self.assertRaisesRegex(ValueError, 'SCOPE_INVALID'):
            storage.finish_input(self.root/'different', run, {}, worker.atomic_json)
        self.assertTrue((self.root/'second-name').exists())

    def test_reparse_or_symlink_input_rejected(self):
        run = self.run_input(1); source = run/'labels.sqlite3'
        actual = source.lstat()
        class Reparse:
            st_mode = actual.st_mode
            st_file_attributes = 1024
        original = Path.lstat
        with patch.object(Path, 'lstat', lambda p, *a, **k: Reparse() if p == source else original(p, *a, **k)):
            with self.assertRaisesRegex(ValueError, 'REPARSE'):
                storage.regular(source)
        self.assertTrue(source.exists())

    def test_error_finishes_input_but_pending_import_does_not(self):
        for i, pending in ((1, False), (2, True)):
            run = self.root / ('refill-20261029T010000Z-'+format(i, '06x')); run.mkdir()
            (run/'labels.sqlite3').write_bytes(b'input')
            state = {'pending_import': {'batch_id': run.name}} if pending else {}
            with self.assertRaisesRegex(RuntimeError, 'inference failed'):
                with storage.tracked_input(self.root, run, state, worker.atomic_json):
                    raise RuntimeError('inference failed')
            value = worker.read_json(run/storage.RECEIPT)
            self.assertEqual(value['finished_at'] is None, pending)

    def test_retained_budget_stops_instead_of_deleting_pending(self):
        run = self.run_input(1, None)
        with patch.object(storage, 'MAX_RETAINED', 1):
            with self.assertRaises(storage.DiskSpaceLow):
                storage.maintain(self.root, {}, worker.atomic_json)
        self.assertTrue((run/'labels.sqlite3').exists())

    def test_low_space_stops_before_remote_snapshot_or_inference(self):
        value, _ = fixture(self.root, 1)
        config_path = self.root/'config.json'; worker.atomic_json(config_path, value)
        usage = shutil._ntuple_diskusage(100*storage.GIB, 99*storage.GIB, storage.GIB)
        with patch.object(storage.shutil, 'disk_usage', return_value=usage), \
             patch.object(worker, 'ssh') as remote, patch.object(worker, 'run_process') as process, \
             patch.object(refill, 'publish_status'):
            result = refill.refill_cycle(config_path)
            self.assertEqual(result['status'], 'waiting_for_disk')
            with self.assertRaises(storage.DiskSpaceLow):
                worker.fetch_snapshot(worker.validate_config(value), 'refill-20261029T010000Z-123abc', self.root/'download.sqlite3')
            remote.assert_not_called(); process.assert_not_called()

    def test_large_download_refused_and_remote_cleaned(self):
        value, _ = fixture(self.root, 1)
        proof = {'status': 'created', 'sha256': 'a'*64, 'bytes': storage.MAX_SNAPSHOT+1}
        with patch.object(worker, 'ssh', side_effect=[json.dumps(proof), '{"status":"cleaned"}']) as remote, \
             patch.object(worker, 'transfer') as transfer:
            with self.assertRaisesRegex(storage.DiskSpaceLow, 'SIZE_LIMIT'):
                worker.fetch_snapshot(worker.validate_config(value), 'refill-20261029T010000Z-123abc', self.root/'download.sqlite3')
            self.assertEqual(remote.call_count, 2); transfer.assert_not_called()

    def test_download_requires_double_incoming_space_above_reserve(self):
        usage = shutil._ntuple_diskusage(100*storage.GIB, 89*storage.GIB, 11*storage.GIB)
        with patch.object(storage.shutil, 'disk_usage', return_value=usage):
            storage.space_check(self.root)
            with self.assertRaisesRegex(storage.DiskSpaceLow, 'SPACE_LOW'):
                storage.space_check(self.root, storage.GIB)

    def test_empty_queue_backoff_and_model_change_or_expiry(self):
        value, _ = fixture(self.root, 1)
        config_path = self.root/'config.json'; worker.atomic_json(config_path, value)
        state_path = Path(value['runs_root'])/'review-refill-state.json'
        state = {'empty_queue_until': time.time()+1800,
                 'empty_queue_model': storage.digest(Path(value['current_detector_run'])/'detector.pt')}
        worker.atomic_json(state_path, state)
        with patch.object(refill, 'publish_status'), patch.object(refill, 'api', return_value={'pending': 0}), \
             patch.object(worker, 'fetch_snapshot') as fetch, \
             patch.object(worker, 'gpu_available', return_value=(False, 'busy')) as gpu:
            self.assertEqual(refill.refill_cycle(config_path)['status'], 'waiting_for_new_images')
            gpu.assert_not_called(); fetch.assert_not_called()
            state['empty_queue_model'] = 'changed'; worker.atomic_json(state_path, state)
            self.assertEqual(refill.refill_cycle(config_path)['status'], 'waiting_for_gpu')
            state['empty_queue_model'] = storage.digest(Path(value['current_detector_run'])/'detector.pt')
            state['empty_queue_until'] = time.time()-1; worker.atomic_json(state_path, state)
            self.assertEqual(refill.refill_cycle(config_path)['status'], 'waiting_for_gpu')
            self.assertEqual(gpu.call_count, 2); fetch.assert_not_called()

    def test_real_sqlite_empty_pass_persists_backoff_and_keeps_human_data(self):
        value, source = fixture(self.root, 2)
        original = source.read_bytes()
        config_path = self.root/'config.json'; worker.atomic_json(config_path, value)
        def process(args, **kwargs):
            args = list(map(str, args))
            refill.generate(value, Path(args[args.index('--snapshot')+1]),
                            Path(args[args.index('--output')+1]),
                            Path(args[args.index('--attempts')+1]), FakeDetector(empty=True))
        with patch.object(refill, 'publish_status'), patch.object(refill, 'api', return_value={'pending': 0}), \
             patch.object(worker, 'gpu_available', return_value=(True, 'available')), \
             patch.object(worker, 'run_process', side_effect=process), \
             patch.object(worker, 'fetch_snapshot', side_effect=lambda c, b, p, **kw: shutil.copyfile(source, p)) as fetch:
            first = refill.refill_cycle(config_path)
            self.assertEqual(first['status'], 'no_more_eligible_images', first)
            second = refill.refill_cycle(config_path)
            self.assertEqual(second['status'], 'waiting_for_new_images', second)
            fetch.assert_called_once()
        self.assertEqual(source.read_bytes(), original)
        receipt = next(Path(value['runs_root']).glob('refill-*/'+storage.RECEIPT))
        self.assertIsInstance(worker.read_json(receipt)['finished_at'], float)
        self.assertEqual((receipt.parent/'labels.sqlite3').read_bytes(), original)


if __name__ == '__main__':
    unittest.main()
