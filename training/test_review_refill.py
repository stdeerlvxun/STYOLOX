import copy
import hashlib
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image
import correction_cycle as worker
import review_refill as refill
from test_correction_cycle import config


class FakeDetector:
    class_agnostic = False

    def __init__(self, empty=False):
        self.calls = 0
        self.empty = empty
        self.thresholds = []

    def predict_image(self, image, **kwargs):
        self.calls += 1
        self.thresholds.append(kwargs.get('threshold'))
        return [] if self.empty else [{'problem': '其他破损', 'score': .8, 'xywh': [.1, .1, .2, .2]}]


def fixture(root, count=4):
    value = {**config(root), 'review_refill_enabled': True, 'fixed_threshold': .05}
    source = Path(value['source_root']); source.mkdir()
    model = Path(value['current_detector_run']); model.mkdir()
    (model/'detector.pt').write_bytes(b'verified model')
    worker.atomic_json(model/'metrics.json', {'model_sha256': hashlib.sha256(b'verified model').hexdigest()})
    snapshot = root/'snapshot.sqlite3'; db = sqlite3.connect(snapshot)
    db.executescript('''CREATE TABLE settings(key TEXT,value TEXT);
        INSERT INTO settings VALUES('root','1');
        CREATE TABLE images(id INTEGER,relative_path TEXT,sha256 TEXT,reported_at TEXT,size INTEGER,mtime_ns INTEGER,available INTEGER,root_id INTEGER);
        CREATE TABLE labels(image_id INTEGER,data TEXT);
        CREATE TABLE drafts(image_id INTEGER);
        CREATE TABLE annotation_documents(image_id INTEGER);
        CREATE TABLE annotation_proposals(image_id INTEGER,status TEXT);
        CREATE TABLE annotation_history(image_id INTEGER);
        CREATE TABLE annotation_feedback(image_id INTEGER);
        CREATE TABLE annotation_proposal_history(image_id INTEGER,status TEXT);
        CREATE TABLE error_reviews(image_id INTEGER,status TEXT);
        CREATE TABLE model_reviews(image_id INTEGER,decision TEXT,status TEXT);''')
    for identity in range(1, count+1):
        path = source/(str(identity)+'.png'); Image.new('RGB', (20, 20), 'white').save(path); stat = path.stat()
        db.execute('INSERT INTO images VALUES(?,?,?,?,?,?,?,?)',
            (identity, path.name, None, '2026-10-'+str(identity).zfill(2), stat.st_size, stat.st_mtime_ns, 1, 1))
        db.execute('INSERT INTO labels VALUES(?,?)', (identity, json.dumps({'defects': ['崩边'], 'image_state': 'defect'})))
    db.commit(); db.close()
    return value, snapshot


class RefillTests(unittest.TestCase):
    def test_eligible_policy_preserves_drafts_boxes_and_all_proposal_statuses(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); value, snapshot = fixture(root, 10)
            db = sqlite3.connect(snapshot)
            db.execute('INSERT INTO drafts VALUES(1)'); db.execute('INSERT INTO annotation_documents VALUES(2)')
            for identity, status in ((3, 'pending'), (4, 'accepted'), (5, 'rejected'), (6, 'stale')):
                db.execute('INSERT INTO annotation_proposals VALUES(?,?)', (identity, status))
            db.execute('UPDATE labels SET data=? WHERE image_id=7', (json.dumps({'defects': ['崩边'], 'image_state': 'unclear'}),))
            db.execute('UPDATE images SET available=0 WHERE id=8'); db.commit(); db.close()
            rows, summary = refill.select_refill_rows(snapshot)
            self.assertEqual([row['image_id'] for row in rows], [10, 9])
            self.assertEqual(summary['excluded']['existing_machine_proposal'], 4)
            self.assertEqual(summary['excluded']['image_state_selected'], 1)

    def test_bounded_newest_first_generation_keeps_actual_predicted_problem(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); value, snapshot = fixture(root, 7)
            value.update(review_refill_batch_size=2, review_refill_scan_limit=4)
            detector = FakeDetector()
            result = refill.generate(value, snapshot, root/'output.json', root/'attempts.json', detector)
            self.assertEqual([row['image_id'] for row in result['items']], [7, 6])
            self.assertEqual(result['items'][0]['boxes'][0]['problem'], '其他破损', 'The detector class is retained instead of copying the human image label')
            self.assertEqual(result['items'][0]['boxes'][0]['type_provenance']['source'],'detector_class_prediction')
            self.assertEqual(result['summary']['attempted_images'], 2)
            self.assertEqual(result['summary']['remaining_eligible_images'], 5)
            self.assertFalse(result['refresh_pending']); self.assertTrue(result['requires_human_review'])
            self.assertEqual(detector.thresholds, [.05, .05])

    def test_unclassified_and_multilabel_images_generate_real_typed_boxes_without_writes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); value, snapshot = fixture(root, 4)
            db = sqlite3.connect(snapshot)
            db.execute('DELETE FROM labels WHERE image_id=4')
            db.execute('UPDATE labels SET data=? WHERE image_id=3', (json.dumps({'defects': ['崩边', '划伤']}),))
            db.execute('UPDATE labels SET data=? WHERE image_id=2', (json.dumps({'defects': []}),))
            db.commit(); db.close()
            before = snapshot.read_bytes()
            detector = FakeDetector()
            result = refill.generate(value, snapshot, root/'output.json', root/'attempts.json', detector)
            self.assertEqual([item['image_id'] for item in result['items']], [4, 3, 2, 1])
            self.assertEqual([item['boxes'][0]['problem'] for item in result['items']], ['其他破损']*4)
            self.assertEqual(result['items'][0]['label_fingerprint'], hashlib.sha256(b'').hexdigest())
            self.assertEqual(result['summary']['eligible_images'], 4)
            self.assertEqual(result['selection_policy'], refill.SELECTION_POLICY)
            self.assertEqual(detector.thresholds, [.05]*4)
            self.assertEqual(snapshot.read_bytes(), before, 'All human data is read-only')
            self.assertFalse(result['classification_results_modified'])
            self.assertFalse(result['source_images_modified'])
            self.assertEqual(result['summary']['typed_boxes'],4)
            self.assertEqual(result['summary']['unassigned_boxes'],0)

    def test_historical_manual_reviews_and_proposals_never_reenter(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); value, snapshot = fixture(root, 14)
            db = sqlite3.connect(snapshot)
            db.execute('INSERT INTO annotation_history VALUES(1)')
            db.execute('INSERT INTO annotation_feedback VALUES(2)')
            for identity, status in ((3, 'pending'), (4, 'accepted'), (5, 'rejected'), (6, 'stale')):
                db.execute('INSERT INTO annotation_proposal_history VALUES(?,?)', (identity, status))
            db.execute("INSERT INTO error_reviews VALUES(7,'reviewed')")
            db.execute("INSERT INTO error_reviews VALUES(8,'pending')")
            db.execute("INSERT INTO model_reviews VALUES(9,'unrelated','accepted')")
            db.execute("INSERT INTO model_reviews VALUES(10,'unrelated','rejected')")
            db.execute("INSERT INTO model_reviews VALUES(11,'unrelated','pending')")
            for identity, state in ((12, 'normal'), (13, 'unrelated'), (14, 'unclear')):
                db.execute('UPDATE labels SET data=? WHERE image_id=?', (json.dumps({'defects': [], 'image_state': state}), identity))
            db.commit(); db.close()
            rows, summary = refill.select_refill_rows(snapshot)
            self.assertEqual([row['image_id'] for row in rows], [11, 10, 8])
            self.assertEqual(summary['excluded']['existing_human_annotation'], 2)
            self.assertEqual(summary['excluded']['existing_machine_proposal'], 4)
            self.assertEqual(summary['excluded']['existing_human_error_review'], 1)
            self.assertEqual(summary['excluded']['confirmed_unrelated_review'], 1)

    def test_empty_predictions_are_not_retried_until_label_source_or_model_changes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); value, snapshot = fixture(root, 1); detector = FakeDetector(empty=True)
            state = root/'attempts.json'
            first = refill.generate(value, snapshot, root/'one.json', state, detector)
            second = refill.generate(value, snapshot, root/'two.json', state, detector)
            self.assertEqual(first['summary']['images_without_boxes'], 1)
            self.assertEqual(second['summary']['attempted_images'], 0); self.assertEqual(detector.calls, 1)
            db = sqlite3.connect(snapshot); db.execute('UPDATE images SET mtime_ns=mtime_ns+1'); db.commit(); db.close()
            refill.generate(value, snapshot, root/'three.json', state, detector); self.assertEqual(detector.calls, 2)
            db = sqlite3.connect(snapshot); db.execute('UPDATE labels SET data=?', (json.dumps({'defects': ['划伤']}),)); db.commit(); db.close()
            refill.generate(value, snapshot, root/'four.json', state, detector); self.assertEqual(detector.calls, 3)
            model = Path(value['current_detector_run']); (model/'detector.pt').write_bytes(b'new verified model')
            worker.atomic_json(model/'metrics.json', {'model_sha256': hashlib.sha256(b'new verified model').hexdigest()})
            refill.generate(value, snapshot, root/'five.json', state, detector); self.assertEqual(detector.calls, 4)

    def test_scan_limit_applies_to_all_unreviewed_images(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); value, snapshot = fixture(root, 8); detector = FakeDetector(empty=True)
            value.update(review_refill_batch_size=2, review_refill_scan_limit=3)
            db = sqlite3.connect(snapshot); db.execute('UPDATE labels SET data=? WHERE image_id=8', (json.dumps({'defects': ['崩边', '划伤']}),)); db.commit(); db.close()
            result = refill.generate(value, snapshot, root/'output.json', root/'attempts.json', detector)
            self.assertEqual(result['summary']['eligible_images'], 8)
            self.assertEqual(result['summary']['eligible_for_current_model'], 8)
            self.assertEqual(result['summary']['attempted_images'], 3)
            self.assertEqual(result['summary']['remaining_eligible_images'], 5)

    def test_old_policy_attempts_do_not_suppress_new_policy_and_missing_source_is_not_proposed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); value, snapshot = fixture(root, 2)
            rows, _ = refill.select_refill_rows(snapshot)
            model_sha = refill.file_hash(Path(value['current_detector_run'])/'detector.pt')
            old_identity = [model_sha, rows[0]['image_id'], rows[0]['label_fingerprint'], value['source_root'],
                            rows[0]['root_id'], rows[0]['relative_path'], rows[0]['sha256'], rows[0]['size'], rows[0]['mtime_ns']]
            old_key = hashlib.sha256(json.dumps(old_identity, ensure_ascii=False).encode()).hexdigest()
            attempts = root/'attempts.json'
            worker.atomic_json(attempts, {'model_sha256': model_sha, 'attempts': {old_key: {'status': 'no_boxes'}}})
            (Path(value['source_root'])/'1.png').unlink()
            result = refill.generate(value, snapshot, root/'output.json', attempts, FakeDetector())
            self.assertEqual([item['image_id'] for item in result['items']], [2])
            self.assertEqual(result['summary']['attempted_images'], 2)
            self.assertEqual(result['summary']['errors'], 1)
            self.assertEqual(result['errors'][0]['image_id'], 1)
            self.assertEqual(worker.read_json(attempts)['selection_policy'], refill.SELECTION_POLICY)

    def test_enough_pending_never_queries_gpu_or_runs_inference(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); value = {**config(root), 'review_refill_enabled': True}; path = root/'config.json'; worker.atomic_json(path, value)
            with patch.object(refill, 'api', return_value={'pending': 10}), patch.object(refill, 'publish_status'), \
                 patch.object(worker, 'gpu_available', side_effect=AssertionError('GPU queried')), \
                 patch.object(worker, 'run_process', side_effect=AssertionError('inference')):
                result = refill.refill_cycle(path)
            self.assertEqual(result['status'], 'queue_sufficient')

    def test_busy_gpu_and_training_lock_wait_without_inference(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); value = {**config(root), 'review_refill_enabled': True}; path = root/'config.json'; worker.atomic_json(path, value)
            with patch.object(refill, 'api', return_value={'pending': 0}), patch.object(refill, 'publish_status'), \
                 patch.object(worker, 'gpu_available', return_value=(False, 'compute busy')), \
                 patch.object(worker, 'run_process', side_effect=AssertionError('inference')):
                self.assertEqual(refill.refill_cycle(path)['status'], 'waiting_for_gpu')
            with worker.ExclusiveLock(Path(value['runs_root'])/'.correction-cycle.lock'), patch.object(refill, 'publish_status'), \
                 patch.object(worker, 'gpu_available', side_effect=AssertionError('GPU queried')):
                self.assertEqual(refill.refill_cycle(path)['status'], 'waiting_for_worker')

    def test_check_only_does_not_create_files_or_query_gpu(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); value = {**config(root), 'review_refill_enabled': True}; path = root/'config.json'; worker.atomic_json(path, value)
            with patch.object(refill, 'api', return_value={'pending': 0}), \
                 patch.object(worker, 'gpu_available', side_effect=AssertionError('GPU queried')):
                self.assertEqual(refill.refill_cycle(path, check_only=True)['status'], 'checked')
            self.assertFalse(Path(value['runs_root']).exists())

    def test_registration_retry_does_not_repeat_inference_or_import(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); value = {**config(root), 'review_refill_enabled': True}; path = root/'config.json'; worker.atomic_json(path, value)
            state_path = Path(value['runs_root'])/'review-refill-state.json'
            worker.atomic_json(state_path, {'pending_registration': {'batch_id': 'refill-test', 'proposal_ids': ['one'],
                'metadata': {'source_model': 'verified'}, 'summary': {'generated_images': 1, 'generated_boxes': 2, 'eligible_images': 3, 'attempted_images': 1}, 'import': {'imported': 1}}})
            calls = []
            def endpoint(_, route, body=None):
                calls.append(route)
                if route.endswith('/register') and calls.count(route) == 1:
                    raise RuntimeError('temporary registration outage')
                return {'pending': 1}
            with patch.object(refill, 'api', side_effect=endpoint), patch.object(refill, 'publish_status'), \
                 patch.object(worker, 'gpu_available', side_effect=AssertionError('GPU queried')), \
                 patch.object(worker, 'ssh', side_effect=AssertionError('Import repeated')):
                self.assertEqual(refill.refill_cycle(path)['status'], 'error')
                self.assertIn('pending_registration', worker.read_json(state_path))
                second = refill.refill_cycle(path)
                self.assertEqual(second['status'], 'refilled'); self.assertEqual(second['imported_images'], 1)
            self.assertNotIn('pending_registration', worker.read_json(state_path))

    def test_full_refill_registers_only_actual_inserted_ids_and_never_refreshes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); value = {**config(root), 'review_refill_enabled': True}; path = root/'config.json'; worker.atomic_json(path, value)
            requests = []; generated = []
            def endpoint(_, route, body=None):
                requests.append((route, body))
                return {'pending': 0 if len(requests) == 1 else 1}
            def process(arguments, **_):
                arguments = list(map(str, arguments)); generated.append(arguments)
                self.assertIn('--infer', arguments)
                manifest = Path(arguments[arguments.index('--output')+1])
                worker.atomic_json(manifest, {'source_checkpoint_sha256': 'verified', 'model_id': 'location-verified',
                    'created_at': '2026-10-16T12:00:00Z', 'refresh_pending': False,
                    'items': [{'image_id': 1, 'boxes': [{}]}, {'image_id': 2, 'boxes': [{}]}],
                    'summary': {'eligible_images': 7, 'remaining_eligible_images': 5,
                                'attempted_images': 2, 'generated_images': 2, 'generated_boxes': 2}})
                return ''
            def remote(_, arguments, **__):
                if '-c' in arguments:
                    return json.dumps({'imported': 1, 'skipped': 1, 'imported_proposal_ids': ['actually-inserted'],
                        'human_data_unchanged': True, 'annotation_data_unchanged': True})
                return ''
            with patch.object(refill, 'api', side_effect=endpoint), patch.object(refill, 'publish_status'), \
                 patch.object(worker, 'gpu_available', return_value=(True, 'available')), \
                 patch.object(worker, 'run_process', side_effect=process), patch.object(worker, 'ssh', side_effect=remote), \
                 patch.object(worker, 'transfer'):
                result = refill.refill_cycle(path)
            self.assertEqual(result['status'], 'refilled'); self.assertEqual(result['imported_images'], 1)
            self.assertEqual(result['eligible_images'], 5); self.assertEqual(len(generated), 1)
            registration = [body for route, body in requests if route.endswith('/register')]
            self.assertEqual(registration[0]['proposal_ids'], ['actually-inserted'])
            self.assertEqual(worker.read_json(Path(value['runs_root'])/'review-refill-state.json'), {})

    def test_refill_only_never_starts_training_and_training_error_preserves_refill_success(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); value = {**config(root), 'review_refill_enabled': True}; path = root/'config.json'; worker.atomic_json(path, value)
            with patch.object(refill, 'refill_cycle', return_value={'status': 'refilled', 'imported_images': 2}), \
                 patch.object(worker, 'training_cycle', side_effect=RuntimeError('training failed')) as training:
                only = worker.cycle(path, refill_only=True); training.assert_not_called()
                self.assertEqual(only['training_status'], 'not_requested')
                combined = worker.cycle(path)
            self.assertEqual(combined['status'], 'review_refilled_training_error')
            self.assertEqual(combined['training_status'], 'error')
            self.assertEqual(combined['review_refill']['status'], 'refilled')

    def test_hard_limits_and_changed_checkpoint_fail_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); value, snapshot = fixture(root, 1)
            for key, bad in (('review_refill_batch_size', 51), ('review_refill_scan_limit', 201), ('review_refill_enabled', 1)):
                changed = copy.deepcopy(value); changed[key] = bad
                with self.assertRaises(ValueError):
                    refill.settings(changed)
            (Path(value['current_detector_run'])/'detector.pt').write_bytes(b'modified without validation')
            with self.assertRaisesRegex(ValueError, 'validated model metadata'):
                refill.generate(value, snapshot, root/'output.json', root/'attempts.json', FakeDetector())

    def test_location_only_model_blocks_instead_of_publishing_blank_types(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);value,snapshot=fixture(root,1);model=FakeDetector();model.class_agnostic=True
            with self.assertRaisesRegex(ValueError,'typed detector'):
                refill.generate(value,snapshot,root/'output.json',root/'attempts.json',model)
            self.assertEqual(model.calls,0)
            self.assertFalse((root/'output.json').exists());self.assertFalse((root/'attempts.json').exists())


if __name__ == '__main__':
    unittest.main()
