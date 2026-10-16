import copy
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch
import json
import sqlite3
import numpy as np
import torch
from vision_classifier import (assign_groups, split_groups, problem_metrics, image_tensor,
    training_exclusion_reason, validate_training_dataset, prepare, train, predict, TRAINING_POLICY)


class DataIntegrityTest(unittest.TestCase):
    def test_only_problem_points_are_eligible_and_states_always_excluded(self):
        for state in ['normal', 'unclear', 'unrelated', 'unknown_state']:
            self.assertEqual(training_exclusion_reason({'image_state': state, 'defects': ['崩边']}), 'image_state_selected')
        for state in ['', 'defect']:
            self.assertEqual(training_exclusion_reason({'image_state': state, 'defects': ['仅一张的新类别'], 'responsibilities': ['待核']}), '')
        self.assertEqual(training_exclusion_reason({'image_state': '', 'defects': [], 'responsibilities': ['生产']}), 'missing_problem_labels')
        self.assertEqual(training_exclusion_reason({'image_state': 'defect', 'defects': ['崩边']}, True), 'unconfirmed_edit')
        with self.assertRaisesRegex(ValueError, 'superseded'):
            validate_training_dataset({'records': []})
        with self.assertRaisesRegex(ValueError, 'image_state_selected'):
            validate_training_dataset({'training_policy': TRAINING_POLICY, 'records': [{'image_id': 1, 'image_state': 'unrelated', 'defects': ['崩边']}]})

    def test_prepare_preserves_one_sample_category_and_excludes_states(self):
        from PIL import Image
        with tempfile.TemporaryDirectory() as folder:
            base = Path(folder); root = base / 'cases'; run = base / 'run'
            (run / 'source').mkdir(parents=True)
            db = sqlite3.connect(run / 'source' / 'labels.sqlite3')
            db.executescript('CREATE TABLE images(id INTEGER,relative_path TEXT,case_id TEXT); CREATE TABLE labels(image_id INTEGER,data TEXT,updated_at TEXT); CREATE TABLE drafts(image_id INTEGER);')
            for i in range(16):
                case = root / '2026-10' / str(i); images = case / 'images' / 'main'; images.mkdir(parents=True)
                image = images / 'sample.png'
                Image.fromarray(np.random.default_rng(i).integers(0, 256, (40, 50, 3), dtype=np.uint8)).save(image)
                (case / 'case.json').write_text('{}')
                state = 'defect' if i < 12 else ['normal', 'unclear', 'unrelated', ''][i - 12]
                label = {'image_state': state, 'defects': ['崩边'] + (['单样本问题'] if i == 0 else []) if i < 15 else [], 'responsibilities': ['待核']}
                db.execute('INSERT INTO images VALUES(?,?,?)', (i, image.relative_to(root).as_posix(), str(i)))
                db.execute('INSERT INTO labels VALUES(?,?,?)', (i, json.dumps(label), 'now'))
            db.commit(); db.close()
            prepare(run, root)
            dataset = json.loads((run / 'dataset.json').read_text(encoding='utf-8'))
            self.assertEqual(len(dataset['records']), 12)
            self.assertEqual(dataset['summary']['excluded_reasons'], {'image_state_selected': 3, 'missing_problem_labels': 1})
            self.assertIn('单样本问题', dataset['summary']['problem_classes'])
            self.assertTrue(any(r['split'] == 'train' and '单样本问题' in r['defects'] for r in dataset['records']))
            validate_training_dataset(dataset)

    def test_problem_only_train_predict_and_old_model_guard(self):
        with tempfile.TemporaryDirectory() as folder:
            run = Path(folder)
            rows = [{'image_id': i, 'relative_path': str(i), 'image_state': 'defect',
                     'defects': ['崩边'] + (['罕见问题'] if i == 0 else []),
                     'split': 'train' if i < 4 else 'validation' if i < 6 else 'test'} for i in range(8)]
            dataset = {'training_policy': TRAINING_POLICY, 'seed': 42, 'records': rows, 'summary': {'problem_classes': ['崩边', '罕见问题']}}
            (run / 'dataset.json').write_text(json.dumps(dataset), encoding='utf-8')
            cache = {'features': torch.arange(32).reshape(8, 4).float(), 'base_sha256': 'base', 'input_size': 640,
                     'manifest_hash': 'manifest', 'feature_extraction': {'device': 'synthetic-test'}}
            with patch('vision_classifier.extract', return_value=cache):
                train(run, epochs=2, balanced_init=True)
            model = torch.load(run / 'classifier.pt', weights_only=True)
            self.assertEqual(model['training_policy'], TRAINING_POLICY)
            self.assertNotIn('state_dict', model)
            self.assertNotIn('states', model)
            self.assertEqual(model['training_config'], {'balanced_init': True, 'learning_rate': .0005, 'weight_decay': .05})
            image = run / 'fixture'; image.write_bytes(b'fixture')
            with patch('vision_classifier.load_backbone', return_value=(torch.nn.Identity(), 'base')), patch('vision_classifier.image_tensor', return_value=torch.ones(3, 2, 2)), patch('vision_classifier.features', return_value=torch.ones(1, 4)):
                predict(run, image, run / 'prediction.json')
            result = json.loads((run / 'prediction.json').read_text(encoding='utf-8'))
            self.assertNotIn('predicted_state', result)
            self.assertNotIn('rule_based_default_responsibilities', result)
            self.assertNotIn('responsibility_source', result)
            self.assertEqual(set(result['problem_scores']), {'崩边', '罕见问题'})
            torch.save({'schema_version': 1}, run / 'classifier.pt')
            with self.assertRaisesRegex(ValueError, 'superseded'):
                predict(run, image, run / 'should-not-exist.json')
            self.assertFalse((run / 'should-not-exist.json').exists())

    def test_linked_orders_and_duplicates_do_not_leak(self):
        rows = []
        for i in range(40):
            rows.append({'case_id': 'case' + str(i), 'order_keys': ['order' + str(i // 2)],
                         'sha256': 'hash' + str(i), 'phash': (1 << i) - 1,
                         'aspect': 1 + i, 'defects': ['崩边'] if i % 3 == 0 else [],
                         'image_state': ['defect', 'unrelated', 'unclear'][i % 3]})
        rows[20]['sha256'] = rows[0]['sha256']
        rows[22]['phash'], rows[22]['aspect'] = rows[2]['phash'], rows[2]['aspect']
        assign_groups(rows)
        split_groups(rows)
        self.assertEqual(rows[0]['split'], rows[20]['split'])
        self.assertEqual(rows[2]['split'], rows[22]['split'])
        for i in range(0, 40, 2):
            self.assertEqual(rows[i]['split'], rows[i + 1]['split'])
        first = [r['split'] for r in rows]
        split_groups(rows)
        self.assertEqual(first, [r['split'] for r in rows])
        self.assertEqual(set(first), {'train', 'validation', 'test'})

    def test_source_change_stops_feature_extraction(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'changed.jpg'
            path.write_bytes(b'changed-source')
            with self.assertRaisesRegex(ValueError, 'changed after snapshot'):
                image_tensor({'image_id': 1, 'full_path': str(path), 'sha256': 'old-hash'}, 640)

    def test_metrics_keep_false_positives_and_missing_support_visible(self):
        metrics = problem_metrics(np.array([[1, 0], [0, 0]]), np.array([[.9, .8], [.8, .1]]), ['崩边', '缺角'])
        self.assertEqual(metrics[0]['precision'], .5)
        self.assertEqual(metrics[0]['recall'], 1)
        self.assertEqual(metrics[1]['support'], 0)
        self.assertIsNone(metrics[1]['recall'])
        self.assertEqual(metrics[1]['fp'], 1)


if __name__ == '__main__':
    unittest.main()
