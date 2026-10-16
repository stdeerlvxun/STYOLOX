"""Focused geometry, data-policy and source-preservation checks for weak localization."""
import hashlib
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest

import numpy as np
from PIL import Image, ImageOps
import torch

from generate_box_proposals import (fingerprint, heatmap_boxes, image_views, project_cams,
                                     safe_source_path, select_images)
from improve_classifier import crop_tensors


class ProposalTests(unittest.TestCase):
    def test_snapshot_selection_preserves_raw_fingerprint_and_excludes_all_human_documents(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'labels.sqlite3'
            db = sqlite3.connect(path)
            db.executescript('''CREATE TABLE settings(key TEXT,value TEXT);
                INSERT INTO settings VALUES('root','1');
                CREATE TABLE images(id INTEGER,relative_path TEXT,sha256 TEXT,reported_at TEXT,available INTEGER,root_id INTEGER);
                CREATE TABLE labels(image_id INTEGER,data TEXT);
                CREATE TABLE drafts(image_id INTEGER);
                CREATE TABLE annotation_documents(image_id INTEGER);''')
            raw = '{"image_state": "defect", "defects": ["崩边"], "note": "原始空格"}'
            for identity in range(1, 9):
                root = 2 if identity == 8 else 1
                state = {2: 'normal', 3: 'unclear', 4: 'unrelated'}.get(identity, 'defect')
                label = raw if identity == 1 else json.dumps({'image_state': state, 'defects': ['崩边']})
                db.execute('INSERT INTO images VALUES(?,?,?,?,?,?)', (identity, f'{identity}.jpg', None,
                           '2026-10-16', identity != 7, root))
                db.execute('INSERT INTO labels VALUES(?,?)', (identity, label))
            db.execute('INSERT INTO drafts VALUES(5)')
            db.execute('INSERT INTO annotation_documents VALUES(6)')
            db.commit(); db.close()
            before = path.read_bytes()
            rows, summary = select_images(path)
            self.assertEqual([row['image_id'] for row in rows], [1])
            self.assertEqual(rows[0]['label_fingerprint'], fingerprint(raw))
            self.assertNotEqual(fingerprint(raw), fingerprint(json.dumps(json.loads(raw))))
            self.assertEqual(summary['excluded']['image_state_selected'], 3)
            self.assertEqual(summary['excluded']['existing_human_annotation'], 1)
            self.assertEqual(path.read_bytes(), before)

    def test_six_views_exactly_match_the_frozen_classifier_including_odd_padding(self):
        with tempfile.TemporaryDirectory() as folder:
            for width, height in ((91, 224), (224, 91), (1279, 1706), (105, 105)):
                path = Path(folder)/f'{width}x{height}.png'
                image = Image.fromarray(np.random.default_rng(3).integers(0, 256, (height, width, 3), dtype='uint8'))
                image.save(path)
                raw = path.read_bytes()
                expected = crop_tensors({'full_path': str(path), 'sha256': hashlib.sha256(raw).hexdigest()})
                actual, geometry = image_views(image)
                self.assertTrue(torch.equal(actual, expected))
                fitted = ImageOps.contain(image, (224, 224))
                self.assertEqual(geometry[0]['content'], [round((224-fitted.width)*.5),
                                 round((224-fitted.height)*.5), fitted.width, fitted.height])
                self.assertEqual(path.read_bytes(), raw)

    def test_crop_projection_returns_to_exif_source_coordinates(self):
        maps = np.ones((1, 14, 14), dtype=np.float32)
        result = project_cams(maps, [{'kind': 'crop', 'region': [100, 200, 200, 200]}], (400, 600), 600)
        self.assertEqual(result.shape, (600, 400))
        self.assertEqual(float(result[210, 110]), 1)
        self.assertEqual(float(result[190, 110]), 0)
        self.assertEqual(float(result[410, 110]), 0)

    def test_pad_projection_removes_letterbox_activation(self):
        maps = np.zeros((1, 224, 224), dtype=np.float32)
        maps[0, :, :56] = 1
        result = project_cams(maps, [{'kind': 'pad', 'content': [56, 0, 112, 224]}], (100, 200))
        self.assertEqual(float(result.max()), 0)

    def test_zero_diffuse_nan_and_whole_image_are_not_defect_boxes(self):
        for heatmap in (np.zeros((100, 100), np.float32), np.ones((100, 100), np.float32),
                        np.full((100, 100), np.nan, np.float32)):
            boxes, reason = heatmap_boxes(heatmap, '缺角', .5)
            self.assertEqual(boxes, [])
            self.assertTrue(reason)

    def test_boundary_and_corner_attention_abstains_instead_of_watermark_box(self):
        for region in ((0, 25, 15, 45), (85, 85, 98, 98)):
            heatmap = np.zeros((100, 100), np.float32)
            x0, y0, x1, y1 = region
            heatmap[y0:y1, x0:x1] = 1
            boxes, _ = heatmap_boxes(heatmap, '其他破损', .4)
            self.assertEqual(boxes, [])

    def test_separate_interior_components_keep_original_class_and_model_origin(self):
        heatmap = np.zeros((100, 200), np.float32)
        heatmap[30:40, 50:70] = 1
        heatmap[65:75, 140:150] = .8
        boxes, reason = heatmap_boxes(heatmap, '崩边', .4)
        self.assertEqual(len(boxes), 2)
        self.assertEqual(reason, '')
        for box in boxes:
            self.assertEqual(box['problem'], '崩边')
            self.assertEqual(box['origin'], 'model')
            self.assertGreater(box['w'], 0)
            self.assertLessEqual(box['x']+box['w'], 1)
            self.assertLessEqual(box['y']+box['h'], 1)

    def test_source_traversal_and_absolute_paths_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            for relative in ('../image.jpg', '/image.jpg', 'C:/image.jpg', '..\\image.jpg', '//server/image.jpg'):
                with self.assertRaises(ValueError):
                    safe_source_path(folder, relative)
            self.assertEqual(safe_source_path(folder, '2026-10/case/image.jpg'),
                             Path(folder).resolve()/'2026-10/case/image.jpg')


if __name__ == '__main__':
    unittest.main()
