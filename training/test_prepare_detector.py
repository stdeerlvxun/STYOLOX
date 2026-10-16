"""Focused tests for detector eligibility, coordinates and independent holdouts."""
import hashlib
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest

import numpy as np
from PIL import Image

from prepare_detector import (COORDINATES, annotation_exclusion, collect_holdout_history, normalize_boxes, prepare,
                              split_records, verify_no_leakage)
from vision_classifier import assign_groups


class DetectorPreparationTests(unittest.TestCase):
    def setUp(self):
        self.label = json.dumps({'defects': ['崩边'], 'image_state': 'defect'}, ensure_ascii=False)
        self.doc = {'status': 'confirmed', 'outcome': 'boxed', 'coordinate_space': COORDINATES,
                    'label_fingerprint': hashlib.sha256(self.label.encode()).hexdigest(),
                    'boxes': [{'id': 'b1', 'problem': '崩边', 'x': .1, 'y': .2, 'w': .3, 'h': .4}]}

    def test_human_confirmation_is_required_even_for_machine_origin(self):
        self.doc['boxes'][0]['origin'] = 'model'
        self.assertEqual(annotation_exclusion(self.label, self.doc), '')
        self.doc['status'] = 'draft'
        self.assertEqual(annotation_exclusion(self.label, self.doc), 'annotation_not_confirmed')

    def test_every_selected_state_and_any_draft_are_excluded(self):
        for state in ('normal', 'unclear', 'unrelated'):
            label = json.dumps({'defects': ['崩边'], 'image_state': state})
            self.assertEqual(annotation_exclusion(label, self.doc), 'image_state_selected')
        self.assertEqual(annotation_exclusion(self.label, self.doc, True), 'unconfirmed_edit')
        self.doc['outcome'] = 'unclear'
        self.assertEqual(annotation_exclusion(self.label, self.doc), 'annotation_not_boxed')

    def test_raw_label_changes_invalidate_confirmation(self):
        self.assertEqual(annotation_exclusion(self.label + ' ', self.doc), 'classification_changed_since_confirmation')

    def test_coordinates_preserve_exact_normalized_and_pixel_boxes(self):
        box = normalize_boxes(self.doc['boxes'], 1000, 500)[0]
        self.assertEqual(box['normalized_xywh'], [.1, .2, .3, .4])
        self.assertEqual(box['xywh'], [100., 100., 300., 200.])
        self.assertEqual(box['xyxy'][:3], [100., 100., 400.])
        for value in (float('nan'), -1., 2.):
            changed = dict(self.doc['boxes'][0], x=value)
            with self.assertRaises(ValueError):
                normalize_boxes([changed], 1000, 500)

    def test_case_order_sha_and_near_duplicates_never_cross_splits(self):
        records = [{'image_id': i, 'case_id': 'case'+str(i), 'sha256': 'sha'+str(i),
                    'order_keys': ['o'+str(i)], 'phash': 0xFFFFFFFFFFFFFFFF << (i*4),
                    'aspect': 1+i*.1, 'boxes': [{'problem': '崩边'}]} for i in range(14)]
        records[1]['case_id'] = records[0]['case_id']
        records[3]['order_keys'] = records[2]['order_keys']
        records[5]['sha256'] = records[4]['sha256']
        records[7]['phash'] = records[6]['phash']
        records[7]['aspect'] = records[6]['aspect']
        assign_groups(records)
        split_records(records)
        verify_no_leakage(records)
        for a, b in ((0, 1), (2, 3), (4, 5), (6, 7)):
            self.assertEqual(records[a]['split'], records[b]['split'])
        records[1]['split'] = next(p for p in ('train', 'test') if p != records[0]['split'])
        with self.assertRaises(ValueError):
            verify_no_leakage(records)

    def test_historical_test_and_validation_stay_held_out(self):
        records = [{'image_id': i, 'case_id': 'c'+str(i), 'sha256': 's'+str(i), 'order_keys': [],
                    'group': i, 'boxes': [{'problem': '崩边'}]} for i in range(12)]
        historical = {'records': [dict(r, split='test' if i < 2 else 'validation')
                                 for i, r in enumerate(records[:4])]}
        split_records(records, historical)
        self.assertEqual([r['split'] for r in records[:4]], ['test', 'test', 'validation', 'validation'])

    def test_temporarily_excluded_holdouts_remain_protected_in_later_rounds(self):
        records=[{'image_id':i,'case_id':'c'+str(i),'sha256':'s'+str(i),'order_keys':[],
            'group':i,'boxes':[{'problem':'崩边'}]} for i in range(12)]
        old=[dict(row,split='test' if i<2 else 'validation') for i,row in enumerate(records[:4])]
        ledger=collect_holdout_history([],{'records':old})
        history={'records':[], 'historical_holdout_records':ledger}
        split_records(records,history)
        self.assertEqual([r['split'] for r in records[:4]],['test','test','validation','validation'])
        self.assertEqual(len(collect_holdout_history([],history)),4)

    def test_too_few_orders_stops_without_fake_split(self):
        records = [{'group': i, 'case_id': str(i), 'sha256': str(i), 'order_keys': [],
                    'boxes': [{'problem': '崩边'}]} for i in range(7)]
        with self.assertRaises(ValueError):
            split_records(records)

    def test_absent_historical_near_duplicate_stays_held_out(self):
        records=[{'image_id':i,'case_id':'new'+str(i),'sha256':'new'+str(i),'order_keys':[],
            'group':i,'phash':(1 << (i+1))-1,'aspect':1+i*.2,
            'boxes':[{'problem':'崩边'}]} for i in range(12)]
        old=dict(records[0],image_id=99,case_id='absent',sha256='absent',split='test')
        old['phash']=records[0]['phash'] ^ 1
        split_records(records,{'records':[], 'historical_holdout_records':[old]})
        self.assertEqual(records[0]['split'],'test')

    def test_prepare_is_read_only_checks_sha_and_preserves_all_box_categories(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            nas, run = root/'nas', root/'run'
            run.mkdir()
            snapshot = root/'snapshot.sqlite3'
            db = sqlite3.connect(snapshot)
            db.executescript('CREATE TABLE settings(key TEXT,value TEXT);INSERT INTO settings VALUES("root","1");'
                'CREATE TABLE images(id INTEGER,root_id INTEGER,relative_path TEXT,case_id TEXT,available INTEGER);'
                'CREATE TABLE labels(image_id INTEGER,data TEXT);CREATE TABLE drafts(image_id INTEGER);'
                'CREATE TABLE annotation_documents(image_id INTEGER,data TEXT,version INTEGER,updated_at TEXT);')
            rng = np.random.default_rng(123)
            originals = {}
            for i in range(13):
                relative = f'2026-10/case{i}/images/main/{i}.png'
                path = nas/relative
                path.parent.mkdir(parents=True)
                Image.fromarray(rng.integers(0, 255, (32+i, 48+i, 3), dtype=np.uint8)).save(path)
                raw = path.read_bytes()
                originals[path] = hashlib.sha256(raw).hexdigest()
                (path.parent.parent.parent/'case.json').write_text(json.dumps({'related_orders': [{'production_no': f'order{i}'}]}), encoding='utf-8')
                label = self.label
                doc = {**self.doc, 'source': {'sha256': originals[path], 'width': 48+i, 'height': 32+i},
                       'boxes': [dict(self.doc['boxes'][0], problem='人工新增框类别')]}
                if i == 12:
                    doc['source']['sha256'] = 'changed'
                db.execute('INSERT INTO images VALUES(?,?,?,?,1)', (i, 1, relative, f'case{i}'))
                db.execute('INSERT INTO labels VALUES(?,?)', (i, label))
                db.execute('INSERT INTO annotation_documents VALUES(?,?,1,"now")', (i, json.dumps(doc)))
            db.commit()
            db.close()
            before = hashlib.sha256(snapshot.read_bytes()).hexdigest()
            result = prepare(run, snapshot, nas, historical_path=None)
            self.assertEqual(len(result['records']), 12)
            self.assertEqual(result['summary']['excluded_reasons'], {'source_hash_mismatch': 1})
            self.assertEqual(result['categories'], [{'id': 0, 'name': '人工新增框类别'}])
            self.assertEqual(before, hashlib.sha256(snapshot.read_bytes()).hexdigest())
            self.assertTrue(all(hashlib.sha256(p.read_bytes()).hexdigest() == h for p, h in originals.items()))
            with self.assertRaises(ValueError):
                prepare(run, snapshot, nas, historical_path=None)


if __name__ == '__main__':
    unittest.main()
