import copy
from contextlib import closing
import json
from pathlib import Path
import sqlite3
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from audit_annotations import audit, exclusion, fingerprint


class AuditTests(unittest.TestCase):
    def setUp(self):
        self.raw = json.dumps({'image_state': 'defect', 'defects': ['崩边']}, ensure_ascii=False)
        self.doc = {'status': 'confirmed', 'outcome': 'boxed', 'label_fingerprint': fingerprint(self.raw),
                    'coordinate_space': 'exif_transposed_normalized_xywh',
                    'boxes': [{'problem': '崩边', 'x': .1, 'y': .1, 'w': .2, 'h': .3}]}

    def test_confirmed_problem_candidate(self):
        self.assertEqual(exclusion(self.raw, False, self.doc), '')

    def test_all_selected_states_excluded(self):
        for state in ('normal', 'unrelated', 'unclear'):
            raw = json.dumps({'image_state': state, 'defects': ['崩边']})
            self.assertEqual(exclusion(raw, False, self.doc), 'image_state_selected')

    def test_classification_draft_excluded(self):
        self.assertEqual(exclusion(self.raw, True, self.doc), 'unconfirmed_classification_edit')

    def test_unconfirmed_annotation_excluded(self):
        self.doc['status'] = 'draft'
        self.assertEqual(exclusion(self.raw, False, self.doc), 'annotation_not_confirmed')

    def test_changed_classification_excluded(self):
        self.assertEqual(exclusion(self.raw + ' ', False, self.doc), 'classification_changed')

    def test_invalid_boxes_excluded(self):
        for value in (True, float('nan'), 2):
            doc = copy.deepcopy(self.doc)
            doc['boxes'][0]['w'] = value
            self.assertEqual(exclusion(self.raw, False, doc), 'invalid_coordinates')

    def test_empty_database_is_read_only_and_not_ready(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'labels.sqlite3'
            with closing(sqlite3.connect(path)) as db:
                db.executescript("""CREATE TABLE settings(key TEXT,value TEXT);
                    INSERT INTO settings VALUES('root','1'); CREATE TABLE labels(image_id INTEGER,data TEXT);
                    CREATE TABLE drafts(image_id INTEGER); CREATE TABLE annotation_documents(image_id INTEGER,data TEXT);
                    CREATE TABLE annotation_history(id INTEGER); CREATE TABLE images(id INTEGER,root_id INTEGER,
                    relative_path TEXT,available INTEGER);""")
                db.commit()
            original = path.read_bytes()
            result = audit(path)
            self.assertFalse(result['ready_for_training'])
            self.assertEqual(result['policy_candidate_images'], 0)
            self.assertEqual(path.read_bytes(), original)


if __name__ == '__main__':
    unittest.main()
