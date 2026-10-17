import copy
import json
from pathlib import Path
import tempfile
import unittest

from annotation_audit import audit_annotations, write_audit


class AnnotationAuditTests(unittest.TestCase):
    def test_categories_are_distinct_overlapping_and_never_automatic_errors(self):
        dataset = {'records': [
            {'image_id': 1, 'split': 'train', 'relative_path': 'one.jpg', 'defects': ['scratch'],
             'boxes': [{'problem': 'chip'}]},
            {'image_id': 2, 'split': 'train', 'defects': ['scratch', 'chip'], 'boxes': [{'problem': 'chip'}]},
            {'image_id': 3, 'split': 'validation', 'defects': ['scratch'], 'boxes': [{'problem': 'chip'}]},
            {'image_id': 4, 'split': 'test', 'defects': ['chip'], 'boxes': [{'problem': 'chip'}, {'problem': 'glue'}]},
            {'image_id': 5, 'split': 'train', 'defects': ['chip'], 'boxes': [{'problem': 'chip'}]}]}
        before = copy.deepcopy(dataset)
        report = audit_annotations(dataset)
        self.assertEqual(dataset, before)
        self.assertEqual(len(report['items']), 4)
        self.assertEqual(report['split_counts']['train']['unique_images'], 2)
        self.assertIn('possible_type_difference', report['items'][0]['audit_categories'])
        self.assertNotIn('possible_type_difference', report['items'][1]['audit_categories'])
        self.assertTrue(all(row['optimization_allowed'] == (row['split'] == 'train') for row in report['items']))
        self.assertTrue(all(row['annotation_error_confirmed'] is False for row in report['items']))

    def test_missing_source_image_paths_are_not_opened_and_holdouts_not_in_html(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = {'records': [
                {'image_id': 1, 'split': 'train', 'relative_path': 'train-only.jpg', 'defects': ['A'], 'boxes': []},
                {'image_id': 2, 'split': 'test', 'relative_path': 'unopened-test.jpg', 'defects': ['A'], 'boxes': []}]}
            path = root / 'dataset.json'
            path.write_text(json.dumps(dataset), encoding='utf-8')
            write_audit(path, root / 'audit')
            html = (root / 'audit/训练标注对应关系核对清单.html').read_text(encoding='utf-8')
            self.assertIn('train-only.jpg', html)
            self.assertNotIn('unopened-test.jpg', html)
            holdout = json.loads((root / 'audit/holdout-metadata-only.json').read_text(encoding='utf-8'))
            self.assertFalse(holdout['image_pixels_read'])
            self.assertFalse(holdout['optimization_allowed'])
            self.assertEqual(holdout['items'][0]['image_id'], 2)


if __name__ == '__main__':
    unittest.main()
