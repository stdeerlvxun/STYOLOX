import hashlib
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest

import torch
from PIL import Image

from roi_feedback import (anchor_centers, feedback_policy, heldout_overlap, prepare_feedback,
                          region_objectness_loss, snapshot_annotation_exclusion, transform_rois)
from train_detector import transferred_state


class NormalRegionFeedbackTests(unittest.TestCase):
    def test_unknown_pixels_ignored_normal_roi_restored_and_positive_wins(self):
        centers = torch.tensor([[1., 1.], [2., 2.], [8., 8.], [9., 9.]])
        truth = torch.tensor([[0.], [1.], [0.], [1.]])
        value = region_objectness_loss(torch.ones_like(truth), truth, [True],
                                      [[[0, 0, 5, 5]]], centers, weight=2.)
        self.assertEqual(value.flatten().tolist(), [2., 1., 0., 1.])

    def test_negative_roi_gradients_do_not_teach_unknown_region_as_normal(self):
        logits = torch.zeros((4, 1), requires_grad=True)
        truth = torch.zeros_like(logits)
        raw = torch.nn.functional.binary_cross_entropy_with_logits(logits, truth, reduction='none')
        loss = region_objectness_loss(raw, truth, [True], [[[0, 0, 4, 4]]],
            torch.tensor([[1., 1.], [3., 3.], [5., 5.], [7., 7.]]))
        loss.sum().backward()
        self.assertEqual(logits.grad.flatten().tolist(), [1., 1., 0., 0.])

    def test_complete_positive_images_and_classification_loss_unchanged(self):
        truth = torch.tensor([[0.], [1.]])
        self.assertEqual(region_objectness_loss(torch.ones_like(truth), truth, [False]).flatten().tolist(), [1., 1.])
        truth = torch.ones((2, 9))
        self.assertEqual(region_objectness_loss(torch.ones_like(truth), truth, [True, True]).sum().item(), 18.)

    def test_horizontal_flip_and_resize_use_identical_roi_geometry(self):
        self.assertEqual(transform_rois([[10, 20, 30, 40]], 100, 2., True), [[140., 40., 180., 80.]])
        self.assertEqual(transform_rois([[10, 20, 30, 40]], 100, 2., False), [[20., 40., 60., 80.]])

    def test_anchor_order_matches_yolox_pyramid_row_major(self):
        centers = anchor_centers(32, 'cpu')
        self.assertEqual(centers.shape, (21, 2))
        self.assertEqual(centers[0].tolist(), [4., 4.])
        self.assertEqual(centers[4].tolist(), [4., 12.])
        self.assertEqual(centers[16].tolist(), [8., 8.])
        self.assertEqual(centers[-1].tolist(), [16., 16.])

    def test_states_drafts_and_changed_fingerprints_remain_excluded(self):
        raw = json.dumps({'defects': ['test'], 'image_state': 'defect'})
        feedback = {'type': 'human_reported_local_false_positive', 'label_scope': 'region_only',
            'whole_image_normal': False, 'human_statement': 'normal process hole',
            'original_classification_fingerprint': hashlib.sha256(raw.encode()).hexdigest()}
        self.assertEqual(feedback_policy(feedback, raw), '')
        self.assertEqual(feedback_policy(feedback, raw, True), 'unconfirmed_edit')
        for state in ('normal', 'unrelated', 'unclear'):
            changed = json.dumps({'defects': ['test'], 'image_state': state})
            self.assertEqual(feedback_policy(feedback, changed), 'image_state_selected')
        self.assertEqual(feedback_policy(feedback, raw+' '), 'classification_changed_since_feedback')
        self.assertEqual(feedback_policy(dict(feedback, whole_image_normal=True), raw), 'ambiguous_normal_region_scope')

    def test_case_order_sha_and_similar_holdouts_block_feedback_training(self):
        record = {'case_id': 'a', 'sha256': 's', 'order_keys': ['order'], 'phash': '15', 'aspect': 1.}
        for updates in ({'case_id': 'a'}, {'sha256': 's'}, {'order_keys': ['order']}, {'phash': '14'}):
            other = dict(record, case_id='b', sha256='t', order_keys=[], phash='65520', split='test')
            other.update(updates)
            self.assertEqual(len(heldout_overlap(record, [], [other])), 1)
        self.assertEqual(heldout_overlap(record, [], [dict(record, split='train')]), [])

    def test_transfer_matches_issue_names_and_leaves_new_class_initialized(self):
        target = {'head.cls_preds.0.bias': torch.tensor([0., 0., 0.]), 'head.obj.weight': torch.ones(1)}
        checkpoint = {'categories': [{'id': 0, 'name': 'A'}, {'id': 1, 'name': 'B'}],
            'model': {'head.cls_preds.0.bias': torch.tensor([1., 2.]), 'head.obj.weight': torch.tensor([3.])}}
        categories = [{'id': 0, 'name': 'B'}, {'id': 1, 'name': 'new'}, {'id': 2, 'name': 'A'}]
        result = transferred_state(target, checkpoint, categories)
        self.assertEqual(result['head.cls_preds.0.bias'].tolist(), [2., 0., 1.])
        self.assertEqual(result['head.obj.weight'].item(), 3.)

    def test_backend_feedback_export_revalidated_and_kept_separate_from_positive_truth(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root/'nas/period/case/images/main/source.png'
            path.parent.mkdir(parents=True)
            Image.new('RGB', (100, 50), 'gray').save(path)
            (path.parent.parent.parent/'case.json').write_text('{}', encoding='utf-8')
            raw = json.dumps({'defects': ['scratch'], 'image_state': 'defect'})
            source = {'sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'width': 100, 'height': 50}
            relative = 'period/case/images/main/source.png'
            snapshot = root/'snapshot.sqlite3'
            with sqlite3.connect(snapshot) as db:
                db.executescript('CREATE TABLE settings(key TEXT,value TEXT);INSERT INTO settings VALUES("root","1");'
                    'CREATE TABLE images(id INTEGER,root_id INTEGER,relative_path TEXT,case_id TEXT,available INTEGER);'
                    'CREATE TABLE labels(image_id INTEGER,data TEXT);CREATE TABLE drafts(image_id INTEGER);'
                    'CREATE TABLE annotation_documents(image_id INTEGER,data TEXT,version INTEGER);')
                db.execute('INSERT INTO images VALUES(7,1,?,?,1)', (relative, 'case'))
                db.execute('INSERT INTO labels VALUES(7,?)', (raw,))
            db.close()
            item = {'feedback_id': 8, 'image_id': 7, 'relative_path': relative, 'source': source,
                'training_eligible_by_label_policy': True, 'feedback': {'status': 'confirmed',
                    'coordinate_space': 'exif_transposed_normalized_xywh', 'whole_image_normal': False,
                    'label_fingerprint': hashlib.sha256(raw.encode()).hexdigest(),
                    'negative_regions': [{'id': 'roi', 'x': .1, 'y': .2, 'w': .3, 'h': .4, 'reason': 'process_hole'}]}}
            export = root/'feedback.json'
            export.write_text(json.dumps({'items': [item]}), encoding='utf-8')
            result = prepare_feedback([export], snapshot, root/'nas', [])
            self.assertEqual(result['included_feedback_ids'], [8])
            self.assertEqual(result['records'][0]['boxes'], [])
            self.assertEqual(result['records'][0]['negative_rois'][0][:3], [10., 10., 40.])
            self.assertEqual(result['records'][0]['split'], 'train')
            with sqlite3.connect(snapshot) as db:
                db.execute('INSERT INTO drafts VALUES(7)')
            db.close()
            result = prepare_feedback([export], snapshot, root/'nas', [])
            self.assertEqual(result['included_feedback_ids'], [])
            self.assertEqual(result['excluded'][0]['reason'], 'unconfirmed_edit')

    def test_exported_review_cannot_reuse_snapshot_draft_unclear_or_later_version(self):
        feedback={'feedback_kind':'annotation_review','annotation_version':3,
            'original_classification_fingerprint':'labels','source':{'sha256':'image'}}
        document={'status':'confirmed','outcome':'boxed','label_fingerprint':'labels','source':{'sha256':'image'}}
        self.assertEqual(snapshot_annotation_exclusion(feedback,document,3),'')
        self.assertEqual(snapshot_annotation_exclusion(feedback,document,4),'annotation_changed_since_feedback_export')
        self.assertEqual(snapshot_annotation_exclusion(feedback,None,0),'annotation_changed_since_feedback_export')
        for updates in ({'status':'draft'},{'outcome':'unclear'}):
            self.assertEqual(snapshot_annotation_exclusion(feedback,{**document,**updates},3),
                'annotation_not_current_confirmed_or_unclear')

    def test_standalone_normal_roi_without_frames_is_allowed_but_stale_document_is_not(self):
        feedback={'feedback_kind':'region_correction','original_classification_fingerprint':'labels','source':{'sha256':'image'}}
        self.assertEqual(snapshot_annotation_exclusion(feedback,None,0),'')
        document={'status':'confirmed','outcome':'boxed','label_fingerprint':'old','source':{'sha256':'image'}}
        self.assertEqual(snapshot_annotation_exclusion(feedback,document,1),'annotation_classification_changed')
        document.update(label_fingerprint='labels',source={'sha256':'old-image'})
        self.assertEqual(snapshot_annotation_exclusion(feedback,document,1),'annotation_source_changed')


if __name__ == '__main__':
    unittest.main()
