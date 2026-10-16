from pathlib import Path
import hashlib
import json
import tempfile
import unittest
from unittest.mock import patch
import torch
from torch import nn
from PIL import Image
from improve_classifier import (crop_tensors, corrected_logits, fit_ridge, ridge_scores,
                                select_threshold, compare, digest)
from vision_classifier import TRAINING_POLICY


class ImprovementTest(unittest.TestCase):
    def test_recall_training_changes_coefficients_and_solves_weighted_objective(self):
        from recall_classifier import fit_recall_ridge
        x=torch.tensor([[1.,0.,0.],[0.,1.,0.],[0.,0.,1.],[1.,1.,0.]])
        y=torch.tensor([[1.,1.],[1.,0.],[1.,0.],[0.,0.]])
        baseline=fit_ridge(x,y,1.)
        control=fit_recall_ridge(x,y,1.,1.)
        self.assertTrue(torch.allclose(baseline['alpha'],control['alpha']))
        model=fit_recall_ridge(x,y,1.,2.)
        self.assertGreater(float((model['alpha']-baseline['alpha']).norm()),.01)
        self.assertEqual(tuple(model['alpha'].shape),(4,2))
        vectors=model['training_vectors'];kernel=vectors@vectors.T
        for c in range(2):
            weights=1+y[:,c]
            left=(kernel+torch.diag(1/weights))@model['alpha'][:,c]
            self.assertTrue(torch.allclose(left,y[:,c]-model['prior'][c],atol=1e-6))

    def test_recall_training_selection_prioritizes_recall_with_bounded_candidates(self):
        from recall_classifier import choose_candidate
        def candidate(cost,recall,precision,count):
            return {'positive_cost':cost,'validation':{'recall':recall,'precision':precision,'candidates_per_image':count}}
        precision_first=candidate(2,.7,.9,1.)
        review=candidate(4,.9,.3,3.)
        too_many=candidate(8,1.,.1,10.)
        self.assertIs(choose_candidate([precision_first,review,too_many]),review)

    def test_recall_loss_learns_and_penalizes_missed_positive_more_than_extra_candidate(self):
        from finetune_classifier import recall_first_loss
        target=torch.tensor([[1.,0.]])
        logits=torch.zeros(1,2,requires_grad=True)
        loss=recall_first_loss(logits,target,torch.ones(2))
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertLess(float(logits.grad[0,0]),0)
        self.assertGreater(float(logits.grad[0,1]),0)
        self.assertGreater(abs(float(logits.grad[0,0])),abs(float(logits.grad[0,1])))
        improved=torch.tensor([[1.,-1.]])
        self.assertLess(float(recall_first_loss(improved,target,torch.ones(2))),float(loss))

    def test_recall_first_cutoff_reduces_missed_problems(self):
        truth=[[1],[1],[1],[1],[0],[0],[0],[0],[0]]
        scores=[[.95],[.9],[.2],[.15],[.3],[.25],[.21],[.18],[.17]]
        balanced=select_threshold(truth,scores,beta=1)
        review=select_threshold(truth,scores)
        self.assertGreater(review['recall'],balanced['recall'])
        self.assertEqual(review['fn'],0)
        self.assertEqual(review['all_issues_covered_images'],9)
        self.assertGreater(review['fp'],balanced['fp'])

    def test_mil_tail_learns_from_bags_without_requiring_every_crop_to_be_positive(self):
        from finetune_classifier import MILTail
        from torchvision.models.convnext import LayerNorm2d
        torch.set_num_threads(4)
        tail=nn.Conv2d(768,768,1)
        model=MILTail(tail,LayerNorm2d(768),19)
        logits=model(torch.randn(2,6,768,2,2))
        self.assertEqual(tuple(logits.shape),(2,19))
        target=torch.zeros(2,19);target[0,0]=1;target[1,18]=1
        nn.BCEWithLogitsLoss()(logits,target).backward()
        self.assertIsNotNone(tail.weight.grad)
        self.assertGreater(float(tail.weight.grad.norm()),0)
        self.assertGreater(float(model.head.weight.grad.norm()),0)

    def test_weight_correction_does_not_treat_weighted_sigmoid_as_probability(self):
        # 2 positives / 10 images, pos_weight=4: weighted sigmoid .5 corresponds to p=.2.
        self.assertAlmostEqual(corrected_logits(torch.zeros(1,1),torch.tensor([2.]),10).item(),.2,places=6)

    def test_ridge_retains_rare_label_and_roundtrips(self):
        x=torch.tensor([[1.,0.,0.],[0.,1.,0.],[0.,0.,1.],[1.,1.,0.]])
        y=torch.tensor([[1.,1.],[1.,0.],[1.,0.],[1.,0.]])
        model=fit_ridge(x,y,1.)
        self.assertEqual(tuple(model['alpha'].shape),(4,2))
        self.assertTrue(torch.allclose(model['center'],x.mean(0)))
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder)/'model.pt';torch.save(model,p)
            restored=torch.load(p,weights_only=True)
            self.assertTrue(torch.allclose(ridge_scores(model,x),ridge_scores(restored,x)))

    def test_local_views_keep_original_file_unchanged(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'source.png';Image.new('RGB',(64,128),'red').save(path)
            before=path.read_bytes(); row={'full_path':str(path),'sha256':digest(path)}
            views=crop_tensors(row)
            self.assertEqual(tuple(views.shape),(6,3,224,224))
            self.assertEqual(path.read_bytes(),before)
            path.write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError,'changed'):
                crop_tensors(row)

    def test_model_and_threshold_selection_ignore_test_labels(self):
        with tempfile.TemporaryDirectory() as folder:
            base=Path(folder);source=base/'source';(source/'candidate-b').mkdir(parents=True)
            rows=[{'image_id':i,'relative_path':str(i),'image_state':'defect','defects':['common']+(['rare'] if i==0 else []),
                   'split':'train' if i<4 else 'validation' if i<6 else 'test','group':i}
                  for i in range(8)]
            dataset={'training_policy':TRAINING_POLICY,'records':rows,'summary':{'problem_classes':['common','rare']}}
            ds=source/'dataset.json';ds.write_text(json.dumps(dataset))
            x=torch.arange(24).reshape(8,3).float()
            cached={'features':x,'manifest_hash':digest(ds),'base_sha256':'yolo'}
            torch.save(cached,source/'features.pt')
            head=torch.nn.Linear(3,2)
            torch.save({'feature_mean':x[:4].mean(0),'feature_std':x[:4].std(0),'problem_state_dict':head.state_dict()},source/'candidate-b'/'classifier.pt')
            conv={'global':x+1,'multi':torch.cat([x,x],dim=1),'encoder_sha256':'conv','seconds':0,'peak_allocated_mib':0}
            out1=base/'one'
            with patch('improve_classifier.extract_classification_features',return_value=conv):
                compare(source,out1)
            first=json.loads((out1/'selection.json').read_text())
            for row in rows:
                if row['split']=='test':row['defects']=['rare']
            ds.write_text(json.dumps(dataset));cached['manifest_hash']=digest(ds);torch.save(cached,source/'features.pt')
            out2=base/'two'
            with patch('improve_classifier.extract_classification_features',return_value=conv):
                compare(source,out2)
            second=json.loads((out2/'selection.json').read_text())
            for key in ['selected','threshold','candidates','train_cv']:
                self.assertEqual(first[key],second[key])
            model=torch.load(out1/'classifier.pt',weights_only=True)
            self.assertEqual(model['problems'],['common','rare'])
            self.assertFalse(first['test_used_for_selection'])


if __name__=='__main__':
    unittest.main()
