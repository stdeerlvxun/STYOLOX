import unittest
import torch
from unrelated_filter import auxiliary_target,conservative_cutoff,fit,POLICY
from vision_classifier import training_exclusion_reason


class UnrelatedTest(unittest.TestCase):
    def test_state_exception_only_applies_to_separate_unrelated_filter(self):
        positive={'image_state':'unrelated','defects':[]}
        self.assertEqual(auxiliary_target(positive,False),1)
        self.assertEqual(training_exclusion_reason(positive),'image_state_selected')
        self.assertIsNone(auxiliary_target(positive,True))
        for state in ['normal','unclear']:
            self.assertIsNone(auxiliary_target({'image_state':state,'defects':[]},False))
        self.assertIsNone(auxiliary_target({'image_state':'unrelated','defects':['崩边']},False))
        self.assertEqual(auxiliary_target({'image_state':'defect','defects':['崩边']},False),0)

    def test_validation_cutoff_protects_known_related_images(self):
        truth=torch.tensor([[0.],[0.],[1.],[1.],[1.]])
        scores=torch.tensor([[.1],[.6],[.5],[.8],[.9]])
        choice=conservative_cutoff(truth,scores)
        self.assertEqual(choice['fp'],0);self.assertEqual(choice['tp'],2)
        self.assertEqual(choice['fn'],1)

    def test_confirmed_normal_is_opt_in_related_only_and_unclear_is_never_allowed(self):
        normal={'image_state':'normal','defects':[]}
        self.assertIsNone(auxiliary_target(normal,False))
        self.assertEqual(auxiliary_target(normal,False,True),0)
        self.assertIsNone(auxiliary_target(normal,True,True))
        self.assertIsNone(auxiliary_target({'image_state':'normal','defects':['崩边']},False,True))
        self.assertIsNone(auxiliary_target({'image_state':'unclear','defects':[]},False,True))
        self.assertEqual(training_exclusion_reason(normal),'image_state_selected')

    def test_related_cost_changes_actual_fitted_weights(self):
        raw=torch.tensor([[1.,0.],[0.,1.],[1.,1.],[.5,1.]])
        target=torch.tensor([[0.],[1.],[1.],[1.]])
        model=fit(raw,target,1.)
        self.assertEqual(tuple(model['alpha'].shape),(4,1))
        self.assertLess(float(model['prior'][0]),float(target.mean()))
        self.assertTrue(torch.isfinite(model['alpha']).all())


if __name__=='__main__':unittest.main()
