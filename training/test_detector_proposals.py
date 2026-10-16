import math
import unittest
from generate_detector_proposals import normalize_predictions


class DetectorProposalTests(unittest.TestCase):
    def test_only_current_human_problem_classes_are_suggested(self):
        predictions=[{'problem':'崩边','score':.8,'xywh':[.1,.2,.3,.4]},
                     {'problem':'划伤','score':.9,'xywh':[.2,.3,.1,.1]}]
        result=normalize_predictions(predictions,{'崩边'})
        self.assertEqual([r['problem'] for r in result],['崩边'])
        self.assertEqual(result[0]['origin'],'model')

    def test_invalid_whole_image_and_nonfinite_boxes_are_abstentions(self):
        predictions=[{'problem':'崩边','score':.8,'xywh':values}
            for values in ([0,0,1,1],[math.nan,.1,.1,.1],[2,2,.1,.1],[.2,.2,0,.1])]
        self.assertEqual(normalize_predictions(predictions,{'崩边'}),[])

    def test_boxes_clip_to_source_bounds_and_have_unique_ids(self):
        predictions=[{'problem':'崩边','score':.8,'xywh':[-.1,.9,.3,.3]}]*2
        result=normalize_predictions(predictions,{'崩边'})
        self.assertEqual(len(result),2);self.assertEqual(result[0]['x'],0)
        self.assertAlmostEqual(result[0]['h'],.1)
        self.assertNotEqual(result[0]['id'],result[1]['id'])


if __name__=='__main__':unittest.main()
