import math
import unittest
from generate_detector_proposals import normalize_predictions,select_generation_rows


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

    def test_default_generation_preserves_all_existing_candidates(self):
        rows=[{'image_id':1,'label_fingerprint':'a'},{'image_id':2,'label_fingerprint':'b'}]
        selected,preserved=select_generation_rows(rows,{1:{'status':'pending','proposal_id':'old','label_fingerprint':'a'}})
        self.assertEqual([r['image_id'] for r in selected],[2]);self.assertEqual(preserved,1)

    def test_refresh_only_current_pending_not_human_rejections_or_stale_labels(self):
        rows=[{'image_id':i,'label_fingerprint':'current'} for i in range(1,6)]
        existing={i:{'status':status,'proposal_id':'p'+str(i),'label_fingerprint':fp}
            for i,status,fp in [(1,'pending','current'),(2,'accepted','current'),
                (3,'rejected','current'),(4,'stale','current'),(5,'pending','old-label')]}
        selected,preserved=select_generation_rows(rows,existing,True)
        self.assertEqual([r['image_id'] for r in selected],[1]);self.assertEqual(preserved,4)
        self.assertEqual(selected[0]['replaces_proposal_id'],'p1')
        self.assertNotIn('replaces_proposal_id',rows[0],'Do not modify the source snapshot records')


if __name__=='__main__':unittest.main()
