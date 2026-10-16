import unittest
from review_comparison import compare_review,summarize_comparisons

def box(identity,x=.1,y=.1,problem='崩边'):
    return dict(id=identity,x=x,y=y,w=.2,h=.2,problem=problem)

class ComparisonTest(unittest.TestCase):
    def test_duplicate_missing_reclassified_and_manual_add(self):
        original=[box('p1'),box('duplicate'),box('false',.7,.7)]
        final=[box('p1',problem='划伤'),box('new',.4,.4)]
        result=compare_review(original,final,[dict(reason='process_hole')])
        self.assertEqual((result['matched_locations'],result['missed_locations'],result['extra_candidates'],result['duplicate_candidates']),(1,1,2,1))
        self.assertEqual(result['problem_type_changed'],1);self.assertEqual(result['human_added_boxes'],1)
        self.assertEqual(result['normal_region_reasons'],{'process_hole':1})
        self.assertEqual(result['location_recall_on_reviewed_examples'],.5)
        self.assertEqual(result['unmatched_human_problems'],{'崩边':1})
    def test_moving_model_box_does_not_count_original_position_as_found(self):
        result=compare_review([box('p1')],[box('p1',.7,.7)])
        self.assertEqual(result['coordinates_changed'],1);self.assertEqual(result['matched_locations'],0)
        self.assertEqual(result['human_added_boxes'],0)
    def test_empty_ground_truth_is_not_100_percent_accuracy(self):
        result=compare_review([box('p1')],[])
        self.assertIsNone(result['location_recall_on_reviewed_examples'])
        summary=summarize_comparisons([{'comparison':result}])
        self.assertIsNone(summary['location_recall_on_reviewed_examples'])
        self.assertEqual(summary['removed_machine_boxes'],1)

if __name__=='__main__':unittest.main()
