import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from training_report import round_summary
from train_detector import evaluate_predictions, correction_selection_key


class TrainingReportTests(unittest.TestCase):
    def test_type_gain_cannot_select_checkpoint_that_forgets_locations(self):
        base={'class_and_location':dict(recall=.2,f2=.2,fp=20,precision=.2),
              'location_only':dict(recall=.6,f2=.5,fp=15)}
        worse={'class_and_location':dict(recall=.4,f2=.4,fp=18,precision=.4),
               'location_only':dict(recall=.5,f2=.45,fp=14)}
        better={'class_and_location':dict(recall=.4,f2=.4,fp=18,precision=.4),
                'location_only':dict(recall=.6,f2=.6,fp=10)}
        self.assertGreater(correction_selection_key(base,base,0),correction_selection_key(worse,base,5))
        self.assertGreater(correction_selection_key(better,base,8),correction_selection_key(base,base,0))

    def test_whole_image_pass_requires_all_found_no_extra_and_correct_type(self):
        records = [dict(image_id=1, relative_path='one', group='a', boxes=[dict(class_id=0, xyxy=[0,0,10,10])]),
                   dict(image_id=2, relative_path='two', group='b', boxes=[dict(class_id=0, xyxy=[0,0,10,10])])]
        predictions = {'1': [dict(class_id=0, xyxy=[0,0,10,10], score=.9)],
                       '2': [dict(class_id=1, xyxy=[0,0,10,10], score=.9)]}
        report, _ = evaluate_predictions(records, predictions, [dict(id=0, name='崩边'), dict(id=1, name='划伤')], .05)
        self.assertEqual(report['whole_image_pass']['class_and_location']['rate'], .5)
        self.assertEqual(report['whole_image_pass']['location_only']['rate'], 1)
        predictions['1'].append(dict(class_id=0, xyxy=[20,20,30,30], score=.8))
        report, _ = evaluate_predictions(records, predictions, [dict(id=0, name='崩边'), dict(id=1, name='划伤')], .05)
        self.assertEqual(report['whole_image_pass']['class_and_location']['rate'], 0)

    def test_summary_keeps_missing_historical_image_pass_unknown(self):
        side = dict(model_sha256='abc', images=3, class_and_location=dict(tp=2, fn=1, fp=2))
        report = dict(training_completed=True, run_id='r', checked_at='2026-10-16', status='trained_not_promoted',
                      validation_comparison=dict(same_validation_images=True, test_used=False, base=side, candidate=side))
        value = round_summary(report)
        self.assertEqual(value['comparison']['before']['found'], 2)
        self.assertIsNone(value['comparison']['after']['whole_image_pass_rate'])
        self.assertIsNone(value['human_pass_rate_after'])


if __name__ == '__main__': unittest.main()
