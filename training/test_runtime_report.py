import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from runtime_report import inference_value, training_value, send, stage


class RuntimeReportTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.root=Path(self.temp.name)
        self.config={'runs_root':str(self.root),'current_detector_run':str(self.root/'model')}

    def tearDown(self):
        self.temp.cleanup()

    def test_batch_retries_do_not_double_count_and_historical_unknown_is_not_zero(self):
        batch={'batch_id':'refill-20261026T010000Z-a','status':'refilled','attempted_images':72,
               'generated_images':50,'generated_boxes':149,'images_without_boxes':22,'errors':0,'imported_images':50}
        first=inference_value(self.config,batch)
        again=inference_value(self.config,batch)
        self.assertEqual(first['total_attempted_images'],72)
        self.assertEqual(again['total_generated_boxes'],149)
        self.assertEqual(again['batches'],1)
        older={**batch,'batch_id':'refill-20261025T010000Z-b'}
        del older['images_without_boxes']
        value=inference_value(self.config,older)
        self.assertEqual(value['total_attempted_images'],144)
        self.assertNotIn('total_images_without_boxes',value)
        self.assertEqual(value['statistics_since'],first['statistics_since'])

    def test_bootstrap_existing_reports_once(self):
        folder=self.root/'refill-old';folder.mkdir()
        (folder/'report.json').write_text(json.dumps({'batch_id':'old','status':'refilled','attempted_images':3}))
        value=inference_value(self.config,{'status':'waiting_for_gpu'})
        self.assertEqual(value['total_attempted_images'],3)
        self.assertEqual(inference_value(self.config,{'status':'waiting_for_gpu'})['batches'],1)

    def test_training_reads_actual_completed_epoch_and_does_not_invent_progress(self):
        report={'status':'training','pending_confirmed_images':20,'max_epochs':50}
        self.assertNotIn('epoch',training_value(report,self.root))
        (self.root/'detector').mkdir()
        (self.root/'detector'/'training-history.json').write_text('[{"epoch": 7}]')
        self.assertEqual(training_value(report,self.root)['epoch'],7)
        (self.root/'detector'/'training-history.json').write_text('[{"epoch":')
        self.assertNotIn('epoch',training_value(report,self.root))

    def test_reporting_failure_cannot_abort_training(self):
        with patch('runtime_report.urllib.request.urlopen',side_effect=OSError('offline')):
            self.assertFalse(send({'api_url':'http://localhost'},'training',{}))
        report={}
        with patch('runtime_report.send',return_value=False):
            with stage(self.config,report,'training'):
                self.assertEqual(report['status'],'training')
        self.assertTrue((self.root/'training-runtime.json').exists())


if __name__=='__main__':unittest.main()
