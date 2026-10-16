import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from fit_roi_verifier import fit


class VerifierPolicyTests(unittest.TestCase):
    def run_rejected(self, records, negative, message):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);source=root/'dataset.json'
            source.write_text(json.dumps({'records':records,'negative_feedback_records':negative}),encoding='utf-8')
            with patch('fit_roi_verifier.load_detector') as load:
                with self.assertRaisesRegex(ValueError,message):fit(source,root/'retained',root/'candidate')
                load.assert_not_called()
            self.assertFalse((root/'candidate/detector.pt').exists())

    def test_source_split_leakage_is_rejected_before_gpu(self):
        self.run_rejected([{'split':'train','group':'same'},{'split':'val','group':'same'}],[], 'split leakage')

    def test_image_status_and_unconfirmed_boxes_cannot_train_verifier(self):
        normal={'split':'train','group':'train','defects':['崩边'],'image_state':'unclear'}
        negative={'split':'train','group':'negative','defects':['崩边'],'feedback_only':True,'negative_rois':[[0,0,1,1]]}
        self.run_rejected([normal],[negative],'image-state')
        normal['image_state']='';normal['annotation_status']='draft'
        self.run_rejected([normal],[negative],'confirmed positive')


if __name__=='__main__':unittest.main()
