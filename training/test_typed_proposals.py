import copy
import hashlib
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest

from test_review_refill import fixture,FakeDetector
import enrich_proposal_types as enrichment
from typed_proposals import normalize_typed_predictions,enrich_boxes


SHA='a'*64


class TypedProposalTest(unittest.TestCase):
    def test_detector_class_is_real_not_existing_label_or_default(self):
        predictions=[{'problem':'划伤','score':.7,'xywh':[.1,.1,.2,.2]},
                     {'problem':'缺角','score':.8,'xywh':[.6,.6,.2,.2]}]
        boxes=normalize_typed_predictions(predictions,SHA)
        self.assertEqual([box['problem'] for box in boxes],['缺角','划伤'])
        self.assertEqual(boxes[0]['type_provenance'],{'source':'detector_class_prediction','model_sha256':SHA,'score':.8})

    def test_class_agnostic_never_renames_location_as_issue_or_arbitrary_other_damage(self):
        predictions=[{'problem':'问题位置','score':.8,'xywh':[.1,.1,.2,.2]}]
        with self.assertRaisesRegex(ValueError,'typed detector'):
            normalize_typed_predictions(predictions,SHA,class_agnostic=True)
        with self.assertRaisesRegex(ValueError,'no trained issue type'):
            normalize_typed_predictions(predictions,SHA)

    def test_cross_class_duplicates_clipping_and_maximum_are_unchanged(self):
        predictions=[{'problem':'划伤','score':.9,'xywh':[-.1,.1,.4,.2]},
                     {'problem':'缺角','score':.8,'xywh':[0,.1,.3,.2]},
                     {'problem':'残胶','score':.6,'xywh':[.6,.6,.2,.2]}]
        boxes=normalize_typed_predictions(predictions,SHA,max_boxes=1)
        self.assertEqual(len(boxes),1);self.assertEqual(boxes[0]['problem'],'划伤');self.assertEqual(boxes[0]['x'],0)
        self.assertEqual(len(normalize_typed_predictions(predictions,SHA)),2)

    def test_enrichment_preserves_ids_geometry_known_types_and_unmatched_blanks(self):
        original=[{'id':'blank','problem':'','x':.1,'y':.1,'w':.2,'h':.2,'origin':'model','score':.8},
                  {'id':'known','problem':'残胶','x':.6,'y':.6,'w':.2,'h':.2,'origin':'model','score':.8},
                  {'id':'unmatched','problem':'','x':.1,'y':.8,'w':.1,'h':.1,'origin':'model','score':.7}]
        before=copy.deepcopy(original)
        boxes,count=enrich_boxes(original,[{'problem':'划伤','score':.9,'xywh':[.11,.1,.2,.2]}],SHA)
        self.assertEqual(count,1);self.assertEqual(original,before)
        self.assertEqual(boxes[0]['problem'],'划伤');self.assertGreater(boxes[0]['type_provenance']['match_iou'],.8)
        for key in ('id','x','y','w','h','origin','score'):self.assertEqual(boxes[0][key],before[0][key])
        self.assertEqual(boxes[1:],before[1:])

    def enrichment_fixture(self,root,count=3):
        config,snapshot=fixture(root,count)
        db=sqlite3.connect(snapshot)
        for name in ('proposal_id','data','label_data'):db.execute('ALTER TABLE annotation_proposals ADD COLUMN '+name+' TEXT')
        model_sha=hashlib.sha256(b'verified model').hexdigest()
        for identity in range(1,count+1):
            path=Path(config['source_root'])/(str(identity)+'.png');stat=path.stat()
            db.execute('DELETE FROM labels WHERE image_id=?',(identity,))
            source={'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'width':20,'height':20,
                    'size':stat.st_size,'mtime_ns':stat.st_mtime_ns}
            data={'source':source,'source_checkpoint_sha256':model_sha,'label_fingerprint':hashlib.sha256(b'').hexdigest(),
                'boxes':[{'id':'original-'+str(identity),'problem':'','x':.1,'y':.1,'w':.2,'h':.2,'origin':'model','score':.8}]}
            db.execute('INSERT INTO annotation_proposals VALUES(?,?,?,?,?)',
                (identity,'pending','pending-'+str(identity),json.dumps(data,ensure_ascii=False),''))
        db.commit();db.close()
        return config,snapshot

    def test_local_enrichment_manifest_is_read_only_cas_bound_and_bounded(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);config,snapshot=self.enrichment_fixture(root)
            before=snapshot.read_bytes();detector=FakeDetector()
            result=enrichment.generate(config,snapshot,root/'enrichment.json',detector,limit=2)
            self.assertEqual([item['image_id'] for item in result['items']],[3,2])
            self.assertEqual(result['summary']['typed_boxes'],2);self.assertEqual(result['summary']['remaining_candidates'],1)
            self.assertEqual(snapshot.read_bytes(),before);self.assertEqual(detector.thresholds,[.05,.05])
            self.assertTrue(result['enrichment_only']);self.assertFalse(result['geometry_modified'])
            db=sqlite3.connect(snapshot)
            for item in result['items']:
                raw=db.execute('SELECT data FROM annotation_proposals WHERE image_id=?',(item['image_id'],)).fetchone()[0]
                self.assertEqual(item['expected_proposal_fingerprint'],hashlib.sha256(raw.encode()).hexdigest())
                old=json.loads(raw);self.assertEqual(item['source'],old['source'])
                for key in ('id','x','y','w','h'):self.assertEqual(item['boxes'][0][key],old['boxes'][0][key])
            db.close()

    def test_enrichment_excludes_human_edits_reviews_different_models_and_changed_source(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);config,snapshot=self.enrichment_fixture(root,7)
            db=sqlite3.connect(snapshot)
            db.execute('INSERT INTO drafts VALUES(1)');db.execute('INSERT INTO annotation_documents VALUES(2)')
            db.execute('INSERT INTO annotation_history VALUES(3)');db.execute("INSERT INTO error_reviews VALUES(4,'reviewed')")
            db.execute("UPDATE annotation_proposals SET status='accepted' WHERE image_id=5")
            raw=json.loads(db.execute('SELECT data FROM annotation_proposals WHERE image_id=6').fetchone()[0]);raw['source_checkpoint_sha256']='b'*64
            db.execute('UPDATE annotation_proposals SET data=? WHERE image_id=6',(json.dumps(raw),))
            db.execute('UPDATE images SET size=size+1 WHERE id=7');db.commit();db.close()
            result=enrichment.generate(config,snapshot,root/'enrichment.json',FakeDetector())
            self.assertEqual(result['items'],[]);self.assertEqual(result['summary']['attempted_images'],0)
            self.assertEqual(result['selection']['excluded']['human_work'],4)

    def test_source_changed_during_inference_never_produces_enrichment(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);config,snapshot=self.enrichment_fixture(root,1)
            class ChangingDetector(FakeDetector):
                def predict_image(self,image,**kwargs):
                    result=super().predict_image(image,**kwargs)
                    (Path(config['source_root'])/'1.png').write_bytes(b'changed source')
                    return result
            result=enrichment.generate(config,snapshot,root/'enrichment.json',ChangingDetector())
            self.assertEqual(result['items'],[]);self.assertEqual(result['summary']['errors'],1)
            self.assertIn('Source changed during inference',result['errors'][0]['error'])


if __name__=='__main__':unittest.main()
