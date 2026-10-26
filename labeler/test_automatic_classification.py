import json
import time
import unittest
from pathlib import Path
from PIL import Image
from app import RESPONSIBILITY_RULES
import test_recommendations as fixtures

class AutomaticTests(unittest.TestCase):
    setUp=fixtures.RecommendationTest.setUp
    tearDown=fixtures.RecommendationTest.tearDown

    def record(self, identity=None, names=('崩边',), **extra):
        image=self.store.image(identity or self.ids[0])
        result={'status':'ready','model_id':'test','candidates':[{'name':n,'score':.8} for n in names],**extra}
        return self.store.automatic.record(image,result,(image['size'],image['mtime_ns']),RESPONSIBILITY_RULES)

    def test_classification_is_persisted_filterable_and_not_a_human_label(self):
        self.assertTrue(self.record())
        row=self.store.detail(self.ids[0])
        self.assertFalse(row['classified'])
        self.assertEqual(row['classification_source'],'machine')
        self.assertEqual(row['label']['defects'],['崩边'])
        self.assertEqual(row['label']['responsibilities'],['生产'])
        self.assertEqual(self.store.counts()['classified'],0)
        listing=self.store.listing({'status':['automatic'],'defect':['崩边']})
        self.assertEqual(listing['total'],1)
        self.assertFalse(listing['items'][0]['automatic_classification']['training_eligible'])
        self.assertNotEqual(self.store.automatic.next_image('test')['id'],self.ids[0])

    def test_human_corrections_win_and_original_machine_result_is_retained(self):
        self.record()
        self.store.commit([(self.ids[0],self.label)])
        self.assertFalse(self.record(names=('划伤',)))
        detail=self.store.detail(self.ids[0])
        self.assertEqual(detail['classification_source'],'human')
        self.assertEqual(detail['label']['responsibilities'],['运输'])
        self.assertEqual(detail['automatic_classification']['label']['responsibilities'],['生产'])
        self.assertEqual(self.store.listing({'status':['automatic']})['total'],0)

    def test_draft_cannot_be_overwritten(self):
        self.store.save_draft(self.ids[0],self.label)
        self.assertFalse(self.record())
        self.assertEqual(self.store.automatic.counts()['waiting'],3)

    def test_empty_prediction_is_unresolved_not_normal(self):
        self.record(names=())
        result=self.store.automatic.current(self.store.image(self.ids[0]))
        self.assertEqual(result['outcome'],'unresolved')
        self.assertEqual(result['label']['image_state'],'')
        self.assertEqual(self.store.automatic.counts()['unresolved'],1)

    def test_unrelated_retains_problem_candidates_for_correction(self):
        self.record(image_state_suggestion={'state':'unrelated','score':.8})
        result=self.store.automatic.current(self.store.image(self.ids[0]))
        self.assertEqual(result['label']['image_state'],'unrelated')
        self.assertEqual(result['candidates'][0]['name'],'崩边')

    def test_failed_image_does_not_starve_others_and_source_change_invalidates(self):
        image=self.store.automatic.next_image('test')
        self.store.automatic.record(image,{'model_id':'test'},(image['size'],image['mtime_ns']),{},error='offline')
        self.assertNotEqual(self.store.automatic.next_image('test')['id'],image['id'])
        self.record()
        with self.store.lock,self.store.db:
            self.store.db.execute('UPDATE images SET size=size+1 WHERE id=?',(self.ids[0],))
        self.assertIsNone(self.store.automatic.current(self.store.image(self.ids[0])))

    def test_recent_scan_discovers_new_images_without_hiding_history(self):
        import datetime as dt
        month=self.source/dt.datetime.now().strftime('%Y-%m');month.mkdir()
        Image.new('RGB',(40,40),'red').save(month/'new.jpg')
        self.app.scanner.start(self.store.current_root(),recent=True)
        while self.app.scanner.status['running']:time.sleep(.01)
        self.assertEqual(self.store.one('SELECT COUNT(*) n FROM images WHERE available=1')['n'],5)
        self.assertEqual(self.store.automatic.counts()['waiting'],5)

    def test_model_refresh_and_source_change_statistics(self):
        self.record()
        counts=self.store.automatic.counts('new-model')
        self.assertEqual(counts['model_refresh'],1)
        self.assertIsNotNone(counts['last_success_at'])
        image=self.store.image(self.ids[1])
        self.store.automatic.record(image,{'model_id':'test'},(image['size'],image['mtime_ns']),{},error='offline')
        with self.store.lock,self.store.db:
            self.store.db.execute('UPDATE images SET size=size+1 WHERE id=?',(image['id'],))
        self.assertEqual(self.store.automatic.counts()['failed'],0)

if __name__=='__main__':unittest.main()
