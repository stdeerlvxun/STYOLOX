import datetime as dt
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from training_rounds import TrainingRounds


class RuntimeStatusTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.store = SimpleNamespace(data_dir=Path(self.temp.name), lock=threading.RLock(),
                                     current_root=lambda: {'id': 1})
        self.rounds = TrainingRounds(self.store)

    def tearDown(self):
        self.temp.cleanup()

    def report(self, minutes=0, **extra):
        return {'kind':'training', 'root_id':1, 'value':{
            'updated_at':(dt.datetime.now(dt.timezone.utc)-dt.timedelta(minutes=minutes)).isoformat(),
            'status':'waiting_for_confirmed_images', 'pending_confirmed_images':17,
            'minimum_confirmed_images':20, **extra}}

    def test_progress_persists_without_creating_training_history(self):
        self.rounds.record_runtime(self.report())
        state = TrainingRounds(self.store).runtime_state()['training']
        self.assertEqual(state['pending_confirmed_images'],17)
        self.assertFalse(state['stale'])
        self.assertFalse(self.rounds.path.exists())

    def test_stale_and_older_delivery_do_not_look_current(self):
        self.rounds.record_runtime(self.report(16))
        self.assertTrue(self.rounds.runtime_state()['training']['stale'])
        self.rounds.record_runtime(self.report(0,status='training'))
        self.assertTrue(self.rounds.record_runtime(self.report(2))['ignored_older'])
        self.assertEqual(self.rounds.runtime_state()['training']['status'],'training')

    def test_invalid_payload_preserves_previous_status(self):
        self.rounds.record_runtime(self.report())
        for invalid in (self.report(pending_confirmed_images=True),self.report(pending_confirmed_images=-1),
                        self.report(updated_at='bad'),self.report(secret='no'),
                        {**self.report(),'root_id':2}):
            with self.assertRaises(ValueError):self.rounds.record_runtime(invalid)
        self.assertEqual(self.rounds.runtime_state()['training']['pending_confirmed_images'],17)

    def test_roots_are_isolated_and_missing_values_stay_missing(self):
        self.rounds.record_runtime(self.report())
        self.assertNotIn('epoch',self.rounds.runtime_state()['training'])
        self.store.current_root=lambda:{'id':2}
        self.assertEqual(self.rounds.runtime_state(),{})


if __name__=='__main__':unittest.main()
