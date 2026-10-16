import copy
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from training_rounds import TrainingRounds


def sample():
    side = dict(found=8, missed=2, false_positive=1, total_boxes=10, images=5,
                whole_image_passed=2, whole_image_pass_rate=.4)
    return dict(round_id='auto-test', status='trained_not_promoted', created_at='2026-10-16',
                reviewed_images=20, same_validation_images=True, test_used=False,
                comparison=dict(objective='issue_type_and_location', before=side, after=dict(side)))


class TrainingRoundsTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.store = SimpleNamespace(data_dir=Path(self.temp.name), lock=threading.RLock())
        self.rounds = TrainingRounds(self.store)

    def tearDown(self):
        self.temp.cleanup()

    def test_round_retries_persist_once_and_do_not_invent_human_pass_rate(self):
        value = sample(); value['human_pass_rate_after'] = .99
        self.rounds.record(value); self.rounds.record(value)
        rows = TrainingRounds(self.store).state()['rounds']
        self.assertEqual(len(rows), 1)
        self.assertIsNone(rows[0]['human_pass_rate_after'])
        self.assertEqual(rows[0]['comparison']['before']['whole_image_pass_rate'], .4)

    def test_bad_cohort_or_rate_cannot_overwrite_good_summary(self):
        self.rounds.record(sample())
        before = self.rounds.path.read_bytes()
        changes = [('total_boxes', 11), ('whole_image_pass_rate', float('nan')), ('images', 6)]
        for key, value in changes:
            invalid = copy.deepcopy(sample()); invalid['comparison']['after'][key] = value
            with self.assertRaises(ValueError): self.rounds.record(invalid)
            self.assertEqual(self.rounds.path.read_bytes(), before)

    def test_old_round_missing_pass_rate_stays_unknown(self):
        value = sample()
        for side in ('before', 'after'):
            value['comparison'][side].update(whole_image_passed=None, whole_image_pass_rate=None)
        self.rounds.record(value)
        self.assertIsNone(self.rounds.state()['rounds'][0]['comparison']['before']['whole_image_pass_rate'])


if __name__ == '__main__': unittest.main()
