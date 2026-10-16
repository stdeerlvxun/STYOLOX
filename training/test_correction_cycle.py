import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import correction_cycle as worker


def item(image_id=1, feedback_id=1):
    label = {'defects':['崩边'], 'image_state':'', 'responsibilities':['生产']}
    raw = json.dumps(label, ensure_ascii=False)
    fingerprint = hashlib.sha256(raw.encode()).hexdigest()
    return {'feedback_id':feedback_id, 'image_id':image_id, 'source':{'sha256':'source'},
            'training_eligible_by_label_policy':True, 'source_sha256_verified':True,
            'included_in_training':False, 'original_classification':label,
            'original_classification_raw':raw,
            'feedback':{'status':'confirmed', 'whole_image_normal':False, 'label_fingerprint':fingerprint,
                        'annotation_version':3, 'negative_regions':[]},
            'annotation':{'status':'confirmed', 'outcome':'boxed', 'label_fingerprint':fingerprint,
                          'boxes':[{'id':'confirmed-box'}]}}


def comparison(base_f2=.4, base_recall=.5, f2=.5, recall=.6, support=10):
    return {'same_validation_images':True, 'problem_name_mapping':True, 'test_used':False,
            'baseline':{'f2':base_f2, 'recall':base_recall, 'tp':5, 'fn':5, 'fp':8},
            'candidate':{'f2':f2, 'recall':recall, 'tp':round(support*recall),
                         'fn':support-round(support*recall), 'fp':6}}


def config(root):
    return {'python_executable':str(root/'python.exe'), 'training_root':str(root/'training'),
            'api_url':'http://192.168.0.101:8878', 'ssh_alias':'local-vm',
            'remote_python':'/opt/labeler/venv/bin/python', 'remote_base':'/opt/labeler',
            'remote_data':'/var/lib/labeler', 'runs_root':str(root/'runs'),
            'source_root':str(root/'nas'), 'current_detector_run':str(root/'old-model'),
            'historical_dataset':str(root/'old-dataset.json'), 'minimum_confirmed_images':20}


class CycleProtectionTests(unittest.TestCase):
    def test_twenty_saves_on_one_image_do_not_trigger_training(self):
        payload = {'items':[item(1, i) for i in range(1, 21)]}
        eligible = worker.eligible_pending(payload)
        self.assertEqual(len(eligible), 20)
        self.assertEqual(len({row['image_id'] for row in eligible}), 1)

    def test_only_current_human_supervision_counts(self):
        invalid = []
        for key, value in [('training_eligible_by_label_policy', False), ('source_sha256_verified', False),
                           ('included_in_training', True)]:
            value_item = item();value_item[key] = value;invalid.append(value_item)
        for key, value in [('status', 'draft'), ('whole_image_normal', True)]:
            value_item = item();value_item['feedback'][key] = value;invalid.append(value_item)
        for change in ({'status':'draft'}, {'outcome':'unclear'}, {'boxes':[]}):
            value_item = item();value_item['annotation'].update(change);invalid.append(value_item)
        value_item = item();value_item['original_classification']['image_state'] = 'normal';invalid.append(value_item)
        value_item = item();value_item['original_classification_raw'] += ' ';invalid.append(value_item)
        self.assertEqual(worker.eligible_pending({'items':invalid}), [])
        normal_roi = item();normal_roi['annotation'] = None
        normal_roi['feedback']['negative_regions'] = [{'id':'roi', 'x':.1, 'y':.1, 'w':.1, 'h':.1, 'reason':'process_hole'}]
        self.assertEqual(len(worker.eligible_pending({'items':[normal_roi]})), 1)

    def test_only_actually_used_train_ids_are_consumed(self):
        positive, holdout, region = item(1, 1), item(2, 2), item(3, 3)
        dataset = {'included_feedback_ids':[1, 2, 3], 'records':[
            {'image_id':1, 'sha256':'source', 'split':'train', 'annotation_version':3,
             'label_fingerprint':positive['feedback']['label_fingerprint'], 'boxes':[{}]},
            {'image_id':2, 'sha256':'source', 'split':'test', 'annotation_version':3,
             'label_fingerprint':holdout['feedback']['label_fingerprint'], 'boxes':[{}]}],
            'negative_feedback_records':[{'image_id':3, 'sha256':'source', 'split':'train',
              'label_fingerprint':region['feedback']['label_fingerprint'], 'feedback_ids':[3], 'negative_rois':[[1, 2, 3, 4]]}]}
        self.assertEqual(worker.actual_feedback_ids(dataset, [positive, holdout, region]), [1, 3])
        self.assertEqual(worker.round_seen_feedback_ids(dataset, [positive, holdout, region]), [1, 2, 3])
        positive['feedback']['annotation_version'] = 4
        self.assertEqual(worker.actual_feedback_ids(dataset, [positive, holdout, region]), [3])
        self.assertEqual(worker.round_seen_feedback_ids(dataset, [positive, holdout, region]), [2, 3], 'Drifted annotations must remain available for a later valid round')

    def test_promotion_requires_strict_improvement_and_recall(self):
        self.assertTrue(worker.promotion_decision(comparison())[0])
        self.assertFalse(worker.promotion_decision(comparison(f2=.4))[0])
        self.assertFalse(worker.promotion_decision(comparison(f2=.6, recall=.4))[0])
        self.assertFalse(worker.promotion_decision(comparison(f2=.5, recall=.3, base_recall=.2))[0])
        self.assertFalse(worker.promotion_decision(comparison(support=9))[0])

    def test_name_alignment_and_no_test_reuse_are_required(self):
        for key, value in [('problem_name_mapping', False), ('same_validation_images', False), ('test_used', True)]:
            report = comparison();report[key] = value
            self.assertFalse(worker.promotion_decision(report)[0])
        report = comparison();report['candidate']['f2'] = float('nan')
        self.assertFalse(worker.promotion_decision(report)[0])

    def test_empty_new_proposals_removed_but_explicit_withdrawal_preserved(self):
        manifest = {'items':[{'image_id':1, 'boxes':[]}, {'image_id':2, 'boxes':[{}]},
            {'image_id':3, 'boxes':[], 'replaces_proposal_id':'old', 'withdraw_if_empty':True},
            {'image_id':4, 'boxes':[], 'replaces_proposal_id':'old2'}]}
        self.assertEqual([i['image_id'] for i in worker.filter_manifest(manifest)['items']], [2, 3])

    def test_check_only_does_not_create_files_query_gpu_or_mutate(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory);path = root/'private.json';worker.atomic_json(path, config(root))
            before = path.read_bytes()
            with patch.object(worker, 'export_feedback', return_value={'items':[item(i, i) for i in range(20)]}), \
                 patch.object(worker, 'gpu_available', side_effect=AssertionError('GPU queried')), \
                 patch.object(worker, 'execute_cycle', side_effect=AssertionError('Mutation')):
                result = worker.cycle(path, check_only=True)
            self.assertTrue(result['ready']);self.assertFalse(result['gpu_queried'])
            self.assertEqual(path.read_bytes(), before);self.assertFalse((root/'runs').exists())

    def test_busy_gpu_skips_training(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory);path = root/'private.json';worker.atomic_json(path, config(root))
            with patch.object(worker, 'export_feedback', return_value={'items':[item(i, i) for i in range(20)]}), \
                 patch.object(worker, 'gpu_available', return_value=(False, 'busy')), \
                 patch.object(worker, 'execute_cycle', side_effect=AssertionError('Must not train')):
                self.assertEqual(worker.cycle(path)['status'], 'skipped_gpu_busy_or_unavailable')

    def test_wddm_desktop_clients_and_hidden_names_do_not_block_idle_gpu(self):
        processes = ('123, C:\\Windows\\explorer.exe\n124, C:\\Chrome\\chrome.exe\n'
                     '125, C:\\WPS\\wps.exe\n126, [Insufficient Permissions]\n'
                     +str(worker.os.getpid())+', C:\\Python\\python.exe\n')
        with patch.object(worker, 'run_process', side_effect=[processes, '4096, 12\n']):
            self.assertEqual(worker.gpu_available(), (True, 'gpu_available'))

    def test_compute_process_or_low_resources_block_gpu(self):
        for process, counters, reason in [
            ('999999, C:\\Python\\python.exe\n', '4096, 1', 'another_gpu_compute_process_is_running'),
            ('999999, llama-server.exe\n', '4096, 1', 'another_gpu_compute_process_is_running'),
            ('999999, chrome.exe\n', '1024, 1', 'insufficient_free_gpu_memory'),
            ('999999, explorer.exe\n', '4096, 65', 'gpu_utilization_above_limit')]:
            with self.subTest(reason=reason), patch.object(worker, 'run_process', side_effect=[process, counters]):
                self.assertEqual(worker.gpu_available(), (False, reason))

    def test_os_lock_rejects_second_owner_and_releases(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'worker.lock'
            with worker.ExclusiveLock(path):
                with self.assertRaises(worker.AlreadyRunning):
                    with worker.ExclusiveLock(path):
                        self.fail('Second worker acquired the exclusive lock')
            with worker.ExclusiveLock(path):
                pass

    def test_three_same_batch_failures_pause(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory);path = root/'private.json';worker.atomic_json(path, config(root))
            with patch.object(worker, 'export_feedback', return_value={'items':[item(i, i) for i in range(20)]}), \
                 patch.object(worker, 'gpu_available', return_value=(True, 'free')), \
                 patch.object(worker, 'execute_cycle', side_effect=RuntimeError('bounded failure')) as execute:
                for _ in range(3):
                    self.assertEqual(worker.cycle(path)['status'], 'error')
                self.assertEqual(worker.cycle(path)['status'], 'paused');self.assertEqual(execute.call_count, 3)

    def test_holdout_reference_updates_even_without_model_promotion(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory);path = root/'private.json';original = config(root);worker.atomic_json(path, original)
            worker.update_config_reference(path, original, root/'new-dataset.json')
            fresh = worker.read_json(path)
            self.assertEqual(fresh['historical_dataset'], str(root/'new-dataset.json'))
            self.assertEqual(fresh['current_detector_run'], original['current_detector_run'])

    def test_twenty_reviewed_images_with_fourteen_train_trigger_exactly_one_round(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory);path = root/'private.json'
            worker.atomic_json(path,{**config(root),'class_agnostic':True,'fixed_threshold':.05})
            payload = {'items':[item(i, i) for i in range(1, 21)]}
            records = [{'image_id':row['image_id'], 'sha256':'source',
                        'split':'train' if row['image_id']<=14 else 'validation',
                        'annotation_version':3, 'label_fingerprint':row['feedback']['label_fingerprint'],
                        'boxes':[{}]} for row in payload['items']]
            dataset = {'records':records, 'included_feedback_ids':list(range(1, 15))}
            trained = []
            def fake_process(arguments, **_):
                arguments = list(map(str, arguments))
                run = Path(arguments[arguments.index('--run')+1])
                if arguments[1].endswith('prepare_detector.py'):
                    worker.atomic_json(run/'dataset.json', dataset)
                elif '--compare-validation-to' in arguments:
                    report = comparison(f2=.3, recall=.3, base_f2=.2, base_recall=.2)
                    report.update(dataset_sha256=hashlib.sha256((run.parent/'dataset.json').read_bytes()).hexdigest())
                    report['candidate']['model_sha256'] = hashlib.sha256((run/'detector.pt').read_bytes()).hexdigest()
                    worker.atomic_json(run/'validation-comparison.json', report)
                else:
                    trained.append((run,arguments));run.mkdir();(run/'detector.pt').write_bytes(b'trained')
                    worker.atomic_json(run/'metrics.json', {'test':None, 'test_evaluation_pending':True,
                        'model_sha256':hashlib.sha256(b'trained').hexdigest(), 'validation':{'f2':.3}})
                return ''
            with patch.object(worker, 'export_feedback', return_value=payload), \
                 patch.object(worker, 'gpu_available', return_value=(True, 'free')), \
                 patch.object(worker, 'run_process', side_effect=fake_process), \
                 patch.object(worker, 'transfer'), patch.object(worker, 'ssh') as remote:
                first = worker.cycle(path)
                self.assertEqual(first['status'], 'trained_not_promoted')
                self.assertEqual(first['feedback_ids_actually_trained'], list(range(1, 15)))
                self.assertEqual(first['round_seen_feedback_ids'], list(range(1, 21)))
                second = worker.cycle(path)
                self.assertEqual(second['status'], 'waiting_for_confirmed_images')
                self.assertEqual(second['pending_confirmed_images'], 0)
                self.assertEqual(len(trained), 1)
                self.assertIn('--class-agnostic',trained[0][1])
                self.assertEqual(trained[0][1][trained[0][1].index('--fixed-threshold')+1],'0.05')
                marker = [call.args[1] for call in remote.call_args_list if 'record_feedback_training.py' in str(call)]
                self.assertEqual(len(marker), 1)
                self.assertEqual(marker[0][marker[0].index('--feedback-ids')+1], ','.join(map(str, range(1, 15))))

    def test_shell_injection_and_low_threshold_configuration_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            valid = config(Path(directory))
            for key, value in [('ssh_alias', '-oProxyCommand=evil'), ('remote_data', '/data/../root'),
                               ('minimum_confirmed_images', 1), ('minimum_validation_boxes', 1)]:
                changed = copy.deepcopy(valid);changed[key] = value
                with self.assertRaises(ValueError):worker.validate_config(changed)


if __name__=='__main__':
    unittest.main()
