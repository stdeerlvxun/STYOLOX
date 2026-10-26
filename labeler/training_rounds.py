"""Small persisted training summaries, independent of human classification data."""
import json
import math
import os
import re
import uuid
import datetime as dt
from pathlib import Path


class TrainingRounds:
    def __init__(self, store):
        self.store = store
        self.path = Path(store.data_dir) / 'training-rounds.json'

    def runtime_state(self):
        root = self.store.current_root()['id']
        path = Path(self.store.data_dir) / ('automation-status-'+str(root)+'.json')
        with self.store.lock:
            value = json.loads(path.read_text(encoding='utf-8')) if path.exists() else {}
        now = dt.datetime.now(dt.timezone.utc)
        for name, item in value.items():
            age = max(0, (now-dt.datetime.fromisoformat(item['updated_at'])).total_seconds())
            item['stale'] = age > (180 if item.get('status') in ('preparing', 'training', 'validating', 'generating') else 900)
        return value

    def record_runtime(self, body):
        if not isinstance(body, dict) or body.get('kind') not in ('training', 'inference'):
            raise ValueError('Invalid automation status kind')
        if body.get('root_id') != self.store.current_root()['id']:
            raise ValueError('Automation status belongs to another image root')
        value = body.get('value')
        text_fields = {'status','updated_at','started_at','run_id','batch_id','model_id','reason',
                       'promotion_reason','statistics_since','last_success_at'}
        count_fields = {'pending_confirmed_images','minimum_confirmed_images','trainable_confirmed_images',
                        'validation_images','test_images','epoch','max_epochs','attempted_images',
                        'generated_images','generated_boxes','images_without_boxes','errors','imported_images',
                        'total_attempted_images','total_generated_images','total_generated_boxes',
                        'total_images_without_boxes','total_errors','total_imported_images','batches',
                        'already_attempted_images','remaining_eligible_images'}
        bool_fields = {'training_completed','promoted'}
        if not isinstance(value, dict) or set(value)-(text_fields|count_fields|bool_fields):
            raise ValueError('Invalid automation status fields')
        cleaned = {}
        for key, item in value.items():
            if key in text_fields and (not isinstance(item, str) or len(item)>1000):
                raise ValueError('Invalid automation status text')
            if key in count_fields and (type(item) is not int or not 0<=item<=10**12):
                raise ValueError('Invalid automation count')
            if key in bool_fields and type(item) is not bool:
                raise ValueError('Invalid automation flag')
            cleaned[key] = item
        for key in ('updated_at', 'started_at', 'statistics_since', 'last_success_at'):
            if key not in cleaned:continue
            try:
                stamp = dt.datetime.fromisoformat(cleaned[key].replace('Z','+00:00'))
                if stamp.tzinfo is None:raise ValueError()
                cleaned[key] = stamp.astimezone(dt.timezone.utc).isoformat()
            except (ValueError, TypeError):
                raise ValueError('Invalid automation timestamp')
        if not cleaned.get('status') or 'updated_at' not in cleaned:
            raise ValueError('Automation status and timestamp required')
        if dt.datetime.fromisoformat(cleaned['updated_at']) > dt.datetime.now(dt.timezone.utc)+dt.timedelta(minutes=5):
            raise ValueError('Automation timestamp is in the future')
        path = Path(self.store.data_dir) / ('automation-status-'+str(body['root_id'])+'.json')
        with self.store.lock:
            state = json.loads(path.read_text(encoding='utf-8')) if path.exists() else {}
            previous = state.get(body['kind'], {})
            if previous.get('updated_at', '') > cleaned['updated_at']:
                return {'ok': True, 'ignored_older': True}
            state[body['kind']] = cleaned
            temporary = path.with_suffix('.'+uuid.uuid4().hex+'.tmp')
            try:
                temporary.write_text(json.dumps(state, ensure_ascii=False), encoding='utf-8')
                os.replace(temporary, path)
            finally:
                temporary.unlink(missing_ok=True)
        return {'ok': True}

    def state(self):
        with self.store.lock:
            rounds = json.loads(self.path.read_text(encoding='utf-8')) if self.path.exists() else []
        return {'rounds': sorted(rounds, key=lambda r: (r['created_at'], r['round_id']), reverse=True),
                'automatic_training': {'minimum_confirmed_images': 20},
                'note': '每累计20张符合原训练规则的已确认图片训练；验证通过才更新模型。验证集通过率不等于人工审核通过率。'}

    def record(self, value):
        identity = value.get('round_id', '')
        if not isinstance(identity, str) or not re.fullmatch(r'[A-Za-z0-9_-]{1,100}', identity):
            raise ValueError('Invalid training round ID')
        if value.get('same_validation_images') is not True or value.get('test_used') is not False:
            raise ValueError('Training comparison must use the same validation images, without test reuse')
        cleaned = {'round_id': identity, 'same_validation_images': True, 'test_used': False,
                   'human_pass_rate_after': None}
        for key in ('status', 'created_at', 'baseline_model_sha256', 'candidate_model_sha256', 'promotion_reason'):
            item = value.get(key, '')
            if not isinstance(item, str) or len(item) > 500:
                raise ValueError('Invalid training summary text')
            cleaned[key] = item
        cleaned['promoted'] = value.get('promoted') is True
        count = value.get('reviewed_images', 0)
        if type(count) is not int or count < 0:
            raise ValueError('Invalid reviewed image count')
        cleaned['reviewed_images'] = count
        comparison = value.get('comparison', {})
        objective = comparison.get('objective')
        if objective not in ('problem_location_only', 'issue_type_and_location'):
            raise ValueError('Invalid comparison objective')
        result = {'objective': objective,
                  'definition': ('同一验证集，IoU≥0.5，整图无漏检且无多余框' +
                                 ('，问题类型也正确' if objective == 'issue_type_and_location' else '；仅衡量位置'))}
        for side in ('before', 'after'):
            row = comparison.get(side, {})
            output = {}
            for key in ('found', 'missed', 'false_positive', 'total_boxes', 'images'):
                number = row.get(key)
                if type(number) is not int or number < 0:
                    raise ValueError('Invalid comparison counts')
                output[key] = number
            if output['found'] + output['missed'] != output['total_boxes']:
                raise ValueError('Found and missed counts do not sum to support')
            passed = row.get('whole_image_passed')
            rate = row.get('whole_image_pass_rate')
            if passed is not None and (type(passed) is not int or not 0 <= passed <= output['images']):
                raise ValueError('Invalid image pass count')
            if rate is not None and (type(rate) not in (int, float) or not math.isfinite(rate) or not 0 <= rate <= 1):
                raise ValueError('Invalid image pass rate')
            expected = passed / output['images'] if passed is not None and output['images'] else None
            if (expected is None) != (rate is None) or (rate is not None and abs(rate-expected) > 1e-8):
                raise ValueError('Image pass rate is inconsistent')
            output.update(whole_image_passed=passed, whole_image_pass_rate=rate)
            result[side] = output
        if result['before']['total_boxes'] != result['after']['total_boxes'] or result['before']['images'] != result['after']['images']:
            raise ValueError('Comparison cohorts differ')
        cleaned['comparison'] = result
        with self.store.lock:
            rows = self.state()['rounds']
            rows = [row for row in rows if row['round_id'] != identity]
            rows.append(cleaned)
            rows.sort(key=lambda row: (row['created_at'], row['round_id']), reverse=True)
            temporary = self.path.with_suffix('.'+uuid.uuid4().hex+'.tmp')
            try:
                temporary.write_text(json.dumps(rows[:200], ensure_ascii=False), encoding='utf-8')
                os.replace(temporary, self.path)
            finally:
                temporary.unlink(missing_ok=True)
        return {'ok': True, 'round_id': identity}
