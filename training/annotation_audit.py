"""Metadata-only label/box reconciliation; never rewrite labels or review queues."""
from __future__ import annotations

import argparse
from collections import Counter
import html
import json
from pathlib import Path


def audit_annotations(dataset):
    """Keep metadata from every split distinct; optimization uses train only.

    A missing correspondence is not a proven annotation error. In particular,
    paired differences can be a more-specific box label or an unboxed problem.
    No image file (including independent test pixels) is opened here.
    """
    items = []
    for record in dataset.get('records', []):
        labelled = {str(value) for value in record.get('defects', [])}
        boxed = {str(box['problem']) for box in record.get('boxes', [])}
        missing, additional = sorted(labelled - boxed), sorted(boxed - labelled)
        if not missing and not additional:
            continue
        kinds = []
        if missing:
            kinds.append('image_problem_without_corresponding_box')
        if additional:
            kinds.append('box_problem_not_in_image_labels')
        if missing and additional:
            kinds.append('possible_type_difference')
        split = record.get('split', 'unknown')
        items.append({'image_id': record.get('image_id'), 'relative_path': record.get('relative_path'),
            'split': split, 'optimization_allowed': split == 'train',
            'image_problem_labels': sorted(labelled), 'human_box_problem_labels': sorted(boxed),
            'image_labels_without_box': missing, 'box_labels_not_in_image_labels': additional,
            'audit_categories': kinds, 'annotation_error_confirmed': False,
            'manual_action_required': 'Check training discrepancies before altering any human annotation.'
                if split == 'train' else 'Holdout metadata only; do not use to optimize or relabel for this comparison.'})
    counts = {}
    for split in sorted({r.get('split', 'unknown') for r in dataset.get('records', [])}):
        rows = [row for row in items if row['split'] == split]
        counts[split] = {'unique_images': len(rows), 'categories': dict(Counter(
            category for row in rows for category in row['audit_categories']))}
    return {'schema': 'annotation_correspondence_audit_v1', 'source': 'frozen eligible dataset metadata',
        'image_pixels_read': False, 'original_data_modified': False, 'reviewed_images_requeued': False,
        'training_filter_changed': False, 'split_counts': counts,
        'note': 'Categories overlap. Possible type differences are not automatically proven errors; only train differences may guide optimization.',
        'items': items}


def write_audit(dataset_path, output):
    dataset_path, output = Path(dataset_path), Path(output)
    dataset = json.loads(dataset_path.read_text(encoding='utf-8-sig'))
    report = audit_annotations(dataset)
    report['dataset_path'] = str(dataset_path)
    output.mkdir(parents=True, exist_ok=True)
    def write(name, value):
        (output / name).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    write('annotation-audit.json', report)
    train = [row for row in report['items'] if row['optimization_allowed']]
    for filename, category in (
            ('train-missing-problem-boxes.json', 'image_problem_without_corresponding_box'),
            ('train-type-difference-candidates.json', 'possible_type_difference'),
            ('train-additional-box-types.json', 'box_problem_not_in_image_labels')):
        write(filename, {'metadata_only': True, 'original_data_modified': False, 'items': [
            row for row in train if category in row['audit_categories']]})
    write('holdout-metadata-only.json', {'image_pixels_read': False, 'optimization_allowed': False,
        'items': [row for row in report['items'] if not row['optimization_allowed']]})
    escape = lambda value: html.escape(str(value))
    rows = ''.join('<tr>' + ''.join('<td>' + escape(value) + '</td>' for value in (
        row['image_id'], row['relative_path'], '、'.join(row['image_problem_labels']),
        '、'.join(row['human_box_problem_labels']), '、'.join(row['image_labels_without_box']) or '—',
        '、'.join(row['box_labels_not_in_image_labels']) or '—')) + '</tr>' for row in train)
    document = '''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>训练标注对应关系核对清单</title>
<style>body{font-family:Microsoft YaHei,sans-serif;line-height:1.6;margin:28px;background:#f5f7fa;color:#17212e}h1{font-size:24px}table{border-collapse:collapse;width:100%;background:white;font-size:13px}td,th{border:1px solid #d5dce5;padding:10px;text-align:left;vertical-align:top}td:nth-child(2){overflow-wrap:anywhere;max-width:420px}th{background:#e8edf5}</style>
<h1>训练标注对应关系核对清单</h1><p>这里只展示训练集的字段差异，不代表已经确认错标。问题框比历史图片标签更具体、图片问题尚未补全框，都可能产生差异。请核对后再决定；程序不自动改标签、改人工框或重新加入日常待审核队列。</p>
<p>本工具只读冻结数据集的元数据，不打开任何图片。验证集和独立测试集的差异另存 holdout-metadata-only.json，明确标记禁止用于本轮训练优化；保持原筛选规则和分组划分。</p>
<p>各分组差异统计：''' + escape(json.dumps(report['split_counts'], ensure_ascii=False)) + '''</p>
<table><thead><tr><th>图片 ID</th><th>原图相对路径</th><th>图片问题点</th><th>人工框问题点</th><th>图片问题暂无对应框</th><th>框类型未在图片标签中</th></tr></thead><tbody>''' + rows + '</tbody></table></html>'
    (output / '训练标注对应关系核对清单.html').write_text(document, encoding='utf-8')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    result = write_audit(args.dataset, args.out)
    print(json.dumps({'split_counts': result['split_counts'], 'unique_images': len(result['items'])}, ensure_ascii=False))
