"""Compare machine proposals with human-reviewed boxes, not a blind accuracy test."""
from collections import Counter

def iou(a,b):
    overlap=max(0,min(a['x']+a['w'],b['x']+b['w'])-max(a['x'],b['x']))*max(0,min(a['y']+a['h'],b['y']+b['h'])-max(a['y'],b['y']))
    union=a['w']*a['h']+b['w']*b['h']-overlap
    return overlap/union if union>0 else 0.

def compare_review(original,final,negative_regions=()):
    candidates=sorted(((iou(p,g),pi,gi) for pi,p in enumerate(original) for gi,g in enumerate(final)),reverse=True)
    used_p=set();used_g=set();matches=[]
    for overlap,pi,gi in candidates:
        if overlap<.5:break
        if pi in used_p or gi in used_g:continue
        used_p.add(pi);used_g.add(gi);matches.append((pi,gi,overlap))
    old={b['id']:b for b in original};moved=reclassified=assigned=added=0;retained=set()
    for box in final:
        identity=box.get('source_box_id',box['id']);before=old.get(identity)
        if before is None:added+=1;continue
        retained.add(identity)
        moved+=any(abs(box[k]-before[k])>1e-6 for k in ('x','y','w','h'))
        reclassified+=bool(before['problem']) and box['problem']!=before['problem']
        assigned+=not before['problem'] and bool(box['problem'])
    duplicates=sum(any(iou(box,final[gi])>=.5 for gi in used_g) for pi,box in enumerate(original) if pi not in used_p)
    return {'scope':'human_reviewed_examples_not_independent_blind_test','iou_threshold':.5,
        'machine_boxes':len(original),'human_boxes':len(final),'matched_locations':len(matches),
        'missed_locations':len(final)-len(matches),'extra_candidates':len(original)-len(matches),
        'duplicate_candidates':duplicates,'coordinates_changed':moved,'problem_type_changed':reclassified,'problem_type_assigned':assigned,
        'human_added_boxes':added,'removed_machine_boxes':len(old)-len(retained),
        'normal_region_reasons':dict(Counter(r['reason'] for r in negative_regions)),
        'location_recall_on_reviewed_examples':len(matches)/len(final) if final else None,
        'unmatched_human_problems':dict(Counter(b['problem'] for i,b in enumerate(final) if i not in used_g))}

def summarize_comparisons(items):
    keys=('machine_boxes','human_boxes','matched_locations','missed_locations','extra_candidates',
          'duplicate_candidates','coordinates_changed','problem_type_changed','problem_type_assigned','human_added_boxes','removed_machine_boxes')
    rows=[item['comparison'] for item in items]
    result={key:sum(row.get(key,0) for row in rows) for key in keys}
    result.update(scope='human_reviewed_examples_not_independent_blind_test',reviewed_images=len(rows),
        location_recall_on_reviewed_examples=result['matched_locations']/result['human_boxes'] if result['human_boxes'] else None)
    for key in ('normal_region_reasons','unmatched_human_problems'):
        counts=Counter()
        for row in rows:counts.update(row[key])
        result[key]=dict(counts)
    return result
