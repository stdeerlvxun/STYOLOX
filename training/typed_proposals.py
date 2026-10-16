"""Keep real detector issue predictions separate from human classification."""
from __future__ import annotations

from generate_detector_proposals import normalize_predictions


def box_iou(first, second):
    overlap=max(0,min(first['x']+first['w'],second['x']+second['w'])-max(first['x'],second['x']))*max(0,min(first['y']+first['h'],second['y']+second['h'])-max(first['y'],second['y']))
    union=first['w']*first['h']+second['w']*second['h']-overlap
    return overlap/union if union>0 else 0.


def normalize_typed_predictions(predictions, model_sha, *, class_agnostic=False, max_boxes=10):
    """Cross-class location NMS retains the class actually predicted for each box."""
    if class_agnostic:raise ValueError('A typed detector is required; retain the current issue-class model before refill')
    boxes=[]
    for prediction in sorted(predictions,key=lambda p:p['score'],reverse=True):
        location_only=class_agnostic or prediction.get('class_agnostic',False) or prediction.get('problem')=='问题位置'
        problem=prediction.get('problem','')
        if location_only or not isinstance(problem,str) or not problem:
            raise ValueError('Detector output has no trained issue type; no blank-type suggestions generated')
        clean=normalize_predictions([{**prediction,'problem':problem}],{problem},max_boxes=1)
        if not clean:continue
        box=clean[0]
        if any(box_iou(box,old)>.5 for old in boxes):continue
        box['type_provenance']={'source':'detector_class_prediction','model_sha256':model_sha,'score':float(prediction['score'])}
        boxes.append(box)
        if len(boxes)>=max_boxes:break
    return boxes


def enrich_boxes(original, predictions, model_sha, *, class_agnostic=False, min_iou=.5):
    """Fill only missing names, preserving proposal box identity and exact geometry."""
    import copy
    result=copy.deepcopy(original)
    candidates=normalize_typed_predictions(predictions,model_sha,class_agnostic=class_agnostic)
    pairs=sorted(((box_iou(box,prediction),i,j) for i,box in enumerate(result) if not box.get('problem')
                  for j,prediction in enumerate(candidates) if prediction.get('problem')),reverse=True)
    used_boxes=set();used_predictions=set()
    for overlap,index,candidate in pairs:
        if overlap<min_iou:break
        if index in used_boxes or candidate in used_predictions:continue
        result[index]['problem']=candidates[candidate]['problem']
        result[index]['type_provenance']={**candidates[candidate]['type_provenance'],'match_iou':overlap}
        used_boxes.add(index);used_predictions.add(candidate)
    return result,len(used_boxes)
