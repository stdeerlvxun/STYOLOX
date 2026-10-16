"""Import explicit human local false-positive feedback without rewriting human annotations."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from app import Store
from annotations import fingerprint
from deployment.snapshot import digest,annotation_digest


def install(data, manifest_path):
    manifest=json.loads(Path(manifest_path).read_text(encoding='utf-8'))
    if manifest.get('type')=='human_reported_local_false_positive':
        if manifest.get('whole_image_normal') is not False or manifest.get('label_scope')!='region_only':
            raise ValueError('Only explicit local human correction is supported')
        original=manifest['rejected_region']
        region={k:original[k] for k in ('x','y','w','h')}
        region.update(id='negative-'+fingerprint(original['id'])[:24],source_box_id=original['id'],reason='process_hole')
        body={'status':'confirmed','source_sha256':manifest['source']['sha256'],
            'label_fingerprint':manifest['original_classification_fingerprint'],
            'negative_regions':[region],'proposal_id':manifest.get('proposal_id'),'model_id':manifest.get('model_id')}
    else:
        body=manifest
    store=Store(data)
    try:
        before=digest(store.db);annotations_before=annotation_digest(store.db)
        result=store.annotations.save_feedback(manifest['image_id'],body)
        assert before==digest(store.db),'Human classification changed during feedback import'
        assert annotations_before==annotation_digest(store.db),'Human annotation changed during feedback import'
        return {**result,'image_id':manifest['image_id'],'human_digest':before,'annotation_digest':annotations_before,
            'feedback_state':store.annotations.feedback_state()}
    finally:store.db.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',type=Path,required=True)
    parser.add_argument('--manifest',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();result=install(args.data,args.manifest)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False))
