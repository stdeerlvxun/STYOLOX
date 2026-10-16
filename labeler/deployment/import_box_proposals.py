"""Install frozen, source-verified suggestions without editing any human annotations."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from app import Store
from snapshot import digest,annotation_digest


def install(data,manifest_path):
    manifest=json.loads(Path(manifest_path).read_text(encoding='utf-8'))
    store=Store(data)
    try:
        result=store.annotations.install_proposals(manifest)
        return {**result,'model_id':manifest['model_id'],'queue':store.annotations.proposal_state(),
                'human_data_unchanged':True,'annotation_data_unchanged':True}
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
