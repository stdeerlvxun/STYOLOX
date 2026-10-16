"""Record feedback IDs actually used by a completed local training run; never promote a model."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from app import Store


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',type=Path,required=True)
    parser.add_argument('--run-id',required=True)
    parser.add_argument('--feedback-ids',required=True,help='Comma-separated IDs actually present in the frozen training manifest')
    parser.add_argument('--model-sha256',required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();identities=[int(v.strip()) for v in args.feedback_ids.split(',') if v.strip()]
    store=Store(args.data)
    try:
        result=store.annotations.record_feedback_training(args.run_id,identities,args.model_sha256)
        result['feedback_state']=store.annotations.feedback_state()
    finally:store.db.close()
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False))
