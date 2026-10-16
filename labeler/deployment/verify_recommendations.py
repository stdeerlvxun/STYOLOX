"""Compare exported CPU inference with the original fixed-test score references."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from recommendations import OnnxPredictor


def verify(model_dir, root, output):
    predictor=OnnxPredictor(model_dir)
    refs=json.loads((model_dir/'references.json').read_text(encoding='utf-8'))
    threshold=predictor.meta['threshold'];records=[]
    for reference in refs:
        raw=(root/reference['relative_path']).read_bytes()
        assert hashlib.sha256(raw).hexdigest()==reference['sha256'],'Source changed'
        started=time.perf_counter();scores=predictor(raw);seconds=time.perf_counter()-started
        delta=max(abs(a-b) for a,b in zip(scores,reference['scores']))
        # CPU ONNX and CUDA PyTorch use different float32 kernels. Require both
        # a bounded score difference and EXACTLY the same candidate decisions.
        assert delta<1e-3,delta
        assert [s>=threshold for s in scores]==[s>=threshold for s in reference['scores']]
        records.append({'image_id':reference['image_id'],'max_score_delta':delta,'seconds':seconds})
    auxiliary_records=[]
    if predictor.meta.get('state_filter'):
        auxiliary=json.loads((model_dir/'auxiliary-references.json').read_text(encoding='utf-8'))
        auxiliary_threshold=predictor.meta['state_filter']['threshold']
        for i,reference in enumerate(auxiliary):
            raw=(root/reference['relative_path']).read_bytes()
            assert hashlib.sha256(raw).hexdigest()==reference['sha256'],'Source changed'
            actual=predictor.predict(raw)['unrelated_score'];expected=reference['unrelated_score']
            assert abs(actual-expected)<1e-3,(reference['image_id'],actual,expected)
            assert (actual>=auxiliary_threshold)==(expected>=auxiliary_threshold),'State decision changed'
            auxiliary_records.append({'image_id':reference['image_id'],'max_score_delta':abs(actual-expected)})
            if i%50==0:print(json.dumps({'stage':'auxiliary_cpu_verification','done':i+1,'total':len(auxiliary)}),flush=True)
    result={'status':'passed','model_id':predictor.meta['model_id'],'images':len(records),
        'max_score_delta':max(r['max_score_delta'] for r in records),
        'median_seconds':statistics.median(r['seconds'] for r in records),
        'max_seconds':max(r['seconds'] for r in records),'cpu_threads':2,'all_candidates_match':True,
        'source_files_read_only':True,'records':records}
    if auxiliary_records:
        result.update({'auxiliary_images':len(auxiliary_records),'auxiliary_all_candidates_match':True,
            'auxiliary_max_score_delta':max(r['max_score_delta'] for r in auxiliary_records)})
    output.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k!='records'},ensure_ascii=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--model',type=Path,required=True)
    parser.add_argument('--root',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();verify(args.model,args.root,args.output)
