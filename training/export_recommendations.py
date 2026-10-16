"""Export the reviewed problem-only checkpoint for CPU inference on server 101."""
import argparse
import hashlib
import json
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
from improve_classifier import classification_encoder, crop_tensors, digest, ridge_scores
from vision_classifier import read_json, write_json, validate_training_dataset, TRAINING_POLICY


class RecommendationModel(nn.Module):
    def __init__(self, encoder, ridge, auxiliary=None):
        super().__init__(); self.encoder=encoder
        self.register_buffer('center',ridge['center'])
        self.register_buffer('coefficients',ridge['training_vectors'].T@ridge['alpha'])
        self.register_buffer('prior',ridge['prior'])
        self.auxiliary=auxiliary is not None
        if self.auxiliary:
            self.auxiliary_feature=auxiliary['feature_kind']
            state=auxiliary['ridge']
            self.register_buffer('auxiliary_center',state['center'])
            self.register_buffer('auxiliary_coefficients',state['training_vectors'].T@state['alpha'])
            self.register_buffer('auxiliary_prior',state['prior'])

    def forward(self, images):
        vectors=self.encoder(images)
        parts=[vectors[0],vectors[1:].mean(0),vectors[1:].max(0).values]
        raw=torch.cat([F.normalize(part,dim=0) for part in parts])
        scores=(F.normalize(raw-self.center,dim=0)@self.coefficients+self.prior).clamp(0,1)
        if not self.auxiliary:return scores
        auxiliary_raw=vectors[0] if self.auxiliary_feature=='global' else raw
        unrelated=(F.normalize(auxiliary_raw-self.auxiliary_center,dim=0)@
            self.auxiliary_coefficients+self.auxiliary_prior).clamp(0,1)
        return scores,unrelated


def export(run, output, auxiliary_run=None):
    checkpoint=torch.load(run/'classifier.pt',map_location='cpu',weights_only=True)
    assert checkpoint['training_policy']==TRAINING_POLICY
    assert checkpoint['feature_kind']=='convnext_local_ridge' and checkpoint['weights_retrained']
    output.mkdir(parents=True,exist_ok=True)
    if (output/'model.json').exists():raise ValueError('Export exists; choose a new folder')
    torch.set_num_threads(4)
    encoder,meta=classification_encoder(torch.device('cpu'))
    assert meta['sha256']==checkpoint['encoder_sha256']
    auxiliary=None
    if auxiliary_run:
        auxiliary=torch.load(auxiliary_run/'unrelated-filter.pt',map_location='cpu',weights_only=True)
        assert auxiliary['training_policy']=='unrelated_auxiliary_v1'
        assert auxiliary['encoder_sha256']==meta['sha256']
    model=RecommendationModel(encoder,checkpoint['ridge'],auxiliary).eval()
    destination=output/'problems.onnx'
    torch.onnx.export(model,torch.zeros(6,3,224,224),str(destination),opset_version=17,
        input_names=['views'],output_names=['scores','unrelated_score'] if auxiliary else ['scores'],dynamo=False)
    source=Path(checkpoint['source_run']);dataset=read_json(source/'dataset.json')
    validate_training_dataset(dataset)
    cache=torch.load(source/'improvement-v1'/'convnext-features.pt',weights_only=True)
    assert cache['manifest_hash']==digest(source/'dataset.json')
    model=model.to('cuda' if torch.cuda.is_available() else 'cpu');device=next(model.parameters()).device
    references=[];max_delta=0.
    with torch.inference_mode():
        for i,row in enumerate(dataset['records']):
            if row['split']!='test':continue
            prediction=model(crop_tensors(row).to(device))
            scores=(prediction[0] if auxiliary else prediction).cpu()
            expected=ridge_scores(checkpoint['ridge'],cache['multi'][i:i+1])[0]
            delta=float((scores-expected).abs().max());max_delta=max(delta,max_delta)
            assert delta<1e-5,delta
            references.append({'image_id':row['image_id'],'relative_path':row['relative_path'],
                'sha256':row['sha256'],'scores':expected.tolist()})
    info={'schema_version':1,'model_id':'recall-trained-v3-'+digest(run/'classifier.pt')[:12],
        'source_checkpoint_sha256':digest(run/'classifier.pt'),'onnx_sha256':digest(destination),
        'training_policy':TRAINING_POLICY,'problems':checkpoint['problems'],'threshold':checkpoint['threshold'],
        'objective':'find_problem_points_for_human_review','responsibility_output':False,
        'requires_human_review':True,'probabilities_calibrated':False,'input_views':6,
        'torch_export_reference_max_delta':max_delta,'source_dataset_sha256':digest(source/'dataset.json')}
    if auxiliary:
        auxiliary_sha=digest(auxiliary_run/'unrelated-filter.pt')
        # Small numerical safety margin for float32 CPU/CUDA differences. This
        # is fixed export policy, never selected from test labels or outcomes.
        cutoff=auxiliary['threshold']+.002
        info['model_id']+='-unrelated-'+auxiliary_sha[:8]
        info['state_filter']={'model_id':'unrelated-'+auxiliary_sha[:12],'training_policy':auxiliary['training_policy'],
            'checkpoint_sha256':auxiliary_sha,'threshold':cutoff,'validation_threshold':auxiliary['threshold'],
            'numerical_safety_margin':.002,'automatic_skip':False,'suppress_problem_candidates':False,
            'requires_human_confirmation':True}
        auxiliary_ds=read_json(auxiliary_run/'dataset.json')
        auxiliary_cache=torch.load(auxiliary_run/'features.pt',weights_only=True)
        assert auxiliary_cache['manifest_hash']==digest(auxiliary_run/'dataset.json')
        references_aux=[]
        for i,row in enumerate(auxiliary_ds['records']):
            if row['split']!='test' and not (row['split']=='validation' and not row['auxiliary_target']):continue
            value=float(ridge_scores(auxiliary['ridge'],auxiliary_cache[auxiliary['feature_kind']][i:i+1])[0,0])
            references_aux.append({'image_id':row['image_id'],'relative_path':row['relative_path'],'sha256':row['sha256'],
                'unrelated_score':value,'is_unrelated':bool(row['auxiliary_target']),'split':row['split']})
        write_json(output/'auxiliary-references.json',references_aux)
        from improve_classifier import summary
        test=[r for r in references_aux if r['split']=='test']
        deployed_metrics=summary([[r['is_unrelated']] for r in test],[[r['unrelated_score']] for r in test],cutoff)
        write_json(output/'auxiliary-deployment-metrics.json',deployed_metrics)
    write_json(output/'model.json',info);write_json(output/'references.json',references)
    print(json.dumps({'model':str(destination),'bytes':destination.stat().st_size,
        'reference_images':len(references),'torch_max_delta':max_delta,**info},ensure_ascii=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--auxiliary-run',type=Path)
    args=parser.parse_args();export(args.run,args.output,args.auxiliary_run)
