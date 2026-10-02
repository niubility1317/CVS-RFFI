"""Strict frozen checkpoint smoke precedes query access; no truth or fitting."""
import argparse
import os
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'code')]
import numpy as np
import torch
from experiments.cvs_sixscene_eval.common import VIEWS,SEEDS,read,write
from experiments.cvs_equivariant_identity.precision import numerical_context,FULL_FP32_POLICY,actual_flags

def check(c,done,initial,contract,expected,resolved,payload):
    method='cvs_residual_identity' if c['condition']=='clean_train' else 'cvs_selected_concat'
    if c['condition'] not in {'clean_train','mid_low_aug'} or c['model_seed'] not in SEEDS:raise ValueError('Unregistered row')
    if any(c.get(k) for k in ('truth','p1_truth','target_truth')):raise ValueError('Truth input forbidden')
    if done.get('status')!='SOURCE_TRAINED' or done.get('epoch')!=200 or done.get('steps')!=10000 or done.get('target_access') is not False or done.get('target_evaluated') is not False:raise ValueError('Source not complete/clean')
    if initial.get('status')!='SCRATCH' or initial.get('scratch_only') is not True or initial.get('checkpoint') is not None or initial.get('ancestors')!=[] or initial.get('checkpoint_sources')!=[] or initial.get('target_access') is not False or initial.get('target_contact') is not False or initial.get('physical_roles')!='EXACT_MATCH' or initial.get('model_seed')!=c['model_seed']:raise ValueError('Checkpoint provenance invalid')
    if any(contract.get(k)!=v for k,v in expected.items()) or contract['equalized']!=1 or contract['out_len']!=256 or contract.get('normalize') is not True or contract.get('classes')!=['14-10','14-7','20-15','20-19','6-15','8-20']:raise ValueError('CHECKPOINT_DATA_CONTRACT_MISMATCH')
    if resolved.get('method')!=method or resolved.get('variant')!='residual_fusion' or resolved.get('model_seed')!=c['model_seed'] or resolved.get('selection')!='fixed_last_epoch' or resolved.get('target_access') is not False or resolved.get('source_counts')!={'L_s':6300,'U_s':56700,'V':27000} or resolved.get('steps_per_epoch')!=50 or resolved.get('domain_backbone') is not False or resolved.get('extra_losses')!=[]:raise ValueError('Resolved source provenance mismatch')
    if payload.get('method')!=method or payload.get('variant')!='residual_fusion' or payload.get('config')!=resolved or payload.get('source_contract')!=contract or payload.get('initialization')!=initial or payload.get('epoch')!=200 or payload.get('selection')!='fixed_last_epoch':raise ValueError('Checkpoint content mismatch')
    if c['condition']=='clean_train':
        if resolved.get('augmentation') is not False:raise ValueError('Pure clean provenance mismatch')
    else:
        from experiments.cvs_selected_concat.source import validate_config
        validate_config(resolved)
        if resolved.get('numerical_policy')!=FULL_FP32_POLICY:raise ValueError('Augmentation precision mismatch')

def predict(c):
    if c.get('numerical_policy')!=FULL_FP32_POLICY:raise ValueError('Inference precision mismatch')
    if any(c.get(k) for k in ('truth','p1_truth','target_truth')):raise ValueError('Truth input forbidden')
    with numerical_context(FULL_FP32_POLICY):_predict(c)

def _predict(c):
    out=Path(c['output_root']);out.mkdir(parents=True,exist_ok=False)
    source=Path(c['source_output']);device=torch.device('cuda:0');torch.set_num_threads(2)
    done,initial,contract,resolved=[read(source/f) for f in ('completion.json','initialization.json','source_contract.json','resolved_config.json')]
    expected=read(c['source_contract']);payload=torch.load(source/'last.pt',map_location='cpu',weights_only=False)
    check(c,done,initial,contract,expected,resolved,payload)
    if c['condition']=='clean_train':
        from experiments.cvs_residual_identity.model import build
        model=build('residual_fusion')
    else:
        from experiments.cvs_selected_concat.model import build
        model=build();frozen=read(source/'source_selection.json')
        if frozen.get('status')!='SOURCE_SELECTION_FROZEN' or frozen.get('epoch')!=200 or frozen.get('variant')!='residual_fusion' or frozen.get('target_access') is not False or frozen.get('target_score_used') is not False or model.contract()!=resolved.get('architecture_actual'):raise ValueError('Augmented source freeze invalid')
    model.load_state_dict(payload['model'],strict=True);model.to(device);model.eval()
    with torch.no_grad():scores=model(torch.zeros(2,2,256,device=device))
    if scores.shape!=(2,6) or not torch.isfinite(scores).all():raise ValueError('Checkpoint smoke failed')
    write(out/'provenance.json',dict(status='VERIFIED',checkpoint=str(source/'last.pt'),scratch_ancestors=[],selection='own_E200',source_roles='EXACT_MATCH',query_fit=False))
    views=Path(c['views_root']);manifest=read(views/'manifest.json')
    if manifest.get('status')!='VALIDATED_ONCE' or manifest.get('count')!=168000 or manifest.get('scenes')!=list(VIEWS[1:]) or manifest.get('classes')!=contract['classes'] or manifest.get('truth_read') is not False:raise ValueError('Sixscene input mismatch')
    with np.load(views/'index.npz',allow_pickle=False) as ix:ids=ix['ids'].copy()
    if len(ids)!=168000 or len(set(ids.tolist()))!=len(ids):raise ValueError('Invalid query IDs')
    write(out/'resolved_config.json',dict(c,pid=os.getpid(),cwd=str(ROOT),python=sys.executable,backend_flags=actual_flags(),batch_size=256,
        torch_version=torch.__version__,hardware=torch.cuda.get_device_name(device),truth_read=False,query_fit=False,
        commit=(ROOT/'release_commit.txt').read_text().strip(),total_parameters=sum(p.numel() for p in model.parameters())))
    torch.cuda.reset_peak_memory_stats();predictions={};timings={};started=time.perf_counter()
    with torch.no_grad():
        for view in VIEWS:
            array=np.load(manifest['clean_ref'] if view=='clean' else views/(view+'.npy'),mmap_mode='r',allow_pickle=False)
            if array.shape!=(len(ids),2,256):raise ValueError('Input shape mismatch')
            tic=time.perf_counter();pred=[]
            for start in range(0,len(ids),256):
                # Explicit buffer avoids N607 Torch2.1/NumPy2 ndarray ABI bridge.
                data=np.ascontiguousarray(array[start:start+256],dtype=np.float32)
                x=torch.frombuffer(bytearray(data.tobytes()),dtype=torch.float32).reshape(data.shape).to(device)
                logits=model(x)
                if logits.shape!=(len(x),6) or not torch.isfinite(logits).all():raise ValueError('Invalid classifier scores')
                pred.extend(logits.argmax(1).cpu().tolist())
            torch.cuda.synchronize();predictions[view]=np.asarray(pred,dtype=np.int64);timings[view]=time.perf_counter()-tic
            print('PREDICT '+str(dict(view=view,count=len(pred),seconds=timings[view],truth_read=False)),flush=True)
    np.savez(out/'predictions.npz',ids=ids,**predictions)
    write(out/'complete.json',dict(status='PREDICTIONS_COMPLETE',count=len(ids),views=list(VIEWS),truth_read=False,query_fit=False,
        inference_seconds=timings,total_seconds=time.perf_counter()-started,peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated()))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);a=p.parse_args();predict(read(a.config))
