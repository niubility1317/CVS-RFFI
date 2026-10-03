"""Measure whole-V branch readout and fixed-weight interventions in one pass."""
import argparse
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

VARIANTS=('response_mean','response_attention','response_order_attention')
SOURCE_COMMIT='83727fa93c090e6daa2b21bd8dd058cfe28b51d1'


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def write(path,data):Path(path).write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def conditions(variant):
    if variant not in VARIANTS:raise ValueError('Unregistered variant')
    return ('all_on','auxiliary_off')


def validate_config(c):
    keys={'source_release','source_commit','source_output','variant','model_seed','output_root','conditions','role','launch_owner'}
    if set(c)!=keys or c['role']!='V' or c['conditions']!=list(conditions(c['variant'])):
        raise ValueError('Only fixed source V diagnostic inputs allowed')
    if c['source_commit']!=SOURCE_COMMIT or c['model_seed'] not in range(2026092701,2026092705):
        raise ValueError('Unexpected frozen source checkpoint')
    project='/home/szu2070436088/2510044040/CV-SincNet'
    rid=c['variant']+'-s'+str(c['model_seed'])
    if c['source_release']!=project+'/releases/cvs_channel_response_identity_20261003_r01' or c['source_output']!=project+'/runs/20261003-phase1-cvs-channel-response-identity-manysig-m12-r01/'+rid+'/source':
        raise ValueError('Unexpected source checkpoint path')
    if c['output_root']!=project+'/runs/20261003-diagnostic-cvs-response-geometry-source-manysig-m12-r01/'+rid:
        raise ValueError('Unexpected exclusive diagnostic output')
    return c


@torch.no_grad()
def components(model,x):
    from experiments.cvs_neural_residual_identity.model import NeuralResidualCVS
    base=NeuralResidualCVS.features(model,x);a=model.response_branch(x)
    joint=base+a['residual']
    coefficients=model.response_branch.compensation.coefficients(x).flatten(1)
    return dict(all_on=model.classify_features(joint),auxiliary_off=model.classify_features(base)),dict(base=base,response=a['residual'],coefficients=coefficients)


def execute(c):
    validate_config(c);release=Path(c['source_release'])
    if (release/'release_commit.txt').read_text().strip()!=c['source_commit']:raise ValueError('Frozen source release differs')
    sys.path[:0]=[str(release),str(release/'code')]
    from experiments.cvs_channel_response_identity.dispatch import read_source_record
    from experiments.cvs_channel_response_identity.model import build
    from experiments.cvs_equivariant_identity.precision import numerical_context,actual_flags
    from experiments.cvs_identity_ce.source import source_args
    from baselines.common.practical_source import build_contract_split
    from baselines.common.cvs_data import make_cvs_loader
    source=Path(c['source_output']);resolved=read(source/'resolved_config.json');contract=read(source/'source_contract.json')
    record=read_source_record(c,read(resolved['source_contract']),'cvs_channel_response_identity')
    initial=read(source/'initialization.json')
    payload=torch.load(source/'last.pt',map_location='cpu',weights_only=False)
    expected=dict(epoch=200,selection='fixed_last_epoch',method='cvs_channel_response_identity',variant=c['variant'],config=resolved,source_contract=contract,initialization=initial,classes=contract['classes'],num_classes=6)
    if any(payload.get(k)!=v for k,v in expected.items()):raise ValueError('Checkpoint payload provenance mismatch')
    out=Path(c['output_root']);out.mkdir(parents=True,exist_ok=False)
    device=torch.device('cuda:0');torch.set_num_threads(2)
    with numerical_context(resolved['numerical_policy']):
        model=build(c['variant']).to(device);model.load_state_dict(payload['model'],strict=True);model.eval();del payload
        for p in model.parameters():p.requires_grad_(False)
        frozen={k:v.detach().clone() for k,v in model.state_dict().items()}
        with torch.no_grad():
            probe=torch.zeros(2,2,256,device=device);probe[1,0]=1.
            scores,_=components(model,probe)
            if not torch.allclose(scores['all_on'],model(probe),atol=1e-6,rtol=1e-6):raise ValueError('No-query full model smoke mismatch')
        split=build_contract_split(source_args(dict(resolved,output_root=str(out/'source_loader'))))
        if split.test or split.named_tests or split.split_info['counts']!={'L_s':6300,'U_s':56700,'V':27000}:raise ValueError('Only source V allowed')
        loader=make_cvs_loader(split.val,batch_size=256,shuffle=False,num_workers=0,device=device,drop_last=False)
        write(out/'resolved_config.json',dict(c,pid=os.getpid(),cwd=os.getcwd(),python=sys.executable,hardware=torch.cuda.get_device_name(device),backend_flags=actual_flags(),source_record=record,target_access=False,optimizer_updates=0,source_selection_changed=False))
        print('RESOLVED_CONFIG '+json.dumps(read(out/'resolved_config.json')),flush=True)
        ids=[];ys=[];rx=[];days=[];values={};tic=time.perf_counter()
        with torch.no_grad():
            for batch in loader:
                x=batch['iq'].to(device);scores,features=components(model,x)
                for name,value in dict(features,**scores).items():
                    if not torch.isfinite(value).all():raise ValueError('Nonfinite geometry')
                    # Use the existing N607-compatible transfer route, independent
                    # of the Torch/NumPy C-ABI bridge installed on that host.
                    values.setdefault(name,[]).append(np.asarray(value.cpu().tolist(),dtype=np.float32))
                ids.extend(m['sample_id'] for m in batch['meta']);ys.extend(batch['label'].tolist());rx.extend(batch['receiver'].tolist());days.extend(batch['day'].tolist())
                if len(ids)%4096==0:print('PROGRESS '+json.dumps(dict(source_v_packets=len(ids))),flush=True)
        if len(ids)!=27000 or len(set(ids))!=27000 or set(ids)!=set(contract['role_ids']['V']) or set(rx)!={1,3,4,6,8} or set(days)!={1,2,3}:raise ValueError('Physical V mismatch')
        arrays={name:np.concatenate(parts) for name,parts in values.items()}
        rows=[]
        for name in conditions(c['variant']):
            prediction=arrays[name].argmax(1);correct=prediction==np.asarray(ys)
            rates={str(r):float(correct[np.asarray(rx)==r].mean()) for r in sorted(set(rx))}
            rows.append(dict(condition=name,count=len(ids),accuracy=float(correct.mean()),rx_accuracy=rates,worst_rx=min(rates.values())))
            print('CONDITION '+json.dumps(rows[-1]),flush=True)
        if abs(rows[0]['accuracy']-record['accuracy'])>1e-12 or abs(rows[0]['worst_rx']-record['worst_rx'])>1e-12:raise ValueError('All-on E200 reproduction differs')
        if any(not torch.equal(v,model.state_dict()[k]) for k,v in frozen.items()):raise ValueError('Model state changed')
        head=model.core.id_backbone.cls_head
        np.savez(out/'source_geometry.npz',ids=np.asarray(ids),truth=np.asarray(ys),receiver=np.asarray(rx),day=np.asarray(days),classifier_weight=np.asarray(head.weight.detach().cpu().tolist(),dtype=np.float32),classifier_scale=np.asarray(head.scale),**arrays)
        done=dict(status='SOURCE_GEOMETRY_COMPLETE',variant=c['variant'],model_seed=c['model_seed'],rows=rows,source_v_count=len(ids),conditions=list(conditions(c['variant'])),seconds=time.perf_counter()-tic,target_access=False,optimizer_updates=0,source_selection_changed=False,model_unchanged=True,peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(device),claim='Complete frozen sourceV representation/logit geometry; descriptive source factors, not causal disentanglement or target generalization')
        write(out/'completion.json',done);print(json.dumps(done),flush=True)

    return done


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);execute(read(p.parse_args().config))
