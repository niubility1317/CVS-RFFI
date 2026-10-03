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

VARIANTS=('channel_dual','channel_order')
SOURCE_COMMIT='8d9600a432f9dcb80e2bfcc9dc90cbb1847e1908'


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def write(path,data):Path(path).write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def conditions(variant):
    if variant not in VARIANTS:raise ValueError('Unregistered variant')
    return ('all_on','auxiliary_off','u_off','v_off','g_identity')+(('d_off',) if variant=='channel_order' else ())


def validate_config(c):
    keys={'source_release','source_commit','source_output','variant','model_seed','output_root','conditions','role','launch_owner'}
    if set(c)!=keys or c['role']!='V' or c['conditions']!=list(conditions(c['variant'])):
        raise ValueError('Only fixed source V diagnostic inputs allowed')
    if c['source_commit']!=SOURCE_COMMIT or c['model_seed'] not in range(2026092701,2026092705):
        raise ValueError('Unexpected frozen source checkpoint')
    project='/home/szu2070436088/2510044040/CV-SincNet'
    rid=c['variant']+'-s'+str(c['model_seed'])
    if c['source_release']!=project+'/releases/cvs_channel_order_identity_20261003_r01' or c['source_output']!=project+'/runs/20261003-phase1-cvs-channel-order-identity-manysig-m16-r01/'+rid+'/source':
        raise ValueError('Unexpected source checkpoint path')
    if c['output_root']!=project+'/runs/20261003-diagnostic-cvs-channel-attribution-source-manysig-m8-r01/'+rid:
        raise ValueError('Unexpected exclusive diagnostic output')
    return c


@torch.no_grad()
def components(model,x):
    from experiments.cvs_neural_residual_identity.model import NeuralResidualCVS
    from experiments.cvs_channel_order_identity.model import cross_order_readout
    base=NeuralResidualCVS.features(model,x)
    a=model.channel_branch(x);branch=model.channel_branch
    ur=branch.readout(a['u']);vr=branch.readout(a['v'])
    pu=branch.u_project(ur);pv=branch.v_project(vr)
    pd=torch.zeros_like(base)
    scalar={}
    base_norm=base.norm(dim=1).clamp_min(1e-12)
    scalar['u_projection_relative']=pu.norm(dim=1)/base_norm
    scalar['v_projection_relative']=pv.norm(dim=1)/base_norm
    scalar['auxiliary_relative']=a['residual'].norm(dim=1)/base_norm
    scalar['uv_readout_cosine']=F.cosine_similarity(ur,vr,dim=1)
    scalar['uv_readout_relative_difference']=(ur-vr).norm(dim=1)/ur.norm(dim=1).clamp_min(1e-12)
    scalar['uv_projection_cosine']=F.cosine_similarity(pu,pv,dim=1)
    scalar['g_input_relative']=(a['compensated_input']-x).flatten(1).norm(dim=1)/x.flatten(1).norm(dim=1).clamp_min(1e-12)
    if branch.d_project is not None:
        cross=cross_order_readout(a['u'],a['difference']);pd=branch.d_project(cross)
        scalar['d_feature_relative']=a['difference'].flatten(1).norm(dim=1)/a['u'].flatten(1).norm(dim=1).clamp_min(1e-12)
        scalar['d_cross_readout_norm']=cross.norm(dim=1)
        scalar['d_projection_relative']=pd.norm(dim=1)/base_norm
        r,i=a['u'].unbind(1);dr,di=a['difference'].unbind(1)
        cr=r*dr+i*di;ci=r*di-i*dr
        coherent=(cr.mean(-1).square()+ci.mean(-1).square()).sqrt()
        absolute=(cr.square()+ci.square()).sqrt().mean(-1)
        # Energy-weighted temporal coherence; zero energy has no defined ratio.
        scalar['d_cross_temporal_coherence']=coherent.sum(1)/absolute.sum(1).clamp_min(1e-12)
    residual=pv+pu+pd
    if not torch.allclose(residual,a['residual'],atol=1e-6,rtol=1e-6):raise ValueError('Branch decomposition differs')
    features=dict(all_on=base+a['residual'],auxiliary_off=base,u_off=base+pv+pd,v_off=base+pu+pd,
                  g_identity=base+branch.v_project(ur)+pu)
    if branch.d_project is not None:features['d_off']=base+pv+pu
    logits={name:model.classify_features(z) for name,z in features.items()}
    scalar['g_identity_logit_distance']=(logits['all_on']-logits['g_identity']).norm(dim=1)
    if branch.d_project is not None:scalar['d_off_logit_distance']=(logits['all_on']-logits['d_off']).norm(dim=1)
    return logits,scalar


def execute(c):
    validate_config(c);release=Path(c['source_release'])
    if (release/'release_commit.txt').read_text().strip()!=c['source_commit']:raise ValueError('Frozen source release differs')
    sys.path[:0]=[str(release),str(release/'code')]
    from experiments.cvs_channel_order_identity.dispatch import read_source_record
    from experiments.cvs_channel_order_identity.model import build
    from experiments.cvs_equivariant_identity.precision import numerical_context,actual_flags
    from experiments.cvs_identity_ce.source import source_args
    from baselines.common.practical_source import build_contract_split
    from baselines.common.cvs_data import make_cvs_loader
    source=Path(c['source_output']);resolved=read(source/'resolved_config.json');contract=read(source/'source_contract.json')
    record=read_source_record(c,read(resolved['source_contract']),'cvs_channel_order_identity')
    initial=read(source/'initialization.json')
    payload=torch.load(source/'last.pt',map_location='cpu',weights_only=False)
    expected=dict(epoch=200,selection='fixed_last_epoch',method='cvs_channel_order_identity',variant=c['variant'],config=resolved,source_contract=contract,initialization=initial,classes=contract['classes'],num_classes=6)
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
        ids=[];ys=[];rx=[];pred={k:[] for k in conditions(c['variant'])};ce={k:0. for k in pred};scalar={};tic=time.perf_counter()
        with torch.no_grad():
            for batch in loader:
                x=batch['iq'].to(device);y=batch['label'].to(device);scores,stats=components(model,x)
                for name,value in scores.items():
                    if not torch.isfinite(value).all():raise ValueError('Nonfinite source logits')
                    pred[name].extend(value.argmax(1).cpu().tolist());ce[name]+=float(F.cross_entropy(value,y,reduction='sum'))
                for name,value in stats.items():
                    if not torch.isfinite(value).all():raise ValueError('Nonfinite diagnostic')
                    scalar.setdefault(name,[]).extend(value.cpu().double().tolist())
                ids.extend(m['sample_id'] for m in batch['meta']);ys.extend(batch['label'].tolist());rx.extend(batch['receiver'].tolist())
                if len(ids)%4096==0:print('PROGRESS '+json.dumps(dict(source_v_packets=len(ids))),flush=True)
        if len(ids)!=27000 or len(set(ids))!=27000 or set(ids)!=set(contract['role_ids']['V']) or set(rx)!={1,3,4,6,8}:raise ValueError('Physical V mismatch')
        rows=[];all_correct=np.asarray(pred['all_on'])==np.asarray(ys)
        for name in pred:
            correct=np.asarray(pred[name])==np.asarray(ys)
            rates={str(r):float(correct[np.asarray(rx)==r].mean()) for r in sorted(set(rx))}
            rows.append(dict(condition=name,count=len(ids),accuracy=float(correct.mean()),ce=ce[name]/len(ids),rx_accuracy=rates,worst_rx=min(rates.values()),prediction_changes=int((np.asarray(pred[name])!=np.asarray(pred['all_on'])).sum()),helped_by_all_on=int((all_correct&~correct).sum()),hurt_by_all_on=int((~all_correct&correct).sum())))
            np.savez(out/(name+'_source_predictions.npz'),ids=np.asarray(ids),truth=np.asarray(ys),receiver=np.asarray(rx),predictions=np.asarray(pred[name]))
            print('CONDITION '+json.dumps(rows[-1]),flush=True)
        baseline=rows[0]
        if abs(baseline['accuracy']-record['accuracy'])>1e-12 or abs(baseline['worst_rx']-record['worst_rx'])>1e-12:raise ValueError('All-on does not reproduce E200 metrics')
        if any(not torch.equal(v,model.state_dict()[k]) for k,v in frozen.items()):raise ValueError('Model state changed')
        np.savez(out/'source_scalar_diagnostics.npz',ids=np.asarray(ids),receiver=np.asarray(rx),**{k:np.asarray(v) for k,v in scalar.items()})
        summaries={k:dict(mean=float(np.mean(v)),p10=float(np.quantile(v,.1)),median=float(np.median(v)),p90=float(np.quantile(v,.9)),maximum=float(np.max(v))) for k,v in scalar.items()}
        done=dict(status='SOURCE_ATTRIBUTION_COMPLETE',variant=c['variant'],model_seed=c['model_seed'],rows=rows,scalar_summary=summaries,source_v_count=len(ids),conditions=list(pred),seconds=time.perf_counter()-tic,target_access=False,optimizer_updates=0,source_selection_changed=False,model_unchanged=True,peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(device),claim='Whole source V within frozen checkpoint; ablated inputs need not be on training distribution; not retraining causal effect or target generalization.')
        write(out/'completion.json',done);print(json.dumps(done),flush=True)
    return done


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);execute(read(p.parse_args().config))
