"""Source-clean F local sensitivity; zero training or target access."""
import argparse
import json
import os
from pathlib import Path
import sys
import time
import numpy as np
import torch
from experiments.pamf_dura_diagnostics.runner import source_indices,tensor,numpy_values
from experiments.pamf_dura_direct_train import runner as base,f_guided
from .head_features import head_feature_basis
from .clean_tangent import clean_jacobian,visible_features,tangent_loss,eligibility

RUN='20261031-diagnostic-DURA-source-cleanTangent-s2026092701-r01'
PREVIOUS='20261031-diagnostic-DURA-source-context-oracles-s2026092701-r01'
RULE=('PrimaryfixedrepeatE30 sourceerrors only. Bothcached01/20 musthave512definedperpacketIQgradcosines, '
 'mean tangent/exact-head-feature gradient cosine>=.5 and>=plainIQmean+.1,atleast384/512nonnegative, '
 'every5sourceRXmeantangentcosine>=0. Onlythen qualifies separate scratchjointE30 with '
 'fixedweight .25/max(two median tangent/IQ NMSE IQgradientnormratios). Noautomatictraining, '
 'no target/TXaccuracyclaim. Clean Jacobian is exactlocalderivative,notgloballyexactfeatureloss.')


def geometry(left,right):
    l=left.flatten(1);r=right.flatten(1);ln=l.norm(dim=1);rn=r.norm(dim=1)
    defined=(ln>1e-12)&(rn>1e-12);dot=(l*r).sum(1)
    # Undefined cosines are masked explicitly; stored zero is never a measured cosine.
    return dict(cosine=torch.where(defined,dot/(ln*rn).clamp_min(1e-24),torch.zeros_like(dot)),
        defined=defined,left_norm=ln,right_norm=rn,dot=dot)


def summarize(a,rx):
    valid=a['tangent_defined'].astype(bool);iqvalid=a['IQ_defined'].astype(bool)
    def mean(value,mask):return float(value[mask].mean(dtype=np.float64)) if mask.any() else None
    ratio_valid=a['IQ_norm']>1e-12
    return dict(count=len(valid),defined_tangent=int(valid.sum()),defined_IQ=int(iqvalid.sum()),
        mean_tangent_cosine=mean(a['tangent_cosine'],valid),mean_IQ_cosine=mean(a['IQ_cosine'],iqvalid),
        nonnegative_tangent=int(((a['tangent_cosine']>=0)&valid).sum()),
        rx_mean_tangent={str(r):mean(a['tangent_cosine'],valid&(rx==r)) for r in np.unique(rx)},
        median_tangent_to_IQ_norm=float(np.median(a['tangent_norm'][ratio_valid]/a['IQ_norm'][ratio_valid])) if ratio_valid.any() else None,
        mean_tangent_loss=float(a['tangent_loss'].mean(dtype=np.float64)),mean_exact_loss=float(a['exact_loss'].mean(dtype=np.float64)),
        median_linearization_relative_error=float(np.median(a['linearization_relative_error'])))


def run(args):
    if args.output!=args.root/'runs'/RUN or args.output.exists() or os.environ.get('CUDA_VISIBLE_DEVICES')!='':
        raise ValueError('New immutable CPU-only source diagnostic required')
    p=base.paths(args.root);clean,roles=base.source_training(p);ids=f_guided.source_ids(p,roles);indices=source_indices(roles)
    previous=args.root/'runs'/PREVIOUS;meta=base.read_json(previous/'results.json')
    if (meta['source_checkpoint']!='20261031-phase1-DURA-repeatViews-E30-s2026092701-r01'
        or meta['source_commit']!='8cf8e22c9ffb54d271b22e621401de8e34c69068' or meta['F_teacher']!=f_guided.teacher_reference()
        or meta['cases'][0]!='native' or not meta['G_unchanged'] or not meta['F_unchanged']
        or any(meta[k]!=0 for k in ['optimizer_updates','TX_labels_read','V_reads','target_reads'])):
        raise ValueError('Matching complete fixed pure-repeat source context outputs required')
    archives={}
    for view in ['1','20']:
        with np.load(previous/f'repeat_E30-{view}.npz',allow_pickle=False) as a:archives[view]={k:a[k].copy() for k in a.files}
        a=archives[view]
        for key,expected in dict(source_positions=indices,physical_ids=ids[indices],roles=roles[indices],clean=clean[indices]).items():
            if not np.array_equal(a[key],expected):raise ValueError('Same original source physical records required')
        if a['IQ'].shape!=(5,512,2,256) or not np.isfinite(a['IQ']).all():raise ValueError('Complete fixedsource native IQ required')
    rx=np.load(p['cache']/'train/rx.npy',allow_pickle=False)[indices]
    if set(rx.tolist())!={1,3,4,6,8} or any(not np.array_equal(a['rx'],rx) for a in archives.values()):raise ValueError('All sourceRX required')
    from experiments.pamf_fcrossrx_readout.frozen import load_reader
    reader=load_reader(p['reader'],p['source_cache'],'cpu').eval().requires_grad_(False)
    before={k:v.clone() for k,v in reader.state_dict().items()};basis=head_feature_basis(reader).float()
    args.output.mkdir();base.write_json(args.output/'launch.json',dict(pid=os.getpid(),cwd=os.getcwd(),argv=sys.argv,
        commit=args.commit,CPU_threads=2,GPU_workers=0,F_teacher=f_guided.teacher_reference(),rule=RULE))
    x=tensor(clean[indices[:2]],'cpu');value,jac=clean_jacobian(reader,x,basis)
    for i in range(2):
        solo,single=clean_jacobian(reader,x[i:i+1],basis)
        torch.testing.assert_close(value[i:i+1],solo,atol=2e-5,rtol=2e-5)
        torch.testing.assert_close(jac[i:i+1],single,atol=2e-5,rtol=2e-5)
    with torch.no_grad():score=reader(x)
    if score.shape!=(2,6) or not torch.isfinite(score).all() or not (jac.abs().sum()>0):raise ValueError('Actualsource F/noquery sensitivity smoke failed')
    base.write_json(args.output/'smoke.json',dict(status='PASS',real_Ls=2,F_inputs=6,G_inputs=0,TX_labels_read=0,target_reads=0,optimizer_updates=0,batched_vs_single_J_close=True))
    print(json.dumps(dict(event='REAL_SOURCE_F_TANGENT_SMOKE_PASS',Ls=2)),flush=True)
    started=time.perf_counter();refs=[];js=[]
    for offset in range(0,512,32):
        phi,jac=clean_jacobian(reader,tensor(clean[indices[offset:offset+32]],'cpu'),basis)
        refs.append(numpy_values(phi));js.append(numpy_values(jac))
    reference=np.concatenate(refs);jacobians=np.concatenate(js);cache_seconds=time.perf_counter()-started
    np.savez(args.output/'source_clean_jacobian.npz',source_positions=indices,physical_ids=ids[indices],roles=roles[indices],rx=rx,
        clean_reference=reference,jacobian=jacobians,basis=numpy_values(basis))
    summaries={};all_vectors={}
    for view,a in archives.items():
        vectors={}
        for offset in range(0,512,32):
            sl=slice(offset,offset+32);x=tensor(clean[indices[sl]],'cpu');r=tensor(a['IQ'][0,sl],'cpu').requires_grad_(True)
            jac=tensor(jacobians[sl],'cpu');ref=tensor(reference[sl],'cpu');phi=visible_features(reader,r,basis)
            exact=(phi-ref).square().sum(1);exact_grad=torch.autograd.grad(exact.sum(),r)[0]
            delta=r.detach()-x;projected=(jac*delta[:,None]).sum((2,3));tangent_grad=2*(projected[:,:,None,None]*jac).sum(1)
            iq_grad=2*delta/(512*x.square().mean((1,2),keepdim=True).clamp_min(1e-4))
            tg=geometry(tangent_grad,exact_grad);ig=geometry(iq_grad,exact_grad)
            local=dict(tangent_loss=tangent_loss(r.detach(),x,jac),exact_loss=exact.detach(),
                tangent_cosine=tg['cosine'],IQ_cosine=ig['cosine'],tangent_defined=tg['defined'].float(),IQ_defined=ig['defined'].float(),
                tangent_norm=tg['left_norm'],IQ_norm=ig['left_norm'],exact_norm=tg['right_norm'],tangent_dot=tg['dot'],IQ_dot=ig['dot'],
                linearization_relative_error=(projected-(phi.detach()-ref)).norm(dim=1)/(phi.detach()-ref).norm(dim=1).clamp_min(1e-8))
            for key,v in local.items():
                if v.shape!=(32,) or not torch.isfinite(v).all():raise FloatingPointError('Finite complete source tangent metrics required')
                vectors.setdefault(key,[]).append(numpy_values(v))
        values={k:np.concatenate(v) for k,v in vectors.items()};summaries[view]=summarize(values,rx);all_vectors[view]=values
        np.savez(args.output/f'source-{view}.npz',source_positions=indices,physical_ids=ids[indices],roles=roles[indices],rx=rx,
            restored=a['IQ'][0],clean=clean[indices],**values)
        print(json.dumps(dict(event='SOURCE_TANGENT_VIEW_COMPLETE',view=view,summary=summaries[view],seconds=time.perf_counter()-started)),flush=True)
    if any(not torch.equal(v,before[k]) for k,v in reader.state_dict().items()) or any(p.grad is not None for p in reader.parameters()):raise ValueError('Frozen F changed or accumulated gradients')
    decision=eligibility(summaries)
    base.write_json(args.output/'results.json',dict(status='ANALYZED_SOURCE_ONLY',summary=summaries,decision=decision,rule=RULE,
        F_teacher=f_guided.teacher_reference(),source_context_run=PREVIOUS,source_checkpoint=meta['source_checkpoint'],source_commit=meta['source_commit'],
        source_records=512,L_s=128,U_s=384,source_views=[1,20],source_F_inputs=1536,smoke_F_inputs=6,new_G_inputs=0,
        source_clean_J_VJP_inputs=2560,smoke_J_VJP_inputs=20,source_restored_VJP_inputs=1024,
        F_unchanged=True,optimizer_updates=0,TX_labels_read=0,V_reads=0,target_reads=0,
        source_J_numeric_bytes=jacobians.nbytes,full63000_J_estimated_numeric_bytes=63000*5*512*4,
        fullE30_cached_J_estimated_numeric_read_bytes=30*63000*5*512*4,
        source_J_cache_seconds=cache_seconds,seconds=time.perf_counter()-started,
        scope='Source trainingpool localclean metric alignment only;nottrueTXaccuracy,notglobalexactloss,nottargetbenefit orrecoverylimit;noautomatictraining'))
    base.write_json(args.output/'complete.json',dict(status='ANALYZED_SOURCE_ONLY',decision=decision,optimizer_updates=0,target_reads=0,goal_complete=False))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ['root','output']:parser.add_argument('--'+key,type=Path,required=True)
    parser.add_argument('--commit',required=True);args=parser.parse_args();args.root=args.root.resolve();args.output=args.output.resolve()
    if Path('release_commit.txt').read_text().strip()!=args.commit:raise ValueError('Pinned release required')
    torch.set_num_threads(2);run(args)


if __name__=='__main__':main()
