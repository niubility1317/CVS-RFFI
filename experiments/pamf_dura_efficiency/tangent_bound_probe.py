"""Complete source metric pressure and bounded-metric falsification; no new G."""
import argparse
import json
import os
from pathlib import Path
import sys
import time
import numpy as np
import torch
from experiments.pamf_dura_direct_train import runner as base,f_guided
from experiments.pamf_dura_diagnostics.runner import source_indices,tensor,numpy_values
from .bounded_tangent import cap_metric,eligibility
from .clean_tangent import visible_features
from .head_features import head_feature_basis
from .tangent_probe import geometry

RUN='20261031-diagnostic-DURA-source-boundedTangent-s2026092701-r01'
PRODUCER='20261031-phase1-DURA-cleanTangent-E30-s2026092701-r01'
PREVIOUS='20261031-diagnostic-DURA-source-cleanTangent-s2026092701-r01'
WEIGHT=.0033130025035573727
RULE=('Fixedweightalphaunchanged. Complete63000source records musthave>=5percent rawrelativeoperatorlambda>1; '
 'actualFP32 boundedfactors mustallhaverelativelambda<=1+2e-5. BOTHfixedrepeatE30sourceviews01/20 need512defined '
 'cosines,rawrealizedguide/IQgradientpressurep95>1,bounded/exactheadfeature mean cosine>=.5 and>=rawmean-.02, '
 '>=384nonnegative,every5sourceRXmean>=0,andmaxboundedrealizedpressure<=1+2e-5. Onlythen supports separate '
 'scratchjointE30 with spectralcap1 andsamealpha;noautomatictraining/no targetTXclaim/no thresholdrelaxation.')


def stats(v):
    v=np.asarray(v,dtype=np.float64)
    return dict(count=len(v),mean=float(v.mean()),median=float(np.median(v)),p95=float(np.quantile(v,.95)),p99=float(np.quantile(v,.99)),maximum=float(v.max()))


def concentration(v):
    v=np.sort(np.asarray(v,dtype=np.float64))[::-1];total=v.sum()
    return {str(f):float(v[:max(1,int(np.ceil(len(v)*f)))].sum()/total) if total>0 else None for f in [.01,.05,.1]}


def run(args):
    if args.output!=args.root/'runs'/RUN or args.output.exists() or os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('Immutable new CPU-only source diagnostic required')
    p=base.paths(args.root);clean,roles=base.source_training(p);ids=f_guided.source_ids(p,roles);indices=source_indices(roles)
    cache=args.root/'runs'/PRODUCER/'scratch_joint/source_clean_J_cache';done=base.read_json(cache/'complete.json')
    expected=dict(status='SOURCE_F_CLEAN_J_COMPLETE',count=63000,shape=[63000,5,2,256],numeric_bytes=645120000,
        F_teacher=f_guided.teacher_reference(),F_inputs=63000,F_J_VJP_inputs=315000,F_updates=0,F_unchanged=True,TX_labels_read=0,V_reads=0,target_reads=0)
    if any(done.get(k)!=v for k,v in expected.items()):raise ValueError('Original source-only F derivative cache required')
    if not np.array_equal(np.load(cache/'physical_ids.npy',allow_pickle=False),ids):raise ValueError('Original full source derivative order required')
    raw=np.load(cache/'jacobian.npy',mmap_mode='r',allow_pickle=False);basis=tensor(np.load(cache/'basis.npy',allow_pickle=False),'cpu')
    if raw.shape!=(63000,5,2,256) or raw.dtype!=np.float32 or basis.shape!=(160,5) or not torch.isfinite(basis).all():raise ValueError('Original complete FP32 source derivative factor required')
    previous=args.root/'runs'/PREVIOUS;old=base.read_json(previous/'results.json')
    if (old['source_checkpoint']!='20261031-phase1-DURA-repeatViews-E30-s2026092701-r01' or old['source_commit']!='8cf8e22c9ffb54d271b22e621401de8e34c69068'
        or old['F_teacher']!=f_guided.teacher_reference() or not old['F_unchanged'] or any(old[k]!=0 for k in ['new_G_inputs','optimizer_updates','TX_labels_read','V_reads','target_reads'])
        or old['decision']['fixed_source_weight']!=WEIGHT):raise ValueError('Fixed pure-repeat source-only diagnostic inputs required')
    with np.load(previous/'source_clean_jacobian.npz',allow_pickle=False) as a:
        for key,value in dict(source_positions=indices,physical_ids=ids[indices],roles=roles[indices]).items():
            if not np.array_equal(a[key],value):raise ValueError('Fixed original source reference positions required')
        old_basis=tensor(a['basis'],'cpu');reference=tensor(a['clean_reference'],'cpu')
    frame=old_basis.T@basis
    torch.testing.assert_close(frame@frame.T,torch.eye(5),atol=1e-5,rtol=1e-5)
    reference=reference@frame
    from experiments.pamf_fcrossrx_readout.frozen import load_reader
    reader=load_reader(p['reader'],p['source_cache'],'cpu').eval().requires_grad_(False)
    before={k:v.clone() for k,v in reader.state_dict().items()};head=head_feature_basis(reader).float()
    torch.testing.assert_close(basis@basis.T,head@head.T,atol=1e-5,rtol=1e-5)
    args.output.mkdir();base.write_json(args.output/'launch.json',dict(pid=os.getpid(),cwd=os.getcwd(),argv=sys.argv,commit=args.commit,CPU_threads=2,GPU_workers=0,rule=RULE))
    with torch.no_grad():scores=reader(tensor(clean[indices[:2]],'cpu'))
    if scores.shape!=(2,6) or not torch.isfinite(scores).all():raise ValueError('Actualsource2Ls F defaulthead noquery smoke failed')
    base.write_json(args.output/'smoke.json',dict(status='PASS',real_Ls=2,new_F_inputs=2,new_G_inputs=0,optimizer_updates=0,TX_labels_read=0,target_reads=0))
    print(json.dumps(dict(event='REAL_SOURCE_F_BOUNDED_METRIC_SMOKE_PASS',Ls=2)),flush=True)
    started=time.perf_counter();out=args.output/'source_metric_cache';out.mkdir()
    factors=np.lib.format.open_memmap(out/'metric_factor.npy',mode='w+',dtype=np.float32,shape=raw.shape);vectors={}
    for offset in range(0,63000,1024):
        end=min(63000,offset+1024);jac=tensor(raw[offset:end],'cpu');x=tensor(clean[offset:end],'cpu')
        energy=512*x.square().mean((1,2)).clamp_min(1e-4);factor,info=cap_metric(jac,energy,WEIGHT)
        if not torch.isfinite(factor).all():raise FloatingPointError('Finite bounded source factors required')
        factors[offset:end]=numpy_values(factor)
        values=dict(raw_trace=info['raw_trace'],raw_relative_max=info['raw_relative_eigenvalues'][:,-1],
            clipped_record=info['clipped_record'].float(),clipped_directions=info['clipped_directions'].float(),bounded_relative_max=info['bounded_relative_max'])
        for key,value in values.items():vectors.setdefault(key,[]).append(numpy_values(value))
    factors.flush();values={k:np.concatenate(v) for k,v in vectors.items()}
    rx=np.load(p['cache']/'train/rx.npy',allow_pickle=False);day=np.load(p['cache']/'train/day.npy',allow_pickle=False)
    np.save(out/'physical_ids.npy',ids,allow_pickle=False)
    np.savez(args.output/'full_source_geometry.npz',physical_ids=ids,roles=roles,rx=rx,day=day,**values)
    masks={'all':np.ones(63000,bool)}
    for key,labels in dict(role=roles,rx=rx,day=day).items():
        for value in np.unique(labels):masks[f'{key}:{value}']=labels==value
    full={key:dict(count=int(mask.sum()),raw_relative_max=stats(values['raw_relative_max'][mask]),
        clipped_fraction=float(values['clipped_record'][mask].mean(dtype=np.float64)),trace_concentration=concentration(values['raw_trace'][mask])) for key,mask in masks.items()}
    cache_seconds=time.perf_counter()-started;maximum=float(values['bounded_relative_max'].max())
    base.write_json(out/'complete.json',dict(status='SOURCE_BOUNDED_METRIC_COMPLETE',count=63000,shape=[63000,5,2,256],numeric_bytes=645120000,
        weight=WEIGHT,relative_eigenvalue_cap=1,maximum_actual_relative_eigenvalue=maximum,F_teacher=f_guided.teacher_reference(),
        source_derivative_cache_run=PRODUCER,new_F_inputs=0,new_G_inputs=0,optimizer_updates=0,TX_labels_read=0,V_reads=0,target_reads=0,seconds=cache_seconds))
    print(json.dumps(dict(event='FULL_SOURCE_BOUNDED_METRIC_COMPLETE',count=63000,seconds=cache_seconds)),flush=True)
    summaries={}
    for view in ['1','20']:
        with np.load(previous/f'source-{view}.npz',allow_pickle=False) as a:data={k:a[k].copy() for k in a.files}
        for key,value in dict(source_positions=indices,physical_ids=ids[indices],roles=roles[indices],rx=rx[indices],clean=clean[indices]).items():
            if not np.array_equal(data[key],value):raise ValueError('Same fixed source observations required')
        metrics={}
        for offset in range(0,512,32):
            sl=slice(offset,offset+32);take=indices[sl];x=tensor(clean[take],'cpu');r=tensor(data['restored'][sl],'cpu').requires_grad_(True)
            phi=visible_features(reader,r,basis);exact=(phi-reference[sl]).square().sum(1);gradient=torch.autograd.grad(exact.sum(),r)[0]
            delta=r.detach()-x;energy=512*x.square().mean((1,2),keepdim=True).clamp_min(1e-4);iq=2*delta/energy
            local={};flags=[]
            for name,matrix in [('raw',raw[take]),('bounded',factors[take])]:
                factor=tensor(matrix,'cpu');projected=(factor*delta[:,None]).sum((2,3));guide=2*WEIGHT*(factor*projected[:,:,None,None]).sum(1)
                g=geometry(guide,gradient);flags.append(g['defined']);pressure=guide.flatten(1).norm(dim=1)/iq.flatten(1).norm(dim=1).clamp_min(1e-12)
                local.update({name+'_cosine':g['cosine'],name+'_defined':g['defined'].float(),name+'_pressure':pressure,
                    name+'_dot':g['dot'],name+'_norm':g['left_norm']})
            local.update(exact_norm=gradient.flatten(1).norm(dim=1),defined=(flags[0]&flags[1]).float())
            for key,value in local.items():
                if not torch.isfinite(value).all():raise FloatingPointError('Finite source pressure/alignment required')
                metrics.setdefault(key,[]).append(numpy_values(value))
        v={k:np.concatenate(value) for k,value in metrics.items()};valid=v['defined'].astype(bool)
        def mean(a,mask):return float(a[mask].mean(dtype=np.float64)) if mask.any() else None
        summaries[view]=dict(count=512,defined=int(valid.sum()),raw_pressure_p95=float(np.quantile(v['raw_pressure'],.95)),
            raw_mean_cosine=mean(v['raw_cosine'],valid),bounded_mean_cosine=mean(v['bounded_cosine'],valid),
            bounded_nonnegative=int(((v['bounded_cosine']>=0)&valid).sum()),
            rx_bounded_mean={str(rxvalue):mean(v['bounded_cosine'],valid&(rx[indices]==rxvalue)) for rxvalue in np.unique(rx[indices])},
            maximum_bounded_pressure=float(v['bounded_pressure'].max()),raw_pressure=stats(v['raw_pressure']),bounded_pressure=stats(v['bounded_pressure']))
        np.savez(args.output/f'source-{view}.npz',source_positions=indices,physical_ids=ids[indices],roles=roles[indices],rx=rx[indices],**v)
        print(json.dumps(dict(event='BOUNDED_SOURCE_VIEW_COMPLETE',view=view,summary=summaries[view])),flush=True)
    if any(not torch.equal(value,before[key]) for key,value in reader.state_dict().items()) or any(p.grad is not None for p in reader.parameters()):raise ValueError('Frozen F changed')
    decision=eligibility(full['all']['clipped_fraction'],maximum,summaries)
    base.write_json(args.output/'results.json',dict(status='ANALYZED_SOURCE_ONLY',full_source=full,summary=summaries,decision=decision,rule=RULE,
        source_records=63000,probe_records=512,source_views=[1,20],F_teacher=f_guided.teacher_reference(),F_unchanged=True,
        source_derivative_cache_run=PRODUCER,source_error_checkpoint=old['source_checkpoint'],source_error_commit=old['source_commit'],weight=WEIGHT,
        maximum_actual_relative_eigenvalue=maximum,new_F_inputs=1024,smoke_F_inputs=2,new_F_VJP_inputs=1024,reused_clean_feature_references=512,
        new_G_inputs=0,optimizer_updates=0,TX_labels_read=0,V_reads=0,target_reads=0,metric_cache_seconds=cache_seconds,seconds=time.perf_counter()-started,
        scope='Source trainingpool only;bounded loss/IQ-output gradient,not Gparametergradient or TXaccuracy/targetbenefit;no automatictraining'))
    base.write_json(args.output/'complete.json',dict(status='ANALYZED_SOURCE_ONLY',decision=decision,optimizer_updates=0,target_reads=0,goal_complete=False))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ['root','output']:parser.add_argument('--'+key,type=Path,required=True)
    parser.add_argument('--commit',required=True);args=parser.parse_args();args.root=args.root.resolve();args.output=args.output.resolve()
    if Path('release_commit.txt').read_text().strip()!=args.commit:raise ValueError('Pinned release required')
    torch.set_num_threads(2);run(args)


if __name__=='__main__':main()
