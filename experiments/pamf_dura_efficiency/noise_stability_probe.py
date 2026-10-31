"""Compare frozen scratch-E30 and DURA under matched source noise perturbations."""
import argparse
import json
import os
from pathlib import Path
import sys
import time
import numpy as np
import torch
from torch.nn import functional as F
from experiments.pamf_dura_direct_train import repeat_views,f_guided
from experiments.pamf_dura_diagnostics.runner import source_indices,tensor,numpy_values
from .context_oracle import validate_source
from .noise_stability import paired_noise,qualifies

base=repeat_views.base
RUN='20261031-diagnostic-DURA-source-noiseStability-s2026092701-r01'
SOURCE_COMMIT='8cf8e22c9ffb54d271b22e621401de8e34c69068'
SEED=2026103113
RULE=('Fixed source512128L384U,cache01/20,same antithetic epsilon per model,real-component variance equals '
 'cached source scalar variance (isotropic stress,not native noise redrawing),clean scene unperturbed. A separately registered scratch jointE30 '
 'noise-replay contrast is eligible only if BOTH views: primary nonclean mean output-drift/input-noise gain '
 '>=1.25 default and primary antithetic MSE/native MSE>=1.10. F clean-choice/feature results are descriptive, '
 'not eligibility or tuning. No automatic training;failure does not relax thresholds;source stress not TX/target proof.')


def aggregate(vectors,mask):
    values={key:dict(mean=float(value[mask].mean(dtype=np.float64)),median=float(np.median(value[mask])),
        p95=float(np.quantile(value[mask],.95))) for key,value in vectors.items()}
    return dict(count=int(mask.sum()),metrics=values)


def run(args):
    if args.output!=args.root/'runs'/RUN or args.output.exists() or os.environ.get('CUDA_VISIBLE_DEVICES')!='':
        raise ValueError('New immutable CPU-only source diagnostic required')
    repeat_views.configure();p=base.paths(args.root)
    validate_source(base.read_json(p['output']/'scratch_joint/training_complete.json'))
    primary=base.load_model(p['output']/'scratch_joint/final.pt','cpu',arm='scratch_joint',commit=SOURCE_COMMIT)
    default,_=base.parent_model(p,'cpu');models=dict(pureG_E30=primary,DURA_default=default)
    for model in models.values():model.eval().requires_grad_(False)
    clean,roles=base.source_training(p);ids=f_guided.source_ids(p,roles);indices=source_indices(roles)
    cache=args.root/'runs'/f_guided.RUN/'scratch_joint/source_feature_cache';meta=base.read_json(cache/'complete.json')
    if (meta['status'],meta['count'],meta['dimension'],meta['F_teacher'],meta['TX_labels_read'],meta['target_reads'])!=(
        'SOURCE_F_FEATURES_COMPLETE',63000,160,f_guided.teacher_reference(),0,0):raise ValueError('Original source clean F cache required')
    if not np.array_equal(np.load(cache/'physical_ids.npy',allow_pickle=False),ids):raise ValueError('Original source physical F order required')
    features=np.load(cache/'features.npy',mmap_mode='r',allow_pickle=False)
    if features.shape!=(63000,160) or features.dtype!=np.float32:raise ValueError('Complete original FP32 source F features required')
    from experiments.pamf_fcrossrx_readout.frozen import load_reader
    reader=load_reader(p['reader'],p['source_cache'],'cpu').eval().requires_grad_(False)
    snapshots={name:{key:value.clone() for key,value in model.state_dict().items()} for name,model in models.items()}
    f_snapshot={key:value.clone() for key,value in reader.state_dict().items()}
    rx=np.load(p['cache']/'train/rx.npy',allow_pickle=False)[indices];day=np.load(p['cache']/'train/day.npy',allow_pickle=False)[indices]
    args.output.mkdir();base.write_json(args.output/'launch.json',dict(pid=os.getpid(),cwd=os.getcwd(),argv=sys.argv,commit=args.commit,
        CPU_threads=2,GPU_workers=0,rule=RULE))
    with torch.no_grad():
        for name,model in models.items():
            score=reader(model(tensor(clean[indices[:2]],'cpu')))
            if score.shape!=(2,6) or not torch.isfinite(score).all():raise ValueError('Actual source2Ls G/F smoke failed')
            print(json.dumps(dict(event='SOURCE_NOISE_STABILITY_SMOKE_PASS',model=name,real_Ls=2)),flush=True)
    base.write_json(args.output/'smoke.json',dict(status='PASS',real_Ls_per_model=2,G_inputs=4,F_inputs=4,TX_labels_read=0,target_reads=0))
    summary={};started=time.perf_counter();baseline_masks={}
    with torch.no_grad():
        for name,model in models.items():
            summary[name]={}
            for view in (1,20):
                folder=p['cache']/'epochs'/f'{view:02d}'
                received=np.load(folder/'received.npy',mmap_mode='r',allow_pickle=False)
                variance=np.load(folder/'variance.npy',mmap_mode='r',allow_pickle=False)
                if received.shape!=(63000,2,256) or received.dtype!=np.float32 or variance.shape!=(63000,) or variance.dtype!=np.float32:
                    raise ValueError('Original source FP32 IQ and real-component variance required')
                with np.load(folder/'groups.npz',allow_pickle=False) as groups:labels={key:groups[key][indices] for key in ['scene','state','lock']}
                if set(labels['scene'].tolist())!=set(base.VIEWS):raise ValueError('All original source scenes required')
                nonclean=labels['scene']!='clean'
                if not nonclean.any():raise ValueError('Nonclean source cohort required')
                baseline_masks[str(view)]=nonclean
                generator=torch.Generator().manual_seed(SEED+view);z=torch.randn(512,2,256,generator=generator)
                vectors={};outputs=[];F_features=[];F_logits=[];noises=[]
                for offset in range(0,512,32):
                    take=indices[offset:offset+32];x=tensor(clean[take],'cpu');y=tensor(received[take],'cpu')
                    v=tensor(variance[take],'cpu');cm=torch.tensor(~nonclean[offset:offset+32],dtype=torch.bool)
                    plus,minus,noise=paired_noise(y,v,cm,z[offset:offset+32]);power=x.square().mean((1,2)).clamp_min(1e-4)
                    reference=tensor(features[take],'cpu');teacher=reader.classify_features(reference)
                    iq=torch.stack([model(value) for value in [y,plus,minus]])
                    embedding=torch.stack([reader.features(value) for value in iq]);logits=torch.stack([reader.classify_features(value) for value in embedding])
                    if not torch.isfinite(iq).all() or not torch.isfinite(embedding).all() or not torch.isfinite(logits).all():
                        raise FloatingPointError('Nonfinite source noise response')
                    MSE=(iq-x[None]).square().mean((2,3))/power[None]
                    cos=1-F.cosine_similarity(embedding,reference[None],dim=2,eps=1e-8)
                    agree=(logits.argmax(2)==teacher.argmax(1)[None]).float()
                    values=dict(native_nmse=MSE[0],plus_nmse=MSE[1],minus_nmse=MSE[2],antithetic_nmse=MSE[1:].mean(0),
                        input_noise_nmse=noise.square().mean((1,2))/power,
                        output_drift_nmse=(iq[1:]-iq[0:1]).square().mean((2,3)).mean(0)/power,
                        native_cosine_error=cos[0],antithetic_cosine_error=cos[1:].mean(0),
                        native_agreement=agree[0],plus_agreement=agree[1],minus_agreement=agree[2])
                    if any(value.shape!=(len(take),) or not torch.isfinite(value).all() for value in values.values()):raise ValueError('Complete finite source metrics required')
                    for key,value in values.items():vectors.setdefault(key,[]).append(numpy_values(value))
                    outputs.append(numpy_values(iq));F_features.append(numpy_values(embedding));F_logits.append(numpy_values(torch.cat([teacher[None],logits],0)))
                    noises.append(numpy_values(noise))
                values={key:np.concatenate(value) for key,value in vectors.items()}
                if any(value.shape!=(512,) for value in values.values()):raise ValueError('Full source512 required')
                np.savez(args.output/f'{name}-{view}.npz',source_positions=indices,physical_ids=ids[indices],roles=roles[indices],rx=rx,day=day,
                    **labels,clean=clean[indices],received=received[indices],variance=variance[indices],noise=np.concatenate(noises),
                    IQ=np.concatenate(outputs,axis=1),F_features=np.concatenate(F_features,axis=1),F_logits=np.concatenate(F_logits,axis=1),
                    clean_F_features=features[indices],**values)
                masks={'all':np.ones(512,bool),'nonclean':nonclean}
                for key,group in dict(role=roles[indices],rx=rx,day=day,**labels).items():
                    for value in np.unique(group):masks[f'{key}:{value}']=group==value
                summary[name][str(view)]={key:aggregate(values,mask) for key,mask in masks.items()}
                print(json.dumps(dict(event='SOURCE_NOISE_STABILITY_CELL_COMPLETE',model=name,view=view,count=512,seconds=time.perf_counter()-started)),flush=True)
    for name,model in models.items():
        if any(not torch.equal(value,snapshots[name][key]) for key,value in model.state_dict().items()) or any(v.grad is not None for v in model.parameters()):
            raise ValueError('Frozen source G changed')
    if any(not torch.equal(value,f_snapshot[key]) for key,value in reader.state_dict().items()) or any(v.grad is not None for v in reader.parameters()):raise ValueError('Frozen F changed')
    decisions=[]
    for view in ['1','20']:
        first=summary['pureG_E30'][view]['nonclean']['metrics'];second=summary['DURA_default'][view]['nonclean']['metrics']
        for metric in [first,second]:
            if metric['input_noise_nmse']['mean']<=0 or metric['native_nmse']['mean']<=0 or metric['output_drift_nmse']['mean']<=0:raise ValueError('Nondegenerate source comparison required')
        gain1=first['output_drift_nmse']['mean']/first['input_noise_nmse']['mean'];gain2=second['output_drift_nmse']['mean']/second['input_noise_nmse']['mean']
        decisions.append(dict(view=view,primary_noise_gain=gain1,default_noise_gain=gain2,noise_gain_ratio=gain1/gain2,
            primary_MSE_ratio=first['antithetic_nmse']['mean']/first['native_nmse']['mean'],
            default_MSE_ratio=second['antithetic_nmse']['mean']/second['native_nmse']['mean']))
    eligible=qualifies(decisions)
    base.write_json(args.output/'results.json',dict(status='ANALYZED_SOURCE_ONLY',summary=summary,decisions=decisions,training_contrast_eligible=eligible,
        source_rule=RULE,source_records=512,source_L=128,source_U=384,epochs=[1,20],noise_seed=SEED,seconds=time.perf_counter()-started,
        G_inputs=6144,G_smoke_inputs=4,F_inputs=6144,F_smoke_inputs=4,F_teacher=f_guided.teacher_reference(),source_F_cache_metadata=meta,
        source_G=repeat_views.RUN,source_G_commit=SOURCE_COMMIT,default_G=base.PARENT,G_unchanged=True,F_unchanged=True,
        optimizer_updates=0,TX_labels_read=0,V_reads=0,target_reads=0,
        scope='Synthetic training-pool noise sensitivity only;not causal overfitting proof/trueTXaccuracy/targetbenefit/recoverylimit;no new trained G or automatic launch'))
    base.write_json(args.output/'complete.json',dict(status='ANALYZED_SOURCE_ONLY',training_contrast_eligible=eligible,optimizer_updates=0,predictions=0,goal_complete=False))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ['root','output']:parser.add_argument('--'+key,type=Path,required=True)
    parser.add_argument('--commit',required=True);args=parser.parse_args()
    if Path('release_commit.txt').read_text().strip()!=args.commit:raise ValueError('Pinned release required')
    torch.set_num_threads(2);run(args)


if __name__=='__main__':main()
