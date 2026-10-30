"""Source-only phase-error decomposition of fixed restorers, never deployment."""
import argparse
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch

from experiments.pamf_dura_direct_train import runner as training, no_variance, f_guided
from experiments.pamf_dura_diagnostics.runner import source_indices, tensor, numpy_values
from experiments.pamf_restoration.models import complex_iq, pair
from .phase_sensitivity import metrics

RUN='20261030-diagnostic-DURA-source-phase-oracle-s2026092701-m2-r01'
SOURCE_COMMIT='cf40edea90667d9645a724b1e8f333027cdf7869'
RULE=('Only fixed noVar control may qualify a separately registered scratch phase-weight study: '
      'BOTH cached01/20 need>=480/512 defined phases,>=10percent totalIQerror removable by phase-only oracle, '
      'clean-F choice gain>=6/512 and no scene loses choices;cleaninput control choices must not fall. '
      'DefaultDURA is descriptive control;oracle consumes sourceclean and is never deployable or an accuracy bound. '
      'If qualified, one new E20 study may double only the exact phase component of IQMSE,not ignore phase.')


def phase_component(restored, clean):
    cross=(complex_iq(restored)*complex_iq(clean.detach()).conj()).mean(1)
    energy=clean.detach().square().mean((1,2)).clamp_min(1e-4)
    return ((cross.abs()-cross.real)/energy).clamp_min(0)


@torch.no_grad()
def align_phase(restored, clean):
    z=complex_iq(restored);x=complex_iq(clean)
    cross=(z*x.conj()).mean(1);ex=x.abs().square().mean(1);er=z.abs().square().mean(1)
    scale=(ex*er).sqrt()
    valid=(ex>1e-8)&(er>1e-8)&(cross.abs()>1e-8*scale)
    # If cross=0 every unit rotation is optimal; identity is a reported
    # neutral representative and valid=False, not an estimated phase.
    unit=torch.where(valid,cross.conj()/cross.abs().clamp_min(1e-20),torch.ones_like(cross))
    aligned=pair(z*unit[:,None]);energy=clean.square().mean((1,2)).clamp_min(1e-4)
    native=(restored-clean).square().mean((1,2))/energy
    corrected=(aligned-clean).square().mean((1,2))/energy
    return aligned,dict(native_nmse=native,aligned_nmse=corrected,phase_nmse_drop=phase_component(restored,clean),
        valid_phase=valid.to(restored.dtype),coherence=(cross.abs()/scale.clamp_min(1e-20)).clamp(0,1),
        absolute_phase_degrees=torch.where(valid,torch.rad2deg(cross.angle()).abs(),torch.zeros_like(ex)))


def phase_decision(summary):
    clean=summary['clean']['all']
    clean_ok=clean['aligned']['teacher_correct']>=clean['native']['teacher_correct']
    rows=[]
    for view in ('1','20'):
        groups=summary[view];overall=groups['all']
        scenes=[v for key,v in groups.items() if key.startswith('scene:')]
        if overall['count']!=512 or len(scenes)!=7:raise ValueError('Full512/allseven source scene groups required')
        passed=(overall['valid_phases']>=480 and overall['phase_fraction']>=.10
            and overall['aligned']['teacher_correct']-overall['native']['teacher_correct']>=6
            and all(g['aligned']['teacher_correct']>=g['native']['teacher_correct'] for g in scenes))
        rows.append(dict(cached_view=view,passed=bool(passed)))
    return dict(clean_control_preserved=bool(clean_ok),views=rows,phase_weight_study_eligible=bool(clean_ok and all(r['passed'] for r in rows)))


def aggregate(v, mask):
    cases={}
    for case in ('native','aligned'):
        current={key.split('/',1)[1]:dict(mean=float(value[mask].mean(dtype=np.float64)),
                    median=float(np.median(value[mask])),p95=float(np.quantile(value[mask],.95)))
                 for key,value in v.items() if key.startswith(case+'/')}
        cases[case]=dict(metrics=current,teacher_correct=int(v[case+'/teacher_agreement'][mask].sum()))
    native_sum=float(v['native/iq_nmse'][mask].sum(dtype=np.float64))
    return dict(count=int(mask.sum()),valid_phases=int(v['phase/valid_phase'][mask].sum()),
        phase_fraction=float(v['phase/phase_nmse_drop'][mask].sum(dtype=np.float64))/native_sum if native_sum>1e-12 else 0.,
        phase_metrics={key.split('/',1)[1]:dict(mean=float(value[mask].mean(dtype=np.float64)),
            median=float(np.median(value[mask])),p95=float(np.quantile(value[mask],.95)))
            for key,value in v.items() if key.startswith('phase/')},**cases)


@torch.no_grad()
def run(args):
    if args.output.name!=RUN or args.output.exists() or os.environ.get('CUDA_VISIBLE_DEVICES')!='':
        raise ValueError('New immutable CPU source-only diagnostic root required')
    no_variance.configure();p=training.paths(args.root)
    done=training.read_json(p['output']/'scratch_joint/training_complete.json')
    if any(done.get(k)!=v for k,v in dict(status='SOURCE_TRAINED',epoch=20,updates=4940,training_TX_labels_read=0,
            training_F_access=False,V_reads=0,target_reads=0,initialization_checkpoint_loads=0).items()):
        raise ValueError('Fixed source-final noVar required')
    novar=training.load_model(p['output']/'scratch_joint/final.pt','cpu',arm='scratch_joint',commit=SOURCE_COMMIT)
    default,_=training.parent_model(p,'cpu')
    models={'noVar':novar.eval().requires_grad_(False),'DURA_default':default.eval().requires_grad_(False)}
    clean,roles=training.source_training(p);ids=f_guided.source_ids(p,roles);indices=source_indices(roles)
    rx=np.load(p['cache']/'train/rx.npy',allow_pickle=False)[indices]
    day=np.load(p['cache']/'train/day.npy',allow_pickle=False)[indices]
    if set(rx.tolist())!={1,3,4,6,8} or set(day.tolist())!={1,2,3}:raise ValueError('Original source-only RX/day cohort required')
    from experiments.pamf_fcrossrx_readout.frozen import load_reader
    reader=load_reader(p['reader'],p['source_cache'],'cpu').eval().requires_grad_(False)
    f_before={k:v.clone() for k,v in reader.state_dict().items()}
    snapshots={name:{k:v.clone() for k,v in model.state_dict().items()} for name,model in models.items()}
    args.output.mkdir()
    training.write_json(args.output/'launch.json',dict(pid=os.getpid(),cwd=os.getcwd(),argv=sys.argv,commit=args.commit,
        CUDA_VISIBLE_DEVICES=os.environ['CUDA_VISIBLE_DEVICES'],CPU_threads=2,GPU_workers=0,F_teacher=f_guided.teacher_reference()))
    for name,model in models.items():
        smoke=reader(model(tensor(clean[indices[:2]],'cpu')))
        if smoke.shape!=(2,6) or not torch.isfinite(smoke).all():raise ValueError('Actual source2Ls G/F checkpoint smoke failed')
        print(json.dumps(dict(event='REAL_SOURCE_G_F_CHECKPOINT_SMOKE_PASS',model=name,Ls=2,target_reads=0)),flush=True)
    started=time.perf_counter();rf=[]
    for offset in range(0,512,32):rf.append(reader.features(tensor(clean[indices[offset:offset+32]],'cpu')).clone())
    reference_features=torch.cat(rf);reference_logits=reader.classify_features(reference_features)
    summary={};all_unchanged=True;power_error=0.
    for name,model in models.items():
        before=snapshots[name]
        cells={}
        for view in ('clean','1','20'):
            if view=='clean':
                received=clean
                labels=dict(scene=np.full(512,'clean'),state=np.full(512,'clean'),lock=np.full(512,'not_applicable'))
            else:
                cache=p['cache']/'epochs'/f'{int(view):02d}'
                received=np.load(cache/'received.npy',mmap_mode='r',allow_pickle=False)
                with np.load(cache/'groups.npz',allow_pickle=False) as groups:labels={k:groups[k][indices] for k in ('scene','state','lock')}
                if received.shape!=(63000,2,256) or received.dtype!=np.float32 or set(labels['scene'].tolist())!=set(training.VIEWS):
                    raise ValueError('Matching full source corruption and allseven scenes required')
            vectors={};iq_x=[];iq_r=[];iq_a=[];features=[];logits=[]
            for offset in range(0,512,32):
                take=indices[offset:offset+32];x=tensor(clean[take],'cpu');r=model(tensor(received[take],'cpu'))
                a,pm=align_phase(r,x)
                local={f'phase/{k}':v for k,v in pm.items() if k not in ('native_nmse','aligned_nmse')}
                ff=[reference_features[offset:offset+32]];zz=[reference_logits[offset:offset+32]]
                for case,value,error in (('native',r,pm['native_nmse']),('aligned',a,pm['aligned_nmse'])):
                    f=reader.features(value);z=reader.classify_features(f);ff.append(f);zz.append(z)
                    local[case+'/iq_nmse']=error
                    local.update({case+'/'+k:v for k,v in metrics(f,z,ff[0],zz[0]).items()})
                power_error=max(power_error,float(((a.square().sum(1)-r.square().sum(1)).abs().mean(1)
                                    /r.square().sum(1).mean(1).clamp_min(1e-8)).max()))
                if not torch.allclose(pm['native_nmse']-pm['aligned_nmse'],pm['phase_nmse_drop'],atol=2e-6,rtol=2e-5):
                    raise ValueError('Source phase-only MSE decomposition mismatch')
                for key,v in local.items():
                    if v.shape!=(len(x),) or not torch.isfinite(v).all():raise FloatingPointError('Nonfinite phase oracle metrics')
                    vectors.setdefault(key,[]).append(numpy_values(v))
                iq_x.append(numpy_values(x));iq_r.append(numpy_values(r));iq_a.append(numpy_values(a))
                features.append(numpy_values(torch.stack(ff)));logits.append(numpy_values(torch.stack(zz)))
                print(json.dumps(dict(event='SOURCE_PHASE_ORACLE_BATCH',model=name,view=view,count=offset+32)),flush=True)
            values={k:np.concatenate(v) for k,v in vectors.items()}
            np.savez(args.output/f'{name}-{view}.npz',source_positions=indices,physical_ids=ids[indices],roles=roles[indices],rx=rx,day=day,
                **labels,clean=np.concatenate(iq_x),native=np.concatenate(iq_r),aligned=np.concatenate(iq_a),
                F_features=np.concatenate(features,axis=1),F_logits=np.concatenate(logits,axis=1),**values)
            masks={'all':np.ones(512,dtype=bool)}
            for key,value in dict(role=roles[indices],rx=rx,day=day,**labels).items():
                for label in np.unique(value):masks[f'{key}:{label}']=value==label
            cells[view]={key:aggregate(values,mask) for key,mask in masks.items()}
        unchanged=all(torch.equal(v,before[k]) for k,v in model.state_dict().items()) and all(p.grad is None for p in model.parameters())
        if not unchanged:raise ValueError('Frozen G state or gradients changed')
        all_unchanged=all_unchanged and unchanged;summary[name]=cells
    if power_error>2e-6 or any(not torch.equal(v,f_before[k]) for k,v in reader.state_dict().items()) or any(p.grad is not None for p in reader.parameters()):
        raise ValueError('Phase-only energy or frozen F state changed')
    decision=phase_decision(summary['noVar'])
    training.write_json(args.output/'results.json',dict(status='ANALYZED_SOURCE_ONLY',summary=summary,decision=decision,source_rule=RULE,
        source_records=512,L_s=128,U_s=384,views=['clean',1,20],models=['noVar','DURA_default'],F_teacher=f_guided.teacher_reference(),
        G_analysis_inputs=3072,G_smoke_inputs=4,F_feature_analysis_inputs=6656,F_smoke_inputs=4,G_unchanged=all_unchanged,F_unchanged=True,
        optimizer_updates=0,TX_labels_read=0,V_reads=0,target_reads=0,power_error_max=power_error,seconds=time.perf_counter()-started,
        scope='Sourceclean phase-only oracle,never deployed or accuracy upperbound;F-informed research selection;not currenttarget efficacy'))
    training.write_json(args.output/'complete.json',dict(status='ANALYZED_SOURCE_ONLY',G_unchanged=True,F_unchanged=True,
        optimizer_updates=0,target_reads=0,goal_complete=False,**decision))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for k in ('root','output'):parser.add_argument('--'+k,type=Path,required=True)
    parser.add_argument('--commit',required=True);args=parser.parse_args()
    if Path('release_commit.txt').read_text().strip()!=args.commit:raise ValueError('Pinned release required')
    torch.set_num_threads(2);run(args)


if __name__=='__main__':main()
