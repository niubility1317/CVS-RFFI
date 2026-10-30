"""Fixed source-only R17 IQ paths; no student training or target selection."""
import argparse
import json
import os
from pathlib import Path
import sys
import time
import numpy as np
import torch
from torch.nn import functional as F
from experiments.pamf_dura_direct_train import runner as base,f_guided
from experiments.pamf_dura_diagnostics.runner import source_indices,tensor,numpy_values

RUN='20261031-diagnostic-DURA-R17-source-IQpaths-s2026092701-r01'
TEACHER_RUN='20261020-phase1-restoration-unrolled-selfsup-s2026092701-m2-r17'
TEACHER_COMMIT='cd97744884b5e460926e48c8b7efb7d78c6da0f4'
REFERENCE='20261030-diagnostic-DURA-source-phase-oracle-s2026092701-m2-r01'
PATHS=('coarse','stage1','stage2','native')


def validate_teacher(value,done,launch,decision):
    required=dict(schema='unrolled_prior_checkpoint_v1',candidate='unroll_unshared',epoch=20,updates=4940,
        prior_initialization='scratch',inherited_prior_epochs=0,training_TX_labels_read=0,training_F_access=False,target_training=False)
    if any(value.get(k)!=v for k,v in required.items()):raise ValueError('Original source-final R17 teacher required')
    expected=dict(coarse_run='20261019-phase1-reception-restoration-selfsup-manysig-m6-r02',
        coarse_commit='7efdba92932a86bf4b0492a565d35377564a26da',
        action_run=base.ACTION,action_commit='c8ccdfdbd03ab5ad8828b1eb9210b99a664ebe7b',
        data_contract='original7/63/30',training_TX_F_reads=0,target_training=False)
    if any(value.get('parent',{}).get(k)!=v for k,v in expected.items()):raise ValueError('R02/R07 original source ancestry required')
    required_done=dict(status='SOURCE_TRAINED',candidate='unroll_unshared',epochs=20,updates=4940,
        L_exposures=126000,U_exposures=1134000,TX_labels_read=0,F_outputs_read=0,F_weights_read=0,
        target_reads=0,frozen_context_updates=0,parameters=4665073,prior_initialization='scratch',inherited_prior_epochs=0)
    if any(done.get(k)!=v for k,v in required_done.items()) or launch.get('commit')!=TEACHER_COMMIT:
        raise ValueError('R17 complete budget and source-only commit required')
    if any(decision.get(k)!=v for k,v in dict(status='SOURCE_DECISION_FROZEN',selected='unroll_unshared',
        target_reads_before_decision=0,training_TX_F_access=False).items()):
        raise ValueError('Original before-target source choice required')


def measures(iq,clean,features,logits,reference_features,reference_logits):
    energy=clean.square().mean((1,2)).clamp_min(1e-4)
    return dict(iq_NMSE=(iq-clean).square().mean((1,2))/energy,
        feature_cosine_error=1-F.cosine_similarity(features,reference_features,dim=1,eps=1e-8),
        teacher_choice_agreement=(logits.argmax(1)==reference_logits.argmax(1)).to(torch.float32))


def aggregate(values,mask):
    return dict(count=int(mask.sum()),teacher_correct=int(values['teacher_choice_agreement'][mask].sum()),
        metrics={k:dict(mean=float(v[mask].mean(dtype=np.float64)),median=float(np.median(v[mask])),
            p95=float(np.quantile(v[mask],.95))) for k,v in values.items()})


@torch.no_grad()
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True);parser.add_argument('--commit',required=True);args=parser.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('Source diagnostic must remain CPU-only')
    if Path('release_commit.txt').read_text().strip()!=args.commit:raise ValueError('Pinned release required')
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    p=base.paths(args.root);clean,roles=base.source_training(p);ids=f_guided.source_ids(p,roles);take=source_indices(roles)
    teacher_root=args.root/'runs'/TEACHER_RUN;row=teacher_root/'unroll_unshared'
    # Preserve the original loader, including complete source/config validation.
    from experiments.pamf_unrolled_restore.runtime import load_checkpoint
    teacher,value=load_checkpoint(row/'native_map_prior/final.pt','cpu')
    done=base.read_json(row/'training_complete.json');launch=base.read_json(teacher_root/'launch.json')
    decision=base.read_json(teacher_root/'source_decision.json');validate_teacher(value,done,launch,decision)
    teacher.eval().requires_grad_(False)
    from experiments.pamf_fcrossrx_readout.frozen import load_reader
    reader=load_reader(p['reader'],p['source_cache'],'cpu').eval().requires_grad_(False)
    g_before={k:v.clone() for k,v in teacher.state_dict().items()};f_before={k:v.clone() for k,v in reader.state_dict().items()}
    output=args.root/'runs'/RUN;output.mkdir()
    base.write_json(output/'launch.json',dict(pid=os.getpid(),cwd=os.getcwd(),argv=sys.argv,commit=args.commit,
        CUDA_VISIBLE_DEVICES='',CPU_threads=2,teacher_checkpoint=str(row/'native_map_prior/final.pt'),teacher_commit=TEACHER_COMMIT))
    smoke=reader(teacher(tensor(clean[np.flatnonzero(roles=='L_s')[:2]],'cpu')))
    if smoke.shape!=(2,6) or not torch.isfinite(smoke).all():raise ValueError('Real R17/F original-head source2Ls smoke failed')
    print(json.dumps(dict(event='REAL_SOURCE_R17_F_NOQUERY_SMOKE_PASS',Ls=2)),flush=True)
    started=time.perf_counter();summary={}
    rx=np.load(p['cache']/'train/rx.npy',allow_pickle=False)[take];day=np.load(p['cache']/'train/day.npy',allow_pickle=False)[take]
    previous=args.root/'runs'/REFERENCE;meta=base.read_json(previous/'results.json')
    if meta['F_teacher']!=f_guided.teacher_reference() or not meta['F_unchanged'] or not meta['G_unchanged']:
        raise ValueError('Complete matching frozen source baseline required')
    for epoch in (1,20):
        cache=p['cache']/'epochs'/f'{epoch:02d}'
        received=np.load(cache/'received.npy',mmap_mode='r',allow_pickle=False)
        with np.load(cache/'groups.npz',allow_pickle=False) as archive:groups={k:archive[k][take] for k in ('scene','state','lock')}
        with np.load(previous/f'noVar-{epoch}.npz',allow_pickle=False) as reference:
            if not np.array_equal(reference['source_positions'],take) or not np.array_equal(reference['physical_ids'],ids[take]) or not np.array_equal(reference['scene'],groups['scene']):
                raise ValueError('Exact old source physical pairing required')
            np.testing.assert_array_equal(reference['clean'],clean[take])
            reference_features=tensor(reference['F_features'][0],'cpu');reference_logits=tensor(reference['F_logits'][0],'cpu')
        iq={k:[] for k in PATHS};features={k:[] for k in PATHS};logits={k:[] for k in PATHS};metrics={k:{} for k in PATHS}
        for offset in range(0,512,32):
            selected=take[offset:offset+32];x=tensor(clean[selected],'cpu')
            stages,variance,action,coarse=teacher.stage_outputs(tensor(received[selected],'cpu'))
            if len(stages)!=3:raise ValueError('Original three R17 stages required')
            paths=dict(coarse=coarse,stage1=stages[0],stage2=stages[1],native=stages[2])
            for name,signal in paths.items():
                f=reader.features(signal);z=reader.classify_features(f)
                current=measures(signal,x,f,z,reference_features[offset:offset+32],reference_logits[offset:offset+32])
                if any(v.shape!=(len(selected),) or not torch.isfinite(v).all() for v in current.values()):raise FloatingPointError('Nonfinite teacher path')
                iq[name].append(numpy_values(signal));features[name].append(numpy_values(f));logits[name].append(numpy_values(z))
                for key,v in current.items():metrics[name].setdefault(key,[]).append(numpy_values(v))
            print(json.dumps(dict(event='R17_SOURCE_PATH_BATCH',epoch=epoch,count=offset+32)),flush=True)
        masks={'all':np.ones(512,dtype=bool)}
        for name,labels in dict(role=roles[take],rx=rx,day=day,**groups).items():
            for label in np.unique(labels):masks[name+':'+str(label)]=labels==label
        cell={}
        for name in PATHS:
            values={k:np.concatenate(v) for k,v in metrics[name].items()}
            cell[name]={group:aggregate(values,mask) for group,mask in masks.items()}
        summary[str(epoch)]=cell
        np.savez(output/f'epoch{epoch:02d}.npz',source_positions=take,physical_ids=ids[take],roles=roles[take],rx=rx,day=day,
            **groups,clean=clean[take],reference_features=numpy_values(reference_features),reference_logits=numpy_values(reference_logits),
            **{'IQ_'+k:np.concatenate(v) for k,v in iq.items()},
            **{'F_features_'+k:np.concatenate(v) for k,v in features.items()},
            **{'F_logits_'+k:np.concatenate(v) for k,v in logits.items()})
    if any(not torch.equal(v,g_before[k]) for k,v in teacher.state_dict().items()) or any(v.grad is not None for v in teacher.parameters()):
        raise ValueError('Frozen R17 changed')
    if any(not torch.equal(v,f_before[k]) for k,v in reader.state_dict().items()) or any(v.grad is not None for v in reader.parameters()):
        raise ValueError('Frozen F changed')
    base.write_json(output/'results.json',dict(status='ANALYZED_SOURCE_ONLY',summary=summary,source_probe=512,source_L=128,source_U=384,
        source_views=[1,20],teacher_run=TEACHER_RUN,teacher_commit=TEACHER_COMMIT,teacher_parameters=4665073,
        teacher_prior_epochs=20,teacher_coarse_epochs=20,teacher_action_epochs=20,teacher_total_component_epochs=60,
        teacher_source_selection_before_target=True,teacher_source_selection_used_source_V_labels=True,
        F_teacher=f_guided.teacher_reference(),G_analysis_inputs=1024,G_smoke_inputs=2,F_analysis_inputs=4096,F_smoke_inputs=2,
        G_unchanged=True,F_unchanged=True,optimizer_updates=0,TX_labels_read=0,V_reads=0,target_reads=0,
        seconds=time.perf_counter()-started,scope='Source training-pool paths;Fcleanchoices notTXaccuracy;no intermediate IQ is presumed physicallyclean or suitable asstudenthint;no selection of target-based teacher or automatic training'))
    print(json.dumps(dict(event='R17_SOURCE_PATHS_COMPLETE',seconds=time.perf_counter()-started)),flush=True)


if __name__=='__main__':main()
