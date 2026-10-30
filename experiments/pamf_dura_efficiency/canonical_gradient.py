"""Output-gradient falsification of head-protected RX feature guidance."""
import argparse
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch
from torch.nn import functional as F

from experiments.pamf_dura_direct_train import runner as training,no_variance,f_guided
from experiments.pamf_dura_diagnostics.runner import source_indices,tensor,numpy_values
from experiments.pamf_restoration.views import sample_nmse
from .rf_geometry import source_outputs,calibration_decision
from . import rx_geometry

RUN='20261031-diagnostic-DURA-source-F-RXgradient-s2026092701-r01'
SOURCE_COMMIT='cf40edea90667d9645a724b1e8f333027cdf7869'
GEOMETRY_COMMIT='aee80ffb0b1e573ec2267c22362a69db30c4b9ac'
OUTPUTS_RUN='20261030-diagnostic-DURA-source-phase-oracle-s2026092701-m2-r01'
RULE=('Fixed original noVar source512 cached01/20,canonical feature target from qualified RXgeometry '
      'alpha.25/fit62488;no G/F updates,TX,V,target. One lossweight=.25/median32(||gCanonical||/||gIQ||). '
      'Each view:median pressure[.125,.5],meanIQcos>=0,negative<=8/16, '
      'median||gCanonical-gOriginalFcos||/||gOriginalFcos||>=.10. '
      'OutputIQ gradients,not parameter gradients. Fail closes recipe without larger shrinkage/reweighting. '
      'Pass permits separately registered matched scratch E20;not target efficacy.')


def gradients(restored,clean,reader,reference,canonical):
    x=restored.detach().requires_grad_(True);features=reader.features(x)
    iq=sample_nmse(x,clean.detach())
    old=1-F.cosine_similarity(features,reference.detach(),dim=1,eps=1e-8)
    new=1-F.cosine_similarity(features,canonical.detach(),dim=1,eps=1e-8)
    gi=torch.autograd.grad(iq.mean(),x)[0]
    gc=torch.autograd.grad(new.mean(),x,retain_graph=True)[0]
    go=torch.autograd.grad(old.mean(),x)[0]
    if any(not torch.isfinite(v).all() for v in (iq,old,new,gi,gc,go)) or any(g.norm()<=0 for g in (gi,gc,go)):
        raise FloatingPointError('Finite nonzero source IQ/oldF/canonical output gradients required')
    return dict(iq_nmse=iq,original_feature_error=old,canonical_feature_error=new),gi,gc,go


def decision(batches):
    result=calibration_decision(batches);result.pop('analytic_RF_study_eligible')
    for row in result['views']:
        delta=float(np.median([v['gradient_delta_to_original'] for v in batches if v['view']==row['view']]))
        row['median_gradient_delta_to_original']=delta
        row['passed']=bool(row['passed'] and np.isfinite(delta) and delta>=.10)
    result['RX_canonical_training_study_eligible']=all(v['passed'] for v in result['views'])
    return result


def run(args):
    if args.output.name!=RUN or args.output.exists() or os.environ.get('CUDA_VISIBLE_DEVICES')!='':
        raise ValueError('New immutable CPU source-only diagnostic root required')
    no_variance.configure();p=training.paths(args.root)
    done=training.read_json(p['output']/'scratch_joint/training_complete.json')
    expected=dict(status='SOURCE_TRAINED',epoch=20,updates=4940,training_TX_labels_read=0,
                  training_F_access=False,V_reads=0,target_reads=0,initialization_checkpoint_loads=0)
    if any(done.get(k)!=v for k,v in expected.items()):raise ValueError('Fixed original source-final noVar required')
    clean,roles=training.source_training(p);ids=f_guided.source_ids(p,roles);indices=source_indices(roles)
    folder=args.root/'runs'/rx_geometry.RUN
    geometry=training.read_json(folder/'results.json');launch=training.read_json(folder/'launch.json')
    if (not geometry['geometry']['source_canonical_feature_calibration_eligible'] or launch['commit']!=GEOMETRY_COMMIT
        or geometry['canonical_alpha']!=.25 or geometry['F_teacher']!=f_guided.teacher_reference()):
        raise ValueError('Fixed qualified source-only RX geometry required')
    with np.load(folder/'geometry.npz',allow_pickle=False) as a:
        for key,expected in dict(source_positions=indices,physical_ids=ids[indices],roles=roles[indices]).items():
            if not np.array_equal(a[key],expected):raise ValueError('Matching source feature holdout roles/IDs required')
        reference=a['probe_features'];canonical=a['canonical_features'];weight=a['normalized_head_weight']
    if reference.shape!=(512,160) or canonical.shape!=reference.shape or not np.isfinite(canonical).all():
        raise ValueError('Complete original/canonical512 source targets required')
    outputs={v:source_outputs(args.root/'runs'/OUTPUTS_RUN/f'noVar-{v}.npz',indices,ids,roles,clean) for v in (1,20)}
    from experiments.pamf_fcrossrx_readout.frozen import load_reader
    reader=load_reader(p['reader'],p['source_cache'],'cpu').eval().requires_grad_(False)
    before={k:v.clone() for k,v in reader.state_dict().items()}
    actual=rx_geometry.unit(np.asarray(reader.core.id_backbone.cls_head.weight.detach().double().tolist()))
    if not np.allclose(actual,weight,atol=1e-12,rtol=0):raise ValueError('Same original F head required')
    model=training.load_model(p['output']/'scratch_joint/final.pt','cpu',arm='scratch_joint',commit=SOURCE_COMMIT).eval().requires_grad_(False)
    old_state={k:v.clone() for k,v in model.state_dict().items()}
    args.output.mkdir()
    training.write_json(args.output/'launch.json',dict(pid=os.getpid(),cwd=os.getcwd(),argv=sys.argv,commit=args.commit,
        CUDA_VISIBLE_DEVICES=os.environ['CUDA_VISIBLE_DEVICES'],CPU_threads=2,GPU_workers=0))
    with torch.no_grad():smoke=reader(model(tensor(clean[indices[:2]],'cpu')))
    if smoke.shape!=(2,6) or not torch.isfinite(smoke).all():raise ValueError('Real source2Ls G/F checkpoint smoke failed')
    if any(not torch.equal(v,old_state[k]) for k,v in model.state_dict().items()):raise ValueError('Frozen G smoke changed state')
    del model,old_state,smoke
    print(json.dumps(dict(event='REAL_SOURCE_G_F_CHECKPOINT_SMOKE_PASS',Ls=2,target_reads=0)),flush=True)
    batches=[];vectors={key:[] for key in ('g_iq','g_canonical','g_original','iq_nmse','original_feature_error','canonical_feature_error','view','source_positions')}
    started=time.perf_counter()
    for view in (1,20):
        for offset in range(0,512,32):
            take=slice(offset,offset+32);x=tensor(clean[indices[take]],'cpu');r=tensor(outputs[view][take],'cpu')
            terms,gi,gc,go=gradients(r,x,reader,tensor(reference[take],'cpu'),tensor(canonical[take],'cpu'))
            ni,nc,no=float(gi.norm()),float(gc.norm()),float(go.norm())
            row=dict(view=view,offset=offset,batch=32,ratio=nc/ni,cosine=float((gi*gc).sum()/(gi.norm()*gc.norm())),
                iq_gradient_norm=ni,canonical_gradient_norm=nc,original_gradient_norm=no,
                gradient_delta_to_original=float((gc-go).norm()/go.norm()),
                canonical_original_gradient_cosine=float((gc*go).sum()/(gc.norm()*go.norm())),
                losses={k:float(v.detach().mean()) for k,v in terms.items()})
            batches.append(row)
            for key,value in dict(g_iq=gi,g_canonical=gc,g_original=go,**terms).items():vectors[key].append(numpy_values(value.detach()))
            vectors['view'].append(np.full(32,view,dtype=np.int64));vectors['source_positions'].append(indices[take])
            print(json.dumps(dict(event='SOURCE_RX_CANONICAL_GRADIENT_BATCH',**row)),flush=True)
    if any(not torch.equal(v,before[k]) for k,v in reader.state_dict().items()) or any(p.grad is not None for p in reader.parameters()):
        raise ValueError('Frozen F state or gradients changed')
    verdict=decision(batches)
    np.savez(args.output/'gradients.npz',**{k:np.concatenate(v) for k,v in vectors.items()})
    training.write_json(args.output/'results.json',dict(status='ANALYZED_SOURCE_ONLY',source_rule=RULE,decision=verdict,batches=batches,
        source_records=512,source_visits=1024,L_s=128,U_s=384,views=[1,20],feature_RX_shrinkage=.25,
        geometry_run=rx_geometry.RUN,geometry_commit=GEOMETRY_COMMIT,F_teacher=f_guided.teacher_reference(),
        G_smoke_inputs=2,F_smoke_inputs=2,G_analysis_inputs=0,F_feature_analysis_inputs=1024,G_unchanged=True,F_unchanged=True,
        optimizer_updates=0,TX_labels_read=0,hard_pseudo_TX_labels=False,V_reads=0,target_reads=0,seconds=time.perf_counter()-started,
        scope='Source outputIQ-gradient feasibility/difference from oldFcos;not Gparameter/identity/target efficacy;F-guided'))
    training.write_json(args.output/'complete.json',dict(status='ANALYZED_SOURCE_ONLY',G_unchanged=True,F_unchanged=True,
        optimizer_updates=0,target_reads=0,goal_complete=False,**verdict))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for k in ('root','output'):parser.add_argument('--'+k,type=Path,required=True)
    parser.add_argument('--commit',required=True);args=parser.parse_args()
    if Path('release_commit.txt').read_text().strip()!=args.commit:raise ValueError('Pinned release required')
    torch.set_num_threads(2);run(args)


if __name__=='__main__':main()
