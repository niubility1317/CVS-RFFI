"""Source-only geometry and deterministic calibration of head-visible guidance."""
import argparse
import json
import os
from pathlib import Path
import statistics
import sys
import time

import numpy as np
import torch
from torch.nn import functional as F

from experiments.pamf_dura_direct_train import runner as training
from experiments.pamf_dura_direct_train import no_variance,f_guided
from experiments.pamf_dura_diagnostics.runner import source_indices,tensor
from experiments.pamf_dura_gradient.runner import geometry
from .model import direct_objective,supervised_action
from .head_features import head_feature_basis,projected_feature_error
from .task_geometry import decision_metrics

RUN='20261030-diagnostic-DURA-head-feature-geometry-s2026092701-r01'
SOURCE_COMMIT='cf40edea90667d9645a724b1e8f333027cdf7869'
RULE=('Calibrate alpha=.25/median(all32 unweighted projected/base gradient norm ratios); '
      'new scratch E20 candidate eligible only if both source views: calibrated median ratio [.125,.5], '
      'base/projected mean cosine>=0 and at most8/16 negative, projected/task mean cosine>cosine/task mean cosine and>0, '
      'teacher agreement<1; all states unchanged/no truth or target; no coefficient grid or target feedback')


def summarize(records):
    ratios=[r['geometry']['all']['base_projected']['right_to_left_norm'] for r in records]
    if len(records)!=32 or any(v is None or not 0<v< float('inf') for v in ratios):
        raise ValueError('Complete finite nonzero projected source gradients required')
    alpha=.25/statistics.median(ratios)
    cells=[]
    for epoch in (1,20):
        items=[r for r in records if r['epoch']==epoch];assert len(items)==16
        stats=[r['geometry']['all'] for r in items]
        cosine=[r['base_projected']['cosine'] for r in stats]
        projected_task=[r['task_projected']['cosine'] for r in stats]
        cosine_task=[r['task_cosine']['cosine'] for r in stats]
        if any(v is None for values in (cosine,projected_task,cosine_task) for v in values):
            raise ValueError('Defined source gradient alignment required')
        ratio=alpha*statistics.median(r['base_projected']['right_to_left_norm'] for r in stats)
        metrics={key:statistics.mean(r['metrics'][key] for r in items) for key in items[0]['metrics']}
        cell=dict(epoch=epoch,calibrated_median_ratio=ratio,base_projected_mean_cosine=statistics.mean(cosine),
            negative_base_projected=sum(v<0 for v in cosine),projected_task_mean_cosine=statistics.mean(projected_task),
            isotropic_task_mean_cosine=statistics.mean(cosine_task),metrics=metrics)
        cell['passed']=(.125<=ratio<=.5 and cell['base_projected_mean_cosine']>=0 and cell['negative_base_projected']<=8
            and cell['projected_task_mean_cosine']>max(0.,cell['isotropic_task_mean_cosine']) and metrics['teacher_argmax_agreement']<1)
        cells.append(cell)
    return dict(alpha=alpha,gradient_pressure=.25,cells=cells,candidate_eligible=all(v['passed'] for v in cells))


def run(args):
    if args.output.name!=RUN or args.output.exists() or os.environ.get('CUDA_VISIBLE_DEVICES')!='':
        raise ValueError('New immutable source-only CPU diagnostic root required')
    no_variance.configure();p=training.paths(args.root)
    done=training.read_json(p['output']/'scratch_joint/training_complete.json')
    expected=dict(status='SOURCE_TRAINED',epoch=20,updates=4940,training_TX_labels_read=0,
        training_F_access=False,V_reads=0,target_reads=0,initialization_checkpoint_loads=0)
    if any(done.get(k)!=v for k,v in expected.items()):raise ValueError('Original matched noVar source-final required')
    model=training.load_model(p['output']/'scratch_joint/final.pt','cpu',arm='scratch_joint',commit=SOURCE_COMMIT)
    model.eval().requires_grad_(True)
    clean,roles=training.source_training(p);ids=f_guided.source_ids(p,roles);indices=source_indices(roles)
    cache=args.root/'runs'/f_guided.RUN/'scratch_joint/source_feature_cache';meta=training.read_json(cache/'complete.json')
    if (meta['status'],meta['count'],meta['dimension'],meta['F_teacher'],meta['TX_labels_read'],meta['target_reads'])!=(
            'SOURCE_F_FEATURES_COMPLETE',63000,160,f_guided.teacher_reference(),0,0):raise ValueError('Frozen legal source F cache required')
    if not np.array_equal(np.load(cache/'physical_ids.npy',allow_pickle=False),ids):raise ValueError('Source F feature order differs')
    targets=np.load(cache/'features.npy',mmap_mode='r',allow_pickle=False)
    if targets.shape!=(63000,160) or targets.dtype!=np.float32:raise ValueError('Original FP32 source F features required')
    from experiments.pamf_fcrossrx_readout.frozen import load_reader
    reader=load_reader(p['reader'],p['source_cache'],'cpu').eval().requires_grad_(False)
    before={key:value.detach().clone() for key,value in model.state_dict().items()}
    reader_before={key:value.detach().clone() for key,value in reader.state_dict().items()}
    basis=head_feature_basis(reader).float()
    args.output.mkdir()
    training.write_json(args.output/'launch.json',dict(pid=os.getpid(),cwd=os.getcwd(),argv=sys.argv,commit=args.commit,CPU_threads=2,GPU_workers=0))
    with torch.no_grad():
        scores=reader(model(tensor(clean[indices[:2]],'cpu')))
        if scores.shape!=(2,6) or not torch.isfinite(scores).all():raise ValueError('Actual source2Ls G/F smoke failed')
    print(json.dumps(dict(event='SOURCE_G_F_CHECKPOINT_SMOKE_PASS',real_Ls=2,TX_labels_read=0,target_reads=0)),flush=True)
    names,parameters=zip(*model.named_parameters());groups={'all':list(range(len(names)))}
    for prefix in ('frozen.coarse.','frozen.estimator.','prior.'):
        groups[prefix]=[i for i,key in enumerate(names) if key.startswith(prefix)]
    records=[];started=time.perf_counter();generator=torch.Generator().manual_seed(2026103005)
    for epoch in (1,20):
        folder=p['cache']/'epochs'/f'{epoch:02d}'
        arrays={key:np.load(folder/(key+'.npy'),mmap_mode='r',allow_pickle=False) for key in ('received','kernel','coefficients','variance')}
        for offset in range(0,512,32):
            take=indices[offset:offset+32];reference=tensor(targets[take],'cpu')
            values={key:tensor(value[take],'cpu') for key,value in arrays.items()}
            probes=[torch.complex(torch.randn(32,256,generator=generator),torch.randn(32,256,generator=generator)) for _ in range(2)]
            _,terms,restored=direct_objective(model,values['received'],tensor(clean[take],'cpu'),
                supervised_action(values['kernel'],values['coefficients']),values['variance'],probes,return_restored=True)
            features=reader.features(restored)
            isotropic=1-F.cosine_similarity(features,reference,dim=1,eps=1e-8)
            projected=projected_feature_error(features,reference,basis)
            actual=reader.classify_features(features)
            with torch.no_grad():
                teacher=reader.classify_features(reference)
                head=reader.core.id_backbone.cls_head
                formula=head.scale*F.linear(F.normalize(reference,dim=1,eps=1e-4),F.normalize(head.weight,dim=1,eps=1e-4))
                torch.testing.assert_close(teacher,formula,atol=1e-6,rtol=1e-6)
            residual=(actual-actual.mean(1,keepdim=True))-(teacher-teacher.mean(1,keepdim=True))
            task=residual.square().mean(1)/(head.scale**2)
            base=terms['iq_nmse']+.1*terms['prior_nmse']+.05*terms['spectral_error']+.05*terms['action_probe_nmse']
            vectors={}
            for key,loss in (('base',base),('cosine',.1*isotropic),('projected',projected),('task',task)):
                gradient=torch.autograd.grad(loss.mean(),parameters,retain_graph=True,allow_unused=True)
                vectors[key]=[torch.zeros_like(p) if g is None else g.detach() for p,g in zip(parameters,gradient)]
                if any(not torch.isfinite(v).all() for v in vectors[key]):raise FloatingPointError('Nonfinite source head-feature gradient')
            stats={}
            for group,selected in groups.items():
                stats[group]={a+'_'+b:geometry([vectors[a][i] for i in selected],[vectors[b][i] for i in selected])
                    for a,b in (('base','projected'),('base','cosine'),('task','projected'),('task','cosine'))}
            with torch.no_grad():decisions=decision_metrics(actual.detach(),teacher)
            metrics=dict(iq_nmse=terms['iq_nmse'],isotropic_feature_error=isotropic,head_projected_feature_error=projected,
                centered_head_error=task,**decisions)
            if any(not torch.isfinite(v).all() for v in metrics.values()):raise FloatingPointError('Nonfinite source metric')
            records.append(dict(epoch=epoch,offset=offset,count=32,L_s=int((roles[take]=='L_s').sum()),U_s=int((roles[take]=='U_s').sum()),
                source_positions=take.tolist(),metrics={key:float(v.detach().mean()) for key,v in metrics.items()},geometry=stats))
            print(json.dumps(dict(event='SOURCE_HEAD_GEOMETRY_BATCH',epoch=epoch,count=offset+32,seconds=time.perf_counter()-started)),flush=True)
    for item,snapshot in ((model,before),(reader,reader_before)):
        if any(not torch.equal(v,snapshot[k]) for k,v in item.state_dict().items()) or any(v.grad is not None for v in item.parameters()):
            raise ValueError('Source diagnostic changed model or accumulated gradient')
    summary=summarize(records)
    training.write_json(args.output/'results.json',dict(status='ANALYZED_SOURCE_ONLY',records=records,summary=summary,
        source_records=512,source_L=128,source_U=384,epochs=[1,20],seconds=time.perf_counter()-started,
        basis_rank=basis.shape[1],basis_from='SVD rowspace of centered normalized frozen original six by160 default head;no fitting',
        model_unchanged=True,F_unchanged=True,optimizer_updates=0,TX_labels_read=0,V_reads=0,target_reads=0,source_rule=RULE,
        scope='Source teacher geometry/calibration only;not TX accuracy or formal generalization;proposed head-space direction predates Fguide target result access'))
    training.write_json(args.output/'complete.json',dict(status='ANALYZED_SOURCE_ONLY',optimizer_updates=0,predictions=0,
        candidate_eligible=summary['candidate_eligible'],goal_complete=False))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ('root','output'):parser.add_argument('--'+key,type=Path,required=True)
    parser.add_argument('--commit',required=True);args=parser.parse_args()
    if Path('release_commit.txt').read_text().strip()!=args.commit:raise ValueError('Pinned release required')
    torch.set_num_threads(2);run(args)


if __name__=='__main__':main()
