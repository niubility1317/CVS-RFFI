"""Source-only frozen-F fidelity and gradient geometry, without TX truth."""
import argparse
import gc
import json
import os
from pathlib import Path
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

RUN='20261030-diagnostic-DURA-source-task-geometry-s2026092701-m2-r01'
MODELS=(('noVarAux',no_variance,'cf40edea90667d9645a724b1e8f333027cdf7869',False),
        ('Fguide010',f_guided,'2cee2bec08165769f787b2aa85d1a8d97177844d',True))
GLOBALS=('RUN','configuration','command','learning_rate','direct_objective','scratch_model','save_model','load_model','smoke','train')


def decision_metrics(actual,reference):
    if actual.shape!=reference.shape or actual.ndim!=2 or actual.shape[1]!=6:
        raise ValueError('Same original six-class frozen F logits required')
    choice=reference.argmax(1)
    chosen=actual.gather(1,choice[:,None]).squeeze(1)
    other=actual.masked_fill(F.one_hot(choice,6).bool(),float('-inf')).max(1).values
    ref=reference.gather(1,choice[:,None]).squeeze(1)
    refother=reference.masked_fill(F.one_hot(choice,6).bool(),float('-inf')).max(1).values
    return dict(teacher_argmax_agreement=(actual.argmax(1)==choice).to(actual.dtype),
                margin_for_clean_F_choice=chosen-other,clean_F_margin=ref-refother)


def run(args):
    if args.output.name!=RUN or args.output.exists() or os.environ.get('CUDA_VISIBLE_DEVICES')!='':
        raise ValueError('New immutable source-only CPU diagnostic root required')
    original={key:getattr(training,key) for key in GLOBALS}
    p=training.paths(args.root);clean,roles=training.source_training(p);indices=source_indices(roles)
    cache=args.root/'runs'/f_guided.RUN/'scratch_joint/source_feature_cache'
    complete=training.read_json(cache/'complete.json')
    if (complete['status'],complete['count'],complete['dimension'],complete['F_teacher'],complete['TX_labels_read'],complete['target_reads']) != (
            'SOURCE_F_FEATURES_COMPLETE',63000,160,f_guided.teacher_reference(),0,0):raise ValueError('Matching authorized source F cache required')
    if not np.array_equal(np.load(cache/'physical_ids.npy',allow_pickle=False),np.load(p['cache']/'train/physical_ids.npy',allow_pickle=False)):
        raise ValueError('Source F cache order differs')
    targets=np.load(cache/'features.npy',mmap_mode='r',allow_pickle=False)
    if targets.shape!=(63000,160) or targets.dtype!=np.float32:raise ValueError('FP32 original160D source features required')
    from experiments.pamf_fcrossrx_readout.frozen import load_reader
    reader=load_reader(p['reader'],p['source_cache'],'cpu').eval().requires_grad_(False)
    reader_before={key:value.detach().clone() for key,value in reader.state_dict().items()}
    args.output.mkdir()
    training.write_json(args.output/'launch.json',dict(pid=os.getpid(),cwd=os.getcwd(),argv=sys.argv,commit=args.commit,CPU_threads=2,GPU_workers=0))
    records=[];started=time.perf_counter()
    for name,module,commit,F_access in MODELS:
        for key,value in original.items():setattr(training,key,value)
        module.configure();paths=training.paths(args.root)
        done=training.read_json(paths['output']/'scratch_joint/training_complete.json')
        expected=dict(status='SOURCE_TRAINED',epoch=20,updates=4940,training_TX_labels_read=0,
            training_F_access=F_access,V_reads=0,target_reads=0,initialization_checkpoint_loads=0)
        if any(done.get(k)!=v for k,v in expected.items()):raise ValueError('Only completed legal source models allowed')
        if F_access and (done.get('F_updates')!=0 or done.get('F_unchanged') is not True or done.get('F_teacher')!=f_guided.teacher_reference()):
            raise ValueError('Frozen authorized F source teacher required')
        model=training.load_model(paths['output']/'scratch_joint/final.pt','cpu',arm='scratch_joint',commit=commit)
        model.eval().requires_grad_(True)
        before={key:value.detach().clone() for key,value in model.state_dict().items()}
        with torch.no_grad():
            smoke=model(tensor(clean[indices[:2]],'cpu'));scores=reader(smoke)
            if scores.shape!=(2,6) or not torch.isfinite(scores).all():raise ValueError('Actual G/F source2Ls smoke failed')
        print(json.dumps(dict(event='SOURCE_G_F_CHECKPOINT_SMOKE_PASS',model=name,real_Ls=2,TX_labels_read=0,target_reads=0)),flush=True)
        names,parameters=zip(*model.named_parameters())
        groups={'all':list(range(len(names)))}
        for prefix in ('frozen.coarse.','frozen.estimator.','prior.'):
            groups[prefix]=[i for i,key in enumerate(names) if key.startswith(prefix)]
        generator=torch.Generator().manual_seed(2026103005)
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
                guide=1-F.cosine_similarity(features,reference,dim=1,eps=1e-8)
                base=terms['iq_nmse']+.1*terms['prior_nmse']+.05*terms['spectral_error']+.05*terms['action_probe_nmse']
                vectors=[]
                for loss in (base.mean(),.1*guide.mean()):
                    derivative=torch.autograd.grad(loss,parameters,retain_graph=True,allow_unused=True)
                    vector=[torch.zeros_like(p) if v is None else v.detach() for p,v in zip(parameters,derivative)]
                    if any(not torch.isfinite(v).all() for v in vector):raise FloatingPointError('Nonfinite source task gradient')
                    vectors.append(vector)
                stats={group:geometry([vectors[0][i] for i in selected],[vectors[1][i] for i in selected]) for group,selected in groups.items()}
                with torch.no_grad():
                    decisions=decision_metrics(reader.classify_features(features.detach()),reader.classify_features(reference))
                metrics=dict(iq_nmse=terms['iq_nmse'],feature_cosine_error=guide,**decisions)
                if any(not torch.isfinite(v).all() for v in metrics.values()):raise FloatingPointError('Nonfinite source task metric')
                records.append(dict(model=name,epoch=epoch,offset=offset,count=32,L_s=int((roles[take]=='L_s').sum()),
                    U_s=int((roles[take]=='U_s').sum()),source_positions=take.tolist(),
                    metrics={key:float(value.detach().mean()) for key,value in metrics.items()},geometry=stats))
                print(json.dumps(dict(event='SOURCE_TASK_BATCH',model=name,epoch=epoch,count=offset+32,seconds=time.perf_counter()-started)),flush=True)
        if any(not torch.equal(v,before[key]) for key,v in model.state_dict().items()) or any(v.grad is not None for v in model.parameters()):
            raise ValueError('Diagnostic changed G or accumulated gradient')
        del model,before,parameters;gc.collect()
    for key,value in original.items():setattr(training,key,value)
    if any(not torch.equal(v,reader_before[key]) for key,v in reader.state_dict().items()) or any(v.grad is not None for v in reader.parameters()):
        raise ValueError('Diagnostic changed F')
    training.write_json(args.output/'results.json',dict(status='ANALYZED_SOURCE_ONLY',records=records,source_records=512,
        source_L=128,source_U=384,epochs=[1,20],seconds=time.perf_counter()-started,model_unchanged=True,F_unchanged=True,
        optimizer_updates=0,TX_labels_read=0,V_reads=0,target_reads=0,
        source_rule='Stronger beta0.3 candidate only if both epochs Fguide/base weighted gradient norm median<=.2,mean cosine>=0,teacher agreement<.98,and Fguide IQ<=1.1 matched noVar;otherwise no beta increase from this diagnostic',
        scope='Frozen F clean-choice agreement and margins are source teacher fidelity, not TX accuracy;batchmean gradients;no new weights or target feedback'))
    training.write_json(args.output/'complete.json',dict(status='ANALYZED_SOURCE_ONLY',optimizer_updates=0,predictions=0,goal_complete=False))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ('root','output'):parser.add_argument('--'+key,type=Path,required=True)
    parser.add_argument('--commit',required=True);args=parser.parse_args()
    if Path('release_commit.txt').read_text().strip()!=args.commit:raise ValueError('Pinned release required')
    torch.set_num_threads(2);run(args)


if __name__=='__main__':main()
