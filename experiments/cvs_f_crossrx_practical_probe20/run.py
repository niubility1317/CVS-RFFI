"""Frozen source-L attribution. No optimizer, fitted head, U/V or target forward."""
import gzip
import json
import os
import sys
import time
from contextlib import ExitStack
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from . import design as d
from .provenance import load
from experiments.cvs_f_practical_source20.trace import ActualTrace, gradient, gradient_comparison, feature_summary
from experiments.cvs_ordinary_fold20.streams import stream
from experiments.cvs_urban_sigtime.channel import imports, configurations, audit, numpy_iq, tensor_iq
from experiments.cvs_urban_sigtime.controls import compare


def decisions(logits,y):
    scaled=logits.detach()/30.
    true=scaled.gather(1,y[:,None]);margins=true-scaled
    other=scaled.clone();other.scatter_(1,y[:,None],-float('inf'))
    return dict(prediction=scaled.argmax(1).cpu().tolist(),cosine_logits=scaled.cpu().tolist(),
        margin=(true[:,0]-other.max(1).values).cpu().tolist(),pairwise_margins=margins.cpu().tolist())


def emit(f,v):f.write(json.dumps(v,allow_nan=False,separators=(',',':'))+'\n')


def run():
    from experiments.cvs_receiver_residual_capacity4.control import lock
    # The worker owns the run for its entire lifetime, including source evaluation.
    with lock(d.BASE/'worker.owner.lock',nonblocking=True):
        _run()


def _run():
    from experiments.cvs_ordinary_pa_source.run import source_L
    from experiments.cvs_ordinary_fold20.design import FULL_FP32_POLICY
    from experiments.cvs_equivariant_identity.precision import numerical_context,actual_flags
    from experiments.cvs_receiver_residual_capacity4.control import lock
    from experiments.cvs_multi_state_action.dispatch import capacity
    with lock(Path(d.PROJECT)/'runs/receiver_residual_capacity4.lock'):
        gpu=int(os.environ['CUDA_VISIBLE_DEVICES'])
        if len(capacity()[gpu]['pids']|{os.getpid()})>3:raise ValueError('All-owner capacity exceeded')
    start=time.perf_counter();torch.set_num_threads(2);device='cuda:0'
    models={};snapshots={};contracts=[];traces={};counts={};derivatives={};roots={}
    try:
        with numerical_context(FULL_FP32_POLICY),ExitStack() as stack:
            for row in d.rows():
                c=d.config(row);a=row['arm'];out=Path(c['output_root']);out.mkdir(exist_ok=False);roots[a]=out
                d.write(out/'resolved_config.json',c)
                models[a],snapshots[a],contract=load(c,device);contracts.append(contract)
                traces[a]=stack.enter_context(ActualTrace(models[a]).installed());counts[a]=2;derivatives[a]=0
            # Both real-checkpoint no-query smokes precede loading source observations.
            L=source_L();idx=d.legal_held_indices(L,contracts)
            with stream(d.DRAW_SEED,0):idx=idx[torch.randperm(len(idx))]
            ch,residual=imports();cfgs=configurations();channel=audit()
            if tuple(cfgs)!=d.SCENES:raise ValueError('Original six practical scenes differ')
            for cfg in cfgs.values():
                if (cfg.fs_hz!=25000000. or cfg.processing_route!='residual' or cfg.mode!='post_sync'
                    or cfg.equalization_enabled or cfg.reference_snr_db!=30. or cfg.boundary!='reflect'):
                    raise ValueError('Unchanged practical residual/post_sync/EQoff/reference30/reflect required')
            d.write(d.BASE/'execution.json',dict(pid=os.getpid(),cwd=os.getcwd(),argv=sys.argv,
                GPU=os.environ.get('CUDA_VISIBLE_DEVICES'),commit=(d.ROOT/'release_commit.txt').read_text().strip(),
                numeric_flags=actual_flags(),channel=channel,source_physical_count=len(idx),source_cache_deserialized=True,
                unused_source_containers='Existing cache may contain other roles; only original labeled RX1 L1260 forwarded, U/V0',
                training_module_file=__import__('SSDG.train_ssdg',fromlist=['x']).__file__,
                input_normalization='Clean source carrier follows original nativeRMS preparation; practical normalize_input before propagation then original AGC. No new packet RMS step in F forward; existing feature normalization layers unchanged',
                counterfactuals='same named draw and actual tensor for both models; total recomputes state/AGC, fixed freezes full state/compensation/equalizer/AGC',
                gradient_loss='Existing default-head CE(label_smoothing=.01), autograd.grad on14 original tensors, eval mode, no optimizer/grad accumulation',
                head_reconstruction='One extra readout-only classify per traced packet to verify joint160; not an extra backbone packet forward',
                source_only=True,target_access=False))
            files={a:stack.enter_context(gzip.open(roots[a]/'packets.jsonl.gz','wt',encoding='utf-8',compresslevel=3)) for a in models}
            grads={a:stack.enter_context(gzip.open(roots[a]/'gradients.jsonl.gz','wt',encoding='utf-8',compresslevel=3)) for a in models}
            for batch,chunk in enumerate(idx.split(63)):
                x=L['x'][chunk].to(device);y=L['y'][chunk].long().to(device);yy=y.cpu().tolist();n=len(chunk)
                ids=[L['ids'][i] for i in chunk.tolist()];days=L['day'][chunk].tolist()
                clean={};clean_features={};clean_grad={}
                for a,m in models.items():
                    output,values=traces[a].forward(x.clone());counts[a]+=n
                    clean[a]=decisions(output['tx_logits'],y)
                    clean_features[a]={k:v.detach() for k,v in values.items()}
                    clean_grad[a]=gradient(F.cross_entropy(output['tx_logits'],y,label_smoothing=.01),m);derivatives[a]+=1
                    summaries=feature_summary(values,clean_features[a])
                    for j in range(n):
                        emit(files[a],dict(physical_id=ids[j],TX=yy[j],day=int(days[j]),RX=1,scene='clean',condition='clean',
                            **{k:v[j] for k,v in clean[a].items()},features={k:{q:v[j] for q,v in z.items()} for k,z in summaries.items()}))
                    emit(grads[a],dict(batch=batch,scene='clean',condition='clean',count=n,
                        parameters=gradient_comparison(clean_grad[a],clean_grad[a])))
                    del output,values
                correct={a:[p==t for p,t in zip(clean[a]['prediction'],yy)] for a in models}
                common=[all(correct[a][j] for a in models) for j in range(n)]
                raw=numpy_iq(x)
                for scene,cfg in cfgs.items():
                    banks={k:[] for k in d.CONDITIONS};metadata=[]
                    for j,pid in enumerate(ids):
                        observation,_=ch.normalize_input(raw[j,0]+1j*raw[j,1])
                        seed=ch.stable_seed(d.DRAW_SEED,'FcrossRX/source/practical_attribution_v1',scene,pid)
                        hardware=ch.receiver_for_session(cfg,d.RECEIVER_SEED,f'rx1:day{int(days[j])}:eq1')
                        views,meta,_=compare(observation,cfg,seed,hardware)
                        if tuple(views)!=d.CONDITIONS:raise ValueError('Shared original source intervention bank differs')
                        metadata.append(meta)
                        for k,w in views.items():banks[k].append(np.stack([w.real,w.imag]))
                    full={}
                    for condition in d.CONDITIONS:
                        view=tensor_iq(np.stack(banks[condition]),device)
                        for a,m in models.items():
                            output,values=traces[a].forward(view.clone());counts[a]+=n
                            result=decisions(output['tx_logits'],y)
                            g=gradient(F.cross_entropy(output['tx_logits'],y,label_smoothing=.01),m);derivatives[a]+=1
                            emit(grads[a],dict(batch=batch,scene=scene,condition=condition,count=n,
                                parameters=gradient_comparison(clean_grad[a],g)))
                            summaries=feature_summary(values,clean_features[a])
                            if condition=='full':full[a]=result['prediction']
                            for j in range(n):
                                emit(files[a],dict(physical_id=ids[j],TX=yy[j],day=int(days[j]),RX=1,scene=scene,condition=condition,
                                    clean_correct=correct[a][j],C0_clean_correct=correct['C0'][j],common_clean_correct=common[j],
                                    full_prediction=full[a][j],**{k:v[j] for k,v in result.items()},
                                    full_channel=metadata[j]['full'],channel=metadata[j][condition],
                                    features={k:{q:v[j] for q,v in z.items()} for k,z in summaries.items()}))
                            del output,values,g
                        del view
                    for a in models:files[a].flush();grads[a].flush()
                    progress=dict(batch=batch+1,batches=20,scene=scene,packet_forwards=counts,derivative_calls=derivatives,
                        wall_seconds=time.perf_counter()-start,optimizer_updates=0,target_access=False)
                    d.write(d.BASE/'progress.json',progress)
                    print(f'PROGRESS batch={batch+1}/20 scene={scene} packets={counts} derivatives={derivatives}',flush=True)
                del clean_features,clean_grad
            results={}
            for a,m in models.items():
                c=d.config(next(r for r in d.rows() if r['arm']==a))
                if (counts[a]!=c['expected_packet_forwards'] or derivatives[a]!=c['expected_derivative_calls']
                    or m.training or any(p.grad is not None for p in m.parameters())
                    or m.id_backbone.encoder._ordinary_eval_packets!=counts[a]
                    or m.id_backbone.encoder._ordinary_train_packets!=0):
                    raise ValueError('Frozen original F budget/mode/gradient state differs')
                if any(not torch.equal(v,m.state_dict()[k].detach().cpu()) for k,v in snapshots[a].items()):
                    raise ValueError('Frozen model parameters or buffers changed')
                results[a]=dict(status='SOURCE_F_CROSSRX_PRACTICAL_DIAGNOSTIC_COMPLETE',config=c,packet_forwards=counts[a],
                    derivative_calls=derivatives[a],parameters_buffers_unchanged=True,optimizer_updates=0,head_fit_updates=0,
                    maximum_logits_reconstruction_error=traces[a].maximum_reconstruction_error,
                    representation=traces[a].representation,learned_gains=traces[a].gates(),
                    V_used=False,U_used=False,target_access=False,source_only=True,new_models=0)
                d.write(roots[a]/'completion.json',results[a])
        d.write(d.BASE/'completion.json',dict(status='SOURCE_F_CROSSRX_PRACTICAL_DIAGNOSTIC_COMPLETE',rows=list(results),
            packet_forwards=sum(counts.values()),derivative_calls=sum(derivatives.values()),unique_source_physical_packets=1260,
            wall_seconds=time.perf_counter()-start,parent_training_epochs_per_model=20,parent_updates_per_model=4440,
            optimizer_updates=0,head_fit_updates=0,new_models=0,V_used=False,U_used=False,target_access=False,
            automatic_training=False,goal_completed=False,source_only=True))
        print('COMPLETE frozen source practical attribution; no training/target',flush=True)
    except Exception as error:
        d.write(d.BASE/'failure.json',dict(status='FAILED',error=repr(error),packet_forwards=counts,derivative_calls=derivatives,no_automatic_retry=True));raise
