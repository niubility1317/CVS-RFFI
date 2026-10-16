import gzip,json,os,sys,time
from contextlib import ExitStack
from pathlib import Path
import numpy as np
import torch
from . import design as d
from .core import factor_views,phase_from_kernel,require_same_practical
from experiments.cvs_f_practical_source20.provenance import load
from experiments.cvs_f_practical_source20.trace import ActualTrace,feature_summary
from experiments.cvs_f_practical_source20.run import decisions,emit
from experiments.cvs_ordinary_fold20.streams import stream
from experiments.cvs_urban_sigtime.channel import imports,configurations,audit,numpy_iq,tensor_iq


def previous():
    d.prerequisite();data={}
    for row in d.rows():
        a=row['arm'];root=d.PREVIOUS/row['row_id'];done=d.read(root/'completion.json')
        if not done['parameters_buffers_unchanged'] or done['config']!=d.prior.config(row):raise ValueError('Previous fixed parent matrix differs')
        data[a]={}
        with gzip.open(root/'packets.jsonl.gz','rt',encoding='utf-8') as f:
            for line in f:
                r=json.loads(line)
                if r['condition'] not in ('full','clean'):continue
                key=(r['physical_id'],r['scene'])
                if key in data[a]:raise ValueError('Duplicate previous source physical prediction')
                data[a][key]={k:r[k] for k in ('TX','day','RX','prediction')}
                if r['condition']=='full':data[a][key]['full_channel']=r['full_channel']
        if len(data[a])!=8820:raise ValueError('Previous clean+6full source matrix required')
    if data['A0'].keys()!=data['W30'].keys():raise ValueError('Same original source observations required')
    return data


def run():
    from experiments.cvs_ordinary_pa_source.run import source_L
    from experiments.cvs_ordinary_fold20.design import FULL_FP32_POLICY
    from experiments.cvs_equivariant_identity.precision import numerical_context,actual_flags
    from experiments.cvs_receiver_residual_capacity4.control import lock
    from experiments.cvs_multi_state_action.dispatch import capacity
    with lock(Path(d.PROJECT)/'runs/receiver_residual_capacity4.lock'):
        if len(capacity()[int(os.environ['CUDA_VISIBLE_DEVICES'])]['pids']|{os.getpid()})>3:raise ValueError('All-owner capacity exceeded')
    torch.set_num_threads(2);start=time.perf_counter();counts={};models={};snapshots={};traces={};contracts=[];roots={};rms=[]
    try:
        with numerical_context(FULL_FP32_POLICY),torch.no_grad(),ExitStack() as stack:
            for row in d.rows():
                c=d.config(row);d.validate(c);a=row['arm'];root=Path(c['output_root']);root.mkdir(exist_ok=False);roots[a]=root
                d.write(root/'resolved_config.json',c)
                original=d.prior.config(row)
                if any(c[k]!=original[k] for k in ('checkpoint','checkpoint_origin','source_contract','model_seed')):raise ValueError('No alternate parent')
                models[a],snapshots[a],contract=load(original,'cuda:0');models[a].requires_grad_(False);contracts.append(contract)
                traces[a]=stack.enter_context(ActualTrace(models[a]).installed());counts[a]=2
            old=previous();L=source_L();idx=d.prior.legal_held_indices(L,contracts)
            with stream(d.DRAW_SEED,0):idx=idx[torch.randperm(len(idx))]
            ch,_=imports();cfgs=configurations();channel=audit()
            require_same_practical(channel['actual_configs'],d.read(d.PREVIOUS/'execution.json')['channel']['actual_configs'])
            files={a:stack.enter_context(gzip.open(roots[a]/'packets.jsonl.gz','wt',encoding='utf-8',compresslevel=3)) for a in models}
            d.write(d.BASE/'execution.json',dict(pid=os.getpid(),cwd=os.getcwd(),argv=sys.argv,GPU=os.environ.get('CUDA_VISIBLE_DEVICES'),
                commit=(d.ROOT/'release_commit.txt').read_text().strip(),channel=channel,numeric_flags=actual_flags(),
                source_count=1260,source_only=True,target_access=False,V_used=False,U_used=False,
                source_cache='Existing source carrier may deserialize other roles; only held original L1260 used/forwarded',
                input_normalization='All four factors share normalize_input before factor; no normalization after delay/phase or added model packetRMS',
                phase_scope='Unit phase of original H largest tap, exactly previous flat_H phase; no TX inversion, oscillator correction or hardware fit',
                delay_scope='24point causal reflect FIR; first24 synthetic-past samples, no future samples; window energy can change',
                cohort_scope='All original full statuses reused only for fixed stratification; not receiver states of synthetic factors',
                extra_head_readouts='One classify sanity reconstruction per traced packet; no extra backbone packet',
                repeated_delay='Same delay waveform evaluated in each of6scene rows as preregistered; not6independent evidence'))
            for batch,chunk in enumerate(idx.split(63)):
                ids=[L['ids'][i] for i in chunk.tolist()];days=L['day'][chunk].tolist();y=L['y'][chunk].long().to('cuda:0');yy=y.cpu().tolist()
                raw=numpy_iq(L['x'][chunk]);unit=[]
                for j in range(len(chunk)):
                    wave,power=ch.normalize_input(raw[j,0]+1j*raw[j,1]);rms.append(float(power));unit.append(wave)
                    p=old['A0'][(ids[j],'clean')]
                    if (p['TX'],p['day'],p['RX'])!=(yy[j],int(days[j]),1):raise ValueError('Old source physical identity differs')
                raw_tensor=tensor_iq(np.stack([np.stack([w.real,w.imag]) for w in unit]),'cuda:0')
                clean={};features={}
                for a,m in models.items():
                    out,z=traces[a].forward(raw_tensor.clone());counts[a]+=len(chunk);clean[a]=decisions(out['tx_logits'],y);features[a]=z
                masks={a:[p==t for p,t in zip(clean[a]['prediction'],yy)] for a in models}
                common=[all(masks[a][j] for a in models) for j in range(len(chunk))]
                for a in models:
                    fs=feature_summary(features[a],features[a])
                    for j,pid in enumerate(ids):
                        emit(files[a],dict(physical_id=pid,TX=yy[j],day=int(days[j]),RX=1,scene='raw',condition='raw',
                            original_cached_clean_prediction=old[a][(pid,'clean')]['prediction'],raw_prediction=clean[a]['prediction'][j],
                            **{k:v[j] for k,v in clean[a].items()},features={k:{q:v[j] for q,v in z.items()} for k,z in fs.items()}))
                for scene,cfg in cfgs.items():
                    if cfg.fractional_delay_half_length!=24 or cfg.boundary!='reflect' or cfg.equalization_enabled:raise ValueError('Fixed origin factor differs')
                    banks={c:[] for c in d.CONDITIONS};meta=[]
                    for j,pid in enumerate(ids):
                        seed=ch.stable_seed(d.DRAW_SEED,'F/source/practical_attribution_v1',scene,pid)
                        hardware=ch.receiver_for_session(cfg,d.SEED,f'rx1:day{int(days[j])}:eq1')
                        phase,m=phase_from_kernel(cfg,seed,hardware);full=old['A0'][(pid,scene)]['full_channel']
                        if m['state']!=full['state'] or m['kernel_length']!=full['kernel_length']:raise ValueError('Previous full latent draw changed')
                        for a in models:
                            if old[a][(pid,scene)]['full_channel']!=full:raise ValueError('Previous two models did not share full physical draw')
                        views=factor_views(unit[j],phase);meta.append(m)
                        for c,w in views.items():banks[c].append(np.stack([w.real,w.imag]))
                    for condition in d.CONDITIONS:
                        view=tensor_iq(np.stack(banks[condition]),'cuda:0')
                        for a,m in models.items():
                            out,z=traces[a].forward(view.clone());counts[a]+=len(chunk);result=decisions(out['tx_logits'],y);fs=feature_summary(z,features[a])
                            for j,pid in enumerate(ids):
                                emit(files[a],dict(physical_id=pid,TX=yy[j],day=int(days[j]),RX=1,scene=scene,condition=condition,
                                    clean_correct=masks[a][j],A0_clean_correct=masks['A0'][j],common_clean_correct=common[j],
                                    raw_prediction=clean[a]['prediction'][j],full_prediction=old[a][(pid,scene)]['prediction'],
                                    full_channel=old['A0'][(pid,scene)]['full_channel'],factor_metadata=meta[j],
                                    input_power=float(np.mean(banks[condition][j][0]**2+banks[condition][j][1]**2)),
                                    **{k:v[j] for k,v in result.items()},features={k:{q:v[j] for q,v in z.items()} for k,z in fs.items()}))
                    for f in files.values():f.flush()
                    d.write(d.BASE/'progress.json',dict(batch=batch+1,batches=20,scene=scene,packet_forwards=counts,wall_seconds=time.perf_counter()-start,optimizer_updates=0,target_access=False))
                    print(f'PROGRESS batch={batch+1}/20 scene={scene} packets={counts}',flush=True)
            for a,m in models.items():
                if counts[a]!=23942 or m.training or any(p.requires_grad or p.grad is not None for p in m.parameters()) or m.id_backbone.encoder._ordinary_eval_packets!=counts[a] or m.id_backbone.encoder._ordinary_train_packets:
                    raise ValueError('Frozen factor budget/state differs')
                if any(not torch.equal(v,m.state_dict()[k].detach().cpu()) for k,v in snapshots[a].items()):raise ValueError('Frozen model state changed')
                d.write(roots[a]/'completion.json',dict(status='SOURCE_F_PHASE_ORIGIN_COMPLETE',config=d.config(next(r for r in d.rows() if r['arm']==a)),
                    packet_forwards=counts[a],parameters_buffers_unchanged=True,requires_grad=False,maximum_logits_reconstruction_error=traces[a].maximum_reconstruction_error,
                    learned_gains=traces[a].gates(),optimizer_updates=0,head_fit_updates=0,derivative_calls=0,source_only=True,target_access=False))
        d.write(d.BASE/'completion.json',dict(status='SOURCE_F_PHASE_ORIGIN_COMPLETE',rows=list(models),packet_forwards=sum(counts.values()),
            optimizer_updates=0,EMA_updates=0,head_fit_updates=0,derivative_calls=0,new_models=0,unique_source_physical_packets=1260,
            source_input_RMS=dict(minimum=min(rms),maximum=max(rms),mean=sum(rms)/len(rms)),wall_seconds=time.perf_counter()-start,
            parent_training_epochs_per_model=20,parent_updates_per_model=4440,source_only=True,target_access=False,V_used=False,U_used=False,automatic_training=False,goal_completed=False))
        print('COMPLETE frozen phase/origin source-only factor;0training0target',flush=True)
    except Exception as error:
        d.write(d.BASE/'failure.json',dict(status='FAILED',error=repr(error),packet_forwards=counts,no_automatic_retry=True));raise
