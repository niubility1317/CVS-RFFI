"""Real frozen default heads and SIG evidence; zero fitting, source L_dev only."""
import copy,gzip,json,time
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from . import design as d
from experiments.cvs_urban_bridge50 import features as feature
from experiments.cvs_urban_bridge50.model import own_model as bridge_model
from experiments.cvs_urban_branch50.model import own_model as nominal_model
from experiments.cvs_urban_branch50.mechanism import developer_panel,contributions
from experiments.cvs_urban_branch50 import branches as b
from experiments.cvs_urban_bridge import signature
from experiments.cvs_urban_bridge.views import coupled
from experiments.cvs_urban_sigtime.channel import imports,configurations,numpy_iq,tensor_iq,usability,audit
from experiments.cvs_urban_sigtime50.mechanism import quality_bin,gates
from experiments.cvs_urban_identity50.mechanism import identity_distances,variance
from experiments.cvs_urban_bridge.probe import head_values

KEYS=(*b.BRANCHES,'J')


@torch.no_grad()
def forward(model,x):
    values=defaultdict(list)
    for start in range(0,len(x),90):
        out,parts=feature.forward(model,x[start:start+90]);values['logits'].append(out['tx_logits'])
        for key in (*KEYS,'D'):values[key].append(parts[key])
    return {key:torch.cat(v) for key,v in values.items()}


def group_errors(student,teacher,scales):
    if student.shape!=teacher.shape or set(scales)!=set(signature.KEYS):raise ValueError('Actual paired raw112 and source scales required')
    if any(not np.isfinite(scales[k]) or not 1e-3<=scales[k]<1e6 for k in signature.KEYS):raise ValueError('Finite source RMS with floor required')
    a,b=signature.split(student),signature.split(teacher)
    values={}
    for key in signature.KEYS:
        error=(a[key]-b[key])/scales[key]
        huber=F.smooth_l1_loss(error,torch.zeros_like(error),reduction='none')
        values[key]=(huber/(1+huber)).flatten(1).mean(1)
    return values


def group_geometry(student,teacher,y):
    if student.shape!=teacher.shape or y.shape!=(len(student),) or bool(((y<0)|(y>=6)).any()):
        raise ValueError('Corresponding legal L raw112 required')
    a,b=signature.split(student),signature.split(teacher);wrong=y[:,None]!=y[None,:]
    if not wrong.any(1).all():raise ValueError('Wrong-TX source comparators required')
    result={}
    for key in signature.KEYS:
        x,z=(F.normalize(v.flatten(1),dim=1,eps=1e-4) for v in (a[key],b[key]))
        same=(x*z).sum(1);nearest=(x@z.T).masked_fill(~wrong,-torch.inf).max(1).values
        result[key]=dict(same=same,wrong=nearest,gap=same-nearest)
    return result


def summarize(items):
    if not items:raise ValueError('Cannot summarize empty source stratum')
    n=len(items);fixed=[r for r in items if r['fixed_NOMINAL_clean_correct']]
    result=dict(count=n,accuracy=sum(r['correct'] for r in items)/n,EMA_accuracy=sum(r['EMA_correct'] for r in items)/n,
        fixed_NOMINAL_clean_correct_count=len(fixed),
        fixed_NOMINAL_retention=sum(r['correct'] for r in fixed)/len(fixed) if fixed else None,
        margin=sum(r['margin'] for r in items)/n,EMA_margin=sum(r['EMA_margin'] for r in items)/n,
        common_local_group_error={k:sum(r['common_local_group_error'][k] for r in items)/n for k in signature.KEYS})
    for name in ('NOMINAL','BRIDGE_CE'):
        field=name+'_full_correct'
        observed=[r for r in fixed if r[field] is not None]
        result[name+'_comparison_count']=len(observed)
        result[name+'_errors_repaired']=sum(r['correct'] and not r[field] for r in observed) if observed else None
        result[name+'_correct_new_errors']=sum(not r['correct'] and r[field] for r in observed) if observed else None
    return result


def source_views(panel):
    ch,_=imports();configs=configurations();views={};proofs=[]
    if tuple(configs)!=tuple(d.training.SCENES):raise ValueError('All six source practical scenes required')
    for scene,cfg in configs.items():
        full=[];HQ=[];details=[];checks=[]
        raw=numpy_iq(panel['x'])
        for i,pid in enumerate(panel['ids']):
            x,_=ch.normalize_input(raw[i,0]+1j*raw[i,1])
            seed=ch.stable_seed(d.SOURCE_STREAM,'bridge_teaching/frozen_source',scene,pid)
            session=f"rx{int(panel['rx'][i])}:day{int(panel['day'][i])}:eq1"
            receiver=ch.receiver_for_session(cfg,d.RECEIVER_SEED,session)
            waves,meta,proof,_=coupled(x,cfg,seed,receiver,noise_improvement_db=6.)
            if not (proof['full_waveform_exact'] and proof['frozen_fields_identical'] and proof['frozen_kernels_identical']):
                raise ValueError('Full/HQ propagation or receiver actions differ')
            full.append(np.stack((waves['full'].real,waves['full'].imag)))
            HQ.append(np.stack((waves['bridge_independent'].real,waves['bridge_independent'].imag)))
            meta.update(q=usability(meta));details.append(meta);checks.append(proof)
        views[scene]=(dict(full=tensor_iq(np.stack(full),'cpu'),HQ_independent=tensor_iq(np.stack(HQ),'cpu')),details)
        proofs.append(dict(scene=scene,count=720,full_waveform_exact=True,receiver_actions_and_kernels_frozen=True,
            numerical_FIR_delay=cfg.fractional_delay_half_length,AGC_gain_frozen=True,
            full_added_noise_SNR_db=sum(m['signal_to_added_noise_db'] for m in details)/720,
            HQ_added_noise_SNR_db=sum(p['bridge_independent_signal_to_added_noise_db'] for p in checks)/720))
    return views,proofs


def load(arm,c):return nominal_model(c,'cpu') if arm=='NOMINAL' else bridge_model(c,'cpu')


def run():
    rows,inner=d.eligible_rows()
    if d.BASE.exists():raise FileExistsError('Existing source attribution must be reconciled, never overwritten')
    torch.set_num_threads(2);started=time.perf_counter();reference=load(*rows[0])
    with torch.no_grad():smoke=reference.forward_identity_only(torch.zeros(2,2,256))['tx_logits']
    if smoke.shape!=(2,6) or not torch.isfinite(smoke).all():raise ValueError('Real default-head checkpoint smoke failed')
    d.BASE.mkdir();d.write(d.BASE/'source_smoke.json',dict(status='PASS',forward_packets=2,query_read=False,source_only=True))
    print('SMOKE PASS real NOMINAL/default head; no query or fitting',flush=True)
    panel=developer_panel(inner);y=panel['y'];fixed=forward(reference,panel['x'])['logits'].argmax(1)==y;del reference
    views,proofs=source_views(panel);forwards=722
    configs=dict(rows);common=d.read(Path(configs['BRIDGE_CE']['output_root'])/'bridge_calibration.json')['scales']
    d.write(d.BASE/'resolved_config.json',dict(config=d.configuration(),actual_engine=audit(),
        source_role='L_dev720',V_forward_packets=0,U_opened=False,common_scales=common,
        commit=(d.ROOT/'release_commit.txt').read_text().strip()))
    results={};baseline={};control={}
    with gzip.open(d.BASE/'sample_identity.jsonl.gz','xt',encoding='utf-8',compresslevel=3) as stream:
        for arm,c in rows:
            model=load(arm,c);cp=torch.load(Path(c['output_root'])/'final_ssdg.pth',map_location='cpu',weights_only=False)
            teacher=copy.deepcopy(model);teacher.load_state_dict(cp['ema_model'],strict=True);teacher.eval().requires_grad_(False)
            before={k:v.clone() for k,v in model.state_dict().items()};tb={k:v.clone() for k,v in teacher.state_dict().items()}
            clean=forward(model,panel['x']);target_clean=forward(teacher,panel['x']);forwards+=1440
            own=common if arm=='NOMINAL' else d.read(Path(c['output_root'])/'bridge_calibration.json')['scales']
            scenes={}
            for scene,(waves,details) in views.items():
                outputs={name:forward(model,x) for name,x in waves.items()}
                targets={name:forward(teacher,x) for name,x in waves.items()};forwards+=2880
                head={name:head_values(v,y) for name,v in outputs.items()}
                th={name:head_values(v,y) for name,v in targets.items()}
                if arm=='NOMINAL':baseline[scene]=head['full']['correct'].clone()
                if arm=='BRIDGE_CE':control[scene]=head['full']['correct'].clone()
                control_correct=control.get(scene);scene_result={}
                for view in d.VIEWS:
                    out=outputs[view];target_HQ=targets['HQ_independent'];h=head[view];t=th[view]
                    errors=group_errors(out['D'],target_HQ['D'],common)
                    geometry=group_geometry(out['D'],target_HQ['D'],y)
                    own_errors=group_errors(out['D'],target_HQ['D'],own)
                    identity={key:identity_distances(out[key],clean[key],target_clean[key],y) for key in KEYS}
                    items=[];strata=defaultdict(list)
                    for i,meta in enumerate(details):
                        item=dict(physical_id=panel['ids'][i],RX=int(panel['rx'][i]),day=int(panel['day'][i]),TX=int(y[i]),
                            state=meta['state_end'],lock=meta['receiver_lock_status'],quality=quality_bin(meta['signal_to_added_noise_db']),
                            added_noise_SNR_db=meta['signal_to_added_noise_db'],receiver_quality_proxy_db=meta['quality_snr_db'],
                            residual_CFO_hz=meta['output_frequency_hz'],residual_kernel_length=meta['residual_kernel_length'],
                            applied_AGC_gain_db=meta['agc_gain_db'],q=meta['q'],
                            fixed_NOMINAL_clean_correct=bool(fixed[i]),NOMINAL_full_correct=bool(baseline[scene][i]),
                            BRIDGE_CE_full_correct=bool(control_correct[i]) if control_correct is not None else None,
                            correct=bool(h['correct'][i]),EMA_correct=bool(t['correct'][i]),
                            margin=float(h['margin'][i]),EMA_margin=float(t['margin'][i]),
                            common_local_group_error={k:float(v[i]) for k,v in errors.items()})
                        items.append(item);strata[item['RX'],item['state'],item['lock'],item['quality']].append(item)
                        stream.write(json.dumps(dict(arm=arm,scene=scene,view=view,**item),allow_nan=False)+'\n')
                    scene_result[view]=dict(overall=summarize(items),
                        strata=[dict(RX=k[0],state=k[1],lock=k[2],quality=k[3],**summarize(v)) for k,v in sorted(strata.items())],
                        branch_identity={key:{name:float(v.mean()) for name,v in values.items()} for key,values in identity.items()},
                        branch_variance=variance({key:out[key] for key in KEYS}),
                        gain_contribution=contributions(model,out),
                        SIG_group_geometry={key:{name:float(v.mean()) for name,v in values.items()} for key,values in geometry.items()},
                        own_scale_local_group_error={k:float(v.mean()) for k,v in own_errors.items()})
                scenes[scene]=scene_result
                d.write(d.BASE/'progress.json',dict(arm=arm,scenes_completed=list(scenes),forward_packets=forwards,source_only=True,query_read=False))
            if any(not torch.equal(v,model.state_dict()[k]) for k,v in before.items()) or any(not torch.equal(v,teacher.state_dict()[k]) for k,v in tb.items()):
                raise ValueError('Frozen attribution changed model or EMA state')
            if any(p.requires_grad or p.grad is not None for p in (*model.parameters(),*teacher.parameters())):
                raise ValueError('Frozen attribution acquired trainable parameters or fitted gradients')
            training_cost=d.read(Path(c['output_root'])/'completion.json')
            results[arm]=dict(clean_accuracy=float((clean['logits'].argmax(1)==y).float().mean()),
                clean_margin=float(head_values(clean,y)['margin'].mean()),scenes=scenes,gates=gates(model),
                clean_branch_variance=variance({key:clean[key] for key in KEYS}),own_scales=own,
                original_training_cost=dict(epochs=50,updates=11100,seconds=training_cost['elapsed_seconds']),
                model_state_unchanged=True,EMA_state_unchanged=True)
            print(json.dumps(dict(arm=arm,forward_packets=forwards,source_only=True)),flush=True)
    if forwards!=d.FORWARD_PACKETS:raise ValueError('Actual frozen-source forward budget differs')
    result=dict(status='ANALYZED',source_only=True,query_read=False,U_truth_read=False,U_opened=False,V_forward_packets=0,
        optimizer_updates=0,EMA_updates=0,head_fit_updates=0,forward_packets=forwards,
        source_count=720,fixed_NOMINAL_clean_correct_count=int(fixed.sum()),rows=results,
        common_scales=common,common_scale_scope=d.configuration()['common_descriptor_scales'],physical_operator_checks=proofs,
        formal_test_IQ_changed=False,seed_count=1,cross_seed_confirmation='NOT_RUN',
        counterfactual_limit='Source same-H noise intervention; trained-row differences also include optimization stochasticity; no unique-cause claim',
        seconds=time.perf_counter()-started,commit=(d.ROOT/'release_commit.txt').read_text().strip())
    d.write(d.BASE/'results.json',result)
    d.write(d.BASE/'complete.json',dict(status='ANALYZED',source_only=True,query_read=False,actual_model_count=3,models_trained=0))
    print(json.dumps({k:result[k] for k in ('status','source_only','forward_packets','seconds')}),flush=True)
    return result


if __name__=='__main__':run()
