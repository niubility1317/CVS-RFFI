"""One frozen matched parent/ownEMA, legal L_dev only, no target or fitting."""
import copy,gzip,json,time
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from . import design as d,signature
from .views import coupled
from experiments.cvs_urban_branch50.model import own_model
from experiments.cvs_urban_branch50.mechanism import developer_panel
from experiments.cvs_urban_sigtime.channel import imports,configurations,numpy_iq,tensor_iq,audit
from experiments.cvs_urban_sigtime50.mechanism import quality_bin


@torch.no_grad()
def forward(model,x):
    logits=[];joint=[];descriptors=[]
    for start in range(0,len(x),90):
        with signature.capture(model) as values:out=model.forward_identity_only(x[start:start+90])
        logits.append(out['tx_logits'].detach());joint.append(out['id_feat_joint'].detach())
        descriptors.append(values['descriptor'].detach())
    return dict(logits=torch.cat(logits),joint=torch.cat(joint),descriptor=torch.cat(descriptors))


def head_values(output,y):
    logits=output['logits'];other=logits.masked_fill(F.one_hot(y,6).bool(),-torch.inf).max(1).values
    probability=logits.softmax(1)
    return dict(correct=logits.argmax(1)==y,margin=(logits.gather(1,y[:,None]).squeeze(1)-other)/30,
        entropy=-(probability*probability.clamp_min(1e-12).log()).sum(1))


def comparisons(student,targets,y):
    x=F.normalize(student['joint'],dim=1,eps=1e-4);wrong=y[:,None]!=y[None,:]
    result={}
    for name,target in targets.items():
        z=F.normalize(target['joint'],dim=1,eps=1e-4);same=(x*z).sum(1)
        nearest=(x@z.T).masked_fill(~wrong,-torch.inf).max(1).values
        result[name]=dict(pair_cosine=same,closest_wrong_TX_cosine=nearest,pair_minus_wrong_cosine=same-nearest)
    return result


def summarize(items):
    n=len(items);fixed=[v for v in items if v['fixed_parent_clean_correct']]
    return dict(count=n,student_accuracy=sum(v['student_correct'] for v in items)/n,
        EMA_accuracy=sum(v['EMA_correct'] for v in items)/n,
        fixed_parent_retention=sum(v['student_correct'] for v in fixed)/len(fixed) if fixed else None,
        repairs=sum(v['student_correct'] and not v['full_student_correct'] for v in fixed),
        new_errors=sum(not v['student_correct'] and v['full_student_correct'] for v in fixed),
        student_margin=sum(v['student_margin'] for v in items)/n,EMA_margin=sum(v['EMA_margin'] for v in items)/n,
        EMA_entropy=sum(v['EMA_entropy'] for v in items)/n)


def run():
    if d.BASE.exists():raise FileExistsError('Reconcile existing source run; no overwrite or automatic resubmission')
    d.BASE.mkdir();started=time.perf_counter();torch.set_num_threads(2);c=d.parent_config()
    source=Path(c['output_root']);inner=d.read(source/'inner_source_contract.json')
    model=own_model(c,'cpu');checkpoint=torch.load(source/'final_ssdg.pth',map_location='cpu',weights_only=False)
    teacher=copy.deepcopy(model);teacher.load_state_dict(checkpoint['ema_model'],strict=True);teacher.eval().requires_grad_(False)
    before={k:v.clone() for k,v in model.state_dict().items()};tb={k:v.clone() for k,v in teacher.state_dict().items()}
    panel=developer_panel(inner);y=panel['y'];ch,_=imports();configs=configurations();forwards=0
    if tuple(configs)!=tuple(d.SCENES):raise ValueError('Declared six source scenes differ from actual channel configurations')
    clean=forward(model,panel['x']);target_clean=forward(teacher,panel['x']);forwards+=1440
    fixed=clean['logits'].argmax(1)==y
    if clean['logits'].shape!=(720,6) or not torch.isfinite(clean['logits']).all():raise ValueError('Actual default checkpoint smoke failed')
    print('SMOKE PASS real A0/ownEMA/default head/internal L_dev; query_read=false; no updates',flush=True)
    d.write(d.BASE/'resolved_config.json',dict(config=d.configuration(),commit=(d.ROOT/'release_commit.txt').read_text().strip(),
        parent_checkpoint=str(source/'final_ssdg.pth'),parent_initialization=d.read(source/'initialization.json'),
        actual_engine=audit(),parameter_count=sum(p.numel() for p in model.parameters())))
    summaries={};identities={};group_differences={};operator_stats=[];all_teacher_HQ=[];scene_cache={}
    with gzip.open(d.BASE/'sample_bridge.jsonl.gz','xt',encoding='utf-8',compresslevel=3) as stream:
        for scene,cfg in configs.items():
            arrays=defaultdict(list);details=[];proofs=[]
            for start in range(0,720,90):
                raw=numpy_iq(panel['x'][start:start+90])
                for j,pid in enumerate(panel['ids'][start:start+90]):
                    record,_=ch.normalize_input(raw[j,0]+1j*raw[j,1])
                    seed=ch.stable_seed(d.DIAGNOSTIC_SEED,'coupled_bridge/source',scene,pid)
                    session=f"rx{int(panel['rx'][start+j])}:day{int(panel['day'][start+j])}:eq1"
                    receiver=ch.receiver_for_session(cfg,d.RECEIVER_SEED,session)
                    waves,meta,proof,traces=coupled(record,cfg,seed,receiver,noise_improvement_db=d.NOISE_IMPROVEMENT_DB)
                    for name in d.VIEWS:arrays[name].append(np.stack((waves[name].real,waves[name].imag)))
                    details.append(meta);proofs.append(proof)
            outputs={};targets={};heads={};teacher_heads={}
            for name in d.VIEWS:
                iq=tensor_iq(np.stack(arrays[name]),'cpu')
                outputs[name]=forward(model,iq);targets[name]=forward(teacher,iq);forwards+=1440
                heads[name]=head_values(outputs[name],y);teacher_heads[name]=head_values(targets[name],y)
            comparisons_by_target=comparisons(outputs['full'],dict(clean_EMA=target_clean,
                HQ_shared_EMA=targets['bridge_shared'],HQ_independent_EMA=targets['bridge_independent']),y)
            identities[scene]={name:{key:float(value.mean()) for key,value in entries.items()} for name,entries in comparisons_by_target.items()}
            all_teacher_HQ.append(targets['bridge_independent']['descriptor'])
            scene_cache[scene]=(outputs['full']['descriptor'],dict(clean_EMA=target_clean['descriptor'],
                full_EMA=targets['full']['descriptor'],HQ_shared_EMA=targets['bridge_shared']['descriptor'],
                HQ_independent_EMA=targets['bridge_independent']['descriptor']))
            summaries[scene]={}
            for name in d.VIEWS:
                items=[];strata=defaultdict(list)
                for i,meta in enumerate(details):
                    item=dict(physical_id=panel['ids'][i],RX=int(panel['rx'][i]),state=meta['state_end'],
                        lock=meta['receiver_lock_status'],quality=quality_bin(meta['signal_to_added_noise_db']),
                        fixed_parent_clean_correct=bool(fixed[i]),student_correct=bool(heads[name]['correct'][i]),
                        EMA_correct=bool(teacher_heads[name]['correct'][i]),full_student_correct=bool(heads['full']['correct'][i]),
                        student_margin=float(heads[name]['margin'][i]),EMA_margin=float(teacher_heads[name]['margin'][i]),
                        EMA_entropy=float(teacher_heads[name]['entropy'][i]),
                        full_added_noise_SNR_db=meta['signal_to_added_noise_db'],full_receiver_quality_proxy_db=meta['quality_snr_db'],
                        residual_CFO_hz=meta['output_frequency_hz'],residual_kernel_length=meta['residual_kernel_length'],
                        applied_AGC_gain_db=meta['agc_gain_db'])
                    stream.write(json.dumps(dict(scene=scene,view=name,**item),allow_nan=False)+'\n');items.append(item)
                    strata[item['RX'],item['state'],item['lock'],item['quality']].append(item)
                summaries[scene][name]=dict(overall=summarize(items),strata=[dict(RX=k[0],state=k[1],lock=k[2],quality=k[3],**summarize(v)) for k,v in sorted(strata.items())])
            operator_stats.append(dict(scene=scene,count=720,full_reconstruction_exact=all(v['full_waveform_exact'] for v in proofs),
                frozen_state_and_operator=all(v['frozen_fields_identical'] and v['frozen_kernels_identical'] for v in proofs),
                shared_noise_SNR_gain_db=sum(v['bridge_shared_signal_to_added_noise_db']-v['full_signal_to_added_noise_db'] for v in proofs)/720,
                independent_noise_SNR_gain_db=sum(v['bridge_independent_signal_to_added_noise_db']-v['full_signal_to_added_noise_db'] for v in proofs)/720,
                nominal_numerical_latency_samples=cfg.fractional_delay_half_length,AGC_gain_frozen=True,sample_hard_clipping_present=False))
            d.write(d.BASE/'progress.json',dict(scenes_completed=list(summaries),source_only=True,query_read=False,forward_packets=forwards))
            print(json.dumps(dict(source_scene=scene,forward_packets=forwards)),flush=True)
    scales=signature.teacher_scales(torch.cat(all_teacher_HQ))
    for scene,(student,targets) in scene_cache.items():
        group_differences[scene]={}
        for name,target in targets.items():
            total,groups=signature.grouped_loss(student,target,scales)
            group_differences[scene][name]=dict(mean_group_loss=float(total),**{key:float(value) for key,value in groups.items()})
    if any(not torch.equal(v,model.state_dict()[k]) for k,v in before.items()) or any(not torch.equal(v,teacher.state_dict()[k]) for k,v in tb.items()):
        raise ValueError('Source bridge probe changed model/EMA state')
    if any(p.grad is not None or p.requires_grad for p in model.parameters()) or any(p.grad is not None or p.requires_grad for p in teacher.parameters()):
        raise ValueError('Frozen source bridge probe acquired training gradients')
    if forwards!=d.EXPECTED_FORWARDS:raise ValueError('Unexpected zero-update source budget')
    result=dict(status='ANALYZED',source_only=True,query_read=False,U_truth_read=False,V_forward_packets=0,
        source_role='L_dev',source_count=720,fixed_parent_clean_correct_count=int(fixed.sum()),
        config=d.configuration(),parent_training_cost=dict(epochs=50,updates=11100,
            seconds=d.read(source/'completion.json')['elapsed_seconds'],weight_inheritance_for_new_training=False),
        forward_packets=forwards,optimizer_updates=0,head_fit_updates=0,feature_modules_added=0,
        model_state_unchanged=True,EMA_state_unchanged=True,summary=summaries,identity_target_comparisons=identities,
        SIG_group_loss=group_differences,SIG_source_scales=scales,
        SIG_scale_scope='RMS from this frozen ownEMA independent HQ source720 x six scenes, floor1e-3; diagnostic, not a warmup training coefficient',
        physical_operator_checks=operator_stats,noise_improvement_db=d.NOISE_IMPROVEMENT_DB,
        original_recording_noise_removed=False,true_total_SNR_known=False,formal_test_IQ_changed=False,
        causal_scope='Controlled simulator-added-noise changes with receiver actions frozen; no trained-method gain claim',
        dropout='eval for all diagnostic views; native training Dropout remains unchanged',
        seconds=time.perf_counter()-started,commit=(d.ROOT/'release_commit.txt').read_text().strip())
    d.write(d.BASE/'results.json',result);d.write(d.BASE/'complete.json',dict(status='ANALYZED',source_only=True,query_read=False,actual_model_count=1,models_trained=0,optimizer_updates=0))
    print(json.dumps({k:result[k] for k in ('status','source_only','forward_packets','seconds')}),flush=True)
    return result


if __name__=='__main__':run()
