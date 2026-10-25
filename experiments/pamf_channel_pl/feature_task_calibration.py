"""Fixed source H/G pairs and no-update whole-parameter task-CE calibration."""
import math
from pathlib import Path
import statistics
import time

from experiments.cric_dura_comparison.io import read,write
from .feature_task_training import RUN,CACHE_RUN,SEED,VARIANTS,GRADIENT_RULE,extra
from .restored_source import validate_context,source_channel,snapshot,unchanged,SCENES

BATCHES=12


def decide(rows):
    names=('GJointHard','GFeatureHard','HFeatureHard')
    if len(rows)!=BATCHES:raise ValueError('Twelve complete source batches required')
    numbers=[r[k] for r in rows for k in ['primary_norm']+[v+'_norm' for v in names]]
    if any(not math.isfinite(v) or v<=0 for v in numbers):raise ValueError('Finite positive whole-parameter source gradients required')
    primary=statistics.median(r['primary_norm'] for r in rows)
    medians={v:statistics.median(r[v+'_norm'] for r in rows) for v in names}
    weights=dict(G=min(.25,.1*primary/max(medians[v] for v in names[:2])),H=min(.25,.1*primary/medians[names[2]]))
    return dict(weights=weights,median_primary_norm=primary,median_extra_norm=medians,
        weighted_median_norm_ratio={v:weights['H' if v.startswith('H') else 'G']*medians[v]/primary for v in names},
        median_gradient_cosine={v:statistics.median(r[v+'_cosine'] for r in rows) for v in names},
        gradient_rule=GRADIENT_RULE,scope='preclip CENTER_UNITS physical whole-parameter norm proxy;not AdamW update guarantee')


def prepare_source(c):
    if (set(c)!={'run_id','source_context','output_root','frozen_sources','code_commit'} or c['run_id']!=RUN):
        raise ValueError('Fixed source preparation without target handles required')
    from experiments.pamf_repl_completion.runner import activate
    activate(c)
    import numpy as np
    import torch
    from torch.nn import functional as F
    from experiments.cric_dura_comparison.f_data import SourceDataset
    from experiments.cric_dura_comparison.external_dura import fp32_copy
    from experiments.pamf_stage2_inference.engine import InferenceEngine
    from experiments.pamf_restored_pl.cache import SourceCache
    from .receiver_reference_source import student_from_parent
    from .restored_dose_source import load_sources,no_native_U
    from .feature_task_training import PairedCache
    torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED)
    start=time.perf_counter();root=Path(c['output_root']);root.mkdir(parents=True,exist_ok=False)
    context=c['source_context'];validate_context(context)
    teachers,configs,payloads,contract,g=load_sources(context,'cuda:0')
    student,cfg=student_from_parent(configs[0],payloads[0]);del payloads
    labeled=SourceDataset(cfg['source_cache'],'L_s',contract)
    unlabeled=SourceDataset(cfg['source_cache'],'U_s',contract)
    if len(labeled)!=6300 or len(unlabeled)!=56700:raise ValueError('Original source roles required')
    engine=InferenceEngine(g,reuse=True,graph=True,native=False)
    x=labeled.x[:2].to('cuda:0');student.eval()
    with torch.no_grad():
        gx=engine(x)[0].clone()
        out=student.forward_identity_only(x,domain_labels=torch.zeros(2,device='cuda:0',dtype=torch.long))
        if not torch.allclose(g(x),gx,atol=1e-5,rtol=1e-5) or not torch.allclose(out['z_id'],teachers[0].features(x),atol=1e-5,rtol=1e-5):
            raise ValueError('Actual same E30/default-head and deployed G source L2 smoke differs')
    q=F.one_hot(torch.zeros(2,device='cuda:0',dtype=torch.long),6).float()
    mask=torch.ones(2,device='cuda:0',dtype=torch.bool);domain=torch.zeros(2,device='cuda:0',dtype=torch.long)
    for shield in (False,True):
        loss=extra(student,gx,domain,q,mask,shield,True)
        if not bool(torch.isfinite(loss)):raise ValueError('Actual task-CE feature graph smoke not finite')
    write(root/'real_checkpoint_smoke.json',dict(status='PASS',role='L_s',physical_packets=2,actual_E30=True,default_head=True,
        recomputed_default_head=True,G_eager_graph_close=True,feature_graph_shield=True,optimizer_updates=0,target_reads=0,U_truth_read=False))
    before=snapshot(student);frozen=[snapshot(m) for m in teachers];gbefore=snapshot(g)
    oldroot=Path(context['project'])/'runs'/CACHE_RUN/'source_U_cache'
    old=read(oldroot/'complete.json')
    if old['G']!=context['G'] or old['source_cases']!=context['cases'] or old['source_namespace']!='PAMF/restored_dose/U/fixed':
        raise ValueError('Original frozen G/teachers and same cached channel namespace required')
    base=SourceCache(dict(cfg,source_cache_output=str(oldroot)))
    iq=np.lib.format.open_memmap(root/'channel_U.npy',mode='w+',dtype=np.float32,shape=(3,56700,2,256))
    offset=0;pair_errors=[]
    with torch.inference_mode():
        for batch in torch.utils.data.DataLoader(unlabeled,batch_size=256,shuffle=False,num_workers=0):
            if not bool((batch[1]==-1).all()) or bool(torch.as_tensor(batch[3]['tx_label_visible']).any()):
                raise ValueError('Only hidden source U is permitted')
            ux=batch[0].to('cuda:0');count=len(ux)
            for k,scene in enumerate(SCENES):
                h=source_channel(ux,batch[3],scene,contract,namespace='PAMF/restored_dose/U/fixed')
                if not bool(torch.isfinite(h).all()):raise ValueError('Finite physical source H pairs required')
                iq[k,offset:offset+count]=fp32_copy(h)
                if offset==0:
                    small={key:value[:2] for key,value in batch[3].items()}
                    _,_,cached=base.take(small,k,'cuda:0',True)
                    actual=engine(h[:2])[0].clone();pair_errors.append(float((actual-cached).abs().max()))
                    if not torch.allclose(actual,cached,atol=1e-5,rtol=1e-5):raise ValueError('New H does not reproduce the existing paired G')
            offset+=count
            if offset%2560==0 or offset==56700:
                write(root/'progress.json',dict(status='SOURCE_H_PAIR_CACHE',U_seen=offset,total=56700,optimizer_updates=0,V_reads=0,target_reads=0),exclusive=False)
                print('SOURCE_H_PAIR_CACHE',offset,56700,flush=True)
    iq.flush();del iq
    if offset!=56700 or not unchanged(g,gbefore) or any(not unchanged(m,b) for m,b in zip(teachers,frozen)):
        raise ValueError('Source H cache incomplete or frozen teacher/G changed')
    write(root/'paired_cache.json',dict(status='SOURCE_HG_CACHE_COMPLETE',source_contract=contract,source_cases=context['cases'],G=context['G'],
        U_seen=56700,namespace='PAMF/restored_dose/U/fixed',scenes=list(SCENES),cached_G_pair_check_pass=True,
        cached_G_pair_max_absolute_errors=pair_errors,U_truth_read=False,V_reads=0,target_reads=0,optimizer_updates=0,
        new_G_packet_forwards=6,teacher_packet_forwards=0,code_commit=c['code_commit']))
    cache=PairedCache(dict(cfg,source_G=context['G'],source_cache_output=str(oldroot),feature_task_output=str(root)))
    student.train();named=[(name,p) for name,p in student.named_parameters() if p.requires_grad];params=[p for _,p in named]
    sinc=student.get_submodule('id_backbone.encoder.core.id_backbone.sinc')
    if sinc.scale_hz!=25000. or sinc.mode!='CENTER_25K':raise ValueError('Original Sinc physical coordinates required')
    def generator():return torch.Generator().manual_seed(SEED)
    ll=iter(torch.utils.data.DataLoader(labeled,batch_size=128,shuffle=True,generator=generator(),num_workers=0,drop_last=True))
    ul=iter(torch.utils.data.DataLoader(unlabeled,batch_size=256,shuffle=True,generator=generator(),num_workers=0,drop_last=True))
    def gradient(loss):
        values=torch.autograd.grad(loss,params,allow_unused=True);parts=[];heads=[]
        for (name,p),v in zip(named,values):
            value=torch.zeros_like(p) if v is None else v
            # The deployed classifier is normalized cls_head.weight. The
            # head's base/PA norms and gain belong to feature fusion before z.
            if name.endswith('.cls_head.weight'):heads.append(value.detach().reshape(-1))
            if name in ('id_backbone.encoder.core.id_backbone.sinc.low_coordinate','id_backbone.encoder.core.id_backbone.sinc.band_coordinate'):
                value=value/25000.
            parts.append(value.reshape(-1))
        result=torch.cat(parts).detach()
        if not bool(torch.isfinite(result).all()):raise ValueError('Finite whole-parameter source gradient required')
        return result,float(torch.cat(heads).norm())
    def restore(state):
        with torch.no_grad():
            for name,value in student.named_buffers():value.copy_(state[name])
    rows=[]
    with no_native_U():
        for step in range(1,BATCHES+1):
            lb,ub=next(ll),next(ul);lx,ly,ld=[v.to('cuda:0') for v in lb[:3]]
            ux=ub[0].to('cuda:0');ud=ub[2].to('cuda:0');rx=ub[3]['rx_i'].to('cuda:0')
            if not bool((ub[1]==-1).all()) or bool(torch.as_tensor(ub[3]['tx_label_visible']).any()):raise ValueError('Hidden source U required')
            q,accepted,hx,gx=cache.take(ub[3],(step-1)%3,'cuda:0');accepted=accepted&(rx!=1)
            student.train()
            with student._phase1_labeled_feature_context():output=student(lx,return_aux=True,domain_labels=ld)
            ce=F.cross_entropy(output['tx_logits'],ly,label_smoothing=.01)
            correction,metrics=student._source_labeled_feature_hook(model=student,labeled_output=output,labels=ly,
                extra=(None,lb[3]),domains=ld,optimizer_step=step,identity_loss=ce,labeled_count=128,labeled_inputs=lx)
            gl,hl=gradient(ce+correction);student.eval()
            with torch.no_grad():confidence,label=student(ux).softmax(1).max(1)
            student.train();logits=student.forward_identity_only(ux,domain_labels=ud)['tx_logits']
            gu,hu=gradient((F.cross_entropy(logits,label,reduction='none')*(confidence>=.95)).mean());primary=gl+gu
            row=dict(step=step,scene=SCENES[(step-1)%3],L_seen=128,U_seen=256,extra_accepted=int(accepted.sum()),held_extra_RX=1,
                primary_norm=float(primary.norm()),L_norm=float(gl.norm()),raw_U_norm=float(gu.norm()),
                primary_L_head_norm=hl,primary_U_head_norm=hu,L_forward_packets=int(metrics['T1_L_packets']),As_KL_weight=float(metrics['T1_As_KL_weight']))
            temporary={name:value.detach().clone() for name,value in student.named_buffers()}
            cpu_rng=torch.random.get_rng_state();cuda_rng=torch.cuda.get_rng_state()
            vectors={}
            for method in VARIANTS[:3]:
                restore(temporary);torch.random.set_rng_state(cpu_rng);torch.cuda.set_rng_state(cuda_rng)
                view=hx if method=='HFeatureHard' else gx;shield=method!='GJointHard'
                loss=sum(extra(student,view,ud,q,accepted,shield,True) for _ in range(2))/2
                ge,head=gradient(loss);vectors[method]=ge
                if (shield and head!=0.) or (not shield and head<=0.):raise ValueError('Auxiliary head-gradient routing differs')
                row.update({method+'_norm':float(ge.norm()),method+'_cosine':float(F.cosine_similarity(primary[None],ge[None],dim=1)[0]),
                    method+'_loss':float(loss.detach()),method+'_classifier_weight_gradient_norm':head})
            if row['L_forward_packets']!=768 or row['As_KL_weight']!=0 or hl<=0:raise ValueError('Original L/default-head primary differs')
            rows.append(row);restore(before)
            write(root/'progress.json',dict(status='SOURCE_FEATURE_TASK_GRADIENTS',batches=step,total=BATCHES,optimizer_updates=0,V_reads=0,target_reads=0),exclusive=False)
            print('SOURCE_FEATURE_TASK_GRADIENTS',step,BATCHES,flush=True)
    current=dict(student.state_dict());current.update(dict(student.named_buffers()))
    if (set(current)!=set(before) or any(not torch.equal(value,before[name]) for name,value in current.items())
            or any(p.grad is not None for p in student.parameters()) or not unchanged(g,gbefore)
            or any(not unchanged(m,b) for m,b in zip(teachers,frozen))):
        raise ValueError('Source preparation retained model/gradient changes')
    result=dict(status='SOURCE_FEATURE_TASK_PREPARED',code_commit=c['code_commit'],model_seed=SEED,rows=rows,decision=decide(rows),
        gradient_rule=GRADIENT_RULE,variants=list(VARIANTS),L_exposures=1536,U_calibration_exposures=3072,U_cache_records=56700,
        full_parameter_gradient_calls=60,all_auxiliary_classifier_weight_gradients_zero=True,
        student_teacher_G_parameters_and_registered_buffers_unchanged=True,temporary_buffers_restored=True,
        optimizer_updates=0,V_reads=0,target_reads=0,U_truth_read=False,G_updates=0,teacher_updates=0,head_fit_updates=0,
        seconds=time.perf_counter()-start,scope='Source IQ pairing and no-update task-gradient calibration;not target performance')
    write(root/'complete.json',result)
    print('SOURCE_FEATURE_TASK_PREPARED',result['seconds'],result['decision']['weights'],flush=True)
