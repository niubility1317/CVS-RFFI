"""No-update source calibration of paired restoration identity relations."""
import math
from pathlib import Path
import statistics
import time
from experiments.cric_dura_comparison.io import read,write
from .asymmetric_hg_training import RUN,CACHE_RUN,PAIR_CACHE_RUN,SEED,VARIANTS,GRADIENT_RULE,inputs
from .asymmetric_hg import extra
from .restored_source import validate_context,snapshot,unchanged,SCENES

BATCHES=12


def decide(rows):
    if len(rows)!=BATCHES:raise ValueError('Twelve complete matched source batches required')
    values=[r[k] for r in rows for k in ['primary_norm']+[v+'_norm' for v in VARIANTS]]
    if any(not math.isfinite(v) or v<=0 for v in values):raise ValueError('Finite positive full-parameter gradients required')
    primary=statistics.median(r['primary_norm'] for r in rows)
    medians={v:statistics.median(r[v+'_norm'] for r in rows) for v in VARIANTS}
    uncapped=.1*primary/max(medians.values());weight=min(.25,uncapped)
    return dict(weight=weight,median_primary_norm=primary,median_extra_norm=medians,dose_upper_cap_active=uncapped>.25,
        weighted_median_norm_ratio={v:weight*medians[v]/primary for v in VARIANTS},
        median_gradient_cosine={v:statistics.median(r[v+'_cosine'] for r in rows) for v in VARIANTS},
        gradient_rule=GRADIENT_RULE,scope='Initial preclip CENTER_UNITS gradient proxy;not AdamW or target guarantee')


def prepare_source(c):
    if set(c)!={'run_id','source_context','output_root','frozen_sources','code_commit'} or c['run_id']!=RUN:
        raise ValueError('Fixed source-only relation calibration required')
    from experiments.pamf_repl_completion.runner import activate
    activate(c)
    import torch
    from torch.nn import functional as F
    from experiments.cric_dura_comparison.f_data import SourceDataset
    from experiments.pamf_stage2_inference.engine import InferenceEngine
    from .receiver_reference_source import student_from_parent
    from .restored_dose_source import load_sources,no_native_U
    from .feature_task_training import PairedCache
    torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED)
    start=time.perf_counter();root=Path(c['output_root']);root.mkdir(parents=True,exist_ok=False)
    context=c['source_context'];validate_context(context)
    teachers,configs,payloads,contract,g=load_sources(context,'cuda:0')
    student,cfg=student_from_parent(configs[0],payloads[0]);del payloads
    labeled=SourceDataset(cfg['source_cache'],'L_s',contract);unlabeled=SourceDataset(cfg['source_cache'],'U_s',contract)
    if len(labeled)!=6300 or len(unlabeled)!=56700:raise ValueError('Original source roles required')
    cache_root=Path(context['project'])/'runs'/CACHE_RUN/'source_U_cache'
    header=read(cache_root/'complete.json')
    if header['G']!=context['G'] or header['source_cases']!=context['cases']:
        raise ValueError('Original aligned frozen G/teachers required')
    cache=PairedCache(dict(cfg,source_G=context['G'],source_cache_output=str(cache_root),
        feature_task_output=context['project']+'/runs/'+PAIR_CACHE_RUN+'/source_preparation'))
    engine=InferenceEngine(g,reuse=True,graph=True,native=False)
    x=labeled.x[:2].to('cuda:0');student.eval()
    with torch.no_grad():
        gx=engine(x)[0].clone();out=student.forward_identity_only(x,domain_labels=torch.zeros(2,device='cuda:0',dtype=torch.long))
        if not torch.allclose(g(x),gx,atol=1e-5,rtol=1e-5) or not torch.allclose(out['z_id'],teachers[0].features(x),atol=1e-5,rtol=1e-5):
            raise ValueError('Actual E30/default-head/G source L2 smoke differs')
    write(root/'real_checkpoint_smoke.json',dict(status='PASS',role='L_s',physical_packets=2,actual_E30=True,default_head=True,
        G_eager_graph_close=True,optimizer_updates=0,target_reads=0,U_truth_read=False))
    before=snapshot(student);frozen=[snapshot(m) for m in teachers];gbefore=snapshot(g)
    student.train();named=[(name,p) for name,p in student.named_parameters() if p.requires_grad];params=[p for _,p in named]
    sinc=student.get_submodule('id_backbone.encoder.core.id_backbone.sinc')
    if sinc.scale_hz!=25000. or sinc.mode!='CENTER_25K':raise ValueError('Original Sinc physical coordinates required')
    def generator():return torch.Generator().manual_seed(SEED)
    ll=iter(torch.utils.data.DataLoader(labeled,batch_size=128,shuffle=True,generator=generator(),num_workers=0,drop_last=True))
    ul=iter(torch.utils.data.DataLoader(unlabeled,batch_size=256,shuffle=True,generator=generator(),num_workers=0,drop_last=True))
    def gradient(loss):
        grads=torch.autograd.grad(loss,params,allow_unused=True);parts=[];heads=[]
        for (name,p),v in zip(named,grads):
            value=torch.zeros_like(p) if v is None else v
            if name.endswith('.cls_head.weight'):heads.append(value.detach().reshape(-1))
            if name in ('id_backbone.encoder.core.id_backbone.sinc.low_coordinate','id_backbone.encoder.core.id_backbone.sinc.band_coordinate'):
                value=value/25000.
            parts.append(value.reshape(-1))
        result=torch.cat(parts).detach()
        if not bool(torch.isfinite(result).all()):raise ValueError('Finite source parameter gradients required')
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
            student.train()
            with student._phase1_labeled_feature_context():output=student(lx,return_aux=True,domain_labels=ld)
            ce=F.cross_entropy(output['tx_logits'],ly,label_smoothing=.01)
            correction,metrics=student._source_labeled_feature_hook(model=student,labeled_output=output,labels=ly,
                extra=(None,lb[3]),domains=ld,optimizer_step=step,identity_loss=ce,labeled_count=128,labeled_inputs=lx)
            gl,hl=gradient(ce+correction);student.eval()
            with torch.no_grad():confidence,label=student(ux).softmax(1).max(1)
            student.train();logits=student.forward_identity_only(ux,domain_labels=ud)['tx_logits']
            gu,hu=gradient((F.cross_entropy(logits,label,reduction='none')*(confidence>=.95)).mean());primary=gl+gu
            row=dict(step=step,scene=SCENES[(step-1)%3],L_seen=128,U_seen=256,held_extra_RX=1,
                primary_norm=float(primary.norm()),L_norm=float(gl.norm()),raw_U_norm=float(gu.norm()),
                primary_L_head_norm=hl,primary_U_head_norm=hu,L_forward_packets=int(metrics['T1_L_packets']),As_KL_weight=float(metrics['T1_As_KL_weight']))
            temporary={name:value.detach().clone() for name,value in student.named_buffers()}
            cpu_rng=torch.random.get_rng_state();cuda_rng=torch.cuda.get_rng_state();paired_counts=None
            for method in VARIANTS:
                restore(temporary);torch.random.set_rng_state(cpu_rng);torch.cuda.set_rng_state(cuda_rng)
                q,mask,views=inputs(cache,ub[3],(step-1)%3,'cuda:0',ux,method);mask=mask&(rx!=1)
                loss,detail=extra(student,views,ud,q,mask,rx,method,True);ge,head=gradient(loss)
                if head!=0. or detail['valid_anchors']<=0:raise ValueError('Actual relation feature gradients and head exclusion required')
                counts={k:detail[k] for k in ('accepted_packets','valid_anchors','positive_pairs','negative_pairs')}
                if paired_counts is None:paired_counts=counts
                elif counts!=paired_counts:raise ValueError('Relation routes must have identical pair membership counts')
                row.update({method+'_norm':float(ge.norm()),method+'_cosine':float(F.cosine_similarity(primary[None],ge[None],dim=1)[0]),
                    method+'_loss':float(loss.detach()),method+'_CE_loss':float(detail['CE_loss']),method+'_CL_loss':float(detail['relation_loss']),
                    method+'_H_feature_gradient_norm':detail['H_feature_gradient_norm'],method+'_G_feature_gradient_norm':detail['G_feature_gradient_norm'],method+'_classifier_weight_gradient_norm':head})
            for key in ('loss','CE_loss','CL_loss'):
                if not math.isclose(row['HJoint_'+key],row['HAnchor_'+key],abs_tol=1e-7,rel_tol=1e-7):raise ValueError('HJoint/HAnchor must have identical values before routing')
            row.update(paired_counts)
            if row['L_forward_packets']!=768 or row['As_KL_weight']!=0 or hl<=0:raise ValueError('Original L/default-head primary differs')
            rows.append(row);restore(before)
            write(root/'progress.json',dict(status='SOURCE_ASYMMETRIC_HG_GRADIENTS',batches=step,total=BATCHES,optimizer_updates=0,V_reads=0,target_reads=0),exclusive=False)
            print('SOURCE_ASYMMETRIC_HG_GRADIENTS',step,BATCHES,flush=True)
    current=dict(student.state_dict());current.update(dict(student.named_buffers()))
    if (set(current)!=set(before) or any(not torch.equal(value,before[name]) for name,value in current.items())
            or any(p.grad is not None for p in student.parameters()) or not unchanged(g,gbefore)
            or any(not unchanged(m,b) for m,b in zip(teachers,frozen))):
        raise ValueError('Source relation calibration retained model or gradient changes')
    result=dict(status='SOURCE_ASYMMETRIC_HG_PREPARED',code_commit=c['code_commit'],model_seed=SEED,rows=rows,decision=decide(rows),
        gradient_rule=GRADIENT_RULE,variants=list(VARIANTS),L_exposures=1536,U_calibration_exposures=3072,full_parameter_gradient_calls=72,
        all_auxiliary_classifier_weight_gradients_zero=True,all_routes_pair_counts_matched=True,HJoint_HAnchor_values_match=True,actual_feature_gradient_routes_verified=True,
        student_teacher_G_parameters_and_registered_buffers_unchanged=True,temporary_buffers_restored=True,
        optimizer_updates=0,V_reads=0,target_reads=0,U_truth_read=False,G_updates=0,teacher_updates=0,head_fit_updates=0,
        seconds=time.perf_counter()-start,scope='No-update source identity-relation calibration;not target performance')
    write(root/'complete.json',result)
    print('SOURCE_ASYMMETRIC_HG_PREPARED',result['seconds'],result['decision']['weight'],flush=True)
