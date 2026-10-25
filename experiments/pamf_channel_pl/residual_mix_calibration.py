"""No-update source calibration for same-packet restoration-residual views."""
import math
from pathlib import Path
import statistics
import time

from experiments.cric_dura_comparison.io import read,write
from .residual_mix_training import RUN,CACHE_RUN,BASE_DOSE_RUN,BASE_WEIGHT,SEED,MIX_SEED,GRADIENT_RULE,extra
from .residual_mix import mix_iq,paired_views
from .restored_source import validate_context,snapshot,unchanged,SCENES

BATCHES=12
ROUTES=('MixMix','MixG','GEndpoint')


def decide(rows):
    if len(rows)!=BATCHES:raise ValueError('Twelve complete matched source batches required')
    numbers=[r[k] for r in rows for k in ['primary_norm']+[v+'_norm' for v in ROUTES]]
    if any(not math.isfinite(v) or v<=0 for v in numbers):raise ValueError('Finite positive physical source gradient norms required')
    primary=statistics.median(r['primary_norm'] for r in rows)
    medians={v:statistics.median(r[v+'_norm'] for r in rows) for v in ROUTES}
    weights={v:min(.25,.1*primary/medians[v]) for v in ROUTES[:2]}
    return dict(weights=weights,base_weight=BASE_WEIGHT,median_primary_norm=primary,median_extra_norm=medians,
        weighted_median_norm_ratio={v:weights[v]*medians[v]/primary for v in ROUTES[:2]},
        base_weighted_median_norm_ratio={v:BASE_WEIGHT*medians[v]/primary for v in ROUTES},
        dose_upper_cap_active={v:.1*primary/medians[v]>.25 for v in ROUTES[:2]},
        median_gradient_cosine={v:statistics.median(r[v+'_cosine'] for r in rows) for v in ROUTES},
        gradient_rule=GRADIENT_RULE,scope='preclip CENTER_UNITS physical gradient proxy;not AdamW or target guarantee')


def prepare_source(c):
    if set(c)!={'run_id','source_context','output_root','frozen_sources','code_commit'} or c['run_id']!=RUN:
        raise ValueError('Fixed source-only residual-view preparation required')
    from experiments.pamf_repl_completion.runner import activate
    activate(c)
    import torch
    from torch.nn import functional as F
    from experiments.cric_dura_comparison.f_data import SourceDataset
    from experiments.pamf_stage2_inference.engine import InferenceEngine
    from experiments.pamf_restored_pl.cache import SourceCache
    from .receiver_reference import metadata_long
    from .receiver_reference_source import student_from_parent
    from .restored_dose_source import load_sources,no_native_U
    torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED)
    start=time.perf_counter();root=Path(c['output_root']);root.mkdir(parents=True,exist_ok=False)
    context=c['source_context'];validate_context(context)
    teachers,configs,payloads,contract,g=load_sources(context,'cuda:0')
    student,cfg=student_from_parent(configs[0],payloads[0]);del payloads
    labeled=SourceDataset(cfg['source_cache'],'L_s',contract);unlabeled=SourceDataset(cfg['source_cache'],'U_s',contract)
    if len(labeled)!=6300 or len(unlabeled)!=56700:raise ValueError('Original source roles required')
    cache_root=Path(context['project'])/'runs'/CACHE_RUN/'source_U_cache';header=read(cache_root/'complete.json')
    base=read(Path(context['project'])/'runs'/BASE_DOSE_RUN/'source_preparation/complete.json')
    if (header['G']!=context['G'] or header['source_cases']!=context['cases'] or header['source_namespace']!='PAMF/restored_dose/U/fixed'
            or base['status']!='SOURCE_FEATURE_TASK_PREPARED' or base['decision']['weights']['G']!=BASE_WEIGHT
            or base['model_seed']!=SEED or any(base[k] for k in ('optimizer_updates','V_reads','target_reads','U_truth_read'))):
        raise ValueError('Same frozen G/teachers, cached pairs and prior source coefficient required')
    cache=SourceCache(dict(cfg,source_cache_output=str(cache_root)))
    engine=InferenceEngine(g,reuse=True,graph=True,native=False)
    x=labeled.x[:2].to('cuda:0');student.eval()
    with torch.no_grad():
        gx=engine(x)[0].clone();out=student.forward_identity_only(x,domain_labels=torch.zeros(2,device='cuda:0',dtype=torch.long))
        if (not torch.allclose(g(x),gx,atol=1e-5,rtol=1e-5) or not torch.allclose(out['z_id'],teachers[0].features(x),atol=1e-5,rtol=1e-5)
                or not torch.equal(mix_iq(x,gx,torch.zeros(2,device='cuda:0')),x)
                or not torch.equal(mix_iq(x,gx,torch.ones(2,device='cuda:0')),gx)):
            raise ValueError('Actual E30/default-head/G and exact interpolation endpoints differ')
    q=F.one_hot(metadata_long(labeled.metadata['y'][:2]).to('cuda:0'),6).float()
    mask=torch.ones(2,device='cuda:0',dtype=torch.bool);domain=torch.zeros(2,device='cuda:0',dtype=torch.long)
    for route in ROUTES[:2]:
        generator=torch.Generator(device='cuda:0').manual_seed(MIX_SEED)
        views,_=paired_views(x,gx,route,generator)
        for view in views:
            loss=extra(student,view,domain,q,mask,True,True)
            if not bool(torch.isfinite(loss)):raise ValueError('Actual residual-view feature graph is not finite')
    write(root/'real_checkpoint_smoke.json',dict(status='PASS',role='L_s',physical_packets=2,actual_E30=True,default_head=True,
        exact_raw_G_endpoints=True,recomputed_default_head=True,G_eager_graph_close=True,optimizer_updates=0,target_reads=0,U_truth_read=False))
    before=snapshot(student);frozen=[snapshot(m) for m in teachers];gbefore=snapshot(g)
    student.train();named=[(name,p) for name,p in student.named_parameters() if p.requires_grad];params=[p for _,p in named]
    sinc=student.get_submodule('id_backbone.encoder.core.id_backbone.sinc')
    if sinc.scale_hz!=25000. or sinc.mode!='CENTER_25K':raise ValueError('Original Sinc physical coordinates required')
    def generator():return torch.Generator().manual_seed(SEED)
    ll=iter(torch.utils.data.DataLoader(labeled,batch_size=128,shuffle=True,generator=generator(),num_workers=0,drop_last=True))
    ul=iter(torch.utils.data.DataLoader(unlabeled,batch_size=256,shuffle=True,generator=generator(),num_workers=0,drop_last=True))
    mix_generators={v:torch.Generator(device='cuda:0').manual_seed(MIX_SEED) for v in ROUTES[:2]}
    def gradient(loss):
        values=torch.autograd.grad(loss,params,allow_unused=True);parts=[];heads=[]
        for (name,p),v in zip(named,values):
            value=torch.zeros_like(p) if v is None else v
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
            q,accepted,gx=cache.take(ub[3],(step-1)%3,'cuda:0',True);accepted=accepted&(rx!=1)
            student.train()
            with student._phase1_labeled_feature_context():output=student(lx,return_aux=True,domain_labels=ld)
            ce=F.cross_entropy(output['tx_logits'],ly,label_smoothing=.01)
            correction,metrics=student._source_labeled_feature_hook(model=student,labeled_output=output,labels=ly,
                extra=(None,lb[3]),domains=ld,optimizer_step=step,identity_loss=ce,labeled_count=128,labeled_inputs=lx)
            gl,hl=gradient(ce+correction);student.eval()
            with torch.no_grad():confidence,label=student(ux).softmax(1).max(1)
            student.train();logits=student.forward_identity_only(ux,domain_labels=ud)['tx_logits']
            gu,hu=gradient((F.cross_entropy(logits,label,reduction='none')*(confidence>=.95)).mean());primary=gl+gu
            flat=ux.flatten(1);gf=gx.flatten(1);raw_norm=flat.norm(dim=1).clamp_min(1e-12)
            realcos=(flat*gf).sum(1)/(raw_norm*gf.norm(dim=1).clamp_min(1e-12))
            row=dict(step=step,scene=SCENES[(step-1)%3],L_seen=128,U_seen=256,extra_accepted=int(accepted.sum()),held_extra_RX=1,
                primary_norm=float(primary.norm()),L_norm=float(gl.norm()),raw_U_norm=float(gu.norm()),
                primary_L_head_norm=hl,primary_U_head_norm=hu,L_forward_packets=int(metrics['T1_L_packets']),As_KL_weight=float(metrics['T1_As_KL_weight']),
                raw_G_real_cosine_min=float(realcos.min()),raw_G_real_cosine_mean=float(realcos.mean()),
                raw_G_relative_distortion_mean=float(((gf-flat).norm(dim=1)/raw_norm).mean()))
            temporary={name:value.detach().clone() for name,value in student.named_buffers()}
            cpu_rng=torch.random.get_rng_state();cuda_rng=torch.cuda.get_rng_state();first_alpha=None
            for route in ROUTES:
                restore(temporary);torch.random.set_rng_state(cpu_rng);torch.cuda.set_rng_state(cuda_rng)
                if route=='GEndpoint':views=(gx,gx);alphas=()
                else:
                    before_cpu=torch.random.get_rng_state();before_cuda=torch.cuda.get_rng_state()
                    views,alphas=paired_views(ux,gx,route,mix_generators[route])
                    if not torch.equal(before_cpu,torch.random.get_rng_state()) or not torch.equal(before_cuda,torch.cuda.get_rng_state()):
                        raise ValueError('Interpolation changed native CPU/CUDA random state')
                    if first_alpha is None:first_alpha=alphas[0].clone()
                    elif not torch.equal(first_alpha,alphas[0]):raise ValueError('Private first-view random streams differ')
                loss=sum(extra(student,view,ud,q,accepted,True,True) for view in views)/2
                ge,head=gradient(loss)
                if head!=0.:raise ValueError('Auxiliary classification weight gradient must be zero')
                row.update({route+'_norm':float(ge.norm()),route+'_cosine':float(F.cosine_similarity(primary[None],ge[None],dim=1)[0]),
                    route+'_loss':float(loss.detach()),route+'_classifier_weight_gradient_norm':head})
                if alphas:
                    aa=torch.cat(alphas)
                    row.update({route+'_alpha_mean':float(aa.double().mean()),route+'_alpha_min':float(aa.min()),route+'_alpha_max':float(aa.max()),
                        route+'_mix_RMS_over_raw_min':min(float((view.flatten(1).norm(dim=1)/raw_norm).min()) for view in views[:len(alphas)])})
            if row['L_forward_packets']!=768 or row['As_KL_weight']!=0 or hl<=0:raise ValueError('Original supervised primary differs')
            rows.append(row);restore(before)
            write(root/'progress.json',dict(status='SOURCE_RESIDUAL_MIX_GRADIENTS',batches=step,total=BATCHES,optimizer_updates=0,V_reads=0,target_reads=0),exclusive=False)
            print('SOURCE_RESIDUAL_MIX_GRADIENTS',step,BATCHES,flush=True)
    current=dict(student.state_dict());current.update(dict(student.named_buffers()))
    if (set(current)!=set(before) or any(not torch.equal(value,before[name]) for name,value in current.items())
            or any(p.grad is not None for p in student.parameters()) or not unchanged(g,gbefore)
            or any(not unchanged(m,b) for m,b in zip(teachers,frozen))):
        raise ValueError('Source residual calibration retained model or gradient changes')
    result=dict(status='SOURCE_RESIDUAL_MIX_PREPARED',code_commit=c['code_commit'],model_seed=SEED,rows=rows,decision=decide(rows),
        gradient_rule=GRADIENT_RULE,routes=list(ROUTES),L_exposures=1536,U_calibration_exposures=3072,full_parameter_gradient_calls=60,
        all_auxiliary_classifier_weight_gradients_zero=True,private_RNG_routes_matched=True,native_RNG_unchanged_by_interpolation=True,
        student_teacher_G_parameters_and_registered_buffers_unchanged=True,temporary_buffers_restored=True,
        optimizer_updates=0,V_reads=0,target_reads=0,U_truth_read=False,G_updates=0,teacher_updates=0,head_fit_updates=0,
        seconds=time.perf_counter()-start,scope='No-update source residual-view calibration;not target performance')
    write(root/'complete.json',result)
    print('SOURCE_RESIDUAL_MIX_PREPARED',result['seconds'],result['decision']['weights'],flush=True)
