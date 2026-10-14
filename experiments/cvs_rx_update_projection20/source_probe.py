"""Source training packets only, temporary optimizer/model copies only."""
import copy
import json
import os
import sys
import time
from pathlib import Path
import torch
from torch.nn import functional as F
from . import design as d
from .projection import project
from .gradient_control import feature_scope,gradient_cone
from .materialize import protected_names,fixed_logits,evaluate,actual_displacement
from experiments.cvs_evidence_recomposition20.capture import forward
from experiments.cvs_sigtime_heldcredit.held_credit import physical_ids
from experiments.cvs_sigtime_heldcredit.source_probe import kernel_changes,norm
from experiments.cvs_u_schedule50.counterfactual import simulate,_equal,_rng_snapshot


def difference(a,b):return tuple(x-y for x,y in zip(a,b))


def dot(a,b):return sum(float((x.double()*y.double()).sum()) for x,y in zip(a,b) if x is not None and y is not None)


def evaluate_split(model,names,after,bank,protected):
    z,overall=evaluate(model,names,after,bank,protected)
    _,labels,receivers=bank
    head=model.id_backbone.encoder.core.id_backbone.cls_head
    with torch.no_grad():
        scores=30.*F.linear(F.normalize(z.double(),dim=1,eps=1e-4),F.normalize(head.weight.detach().double(),dim=1,eps=1e-4))
        losses=F.cross_entropy(scores,labels,reduction='none')
    if len(z)!=360:raise ValueError('Fixed risk180+independentheld180 required')
    result=dict(all=overall)
    for label,part in [('risk',slice(0,180)),('held',slice(180,360))]:
        ls,rr=losses[part],receivers[part]
        if any(int((rr==r).sum())!=36 for r in d.SOURCE_RX):raise ValueError('Fixed L36/RX evaluation required')
        per={str(r):float(ls[rr==r].mean()) for r in d.SOURCE_RX}
        result[label]=dict(mean_CE=float(ls.mean()),by_RX=per,worst_RX_CE=max(per.values()))
    return z,result


def banks(dataset):
    if len(dataset)!=6300 or getattr(dataset,'hide',True):raise ValueError('Visible source L6300 required')
    metadata=[];cells={}
    for i,it in enumerate(dataset.index):
        key=(int(it.tx_i),int(it.rx_i),int(it.day_i))
        if key[0] not in range(6) or key[1] not in d.SOURCE_RX or key[2] not in (1,2,3):
            raise ValueError('Source L TX/RX/day differs')
        pid=(int(it.eq_i),int(it.rx_i),int(it.day_i),int(it.sig_i),int(dataset.selected[i]))
        metadata.append(dict(y=key[0],rx=key[1],id=pid));cells.setdefault(key,[]).append(i)
    physical_ids([m['id'] for m in metadata],6300)
    if len(cells)!=90 or any(len(v)!=70 for v in cells.values()):raise ValueError('Original 90x70 L cells differ')
    keys=sorted(cells)
    groups={str(o):[cells[k][o] for k in keys] for o in d.ORDINALS}
    groups.update(risk=[cells[k][o] for k in keys for o in d.RISK_ORDINALS],
                  held=[cells[k][o] for k in keys for o in d.HELD_ORDINALS])
    selected=[i for indices in groups.values() for i in indices]
    if len(selected)!=720 or len(set(selected))!=720:raise ValueError('Current/risk/held L packets must be physically disjoint')
    result={}
    for name,indices in groups.items():
        xs=[];ys=[];rxs=[];ids=[]
        for i in indices:
            x,y,_,meta=dataset[i];pid=tuple(int(meta[k]) for k in ('eq_i','rx_i','day_i','sig_i','base_index'))
            if pid!=metadata[i]['id'] or int(y)!=metadata[i]['y']:raise ValueError('Actual L IQ/labels/physical metadata differ')
            xs.append(x);ys.append(int(y));rxs.append(metadata[i]['rx']);ids.append(pid)
        physical_ids(ids,len(indices))
        result[name]=(torch.stack(xs),torch.tensor(ys),torch.tensor(rxs),ids)
    return result


def changes(model,names,raw,other):
    shallow_tokens=('.sinc.','.time_fuse.','.t1.','.t2.','.sequence_pool.')
    protected=protected_names(model)
    changed=[n for n,a,b in zip(names,raw,other) if n.startswith('id_backbone.encoder.') and n not in protected and not torch.equal(a,b)]
    return dict(changed_feature_tensors=len(changed),shallow_middle_tensors=sum(any(t in n for t in shallow_tokens) for n in changed),
        layer_delta_norm={label:norm([a.double()-b.double() for n,a,b in zip(names,other,raw) if token in n and n not in protected])
            for label,token in [('Sinc','.sinc.'),('time_fuse','.time_fuse.'),('t1','.t1.'),('t2','.t2.'),
                                 ('SIG_TIME','.sequence_pool.'),('PA','pa_'),('reference','.response_projection.')]},
        protected_W_gain_delta_from_RAW={n:float((a.double()-b.double()).norm()) for n,a,b in zip(names,other,raw) if n in protected})


def run(request):
    d.validate(request);out=Path(d.OUTPUT)
    if out.exists():raise FileExistsError('Preserve existing source assay, no relaunch')
    out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(2);started=time.perf_counter()
    runtime=dict(pid=os.getpid(),cwd=os.getcwd(),argv=sys.argv,device='cpu',GPU_allocations=0)
    d.write(out/'runtime.json',runtime);print('RUNTIME '+json.dumps(runtime),flush=True)
    from experiments.cvs_crossrx_sharpness20 import design as parent,model as api
    from experiments.cvs_crossrx_sharpness20.provenance import source_proof,validate_checkpoint
    from experiments.cvs_phase1_stack.runtime import installed
    from experiments.cvs_phase1_stack.fast_execution import configure
    c=parent.config(next(r for r in parent.rows() if r['row_id']==request['source_row']))
    source=source_proof(c) # Exact roles/ancestry/final selection before either model deserialization.
    ck=torch.load(source/'final_ssdg.pth',map_location='cpu',weights_only=False);validate_checkpoint(ck,c)
    initial=torch.load(source/'initial_smoke.pt',map_location='cpu',weights_only=False)
    if set(initial)!={'model','scratch_only'} or initial['scratch_only'] is not True:
        raise ValueError('Own scratch initial checkpoint schema differs')
    model=api.build(c,torch.device('cpu')).eval();args=parent.make_args(c,'cpu')
    named=[(n,p) for n,p in model.named_parameters() if p.requires_grad]
    names=[n for n,p in named];parameters=[p for n,p in named]
    if len(parameters)!=188 or len(ck['optimizer']['param_groups'])!=1:raise ValueError('Original native optimizer traversal differs')
    scope=feature_scope(model,parameters,names);indices=scope['indices'];fnames=scope['names']
    protected=protected_names(model);head=api.classifier(model)
    wi=next(i for i,p in enumerate(parameters) if p is head.weight)
    for state,weights in [('INITIAL',initial['model']),('FINAL20',ck['model'])]:
        model.load_state_dict(weights,strict=True)
        with torch.no_grad():seen=forward(model,torch.zeros(2,2,256))
        if seen['logits'].shape!=(2,6):raise ValueError('Real-checkpoint no-query smoke failed')
        print('SMOKE PASS '+state+' strict own checkpoint/query_read=false',flush=True)
    with installed(c) as native:
        context=native._build_ssdg_wisig_data(configure(parent.make_args(c,'cpu')),torch.device('cpu'))
        from cvsrffi.xuc_fusion.native import role_ids_from_native
        if context['named_test_loaders'] or role_ids_from_native(context)!=parent.read(parent.SOURCE)['role_ids']:
            raise ValueError('Original source-only role handles differ')
    bank=banks(context['train_loader'].dataset)
    rx,ry,rr,rids=bank['risk'];hx,hy,hr,hids=bank['held']
    evaluation_bank=(torch.cat([rx,hx]),torch.cat([ry,hy]),torch.cat([rr,hr]))
    records=[];forwards=4;temporary_steps=0
    for state in d.STATES:
        model.load_state_dict(initial['model'] if state=='INITIAL' else ck['model'],strict=True)
        model.eval() # Do not unfreeze parameters not in the original optimizer traversal.
        opt=torch.optim.AdamW(parameters,lr=float(args.lr),weight_decay=float(args.weight_decay))
        if state=='FINAL20':opt.load_state_dict(ck['optimizer'])
        before=copy.deepcopy(model.state_dict());opt_before=copy.deepcopy(opt.state_dict());rng_before=_rng_snapshot()
        pre=tuple(p.detach().clone() for p in parameters);preF=tuple(pre[i] for i in indices)
        observed=forward(model,rx);forwards+=180
        losses=F.cross_entropy(fixed_logits(model,observed),ry,reduction='none')
        risk={};risk_norm={}
        for receiver in d.SOURCE_RX:
            mask=rr==receiver
            if int(mask.sum())!=36:raise ValueError('Fixed risk L36 per RX required')
            full=torch.autograd.grad(losses[mask].mean(),parameters,retain_graph=True,allow_unused=True)
            if any(g is not None for n,g in zip(names,full) if n in protected):
                raise ValueError('Risk identity graph must detach W/all three gains')
            if any(g is not None for i,g in enumerate(full) if i not in indices):
                raise ValueError('Risk derivative escaped static deployed feature scope')
            risk[receiver]=tuple(None if full[i] is None else full[i].detach().double() for i in indices)
            risk_norm[str(receiver)]=norm(risk[receiver])
        del observed,losses
        # Evaluate two disjoint L sets once before any hypothetical step.
        z_before,eval_before=evaluate_split(model,names,pre,evaluation_bank,protected);forwards+=360
        for ordinal in d.ORDINALS:
            x,y,_,ids=bank[str(ordinal)];observed=forward(model,x);forwards+=90
            main=F.cross_entropy(observed['logits'],y)
            base=torch.autograd.grad(main,parameters,allow_unused=True)
            if base[wi] is None or norm([base[wi]])<=0:raise ValueError('Main CE must jointly train native W')
            control=gradient_cone(base,parameters,indices,risk)
            proposal=simulate(opt,base,control['gradients'],names=names,max_grad_norm=5.);temporary_steps+=2
            raw=proposal['base_after'];grad=proposal['total_after']
            rawF=tuple(raw[i] for i in indices);gradF=tuple(grad[i] for i in indices)
            raw_delta=tuple(a.double()-b.double() for a,b in zip(rawF,preF))
            projected=project(raw_delta,risk,role='L_s')
            rounded=actual_displacement(preF,projected['projected_delta'],risk)
            update=list(raw)
            for i,value in zip(indices,rounded['after_F']):update[i]=value
            update=tuple(update);actual_delta=rounded['actual_delta_FP64']
            modes={};zs={};risk_ok=True
            for mode,after in [('RAW',raw),('GRAD_CONE',grad),('UPDATE_CONE',update)]:
                z,evaluation=evaluate_split(model,names,after,evaluation_bank,protected);forwards+=360
                zs[mode]=z
                physical=kernel_changes(model,names,dict(base_after=raw,total_after=after))
                delta=tuple(after[i].double()-pre[i].double() for i in indices)
                modes[mode]=dict(evaluation=evaluation,actual_feature_displacement_norm=norm(delta),
                    directional_risk_by_RX={str(r):dot(risk[r],delta) for r in d.SOURCE_RX},
                    physical_Sinc=physical,changes_from_RAW=changes(model,names,raw,after),
                    pre_step_W_and_all_gains_fixed_in_evaluation=True)
            modes['UPDATE_CONE']['actual_displacement']=rounded['summary']
            modes['UPDATE_CONE']['projection']=projected['summary']
            modes['GRAD_CONE']['gradient_projection']=control['projection_summary']
            for receiver in d.SOURCE_RX:
                scalar=rounded['summary']['by_RX'][str(receiver)]
                bound=scalar['measured_rounding_error_bound']+projected['summary']['by_RX'][str(receiver)]['FP64_raw_risk_tolerance']
                risk_ok &= scalar['actual_directional_risk']<=bound
            held_before=eval_before['held'];raw_eval=modes['RAW']['evaluation']['held'];upd_eval=modes['UPDATE_CONE']['evaluation']['held']
            worst_delta=lambda value:max(value['by_RX'][str(r)]-held_before['by_RX'][str(r)] for r in d.SOURCE_RX)
            change=difference(actual_delta,raw_delta);grad_delta=tuple(a.double()-b.double() for a,b in zip(gradF,preF))
            raw_norm=norm(raw_delta)
            record=dict(state=state,ordinal=ordinal,current_L_CE=float(main.detach()),
                learning_rate=float(opt.param_groups[0]['lr']),optimizer_moments='fresh hypothetical initial' if state=='INITIAL' else 'actual saved final20',
                risk_gradient_norm_by_RX=risk_norm,modes=modes,pre_step_evaluation=eval_before,
                RAW_positive_directional_risk_RX=sum(dot(risk[r],raw_delta)>projected['summary']['by_RX'][str(r)]['FP64_raw_risk_tolerance'] for r in d.SOURCE_RX),
                UPDATE_RAW_relative_displacement_difference=norm(change)/max(raw_norm,1e-300),
                UPDATE_GRAD_relative_displacement_difference=norm(difference(actual_delta,grad_delta))/max(raw_norm,1e-300),
                UPDATE_retained_actual_displacement_ratio=norm(actual_delta)/raw_norm if raw_norm else 1.,
                UPDATE_RAW_held_worst_RX_CE_delta_reduction=worst_delta(raw_eval)-worst_delta(upd_eval),
                UPDATE_RAW_held_mean_CE_increase=upd_eval['mean_CE']-raw_eval['mean_CE'],
                UPDATE_RAW_held_feature_RMS=float((zs['UPDATE_CONE'][180:].double()-zs['RAW'][180:].double()).square().mean().sqrt()),
                shallow_middle_tensors_changed_from_RAW=modes['UPDATE_CONE']['changes_from_RAW']['shallow_middle_tensors'],
                actual_risk_within_solver_and_rounding_bound=bool(risk_ok),
                temporary_AdamW_global_summary={k:v for k,v in proposal['summary'].items() if k!='groups'})
            records.append(record);d.write(out/(state+'-ordinal'+str(ordinal)+'.json'),record)
            print('SOURCE_CELL '+json.dumps({k:v for k,v in record.items() if k not in ('modes','pre_step_evaluation','temporary_AdamW_global_summary')}),flush=True)
        if (not _equal(before,model.state_dict()) or not _equal(opt_before,opt.state_dict())
            or any(p.grad is not None for p in model.parameters()) or not _equal(rng_before,_rng_snapshot())):
            raise ValueError('Source assay changed live model/optimizer/gradients/RNG')
    gate=d.source_gate(records)
    if forwards!=10444 or temporary_steps!=16:raise ValueError('Fixed source diagnostic computation counts differ')
    summary=dict(status='SOURCE_SCREEN_PASSED' if gate['passed'] else 'SOURCE_SCREEN_NOT_PASSED',
        run_id=d.RUN,analysis_code_commit=request['analysis_code_commit'],gate_result=gate,source_falsification=d.GATE,
        checkpoint_source=dict(run_id=parent.RUN,row_id=c['row_id'],seed=2026092701,epoch=20,
            source_roles='EXACT_MATCH',ancestors=[],selection='Preregistered BASE_CE control firstseed own initial/final20; no target ranking'),
        feature_scope=scope['summary'],source_L_IQ_reads=720,source_metadata_packets=6300,
        current_L_unique=360,risk_L_unique=180,independent_held_L_unique=180,
        network_IQ_forwards=forwards,temporary_AdamW_steps=temporary_steps,
        actual_optimizer_updates=0,head_fit_updates=0,EMA_updates=0,U_IQ_reads=0,V_IQ_reads=0,
        U_truth_read=False,target_access=False,default_head_main_joint=True,
        live_parameters_buffers_optimizer_gradients_RNG_unchanged=True,
        current_risk_held_physical_ID_disjoint=True,previously_trained_source_L=True,
        CPU_counterfactual_not_contemporaneous_CUDA=True,runtime=runtime,records=records,
        wall_seconds=time.perf_counter()-started,
        limitation='Fixed eval-mode L-only two-state virtual diagnosis; INITIAL explicitly fresh moments, FINAL20 actual saved moments. No mid-state, actual training, unseenRX efficacy, target feedback or novelty proof; finite step observations do not inherit first-order guarantees.')
    d.write(out/'summary.json',summary);print('SOURCE_COMPLETE '+json.dumps(gate),flush=True)
    return summary
