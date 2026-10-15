"""Cold-snapshot RX-blocked one-step attribution with no persistent updates."""
import copy,json,os,time
from pathlib import Path
import torch
from torch.nn import functional as F
from . import design as d
from .objective import subset,losses,matched_value,validate_bank
from experiments.cvs_evidence_recomposition20.capture import forward
from experiments.cvs_evidence_recomposition20.source_probe import total_gradient,protected_names
from experiments.cvs_identity_correspondence20.bank import banks
from experiments.cvs_sigtime_heldcredit.source_probe import norm
from experiments.cvs_u_schedule50.counterfactual import simulate,_equal,_rng_snapshot,_preserve_rng

def dot(a,b):return sum(float((x.double()*y.double()).sum()) for x,y in zip(a,b) if x is not None and y is not None)

def evaluate(model,names,after,bank,wi,pre_weight):
    labels=validate_bank(bank)
    if bank['training']:raise ValueError('Readout requires the held-RX evaluation bank')
    before=copy.deepcopy(model.state_dict());rng=_rng_snapshot()
    with _preserve_rng(),torch.no_grad():
        temporary=copy.deepcopy(model).eval();parameters=dict(temporary.named_parameters())
        for name,value in zip(names,after):parameters[name].copy_(value)
        features=forward(temporary,bank['x']);z=features['joint'].detach().clone()
        def readout(weight):
            logits=30.*F.linear(F.normalize(z.double(),dim=1,eps=1e-4),F.normalize(weight.double(),dim=1,eps=1e-4))
            other=logits.clone();other.scatter_(1,labels[:,None],float('-inf'))
            return dict(count=36,mean_CE=float(F.cross_entropy(logits,labels)),
                accuracy=float((logits.argmax(1)==labels).double().mean()),
                errors=int((logits.argmax(1)!=labels).sum()),
                margin=float((logits.gather(1,labels[:,None]).flatten()-other.max(1).values).mean()))
        actual=readout(after[wi]);pre_w=readout(pre_weight)
        native=features['logits'].double()
        formula=30.*F.linear(F.normalize(z.double(),dim=1,eps=1e-4),F.normalize(after[wi].double(),dim=1,eps=1e-4))
        if not torch.allclose(formula,native,atol=2e-5,rtol=2e-6):raise ValueError('Default-head actual formula differs')
    if not _equal(before,model.state_dict()) or not _equal(rng,_rng_snapshot()):raise ValueError('Readout changed live state/RNG')
    return z,dict(ACTUAL=actual,PRE_W=pre_w,head_fit_updates=0,actual_optimizer_updates=0)

def run(request):
    d.validate(request);out=Path(d.OUTPUT)
    out.mkdir(parents=True,exist_ok=False)
    try:return _run_owned(request,out)
    except Exception as error:
        if not (out/'summary.json').exists():
            try:
                with (out/'failure.json').open('x',encoding='utf-8') as handle:
                    json.dump(dict(status='FAILED',exception=type(error).__name__,reason=str(error),preserve_outputs=True,automatic_retry=False),handle,ensure_ascii=False,indent=2)
                    handle.write('\n')
            except FileExistsError:pass
        raise

def _run_owned(request,out):
    torch.set_num_threads(2);started=time.perf_counter()
    d.write(out/'runtime.json',dict(pid=os.getpid(),cwd=os.getcwd(),CPU_only=True,GPU_allocations=0))
    from experiments.cvs_evidence_responsibility_train20 import design as parent,model as api
    from experiments.cvs_evidence_responsibility_train20.provenance import source_proof
    from experiments.cvs_evidence_responsibility_train20.callback import selected_names
    from experiments.cvs_phase1_stack.runtime import installed
    from experiments.cvs_phase1_stack.fast_execution import configure
    c=parent.config(next(r for r in parent.rows() if r['row_id']==request['source_row']))
    source=source_proof(c)  # Existing source contract/ancestry, before deserialization.
    initial=torch.load(source/'initial_smoke.pt',map_location='cpu',weights_only=False)
    if set(initial)!={'model','scratch_only'} or initial['scratch_only'] is not True:
        raise ValueError('Own untrained initial snapshot required')
    model=api.build(c,'cpu').eval();model.load_state_dict(initial['model'],strict=True)
    named=[(name,p) for name,p in model.named_parameters() if p.requires_grad]
    names=[name for name,p in named];parameters=[p for name,p in named]
    if len(parameters)!=188:raise ValueError('Original188 optimizer parameters required')
    protected=protected_names(model);wi=next(i for i,name in enumerate(names) if name in protected and name.endswith('.weight'))
    fi=[i for i,name in enumerate(names) if name.startswith('id_backbone.encoder.') and i!=wi]
    selected=set(selected_names(model))
    head=api.classifier(model);encoder=model.id_backbone.encoder
    gains=(head.gain,encoder.response_gain,encoder.sequence_pool.gain_raw)
    with torch.no_grad():smoke=forward(model,torch.zeros(2,2,256))
    if smoke['logits'].shape!=(2,6) or not bool(torch.isfinite(smoke['logits']).all()):raise ValueError('Real initial checkpoint smoke failed')
    print('SMOKE PASS owninitial/query_access=false',flush=True)
    with installed(c) as native:
        context=native._build_ssdg_wisig_data(configure(parent.make_args(c,'cpu')),torch.device('cpu'))
        from cvsrffi.xuc_fusion.native import role_ids_from_native
        if context['named_test_loaders'] or role_ids_from_native(context)!=parent.read(parent.SOURCE)['role_ids']:
            raise ValueError('Unchanged source physical roles differ')
    bank=banks(context['train_loader'].dataset)
    optimizer=torch.optim.AdamW(parameters,lr=d.LR,weight_decay=1e-4)
    before=copy.deepcopy(model.state_dict());opt_before=copy.deepcopy(optimizer.state_dict());rng_before=_rng_snapshot()
    pre=tuple(p.detach().clone() for p in parameters)
    records=[];pairs=[];forwards=2;temporary_steps=0
    for held_rx in d.RX:
        held=subset(bank['held'],held_rx,False)
        before_z,held_before=evaluate(model,names,pre,held,wi,pre[wi]);forwards+=36
        for ordinal in d.ORDINALS:
            current=subset(bank['current'][ordinal],held_rx,True)
            if set(map(tuple,current['ids'])) & set(map(tuple,held['ids'])):raise ValueError('Physical training/evaluation overlap')
            components=forward(model,current['x']);forwards+=144
            main=F.cross_entropy(components['logits'],current['y'])
            base=torch.autograd.grad(main,parameters,retain_graph=True,allow_unused=True)
            if base[wi] is None or norm([base[wi]])<=0:raise ValueError('Joint main default W gradient missing')
            values,facts=losses(components,current,parameters[wi],gains)
            raw={mode:torch.autograd.grad(values[mode],parameters,retain_graph=True,allow_unused=True) for mode in ('ZERO','JOINT_GAP','ROUTED_GAP')}
            gj,gr=([raw[mode][i] for i in fi] for mode in ('JOINT_GAP','ROUTED_GAP'))
            if norm(gj)==0:raise ValueError('JOINT auxiliary norm is zero; norm attribution undefined')
            scale=norm(gr)/norm(gj)
            values['JOINT_NORM']=matched_value(values['JOINT_GAP'],scale)
            raw['JOINT_NORM']=torch.autograd.grad(values['JOINT_NORM'],parameters,retain_graph=True,allow_unused=True)
            matched_norm=norm([raw['JOINT_NORM'][i] for i in fi])
            if abs(matched_norm-norm(gr))>5e-6*max(norm(gr),1e-12):raise ValueError('Auxiliary F gradient norms do not match')
            for mode in ('JOINT_GAP','JOINT_NORM','ROUTED_GAP'):
                if not torch.equal(values[mode].detach(),values['JOINT_GAP'].detach()):raise ValueError('Same scalar attribution differs')
            afters={};zs={};evaluations={}
            for mode in d.MODES:
                auxiliary=raw[mode]
                if any(g is not None for name,g in zip(names,auxiliary) if name not in selected):
                    raise ValueError('Auxiliary reaches W/gain or a non-feature parameter')
                if any(g is not None for g,b in zip(auxiliary,base) if b is None):
                    raise ValueError('Auxiliary activates a main-unused parameter')
                proposal=simulate(optimizer,base,total_gradient(base,auxiliary),names=names,max_grad_norm=5.)
                temporary_steps+=2;after=proposal['total_after'];afters[mode]=after
                if mode!='ZERO' and not _equal(afters['ZERO'],proposal['base_after']):raise ValueError('Shared main CE update differs')
                z,result=evaluate(model,names,after,held,wi,pre[wi]);forwards+=36
                zs[mode]=z;evaluations[mode]=result
                total_norm=norm([auxiliary[i] for i in fi]);main_norm=norm([base[i] for i in fi])
                row=dict(held_rx=held_rx,ordinal=ordinal,mode=mode,current_receivers=facts['current_receivers'],
                    training_packets=144,evaluation_packets=36,current_main_CE=float(main.detach()),
                    auxiliary_value=float(values[mode].detach()),auxiliary_F_gradient_norm=total_norm,
                    gradient_norm_control_scale=scale,gradient_cosine_main=dot([auxiliary[i] for i in fi],[base[i] for i in fi])/(total_norm*main_norm) if total_norm*main_norm else None,
                    current_active=facts,held_before=held_before,held_after=result,
                    feature_parameter_delta_from_before=norm([after[i].double()-pre[i].double() for i in fi]),
                    feature_parameter_increment_vs_ZERO=norm([after[i].double()-afters['ZERO'][i].double() for i in fi]),
                    main_W_delta_norm=float((after[wi].double()-pre[wi].double()).norm()),
                    held_feature_RMS_from_before=float((z.double()-before_z.double()).square().mean().sqrt()),
                    W_three_gain_aux_grad_None=True,main_default_head_joint=True,temporary_AdamW_steps=2,
                    head_fit_updates=0,actual_optimizer_updates=0,held_RX_used_for_gradient=False,
                    temporary_summary={k:v for k,v in proposal['summary'].items() if k!='groups'})
                records.append(row);d.write(out/(str(held_rx)+'-'+str(ordinal)+'-'+mode+'.json'),row)
                print('CELL '+json.dumps(dict(held_rx=held_rx,ordinal=ordinal,mode=mode,PRE_W_CE=result['PRE_W']['mean_CE'],F_grad_norm=total_norm)),flush=True)
            comparisons={}
            for control in ('ZERO','JOINT_GAP','JOINT_NORM'):
                comparisons[control]=dict(
                    **{readout+'_CE_difference':evaluations['ROUTED_GAP'][readout]['mean_CE']-evaluations[control][readout]['mean_CE'] for readout in ('ACTUAL','PRE_W')},
                    **{readout+'_error_difference':evaluations['ROUTED_GAP'][readout]['errors']-evaluations[control][readout]['errors'] for readout in ('ACTUAL','PRE_W')},
                    feature_delta_difference=norm([afters['ROUTED_GAP'][i].double()-afters[control][i].double() for i in fi]),
                    held_feature_RMS_difference=float((zs['ROUTED_GAP'].double()-zs[control].double()).square().mean().sqrt()))
            pairs.append(dict(held_rx=held_rx,ordinal=ordinal,ROUTED_vs=comparisons,
                norm_matched_auxiliary=True,actual_AdamW_increment_norms_not_forced_equal=True))
    if not _equal(before,model.state_dict()) or not _equal(opt_before,optimizer.state_dict()) or not _equal(rng_before,_rng_snapshot()) or any(p.grad is not None for p in parameters):
        raise ValueError('Audit changed live model/optimizer/gradients/RNG')
    if len(records)!=80 or len(pairs)!=20 or forwards!=5942 or temporary_steps!=160:raise ValueError('Fixed one-step attribution budget differs')
    summary=dict(status='SOURCE_ATTRIBUTION_COMPLETE',run_id=d.RUN,analysis_code_commit=request['analysis_code_commit'],
        records=records,comparisons=pairs,source_bank=bank['summary'],network_IQ_forwards=forwards,
        temporary_AdamW_steps=160,actual_optimizer_updates=0,head_fit_updates=0,
        U_IQ_reads=0,V_IQ_reads=0,U_truth_read=False,target_access=False,
        live_model_optimizer_gradients_RNG_unchanged=True,wall_seconds=time.perf_counter()-started,
        limitation=request['interpretation'],main_run_qualification=False,source_snapshot='Own untrained scratch0, no trained final weights or moments loaded')
    d.write(out/'summary.json',summary);print('SOURCE_COMPLETE '+json.dumps(dict(cells=80,paired=20,main_run_qualification=False)),flush=True)
    return summary
