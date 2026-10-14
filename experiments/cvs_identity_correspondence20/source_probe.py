"""True-label correspondence learning in actual joint160, source L only."""
import copy,json,os,sys,time
from pathlib import Path
import torch
from torch.nn import functional as F
from . import design as d
from .relationship import losses
from experiments.cvs_restart_state20.optimizer import make_optimizer
from experiments.cvs_restart_state20.evaluate import evaluate
from experiments.cvs_restart_state20.source_probe import step_metrics
from .bank import banks
from experiments.cvs_evidence_recomposition20.capture import forward
from experiments.cvs_evidence_recomposition20.source_probe import protected_names,total_gradient
from experiments.cvs_sigtime_heldcredit.source_probe import norm,kernel_changes
from experiments.cvs_u_schedule50.counterfactual import simulate,_equal,_rng_snapshot

def dot(a,b):return sum(float((x.double()*y.double()).sum()) for x,y in zip(a,b) if x is not None and y is not None)

def compact(summaries):
    result={}
    for mode,summary in summaries.items():
        result[mode]={k:v for k,v in summary.items() if k not in ('matrix_diagnostics','actual_pairs')}
        per_rx={}
        for rx in d.SOURCE_RX:
            matrices=[r['matching'] for r in summary['matrix_diagnostics'] if r['anchor_RX']==rx]
            masses=[m for q in matrices for side in ('row','column') for m in q['diagmass'][side]]
            per_rx[str(rx)]=dict(mean_correct_matching_mass=sum(masses)/len(masses),
                matching_errors_total=sum(q['argmaxerrors']['row']+q['argmaxerrors']['column'] for q in matrices),
                active_rows_and_columns=sum(q['lowmass']['row']+q['lowmass']['column'] for q in matrices),
                matrix_count=len(matrices),directional_decisions=len(masses))
        result[mode]['by_anchor_RX']=per_rx
    return result

def matching(z,bank):
    with torch.no_grad(): _,summary=losses(z,bank,source_role='L_s')
    return compact(summary)

def compare(names,fi,after,control,z,control_z,result,control_result):
    tokens=('.sinc.','.time_fuse.','.t1.','.t2.','.sequence_pool.')
    changed=[names[i] for i in fi if not torch.equal(after[i],control[i])]
    a=result['identity_matching']['CROSS_ROW'];b=control_result['identity_matching']['CROSS_ROW']
    return dict(actual_F_difference_norm=norm([after[i].double()-control[i].double() for i in fi]),
        feature_difference_tensors=len(changed),shallow_middle_difference_tensors=sum(any(t in name for t in tokens) for name in changed),
        held_feature_RMS_difference=float((z.double()-control_z.double()).square().mean().sqrt()),
        held_correct_matching_mass_gain=a['mean_correct_matching_mass']-b['mean_correct_matching_mass'],
        held_matching_errors_increase=a['matching_errors_total']-b['matching_errors_total'],
        raw_identity_margin_gain=result['geometry']['identity_margin']-control_result['geometry']['identity_margin'],
        PRE_W_CE_difference=result['PRE_W']['mean_CE']-control_result['PRE_W']['mean_CE'],
        ACTUAL_CE_difference=result['ACTUAL']['mean_CE']-control_result['ACTUAL']['mean_CE'],
        ACTUAL_accuracy_difference=result['ACTUAL']['accuracy']-control_result['ACTUAL']['accuracy'],
        PRE_W_accuracy_difference=result['PRE_W']['accuracy']-control_result['PRE_W']['accuracy'])

def run(request):
    d.validate(request);out=Path(d.OUTPUT)
    if out.exists():raise FileExistsError('Preserve existing identity matching assay; reconcile before retry')
    out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(2);started=time.perf_counter()
    runtime=dict(pid=os.getpid(),cwd=os.getcwd(),argv=sys.argv,device='cpu',GPU_allocations=0)
    d.write(out/'runtime.json',runtime);print('RUNTIME '+json.dumps(runtime),flush=True)
    from experiments.cvs_crossrx_sharpness20 import design as parent,model as api
    from experiments.cvs_crossrx_sharpness20.provenance import source_proof,validate_checkpoint
    from experiments.cvs_phase1_stack.runtime import installed
    from experiments.cvs_phase1_stack.fast_execution import configure
    c=parent.config(next(r for r in parent.rows() if r['row_id']==request['source_row']))
    source=source_proof(c)
    checkpoint=torch.load(source/'final_ssdg.pth',map_location='cpu',weights_only=False);validate_checkpoint(checkpoint,c)
    cp_opt_before=copy.deepcopy(checkpoint['optimizer'])
    model=api.build(c,torch.device('cpu')).eval()
    named=[(name,p) for name,p in model.named_parameters() if p.requires_grad]
    names=[name for name,p in named];parameters=[p for name,p in named]
    if len(parameters)!=188:raise ValueError('Original188 optimizer traversal differs')
    protected=protected_names(model);weight_name=next(name for name in protected if name.endswith('.weight'))
    wi=names.index(weight_name);fi=[i for i,name in enumerate(names) if name.startswith('id_backbone.encoder.') and name!=weight_name]
    if len(fi)!=71 or sum(parameters[i].numel() for i in fi)!=195041:raise ValueError('Actual F71 including3gains differs')
    model.load_state_dict(checkpoint['model'],strict=True)
    with torch.no_grad():observed=forward(model,torch.zeros(2,2,256))
    if observed['logits'].shape!=(2,6):raise ValueError('Real checkpoint noquery smoke failed')
    print('SMOKE PASS FINAL20 strict own checkpoint/query_read=false',flush=True)
    with installed(c) as native:
        context=native._build_ssdg_wisig_data(configure(parent.make_args(c,'cpu')),torch.device('cpu'))
        from cvsrffi.xuc_fusion.native import role_ids_from_native
        if context['named_test_loaders'] or role_ids_from_native(context)!=parent.read(parent.SOURCE)['role_ids']:
            raise ValueError('Unchanged source role handles differ')
    bank=banks(context['train_loader'].dataset);held=bank['held'];forwards=2;temporary_steps=0;records=[];paired=[]
    for state in d.STATES:
        model.load_state_dict(checkpoint['model'],strict=True);model.eval()
        saved=copy.deepcopy(checkpoint['optimizer'])
        optimizer=make_optimizer(parameters,saved,'RESTART_KEEP',restart_lr=d.LR)
        before=copy.deepcopy(model.state_dict());opt_before=copy.deepcopy(optimizer.state_dict());rng_before=_rng_snapshot()
        pre=tuple(p.detach().clone() for p in parameters)
        z_before,held_before=evaluate(model,names,pre,held);forwards+=180
        held_before['identity_matching']=matching(z_before,held)
        for ordinal in d.ORDINALS:
            current=bank['current'][ordinal];components=forward(model,current['x']);forwards+=180
            ce=F.cross_entropy(components['logits'],current['y'])
            base=torch.autograd.grad(ce,parameters,retain_graph=True,allow_unused=True)
            if base[wi] is None or norm([base[wi]])<=0:raise ValueError('DefaultW originaljoint CE gradient missing')
            objectives,current_matching=losses(components['joint'],current,source_role='L_s')
            current_matching=compact(current_matching)
            afters={};features={};evaluations={}
            for mode in d.MODES:
                auxiliary=torch.autograd.grad(objectives[mode],parameters,retain_graph=True,allow_unused=True)
                if auxiliary[wi] is not None:raise ValueError('Matching loss depends on concrete class W')
                if any(g is not None for g,b in zip(auxiliary,base) if b is None):
                    raise ValueError('Matching activated main-unused parameters')
                if mode=='ZERO' and (norm(auxiliary)!=0 or float(objectives[mode])!=0):raise ValueError('ZERO matching nonzero')
                proposal=simulate(optimizer,base,total_gradient(base,auxiliary),names=names,max_grad_norm=5.);temporary_steps+=2
                after=proposal['total_after'];afters[mode]=after
                if mode=='ZERO' and not _equal(after,proposal['base_after']):raise ValueError('ZERO CE controls differ')
                if mode!='ZERO' and not _equal(afters['ZERO'],proposal['base_after']):raise ValueError('Same-state base CE step differs')
                z,evaluation=evaluate(model,names,after,held);forwards+=180
                evaluation['identity_matching']=matching(z,held);features[mode]=z;evaluations[mode]=evaluation
                ng=norm([auxiliary[i] for i in fi]);nb=norm([base[i] for i in fi])
                row=dict(state=state,ordinal=ordinal,mode=mode,learning_rate=d.LR,
                    optimizer_state_entries=len(optimizer.state),optimizer_moments='saved final20 m/v/step, newstageLR',
                    current_L_CE=float(ce.detach()),auxiliary_loss=float(objectives[mode].detach()),
                    auxiliary_gradient_norm=norm(auxiliary),main_gradient_norm=norm(base),
                    auxiliary_F_gradient_cosine_main=dot([auxiliary[i] for i in fi],[base[i] for i in fi])/(ng*nb) if ng*nb else None,
                    auxiliary_F_gradient_by_layer={label:norm([g for name,g in zip(names,auxiliary) if token in name])
                        for label,token in [('Sinc','.sinc.'),('time_fuse','.time_fuse.'),('t1','.t1.'),('t2','.t2.'),('SIG_TIME','.sequence_pool.'),('PA','pa_'),('reference','.response_projection.')]},
                    auxiliary_default_W_None=True,actual_three_gains_in_feature_graph=True,current_matching=current_matching,
                    **step_metrics(names,pre,after,weight_name),physical_Sinc=kernel_changes(model,names,dict(base_after=pre,total_after=after)),
                    held_before=held_before,held_after=evaluation,
                    held_feature_RMS_from_before=float((z.double()-z_before.double()).square().mean().sqrt()),
                    PRE_W_held_CE_reduction_from_before=held_before['PRE_W']['mean_CE']-evaluation['PRE_W']['mean_CE'],
                    ACTUAL_held_CE_reduction_from_before=held_before['ACTUAL']['mean_CE']-evaluation['ACTUAL']['mean_CE'],
                    held_raw_identity_margin_gain_from_before=evaluation['geometry']['identity_margin']-held_before['geometry']['identity_margin'],
                    temporary_AdamW={k:v for k,v in proposal['summary'].items() if k!='groups'})
                records.append(row);d.write(out/(state+'-'+mode+'-ordinal'+str(ordinal)+'.json'),row)
                print('SOURCE_CELL '+json.dumps(dict(state=state,ordinal=ordinal,mode=mode,auxiliary_loss=row['auxiliary_loss'],
                    auxgradient=row['auxiliary_gradient_norm'],held_mass=evaluation['identity_matching']['CROSS_ROW']['mean_correct_matching_mass'],
                    held_matching_errors=evaluation['identity_matching']['CROSS_ROW']['matching_errors_total'])),flush=True)
            pair=dict(state=state,ordinal=ordinal)
            for control in d.GATE['controls']:
                pair['vs_'+control]=compare(names,fi,afters['CROSS_ROW'],afters[control],features['CROSS_ROW'],features[control],
                    evaluations['CROSS_ROW'],evaluations[control])
            paired.append(pair)
        if (not _equal(before,model.state_dict()) or not _equal(opt_before,optimizer.state_dict())
            or any(p.grad is not None for p in model.parameters()) or not _equal(rng_before,_rng_snapshot())):
            raise ValueError('Source matching changed live model/optimizer/gradients/RNG')
    if not _equal(cp_opt_before,checkpoint['optimizer']):raise ValueError('Checkpoint optimizer payload changed')
    if forwards!=3062 or temporary_steps!=24:raise ValueError('Fixed matching source computation budget differs')
    verdict=d.source_gate(records,paired)
    summary=dict(status='SOURCE_SCREEN_PASSED' if verdict['passed'] else 'SOURCE_SCREEN_NOT_PASSED',
        run_id=d.RUN,analysis_code_commit=request['analysis_code_commit'],gate_result=verdict,source_falsification=d.GATE,
        checkpoint_source=dict(run_id=parent.RUN,row_id=c['row_id'],seed=2026092701,epoch=20,
            source_roles='EXACT_MATCH',ancestors=[],selection='fixed ownBASECE firstseed final20, no target ranking'),
        source_bank=bank['summary'],source_L_IQ_reads=900,source_L_metadata_packets=6300,
        network_IQ_forwards=forwards,temporary_AdamW_steps=temporary_steps,actual_optimizer_updates=0,
        head_fit_updates=0,EMA_updates=0,U_IQ_reads=0,V_IQ_reads=0,U_truth_read=False,target_access=False,
        default_head_main_joint=True,source_feature_scope_tensors=71,source_feature_scope_coordinates=195041,
        live_model_optimizer_checkpoint_gradients_RNG_unchanged=True,records=records,paired_regime_comparisons=paired,
        runtime=runtime,wall_seconds=time.perf_counter()-started,
        limitation='Independent120 ROW computations only, zero BI/Sinkhorn. FixedFINAL20 savedm/v/step andLR2e-4; new900 physically disjoint from prior900, all originalL was seen by parent. Source CPU one-step signal, not unseenRX/DG. No realtraining/U coupling/head fitting or target selection.')
    d.write(out/'summary.json',summary);print('SOURCE_COMPLETE '+json.dumps(verdict),flush=True)
    return summary
