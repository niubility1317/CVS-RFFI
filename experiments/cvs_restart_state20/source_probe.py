"""Same pure-CE gradient at three declared continuation states, source L only."""
import copy
import json
import os
import sys
import time
from pathlib import Path
import torch
from torch.nn import functional as F
from . import design as d
from .optimizer import make_optimizer
from .evaluate import evaluate
from experiments.cvs_identity_difference20.bank import banks
from experiments.cvs_evidence_recomposition20.capture import forward
from experiments.cvs_evidence_recomposition20.source_probe import protected_names
from experiments.cvs_sigtime_heldcredit.source_probe import norm,kernel_changes
from experiments.cvs_u_schedule50.counterfactual import simulate,_equal,_rng_snapshot


def step_metrics(names,pre,after,weight_name):
    indices=[i for i,name in enumerate(names) if name.startswith('id_backbone.encoder.') and name!=weight_name]
    delta=[after[i].double()-pre[i].double() for i in indices]
    tokens=('.sinc.','.time_fuse.','.t1.','.t2.','.sequence_pool.')
    changed=[names[i] for i in indices if not torch.equal(after[i],pre[i])]
    return dict(actual_F_step_norm=norm(delta),feature_tensors_changed_from_before=len(changed),
        shallow_middle_tensors_changed_from_before=sum(any(token in name for token in tokens) for name in changed),
        feature_scope_tensors=len(indices),feature_scope_coordinates=sum(pre[i].numel() for i in indices),
        W_step_norm=norm([a.double()-b.double() for name,a,b in zip(names,after,pre) if name==weight_name]),
        feature_step_norm_by_layer={label:norm([a.double()-b.double() for name,a,b in zip(names,after,pre) if token in name and name!=weight_name])
            for label,token in [('Sinc','.sinc.'),('time_fuse','.time_fuse.'),('t1','.t1.'),('t2','.t2.'),('SIG_TIME','.sequence_pool.'),('PA','pa_'),('reference','.response_projection.')]})


def run(request):
    d.validate(request);out=Path(d.OUTPUT)
    if out.exists():raise FileExistsError('Preserve existing stage assay, reconcile before retry')
    out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(2);started=time.perf_counter()
    runtime=dict(pid=os.getpid(),cwd=os.getcwd(),argv=sys.argv,device='cpu',GPU_allocations=0)
    d.write(out/'runtime.json',runtime);print('RUNTIME '+json.dumps(runtime),flush=True)
    from experiments.cvs_crossrx_sharpness20 import design as parent,model as api
    from experiments.cvs_crossrx_sharpness20.provenance import source_proof,validate_checkpoint
    from experiments.cvs_phase1_stack.runtime import installed
    from experiments.cvs_phase1_stack.fast_execution import configure
    c=parent.config(next(r for r in parent.rows() if r['row_id']==request['source_row']))
    source=source_proof(c) # Same source roles/scratch ancestry/final-only selection BEFORE load.
    checkpoint=torch.load(source/'final_ssdg.pth',map_location='cpu',weights_only=False);validate_checkpoint(checkpoint,c)
    model=api.build(c,torch.device('cpu')).eval();model.load_state_dict(checkpoint['model'],strict=True)
    named=[(name,p) for name,p in model.named_parameters() if p.requires_grad]
    names=[name for name,p in named];parameters=[p for name,p in named]
    if len(parameters)!=188:raise ValueError('Original optimizer parameter traversal differs')
    protected=protected_names(model);weight_name=next(name for name in protected if name.endswith('.weight'))
    wi=names.index(weight_name);fi=[i for i,name in enumerate(names) if name.startswith('id_backbone.encoder.') and name!=weight_name]
    if len(fi)!=71 or sum(parameters[i].numel() for i in fi)!=195041:raise ValueError('Actual f_theta including gains differs')
    optimizers={mode:make_optimizer(parameters,checkpoint['optimizer'],mode,restart_lr=d.RESTART_LR) for mode in d.REGIMES}
    if float(optimizers['END_SAVED'].param_groups[0]['lr'])!=1e-6:raise ValueError('Parent fixed endpoint LR differs')
    with torch.no_grad():observed=forward(model,torch.zeros(2,2,256))
    if observed['logits'].shape!=(2,6):raise ValueError('Real checkpoint/query-free smoke failed')
    print('SMOKE PASS FINAL20 strict own checkpoint/query_read=false',flush=True)
    with installed(c) as native:
        context=native._build_ssdg_wisig_data(configure(parent.make_args(c,'cpu')),torch.device('cpu'))
        from cvsrffi.xuc_fusion.native import role_ids_from_native
        if context['named_test_loaders'] or role_ids_from_native(context)!=parent.read(parent.SOURCE)['role_ids']:
            raise ValueError('Unchanged source role handles differ')
    bank=banks(context['train_loader'].dataset);held=bank['held']
    before=copy.deepcopy(model.state_dict());opt_before={mode:copy.deepcopy(opt.state_dict()) for mode,opt in optimizers.items()}
    cp_opt_before=copy.deepcopy(checkpoint['optimizer']);rng_before=_rng_snapshot()
    pre=tuple(p.detach().clone() for p in parameters)
    z_before,held_before=evaluate(model,names,pre,held);forwards=182;temporary_steps=0;records=[];paired=[]
    for ordinal in d.ORDINALS:
        current=bank['current'][ordinal];components=forward(model,current['x']);forwards+=180
        loss=F.cross_entropy(components['logits'],current['y'])
        gradients=torch.autograd.grad(loss,parameters,allow_unused=True)
        if gradients[wi] is None or norm([gradients[wi]])<=0:raise ValueError('Original CE must jointly train default W')
        afters={};evaluations={};features={}
        for mode in d.REGIMES:
            optimizer=optimizers[mode]
            proposal=simulate(optimizer,gradients,gradients,names=names,max_grad_norm=5.);temporary_steps+=2
            if not _equal(proposal['base_after'],proposal['total_after']):raise ValueError('Identical pureCE branches differ')
            after=proposal['total_after'];afters[mode]=after
            z,result=evaluate(model,names,after,held);forwards+=180;features[mode]=z;evaluations[mode]=result
            delta=step_metrics(names,pre,after,weight_name)
            record=dict(ordinal=ordinal,regime=mode,state='FINAL20',current_L_CE=float(loss.detach()),
                learning_rate=float(optimizer.param_groups[0]['lr']),optimizer_state_entries=len(optimizer.state),
                gradient_norm=norm(gradients),main_W_gradient_norm=norm([gradients[wi]]),same_original_CE_gradient_all_regimes=True,
                **delta,physical_Sinc=kernel_changes(model,names,dict(base_after=pre,total_after=after)),
                held_before=held_before,held_after=result,
                held_feature_RMS_from_before=float((z.double()-z_before.double()).square().mean().sqrt()),
                PRE_W_held_CE_reduction_from_before=held_before['PRE_W']['mean_CE']-result['PRE_W']['mean_CE'],
                ACTUAL_held_CE_reduction_from_before=held_before['ACTUAL']['mean_CE']-result['ACTUAL']['mean_CE'],
                held_raw_identity_margin_gain_from_before=result['geometry']['identity_margin']-held_before['geometry']['identity_margin'],
                gains_are_updated_feature_parameters=True,default_W_updates_are_not_feature_parameters=True,
                temporary_AdamW={key:value for key,value in proposal['summary'].items() if key!='groups'})
            records.append(record);d.write(out/(mode+'-ordinal'+str(ordinal)+'.json'),record)
            print('SOURCE_CELL '+json.dumps({key:value for key,value in record.items() if key not in ('held_before','held_after','temporary_AdamW','physical_Sinc','feature_step_norm_by_layer')}),flush=True)
        entry=dict(ordinal=ordinal,
            KEEP_END_actual_F_step_difference_norm=norm([afters['RESTART_KEEP'][i].double()-afters['END_SAVED'][i].double() for i in fi]),
            RESET_KEEP_actual_F_step_difference_norm=norm([afters['RESTART_RESET'][i].double()-afters['RESTART_KEEP'][i].double() for i in fi]),
            KEEP_END_held_feature_RMS=float((features['RESTART_KEEP'].double()-features['END_SAVED'].double()).square().mean().sqrt()),
            RESET_KEEP_held_feature_RMS=float((features['RESTART_RESET'].double()-features['RESTART_KEEP'].double()).square().mean().sqrt()),
            RESET_KEEP_preW_CE_difference=evaluations['RESTART_RESET']['PRE_W']['mean_CE']-evaluations['RESTART_KEEP']['PRE_W']['mean_CE'])
        paired.append(entry)
    if (not _equal(before,model.state_dict()) or not _equal(cp_opt_before,checkpoint['optimizer'])
        or any(not _equal(opt_before[mode],opt.state_dict()) for mode,opt in optimizers.items())
        or any(p.grad is not None for p in model.parameters()) or not _equal(rng_before,_rng_snapshot())):
        raise ValueError('Stage assay changed live model/optimizer/payload/gradients/RNG')
    if forwards!=3062 or temporary_steps!=24:raise ValueError('Fixed stage computation budget differs')
    verdict=d.source_gate(records)
    summary=dict(status='SOURCE_SCREEN_PASSED' if verdict['passed'] else 'SOURCE_SCREEN_NOT_PASSED',
        run_id=d.RUN,analysis_code_commit=request['analysis_code_commit'],gate_result=verdict,source_falsification=d.GATE,
        checkpoint_source=dict(run_id=parent.RUN,row_id=c['row_id'],seed=2026092701,epoch=20,
            source_roles='EXACT_MATCH',ancestors=[],selection='fixed own BASE_CE firstseed final20; no target ranking'),
        source_bank=bank['summary'],source_L_IQ_reads=900,source_L_metadata_packets=6300,
        network_IQ_forwards=forwards,temporary_AdamW_steps=temporary_steps,actual_optimizer_updates=0,
        head_fit_updates=0,EMA_updates=0,U_IQ_reads=0,V_IQ_reads=0,U_truth_read=False,target_access=False,
        default_head_main_joint=True,live_model_optimizer_checkpoint_gradients_RNG_unchanged=True,
        source_feature_scope_tensors=71,source_feature_scope_coordinates=195041,records=records,paired_regime_comparisons=paired,
        runtime=runtime,wall_seconds=time.perf_counter()-started,
        limitation='PureCE fixed eval CPU single-step at own final20 only; not actual continuation/dropout/epoch efficacy. Gains are updated features; PRE_W uses only pre-step W, no fitted head. Independent heldL was seen by parent training; source evidence is not unseenRX efficacy, target selection or revival of closed auxiliary candidates.')
    d.write(out/'summary.json',summary);print('SOURCE_COMPLETE '+json.dumps(verdict),flush=True)
    return summary
