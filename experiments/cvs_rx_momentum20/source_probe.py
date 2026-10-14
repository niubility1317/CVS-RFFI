"""One source graph: original CE plus five RX gradient contrasts, history only."""
import copy
import json
import os
import sys
import time
from pathlib import Path
import torch
from torch.nn import functional as F
from . import design as d
from .state import make_filtered_optimizer
from experiments.cvs_restart_state20.evaluate import evaluate
from experiments.cvs_restart_state20.source_probe import step_metrics
from experiments.cvs_identity_difference20.bank import banks
from experiments.cvs_evidence_recomposition20.capture import forward
from experiments.cvs_evidence_recomposition20.source_probe import protected_names
from experiments.cvs_sigtime_heldcredit.source_probe import norm,kernel_changes
from experiments.cvs_u_schedule50.counterfactual import simulate,_equal,_rng_snapshot


def compare(names,fi,after,control,z,control_z,result,control_result):
    tokens=('.sinc.','.time_fuse.','.t1.','.t2.','.sequence_pool.')
    changed=[names[i] for i in fi if not torch.equal(after[i],control[i])]
    return dict(actual_F_difference_norm=norm([after[i].double()-control[i].double() for i in fi]),
        feature_difference_tensors=len(changed),
        shallow_middle_difference_tensors=sum(any(t in name for t in tokens) for name in changed),
        held_feature_RMS_difference=float((z.double()-control_z.double()).square().mean().sqrt()),
        raw_identity_margin_gain=result['geometry']['identity_margin']-control_result['geometry']['identity_margin'],
        PRE_W_CE_difference=result['PRE_W']['mean_CE']-control_result['PRE_W']['mean_CE'],
        ACTUAL_CE_difference=result['ACTUAL']['mean_CE']-control_result['ACTUAL']['mean_CE'])


def run(request):
    d.validate(request);out=Path(d.OUTPUT)
    if out.exists():raise FileExistsError('Preserve existing momentum assay, reconcile before retry')
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
    model=api.build(c,torch.device('cpu')).eval();model.load_state_dict(checkpoint['model'],strict=True)
    named=[(name,p) for name,p in model.named_parameters() if p.requires_grad]
    names=[name for name,p in named];parameters=[p for name,p in named]
    if len(parameters)!=188:raise ValueError('Original optimizer parameter traversal differs')
    protected=protected_names(model);weight_name=next(name for name in protected if name.endswith('.weight'))
    wi=names.index(weight_name);fi=[i for i,name in enumerate(names) if name.startswith('id_backbone.encoder.') and name!=weight_name]
    if len(fi)!=71 or sum(parameters[i].numel() for i in fi)!=195041:raise ValueError('Actual f_theta including gains differs')
    with torch.no_grad():observed=forward(model,torch.zeros(2,2,256))
    if observed['logits'].shape!=(2,6):raise ValueError('Real checkpoint/query-free smoke failed')
    print('SMOKE PASS FINAL20 strict own checkpoint/query_read=false',flush=True)
    with installed(c) as native:
        context=native._build_ssdg_wisig_data(configure(parent.make_args(c,'cpu')),torch.device('cpu'))
        from cvsrffi.xuc_fusion.native import role_ids_from_native
        if context['named_test_loaders'] or role_ids_from_native(context)!=parent.read(parent.SOURCE)['role_ids']:
            raise ValueError('Unchanged source role handles differ')
    bank=banks(context['train_loader'].dataset);held=bank['held']
    before=copy.deepcopy(model.state_dict());cp_opt_before=copy.deepcopy(checkpoint['optimizer']);rng_before=_rng_snapshot()
    pre=tuple(p.detach().clone() for p in parameters)
    z_before,held_before=evaluate(model,names,pre,held);forwards=182;temporary_steps=0;records=[];paired=[]
    for ordinal in d.ORDINALS:
        current=bank['current'][ordinal];components=forward(model,current['x']);forwards+=180
        per_packet=F.cross_entropy(components['logits'],current['y'],reduction='none');loss=per_packet.mean()
        rx_grads=[]
        for rx in d.SOURCE_RX:
            mask=current['rx']==rx
            if int(mask.sum())!=36:raise ValueError('Fixed balanced source RX36 differs')
            rx_grads.append(torch.autograd.grad(per_packet[mask].mean(),parameters,retain_graph=True,allow_unused=True))
        gradients=torch.autograd.grad(loss,parameters,allow_unused=True)
        if gradients[wi] is None or norm([gradients[wi]])<=0:raise ValueError('Original CE must jointly train default W')
        # The original whole-batch gradient remains the main gradient; this is diagnostic only.
        mean_rx=[None if gradients[i] is None else sum(gs[i] for gs in rx_grads)/5 for i in range(len(parameters))]
        gradient_mean_discrepancy=norm([a.double()-b.double() for a,b in zip(gradients,mean_rx) if a is not None])
        afters={};evaluations={};features={}
        for mode in d.REGIMES:
            optimizer,filter_summary=make_filtered_optimizer(parameters,names,checkpoint['optimizer'],gradients,
                tuple(rx_grads),mode,restart_lr=d.RESTART_LR,max_grad_norm=5.)
            opt_before=copy.deepcopy(optimizer.state_dict())
            proposal=simulate(optimizer,gradients,gradients,names=names,max_grad_norm=5.);temporary_steps+=2
            if not _equal(proposal['base_after'],proposal['total_after']):raise ValueError('Identical CE branches differ')
            if not _equal(opt_before,optimizer.state_dict()):raise ValueError('Simulation changed input optimizer')
            after=proposal['total_after'];afters[mode]=after
            z,result=evaluate(model,names,after,held);forwards+=180;features[mode]=z;evaluations[mode]=result
            record=dict(ordinal=ordinal,regime=mode,state='FINAL20',current_L_CE=float(loss.detach()),
                learning_rate=float(optimizer.param_groups[0]['lr']),optimizer_state_entries=len(optimizer.state),
                gradient_norm=norm(gradients),main_W_gradient_norm=norm([gradients[wi]]),
                RX_mean_vs_original_gradient_norm=gradient_mean_discrepancy,
                same_original_CE_gradient_all_regimes=True,RX_gradients_from_same_forward=True,
                momentum_filter=filter_summary,**step_metrics(names,pre,after,weight_name),
                physical_Sinc=kernel_changes(model,names,dict(base_after=pre,total_after=after)),
                held_before=held_before,held_after=result,
                held_feature_RMS_from_before=float((z.double()-z_before.double()).square().mean().sqrt()),
                PRE_W_held_CE_reduction_from_before=held_before['PRE_W']['mean_CE']-result['PRE_W']['mean_CE'],
                ACTUAL_held_CE_reduction_from_before=held_before['ACTUAL']['mean_CE']-result['ACTUAL']['mean_CE'],
                held_raw_identity_margin_gain_from_before=result['geometry']['identity_margin']-held_before['geometry']['identity_margin'],
                gains_are_updated_feature_parameters=True,default_W_updates_are_not_filtered=True,
                temporary_AdamW={key:value for key,value in proposal['summary'].items() if key!='groups'})
            records.append(record);d.write(out/(mode+'-ordinal'+str(ordinal)+'.json'),record)
            print('SOURCE_CELL '+json.dumps(dict(ordinal=ordinal,regime=mode,actual_F_step_norm=record['actual_F_step_norm'],
                PRE_W_CE_reduction=record['PRE_W_held_CE_reduction_from_before'],ACTUAL_CE_reduction=record['ACTUAL_held_CE_reduction_from_before'],
                raw_identity_margin_gain=record['held_raw_identity_margin_gain_from_before'],filter_summary=filter_summary)),flush=True)
        if any(not torch.equal(afters['KEEP'][wi],afters[mode][wi]) for mode in d.REGIMES):
            raise ValueError('Unfiltered normal default W step differs between methods')
        pair=dict(ordinal=ordinal,W_after_exactly_same=True)
        for control in d.REGIMES[:2]:
            pair['vs_'+control]=compare(names,fi,afters['RX_M_FILTER'],afters[control],features['RX_M_FILTER'],features[control],
                evaluations['RX_M_FILTER'],evaluations[control])
        paired.append(pair)
    if (not _equal(before,model.state_dict()) or not _equal(cp_opt_before,checkpoint['optimizer'])
        or any(p.grad is not None for p in model.parameters()) or not _equal(rng_before,_rng_snapshot())):
        raise ValueError('Momentum assay changed live model/payload/gradients/RNG')
    if forwards!=3062 or temporary_steps!=24:raise ValueError('Fixed source computation budget differs')
    verdict=d.source_gate(records,paired)
    summary=dict(status='SOURCE_SCREEN_PASSED' if verdict['passed'] else 'SOURCE_SCREEN_NOT_PASSED',
        run_id=d.RUN,analysis_code_commit=request['analysis_code_commit'],gate_result=verdict,source_falsification=d.GATE,
        checkpoint_source=dict(run_id=parent.RUN,row_id=c['row_id'],seed=2026092701,epoch=20,
            source_roles='EXACT_MATCH',ancestors=[],selection='fixed own BASE_CE firstseed final20; no target ranking'),
        source_bank=bank['summary'],source_L_IQ_reads=900,source_L_metadata_packets=6300,
        network_IQ_forwards=forwards,temporary_AdamW_steps=temporary_steps,actual_optimizer_updates=0,
        head_fit_updates=0,EMA_updates=0,U_IQ_reads=0,V_IQ_reads=0,U_truth_read=False,target_access=False,
        default_head_main_joint=True,live_model_checkpoint_gradients_RNG_unchanged=True,
        source_feature_scope_tensors=71,source_feature_scope_coordinates=195041,records=records,paired_regime_comparisons=paired,
        runtime=runtime,wall_seconds=time.perf_counter()-started,
        limitation='Same fixed own final20, eval CPU single-step source assay. Source gradient contrasts may contain useful identity interactions. PRE_W fixes only W for nonfit analysis; after gains remain actual. HeldL was seen by parent, not unseenRX/blind/DG efficacy. No main-training frequency or target-selected choice.')
    d.write(out/'summary.json',summary);print('SOURCE_COMPLETE '+json.dumps(verdict),flush=True)
    return summary
