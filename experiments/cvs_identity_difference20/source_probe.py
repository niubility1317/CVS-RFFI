"""Real source-L graphs and hypothetical AdamW only; no live training."""
import copy
import json
import os
import sys
import time
from pathlib import Path
import torch
from torch.nn import functional as F
from . import design as d
from .bank import banks
from .objective import objective
from experiments.cvs_evidence_recomposition20.capture import forward
from experiments.cvs_evidence_recomposition20.source_probe import protected_names,total_gradient,gradient_difference
from experiments.cvs_crossrx_sharpness20.source_features import geometry
from experiments.cvs_sigtime_heldcredit.source_probe import norm,kernel_changes
from experiments.cvs_u_schedule50.counterfactual import simulate,_equal,_rng_snapshot


def evaluate(model,names,after,bank,protected):
    temporary=copy.deepcopy(model).eval().requires_grad_(False)
    target=dict(temporary.named_parameters());original=dict(model.named_parameters())
    with torch.no_grad():
        for name,value in zip(names,after):target[name].copy_(original[name] if name in protected else value)
        components=forward(temporary,bank['x']);z=components['joint']
        weight=original[next(name for name in protected if name.endswith('.weight'))].detach()
        logits=30.*F.linear(F.normalize(z.double(),dim=1,eps=1e-4),F.normalize(weight.double(),dim=1,eps=1e-4))
        losses=F.cross_entropy(logits,bank['y'],reduction='none')
        result=dict(geometry(z,bank['y'],bank['rx'],bank['day']),mean_CE=float(losses.mean()),
            per_RX_CE={str(r):float(losses[bank['rx']==r].mean()) for r in d.SOURCE_RX},
            per_RX_identity_margin={str(r):float((logits.gather(1,bank['y'][:,None]).squeeze(1)
                -logits.masked_fill(F.one_hot(bank['y'],6).bool(),float('-inf')).amax(1))[bank['rx']==r].mean()) for r in d.SOURCE_RX},
            fixed_pre_step_default_W_and_all_gains=True)
    return z.detach(),result


def difference_norm(a,b):return norm([x.double()-y.double() for x,y in zip(a,b)])


def changes(names,raw,other,protected):
    changed=[name for name,a,b in zip(names,raw,other) if name.startswith('id_backbone.encoder.') and name not in protected and not torch.equal(a,b)]
    tokens=('.sinc.','.time_fuse.','.t1.','.t2.','.sequence_pool.')
    return dict(feature_tensors_changed=len(changed),shallow_middle_tensors_changed=sum(any(t in name for t in tokens) for name in changed),
        layer_delta_norm={label:norm([a.double()-b.double() for name,a,b in zip(names,other,raw) if token in name and name not in protected])
            for label,token in [('Sinc','.sinc.'),('time_fuse','.time_fuse.'),('t1','.t1.'),('t2','.t2.'),('SIG_TIME','.sequence_pool.'),('PA','pa_'),('reference','.response_projection.')]})


def run(request):
    d.validate(request);out=Path(d.OUTPUT)
    if out.exists():raise FileExistsError('Preserve prior source assay, reconcile before retry')
    out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(2);started=time.perf_counter()
    runtime=dict(pid=os.getpid(),cwd=os.getcwd(),argv=sys.argv,device='cpu',GPU_allocations=0)
    d.write(out/'runtime.json',runtime);print('RUNTIME '+json.dumps(runtime),flush=True)
    from experiments.cvs_crossrx_sharpness20 import design as parent,model as api
    from experiments.cvs_crossrx_sharpness20.provenance import source_proof,validate_checkpoint
    from experiments.cvs_phase1_stack.runtime import installed
    from experiments.cvs_phase1_stack.fast_execution import configure
    c=parent.config(next(r for r in parent.rows() if r['row_id']==request['source_row']))
    source=source_proof(c) # Exact roles, scratch ancestry, final-only source selection BEFORE either load.
    checkpoint=torch.load(source/'final_ssdg.pth',map_location='cpu',weights_only=False);validate_checkpoint(checkpoint,c)
    initial=torch.load(source/'initial_smoke.pt',map_location='cpu',weights_only=False)
    if set(initial)!={'model','scratch_only'} or initial['scratch_only'] is not True:raise ValueError('Own initial scratch state differs')
    model=api.build(c,torch.device('cpu')).eval();args=parent.make_args(c,'cpu')
    named=[(name,p) for name,p in model.named_parameters() if p.requires_grad]
    names=[name for name,p in named];parameters=[p for name,p in named]
    if len(parameters)!=188 or len(checkpoint['optimizer']['param_groups'])!=1:raise ValueError('Original optimizer traversal differs')
    protected=protected_names(model);head=api.classifier(model);wi=next(i for i,p in enumerate(parameters) if p is head.weight)
    fi=[i for i,name in enumerate(names) if name.startswith('id_backbone.encoder.') and name not in protected]
    if len(fi)!=68 or sum(parameters[i].numel() for i in fi)!=194561:raise ValueError('Fixed deployed feature scope differs')
    for label,state in [('INITIAL',initial['model']),('FINAL20',checkpoint['model'])]:
        model.load_state_dict(state,strict=True)
        with torch.no_grad():observed=forward(model,torch.zeros(2,2,256))
        if observed['logits'].shape!=(2,6):raise ValueError('Real checkpoint/query-free smoke failed')
        print('SMOKE PASS '+label+' strict own checkpoint/query_read=false',flush=True)
    with installed(c) as native:
        context=native._build_ssdg_wisig_data(configure(parent.make_args(c,'cpu')),torch.device('cpu'))
        from cvsrffi.xuc_fusion.native import role_ids_from_native
        if context['named_test_loaders'] or role_ids_from_native(context)!=parent.read(parent.SOURCE)['role_ids']:
            raise ValueError('Unchanged legal source role handles differ')
    bank=banks(context['train_loader'].dataset);held=bank['held'];records=[];forwards=4;temporary_steps=0
    for state in d.STATES:
        model.load_state_dict(initial['model'] if state=='INITIAL' else checkpoint['model'],strict=True);model.eval()
        optimizer=torch.optim.AdamW(parameters,lr=float(args.lr),weight_decay=float(args.weight_decay))
        if state=='FINAL20':optimizer.load_state_dict(checkpoint['optimizer'])
        before=copy.deepcopy(model.state_dict());opt_before=copy.deepcopy(optimizer.state_dict());rng_before=_rng_snapshot()
        pre=tuple(p.detach().clone() for p in parameters)
        z_before,held_before=evaluate(model,names,pre,held,protected);forwards+=180
        for ordinal in d.ORDINALS:
            current=bank['current'][ordinal];components=forward(model,current['x']);forwards+=180
            ce=F.cross_entropy(components['logits'],current['y'])
            base=torch.autograd.grad(ce,parameters,retain_graph=True,allow_unused=True)
            if base[wi] is None or norm([base[wi]])<=0:raise ValueError('Original main CE must train default W')
            gradients={};summaries={};after={};proposals={}
            for mode in d.MODES:
                value=objective(model,components,current['y'],current['rx'],current['side'],current['ids'],mode,role='L_s')
                gradient=torch.autograd.grad(value['loss'],parameters,retain_graph=True,allow_unused=True)
                if any(g is not None for name,g in zip(names,gradient) if name in protected):raise ValueError('Auxiliary W/gain gradient escaped')
                if any(g is not None for g,b in zip(gradient,base) if b is None):raise ValueError('Auxiliary activated a baseline-unused optimizer parameter')
                gradients[mode]=gradient;summaries[mode]=dict(value['summary'],auxiliary_gradient_norm=norm(gradient),
                    auxiliary_to_main_gradient_ratio=norm(gradient)/max(norm(base),1e-12),
                    auxiliary_gradient_by_layer={label:norm([g for name,g in zip(names,gradient) if token in name])
                        for label,token in [('Sinc','.sinc.'),('time_fuse','.time_fuse.'),('t1','.t1.'),('t2','.t2.'),('SIG_TIME','.sequence_pool.'),('PA','pa_'),('reference','.response_projection.')]})
                if mode=='ZERO' and (norm(gradient)!=0 or float(value['loss'])!=0):raise ValueError('ZERO control is nonzero')
                if mode!='ZERO':
                    proposal=simulate(optimizer,base,total_gradient(base,gradient),names=names,max_grad_norm=5.)
                    temporary_steps+=2;after[mode]=proposal['total_after'];proposals[mode]=proposal
                    summaries[mode]['virtual_AdamW']={key:z for key,z in proposal['summary'].items() if key!='groups'}
            raw=proposals['SELF_DIFF']['base_after']
            if not _equal(raw,proposals['CROSS_DIFF']['base_after']):raise ValueError('Repeated same-state CE AdamW control differs')
            after['ZERO']=raw;zs={};evaluations={}
            for mode in d.MODES:
                z,evaluation=evaluate(model,names,after[mode],held,protected);forwards+=180
                zs[mode]=z;evaluations[mode]=evaluation
                summaries[mode].update(held_after=evaluation,changes_from_ZERO=changes(names,raw,after[mode],protected),
                    physical_Sinc=kernel_changes(model,names,dict(base_after=raw,total_after=after[mode])))
            cross=value['tensors']['cos_CROSS300'];deltaA=value['tensors']['delta_A'];deltaB=value['tensors']['delta_B']
            rawdelta=[raw[i].double()-pre[i].double() for i in fi]
            cross_delta=[after['CROSS_DIFF'][i].double()-raw[i].double() for i in fi]
            row=dict(state=state,ordinal=ordinal,current_L_CE=float(ce.detach()),main_gradient_norm=norm(base),
                main_protected_gradient_norm={name:norm([base[i]]) for i,name in enumerate(names) if name in protected},
                learning_rate=float(optimizer.param_groups[0]['lr']),optimizer_moments='explicitly fresh hypothetical initial' if state=='INITIAL' else 'actual saved final20',
                modes=summaries,held_before=held_before,CROSS_active_direction_pairs=int((cross<.9).sum()),
                minimum_class_difference_norm=float(torch.cat([deltaA.detach().norm(dim=-1).flatten(),deltaB.detach().norm(dim=-1).flatten()]).min()),
                CROSS_SELF_auxiliary_gradient_relative_difference=norm(gradient_difference(gradients['CROSS_DIFF'],gradients['SELF_DIFF']))/max(norm(gradients['SELF_DIFF']),1e-12),
                CROSS_ZERO_actual_F_displacement_relative_difference=norm(cross_delta)/max(norm(rawdelta),1e-12),
                CROSS_SELF_actual_F_displacement_relative_difference=difference_norm([after['CROSS_DIFF'][i] for i in fi],[after['SELF_DIFF'][i] for i in fi])/max(norm(rawdelta),1e-12),
                CROSS_ZERO_held_feature_RMS=float((zs['CROSS_DIFF'].double()-zs['ZERO'].double()).square().mean().sqrt()),
                CROSS_SELF_held_feature_RMS=float((zs['CROSS_DIFF'].double()-zs['SELF_DIFF'].double()).square().mean().sqrt()),
                CROSS_ZERO_shallow_middle_tensors_changed=summaries['CROSS_DIFF']['changes_from_ZERO']['shallow_middle_tensors_changed'],
                CROSS_ZERO_held_identity_margin_gain=evaluations['CROSS_DIFF']['identity_margin']-evaluations['ZERO']['identity_margin'],
                CROSS_SELF_held_identity_margin_gain=evaluations['CROSS_DIFF']['identity_margin']-evaluations['SELF_DIFF']['identity_margin'],
                CROSS_ZERO_held_mean_CE_increase=evaluations['CROSS_DIFF']['mean_CE']-evaluations['ZERO']['mean_CE'],
                CROSS_SELF_held_mean_CE_increase=evaluations['CROSS_DIFF']['mean_CE']-evaluations['SELF_DIFF']['mean_CE'])
            records.append(row);d.write(out/(state+'-ordinal'+str(ordinal)+'.json'),row)
            print('SOURCE_CELL '+json.dumps({key:z for key,z in row.items() if key not in ('modes','held_before')}),flush=True)
        if (not _equal(before,model.state_dict()) or not _equal(opt_before,optimizer.state_dict())
            or any(p.grad is not None for p in model.parameters()) or not _equal(rng_before,_rng_snapshot())):
            raise ValueError('Source assay changed live model/optimizer/gradients/RNG')
    if forwards!=6124 or temporary_steps!=32:raise ValueError('Fixed computation budget differs')
    verdict=d.source_gate(records)
    summary=dict(status='SOURCE_SCREEN_PASSED' if verdict['passed'] else 'SOURCE_SCREEN_NOT_PASSED',run_id=d.RUN,
        analysis_code_commit=request['analysis_code_commit'],gate_result=verdict,source_falsification=d.GATE,
        checkpoint_source=dict(run_id=parent.RUN,row_id=c['row_id'],seed=2026092701,epoch=20,source_roles='EXACT_MATCH',
            ancestors=[],selection='fixed own BASE_CE firstseed initial/final20; no target ranking'),
        source_bank=bank['summary'],source_L_IQ_reads=900,source_L_metadata_packets=6300,
        network_IQ_forwards=forwards,temporary_AdamW_steps=temporary_steps,actual_optimizer_updates=0,head_fit_updates=0,EMA_updates=0,
        U_IQ_reads=0,V_IQ_reads=0,U_truth_read=False,target_access=False,live_state_RNG_unchanged=True,
        default_head_main_joint=True,static_feature_tensors=68,static_feature_coordinates=194561,
        runtime=runtime,records=records,wall_seconds=time.perf_counter()-started,
        limitation='Fixed eval-mode CPU two-state sourceL finite-step diagnosis. INITIAL fresh moments/FINAL20 saved moments, not reset continuation. Independent heldL was seen by parent training. No contemporaneous CUDA step, U/mid-state, unseenRX efficacy, target or novelty proof; detached aux endpoints do not freeze globalclip-coupled/shared main updates.')
    d.write(out/'summary.json',summary);print('SOURCE_COMPLETE '+json.dumps(verdict),flush=True)
    return summary
