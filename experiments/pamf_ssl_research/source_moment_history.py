"""Isolate inherited U moments at common parameters using discarded source copies."""
from contextlib import contextmanager
import math
import torch
from experiments.cric_dura_comparison.f_optimizer import build,clip_physical
from .optimizer_reference import container,labeled_step
from .parameter_step_projection import project_parameters
from .branch_gradient_projection import branch_projected_gradients,GROUPS,ANCHORS
from .u_gradprotect_objective import parameter_group
from . import source_branch_gradients as source
from .source_optimizer_counterfactual import _first_step_options

KIND='fixed_original_E20_source_moment_history_v1'
RULES=dict(material_cosine_threshold=-1e-6,minimum_negative_history_drift_anchors_per_seed=1,
    current_copied_reference_zero_increment_required=True,all_three_fixed_seeds_required=True)


def validate_plan(plan):
    if (plan.get('kind')!=KIND or plan.get('source_rules')!=RULES
            or plan.get('seeds')!=[2026092701,2026092702,2026092703] or plan.get('source_epoch')!=20
            or plan.get('steps_per_seed')!=1 or plan.get('history_steps')!=2
            or plan.get('L_batch')!=128 or plan.get('U_batch')!=256
            or any(plan.get(k)!=0 for k in ('V_reads','target_reads','U_truth_reads','G_training_reads'))):
        raise ValueError('Only the preregistered original-E20 two-step source moment comparison allowed')


@torch.no_grad()
def history_probe(model,named,full,clean,physical,joint,*,base_lr,weight_decay):
    """Reuse fixed source derivatives; recenter copies to isolate history,not a trajectory."""
    named=list(named);anchors=(full,clean,physical)
    if (not named or any(p.device.type!='cpu' or p.dtype!=torch.float32 for _,p in named)
            or any(len(g)!=len(named) for g in (*anchors,joint))
            or not math.isfinite(base_lr) or base_lr<=0 or not math.isfinite(weight_decay) or weight_decay<0):
        raise ValueError('Finite actual CPU FP32 source gradients and native optimizer options required')
    origin=tuple(p.detach().clone() for _,p in named)
    copies=[container(model,named) for _ in range(2)]
    parameters=[[dict(c.named_parameters())[name] for name,_ in named] for c in copies]
    optimizers=[build(c,ps,'CENTER_UNITS',lr=base_lr,weight_decay=weight_decay) for c,ps in zip(copies,parameters)]

    def step(index,gradients):
        opt=optimizers[index];ps=parameters[index];opt.zero_grad(set_to_none=True)
        for (_,original),p,g in zip(named,ps,gradients):
            if g is not None:
                if (g.shape!=original.shape or g.dtype!=original.dtype or g.device!=original.device
                        or not bool(torch.isfinite(g).all())):raise ValueError('Finite matching source gradients required')
                p.grad=g.detach().clone()
        norm=clip_physical(opt,ps,5.)
        if not bool(torch.isfinite(norm)):raise ValueError('Nonfinite counterfactual physical gradient norm')
        opt.step()
        values=tuple(p.detach().clone() for p in ps)
        if any(not bool(torch.isfinite(p).all()) for p in values):raise ValueError('Nonfinite source parameter copy')
        return values

    first_L=step(0,full);first_joint=step(1,joint)
    protected,first_details=project_parameters(named,first_L,first_joint,*anchors)
    for p,v in zip(parameters[1],protected):p.copy_(v)
    moment_differences={}
    for group in GROUPS:
        totals={key:0. for key in ('exp_avg','exp_avg_sq')}
        for (name,_),left,right in zip(named,*parameters):
            if parameter_group(name)!=group:continue
            a=optimizers[0].state.get(left,{});b=optimizers[1].state.get(right,{})
            for key in totals:
                av=a.get(key,torch.zeros_like(left));bv=b.get(key,torch.zeros_like(right))
                totals[key]+=float((av.double()-bv.double()).square().sum())
        moment_differences[group]={key:math.sqrt(value) for key,value in totals.items()}
    # Same coordinates and fixed gradients remove parameter-path differences.
    # Keep both distinct first-step optimizer histories unchanged.
    for ps in parameters:
        for p,v in zip(ps,origin):p.copy_(v)
    reference,reference_meta=labeled_step(copies[1],list(zip((n for n,_ in named),parameters[1])),
        optimizers[1],full,clip_impl=torch.nn.utils.clip_grad_norm_)
    current_L=step(0,full);current_joint_history_L=step(1,full)
    if any(not torch.equal(a,b) for a,b in zip(reference,current_joint_history_L)):
        raise ValueError('Current copied-moment L reference must equal the actual U-off L step')
    unchanged,current_details=project_parameters(named,reference,current_joint_history_L,*anchors)
    if (any(not torch.equal(a,b) for a,b in zip(unchanged,current_joint_history_L))
            or any(v['raw_increment_norm']!=0 or v['applied_increment_norm']!=0
                or not v['actual_parameter_check'] or not v['norm_nonexpansion'] for v in current_details.values())):
        raise ValueError('The current reference guard must see exactly zero new U increment')
    history_extra=tuple(a-b for a,b in zip(current_L,current_joint_history_L))
    relations={group:{anchor:source._relation(named,g,history_extra,group)
        for anchor,g in zip(ANCHORS,anchors)} for group in GROUPS}
    negative=[dict(group=group,anchor=anchor,**value) for group,values in relations.items()
        for anchor,value in values.items() if value['cosine'] is not None and value['cosine']<RULES['material_cosine_threshold']]
    if any(o._sinc_unit_steps!=2 or o._sinc_physical_clip_calls!=2 for o in optimizers):
        raise ValueError('Two actual native physical steps per history copy required')
    return dict(optimizer_steps=5,physical_clip_calls=5,current_U_gradient_zero=True,
        common_parameters_reset=True,fixed_frozen_source_derivatives=True,
        first_actual_guard_groups=first_details,current_guard_groups=current_details,
        current_copied_reference_matches_actual_L_step=True,copied_reference_metadata=reference_meta,
        moment_difference_norms=moment_differences,history_extra_descent_relations=relations,
        negative_history_drift_anchors=negative,
        shadow_parameter_gradient_tensors=4*sum(g is not None for g in full)+sum(g is not None for g in joint),
        original_parameter_or_gradient_writes=0,extra_model_forwards=0,extra_autograd_calls=0,
        interpretation='Two fixed-gradient histories at reset common original-E20 coordinates,not two training epochs or an observed current trajectory.U is present only in the first history step.Current U=0;copied joint moments make its L-only reference identical to the current update.Negative history-extra anchor dot means worse first-order anchor change relative to the L-only-moment counterfactual,not a measured loss increase or target effect.')


@contextmanager
def installed_counterfactual():
    original_facts=source.gradient_facts;original_class=source.BranchDiagnostic;current={}
    class MomentDiagnostic(original_class):
        def guide(self,labeled,student):
            if current:raise ValueError('One active disposable source model required')
            current.update(model=student,config=self.c)
            try:return super().guide(labeled,student)
            finally:current.clear()
    def observe(named,losses,*,reference=None):
        if reference is None or not current:raise ValueError('Exact original source derivatives required')
        derivatives=[];original_grad=torch.autograd.grad
        def capture(*args,**kwargs):
            values=original_grad(*args,**kwargs);derivatives.append(values);return values
        torch.autograd.grad=capture
        try:facts,cap1=original_facts(named,losses,reference=reference)
        finally:torch.autograd.grad=original_grad
        if len(derivatives)!=6:raise ValueError('Exactly the six existing component derivatives required')
        joint,guard=branch_projected_gradients(named,reference[0],reference[1],derivatives[0],derivatives[1],max_u_ratio=None)
        facts['moment_history_counterfactual']=history_probe(current['model'],named,reference[0],derivatives[0],
            derivatives[1],joint,**_first_step_options(current['config']))
        facts['moment_history_counterfactual']['raw_uncapped_gradient_groups']=guard
        return facts,cap1
    source.gradient_facts=observe;source.BranchDiagnostic=MomentDiagnostic
    try:yield
    finally:source.gradient_facts=original_facts;source.BranchDiagnostic=original_class;current.clear()


def collect(runtime,plan,analysis_commit):
    validate_plan(plan)
    with installed_counterfactual():result=source.collect(runtime,plan,analysis_commit)
    rows=[r['source_branch_facts']['moment_history_counterfactual'] for r in result['rows']]
    if len(rows)!=3 or result['actual_autograd_calls']!=27 or any(r['optimizer_steps']!=5 for r in rows):
        raise ValueError('All three fixed source models,27 derivatives and15 discarded native optimizer steps required')
    passed=[len(r['negative_history_drift_anchors'])>=RULES['minimum_negative_history_drift_anchors_per_seed'] for r in rows]
    original={key:result[key] for key in ('optimizer_steps','parameter_updates','parameter_grad_writes','persisted_buffer_updates')}
    result.update(status='VERIFIED_FIXED_SOURCE_MOMENT_HISTORY',kind=KIND,source_rules=RULES,
        source_rule_pass=all(passed),each_seed_negative_history_drift_rule_pass=passed,
        scientific_status='SOURCE_HISTORY_LIMITATION_SUPPORTED' if all(passed) else 'SCIENTIFIC_FALSIFIED_SOURCE',
        negative_history_drift_anchors=sum(len(r['negative_history_drift_anchors']) for r in rows),
        optimizer_steps=15,shadow_optimizer_steps=15,parameter_updates=15,
        parameter_grad_writes=sum(r['shadow_parameter_gradient_tensors'] for r in rows),
        original_model_update_counts=original,shadow_parameter_updates_discarded=True,new_formal_training_updates=0,
        automatic_candidate_launch=False,goal_complete=False,
        interpretation='One fixed original-E20 L128/U256 batch per seed;two recentered parameter-only history steps,not a training trajectory.15 disposable optimizer steps/27 original derivatives/7686source packet views.Original model/gradient/persisted buffer updates0.No V/Utruth/target/G reads,no healthy-job changes or performance claim.')
    return result
