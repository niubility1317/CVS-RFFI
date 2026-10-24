"""One fixed source comparison removing only the raw gradient magnitude cap."""
from contextlib import contextmanager
import torch
from . import source_branch_gradients as source
from . import source_step_projection as step_source
from . import source_optimizer_counterfactual as optimizer_source
from . import source_optimizer_shadow as shadow
from .branch_gradient_projection import branch_projected_gradients
from .parameter_step_projection import project_parameters

KIND='fixed_original_E20_source_uncapped_step_v1'
RULES=dict(minimum_actual_increment_norm_retention_per_nonzero_group=.5,
    minimum_uncapped_vs_capped_protected_increment_ratio_in_any_group_each_seed=1.05,
    minimum_removed_material_conflicts=1,material_cosine_threshold=-1e-6,
    all_actual_parameter_constraints_and_nonexpansion_required=True,all_three_fixed_seeds_required=True)


def validate_plan(plan):
    if (plan.get('kind')!=KIND or plan.get('source_rules')!=RULES
            or plan.get('seeds')!=[2026092701,2026092702,2026092703] or plan.get('source_epoch')!=20
            or plan.get('steps_per_seed')!=1 or plan.get('L_batch')!=128 or plan.get('U_batch')!=256
            or any(plan.get(k)!=0 for k in ('V_reads','target_reads','U_truth_reads','G_training_reads'))):
        raise ValueError('Only the fixed original-E20 source uncapped-step scope and preregistered rules allowed')


@contextmanager
def installed_counterfactual():
    original_build=shadow.build;snapshots=[];current={}
    def build(*args,**kwargs):
        optimizer=original_build(*args,**kwargs);parameters=list(args[1])
        def capture(*unused):snapshots.append(tuple(p.detach().clone() for p in parameters))
        optimizer.register_step_post_hook(capture);return optimizer
    shadow.build=build
    try:
        with step_source.installed_counterfactual():
            original_facts=source.gradient_facts;original_step=optimizer_source.shadow_step
            def actual_step(*args,**kwargs):
                current.update(model=args[0],options=kwargs)
                return original_step(*args,**kwargs)
            def observe(named,losses,*,reference=None):
                snapshots.clear();derivatives=[];original_grad=torch.autograd.grad
                def capture(*args,**kwargs):
                    value=original_grad(*args,**kwargs);derivatives.append(value);return value
                torch.autograd.grad=capture
                try:
                    facts,cap1=original_facts(named,losses,reference=reference)
                finally:torch.autograd.grad=original_grad
                try:
                    if reference is None or len(derivatives)!=6 or len(snapshots)!=4 or not current:
                        raise ValueError('Exact original component derivatives and four baseline optimizer copies required')
                    gradients,gradient_details=branch_projected_gradients(named,reference[0],reference[1],derivatives[0],derivatives[1],max_u_ratio=None)
                    _,metadata=shadow.shadow_step(current['model'],named,gradients,**current['options'])
                    if len(snapshots)!=5:raise ValueError('Exactly one additional uncapped optimizer copy required')
                    _,details=project_parameters(named,snapshots[0],snapshots[4],reference[0],derivatives[0],derivatives[1])
                    baseline=facts['parameter_step_projection']['groups'];comparisons={};removed=[]
                    for group,value in details.items():
                        if not value['actual_parameter_check'] or not value['norm_nonexpansion']:
                            raise ValueError('Actual uncapped FP32 constraints and norm nonexpansion required')
                        old=baseline[group]['applied_increment_norm'];new=value['applied_increment_norm']
                        comparisons[group]=dict(capped_protected_increment_norm=old,uncapped_protected_increment_norm=new,
                            uncapped_to_capped_norm_ratio=new/old if old else None,capped_zero=old==0,
                            amplification_rule_pass=new>0 and new>=1.05*old)
                        for anchor,before in value['anchor_cosines_before'].items():
                            after=value['anchor_cosines_after'][anchor]
                            if before is not None and before<-1e-6 and (after is None or after>=-1e-6):
                                removed.append(dict(group=group,anchor=anchor,before_cosine=before,after_cosine=after,
                                    zero_increment=new==0,quantization_fallback=value['quantization_fallback']))
                    facts['uncapped_step_counterfactual']=dict(kind=KIND,raw_gradient_groups=gradient_details,
                        uncapped_parameter_groups=details,comparisons=comparisons,removed_material_conflicts=removed,
                        uncapped_optimizer=metadata,actual_optimizer_copies_observed=5,extra_optimizer_steps=1,
                        extra_autograd_calls=0,extra_model_forwards=0,protected_parameters_applied_to_original_model=False,
                        interpretation='Only raw U/L magnitude cap removed;the same three gradient anchors and actual-step operator retained.Fresh moments,one fixed source batch;all shadow updates discarded.')
                    return facts,cap1
                finally:snapshots.clear();current.clear()
            optimizer_source.shadow_step=actual_step;source.gradient_facts=observe
            try:yield
            finally:optimizer_source.shadow_step=original_step;source.gradient_facts=original_facts
    finally:shadow.build=original_build;snapshots.clear();current.clear()


def collect(runtime,plan,analysis_commit):
    validate_plan(plan)
    with installed_counterfactual():result=source.collect(runtime,plan,analysis_commit)
    rows=[row['source_branch_facts']['uncapped_step_counterfactual'] for row in result['rows']]
    if len(rows)!=3 or any(r['actual_optimizer_copies_observed']!=5 for r in rows):
        raise ValueError('All three fixed source models and five actual optimizer copies each required')
    retentions=[v['retention'] for row in rows for v in row['uncapped_parameter_groups'].values() if v['raw_increment_norm']>0]
    if not retentions:raise ValueError('Nonzero source incremental motion required')
    removed=sum(len(r['removed_material_conflicts']) for r in rows)
    amplification=[any(v['amplification_rule_pass'] for v in row['comparisons'].values()) for row in rows]
    passed=min(retentions)>=.5 and all(amplification) and removed>=1
    original_counts={k:result[k] for k in ('optimizer_steps','parameter_updates','parameter_grad_writes','persisted_buffer_updates')}
    writes=0
    for row in result['rows']:
        old=row['source_branch_facts']['optimizer_counterfactual'];new=row['source_branch_facts']['uncapped_step_counterfactual']
        writes+=old['labeled_only_optimizer']['shadow_parameter_gradient_tensors']
        writes+=sum(v['optimizer']['shadow_parameter_gradient_tensors'] for v in old['comparisons'].values())
        writes+=new['uncapped_optimizer']['shadow_parameter_gradient_tensors']
    result.update(status='VERIFIED_FIXED_SOURCE_UNCAPPED_STEP',kind=KIND,source_rules=RULES,source_rule_pass=passed,
        scientific_status='SOURCE_COUNTERFACTUAL_PASSED' if passed else 'SCIENTIFIC_FALSIFIED_SOURCE',
        minimum_nonzero_group_increment_norm_retention=min(retentions),each_seed_amplification_rule_pass=amplification,
        removed_material_conflicts=removed,quantization_fallback_groups=sum(v['quantization_fallback'] for r in rows for v in r['uncapped_parameter_groups'].values()),
        optimizer_steps=15,shadow_optimizer_steps=15,parameter_updates=15,parameter_grad_writes=writes,
        original_model_update_counts=original_counts,shadow_parameter_updates_discarded=True,new_formal_training_updates=0,
        protected_parameters_applied_to_original_model=False,automatic_candidate_launch=False,goal_complete=False,
        interpretation='Three original-E20 fresh first steps only;15 disposable optimizer copies/27 original derivatives/7686source packet views.Original model/grad/buffer updates0,no new forward/V/Utruth/target/G reads.No automatic training or target-performance claim.')
    return result
