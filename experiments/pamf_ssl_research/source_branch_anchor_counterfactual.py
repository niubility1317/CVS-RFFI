"""Fixed source-only comparison of one full-L guard and three simultaneous anchors."""
from contextlib import contextmanager
import math
import torch
from . import source_branch_gradients as source
from .branch_gradient_projection import branch_projected_gradients,GROUPS,ANCHORS
from .source_cap2_projection_reference import projected_gradients

KIND='fixed_original_E20_source_branch_anchor_counterfactual_v1'
RULES=dict(minimum_whole_applied_U_norm_retention_vs_native_cap2=.5,
           minimum_removed_material_component_conflicts=1,material_cosine_threshold=-1e-6,
           require_all_three_exact_source_seed_steps=True,parameter_updates=0,U_truth_reads=0,V_reads=0,target_reads=0)


def subtract(named,combined,labeled):
    return tuple(None if a is None and b is None else
        (torch.zeros_like(p) if a is None else a)-(torch.zeros_like(p) if b is None else b)
        for (_,p),a,b in zip(named,combined,labeled))


@contextmanager
def installed_counterfactual():
    """Observe the existing six component derivatives; no additional autograd call."""
    original=source.gradient_facts
    def observe(named,losses,*,reference=None):
        if reference is None:raise ValueError('Exact original full-L/U source reference required')
        derivatives=[];grad=torch.autograd.grad
        def capture(*args,**kwargs):
            result=grad(*args,**kwargs);derivatives.append(result);return result
        torch.autograd.grad=capture
        try:facts,native=original(named,losses,reference=reference)
        finally:torch.autograd.grad=grad
        if len(derivatives)!=6:raise ValueError('Reuse exactly six existing source component derivatives')
        full_l,full_u=reference
        baseline,baseline_details=projected_gradients(named,full_l,full_u,max_u_ratio=2.)
        alternative,alternative_details=branch_projected_gradients(named,full_l,full_u,derivatives[0],derivatives[1],max_u_ratio=2.)
        applied_native=subtract(named,baseline,full_l)
        baseline_components={}
        for group in GROUPS:
            baseline_components[group]={anchor:source._relation(named,vector,applied_native,group)
                for anchor,vector in zip(ANCHORS,(full_l,derivatives[0],derivatives[1]))}
        native_norm=math.sqrt(sum(d['applied_U_norm']**2 for d in baseline_details.values()))
        alternative_norm=math.sqrt(sum(d['applied_U_norm']**2 for d in alternative_details.values()))
        if not native_norm or not math.isfinite(alternative_norm):raise ValueError('Nonzero finite native source U reference required')
        removed=[]
        for group in GROUPS:
            for anchor in ('L_clean','L_As'):
                before=baseline_components[group][anchor]['cosine']
                after=alternative_details[group]['anchor_cosines_after'][anchor]
                if before is not None and before<RULES['material_cosine_threshold'] and after is not None and after>=RULES['material_cosine_threshold']:
                    removed.append(dict(group=group,anchor=anchor,before_cosine=before,after_cosine=after))
        facts['branch_anchor_counterfactual']=dict(kind=KIND,
            native_full_L_cap2_details=baseline_details,native_full_L_cap2_component_relations=baseline_components,
            simultaneous_three_anchor_cap2_details=alternative_details,
            whole_native_applied_U_norm=native_norm,whole_three_anchor_applied_U_norm=alternative_norm,
            whole_U_norm_retention_vs_native_cap2=alternative_norm/native_norm,
            removed_material_component_conflicts=removed,extra_autograd_calls=0,
            extra_model_forward_calls=0,extra_teacher_forward_calls=0,
            projection_interpretation='Counterfactual parameter gradients only;complete original L unchanged,only U projected onto fullL/nonphysicalL/physicalL halfspaces then cap2.None of these gradients is applied to a model.')
        return facts,native
    source.gradient_facts=observe
    try:yield
    finally:source.gradient_facts=original


def collect(rt,plan,analysis_commit):
    if plan.get('kind')!=KIND or plan.get('source_rules')!=RULES:
        raise ValueError('One preregistered branch projection and unchanged source rules required')
    with installed_counterfactual():result=source.collect(rt,plan,analysis_commit)
    observations=[row['source_branch_facts']['branch_anchor_counterfactual'] for row in result['rows']]
    minimum=min(row['whole_U_norm_retention_vs_native_cap2'] for row in observations)
    removed=sum(len(row['removed_material_component_conflicts']) for row in observations)
    passed=minimum>=RULES['minimum_whole_applied_U_norm_retention_vs_native_cap2'] and removed>=RULES['minimum_removed_material_component_conflicts']
    result.update(status='VERIFIED_FIXED_SOURCE_BRANCH_ANCHOR_COUNTERFACTUAL',kind=KIND,source_rules=RULES,
        source_rule_pass=passed,minimum_whole_U_norm_retention_vs_native_cap2=minimum,
        removed_material_component_conflicts=removed,counterfactual_extra_autograd_calls=0,
        scientific_status='SOURCE_COUNTERFACTUAL_PASSED' if passed else 'SCIENTIFIC_FALSIFIED_SOURCE',
        actual_native_guard='Existing cap1 recipe,bitwise bridge verified;cap2 and three-anchor cap2 are algebraic comparisons on these same derivatives',
        future_training='No automatic launch.This source result can support a separately fixed candidate,independent review,real-checkpoint smoke and original-E20 training;not an optimizer or target result.')
    return result
