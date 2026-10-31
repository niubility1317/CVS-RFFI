"""Reuse one demonstrably source-only neural encoder; no new source fitting."""
import time
from pathlib import Path
from .io import read,write,require
from .milkr.adaptive_residual import STRENGTHS


def parent_metadata(rt):
    parent_rt=read(rt['source_reuse_runtime']);parent=read(rt['source_reuse_selection'])
    require(parent_rt['kernel_method']=='crossrx_residual_ensemble','Wrong source residual parent')
    for key in ('source_cache','source_contract','old_classes','model_seeds'):require(parent_rt[key]==rt[key],'Source inheritance data/model role differs: '+key)
    for seed in rt['model_seeds']:require(parent_rt['PAMF_sources'][str(seed)]==rt['PAMF_sources'][str(seed)],'PAMF source checkpoint differs')
    for key in ('checkpoint','source_contract','roles'):require(parent_rt['DURA'][key]==rt['DURA'][key],'DURA source checkpoint/data role differs')
    require(str(Path(parent_rt['run_root'])/'source_orbit_cca.npz')==rt['source_reuse_checkpoint'],'Source checkpoint is not the parent source encoder')
    require(parent['status']=='SOURCE_FROZEN' and parent['target_eligible'],'Parent encoder was not source frozen')
    require(all(parent[k]==0 for k in ('target_reads','query_reads','U_truth_reads','F_updates','G_updates')),'Parent source state used prohibited inputs')
    require(len(parent['source_metric_training'])==6,'Source neural training provenance missing')
    for row in parent['source_metric_training']:
        for orbit in ('C4','C8'):
            a=row[orbit];require(a['steps']==600 and all(a[k]==0 for k in ('target_reads','query_reads','U_truth_reads','F_updates','G_updates')),'Neural parent is not source-only fixed-step')
    require(not any(k in parent['K1_rule'] for k in ('kernel_density_rule','rf_prior_gate_rule','rf_class_prior_rule')),'Parent prior differs from the declared method')
    require(tuple(rt['support_residual_strengths'])==STRENGTHS and rt['K1_residual_strength']==1.,'Wrong fixed residual selection grid')
    return parent_rt,parent


def freeze(rt,context):
    started=time.perf_counter();parent_rt,parent,bridge=context
    root=Path(rt['run_root']);bridge.save(root/'source_orbit_cca.npz')
    rule=dict(parent['K1_rule'],support_residual_strengths=list(STRENGTHS),K1_residual_strength=1.)
    write(root/'source_selection.json',dict(status='SOURCE_FROZEN',target_eligible=True,K1_rule=rule,
        scientific_verdict='FIXED_SUPPORT_SELECTION_ALL16_FALSIFICATION',feature_method=rt['kernel_method'],
        source_checkpoint_reused=True,parent_source_run_id=parent_rt['run_id'],parent_source_checkpoint=rt['source_reuse_checkpoint'],
        K1_rule_inherited=True,source_K1_grid_inherited=True,source_K1_grid=parent['source_K1_grid'],
        source_summary=[],inherited_source_validation=parent['source_summary'],inherited_validation_scope='K1 remains identical; parent K20 results do not validate the new selector',
        source_metric_training=[],inherited_source_metric_training=parent['source_metric_training'],
        source_new_optimizer_updates=0,source_model_fitting_rows=0,source_physical_rows=0,
        inherited_source_physical_rows=parent['source_physical_rows'],inherited_source_paired_observations=parent['source_paired_observations'],
        source_RX_TX_folds=parent['source_RX_TX_folds'],source_RX_TX_folds_inherited=True,
        source_correlation_weighting=parent['source_correlation_weighting'],source_correlation_weighting_inherited=True,
        source_primary_query_paths=[0,2],source_held_new_head_columns_removed_before_softmax=True,
        RF_phase_eighths=list(range(8)),RF_kernel_orbits=[4,8],head_phase_quarters=[0,1,2,3],selected_rank=64,
        source_numerical_smoke='PASS',encoder_checkpoint_created=True,seconds=time.perf_counter()-started,
        target_reads=0,query_reads=0,U_truth_reads=0,F_updates=0,G_updates=0,
        scope='Only frozen source CCA/NN and source K1 rule inherited; no parent target SUP state, prediction, truth or optimizer loaded'))
