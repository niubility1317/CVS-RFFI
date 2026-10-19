"""Create the fixed matrix and a standard experiment registration."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

RUN = '20261019-phase1-pamf-dg-nativebudget-m24-r01'
SEEDS = (2026092701, 2026092702, 2026092703)
BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[1]


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')


def main():
    rows = []
    for family, epochs, batch, lr, optimizer in [
        ('ibrff', 200, 128, .001, 'Adam'), ('sdg', 200, 32, .001, 'Adam'),
        ('riei', 100, 64, .0001, 'SGD momentum0'), ('drift', 100, 256, .0001, 'Adam')]:
        for model_seed in SEEDS:
            for variant in ('full', 'ce'):
                rows.append(dict(row_id='%s-%s-s%d' % (family, variant, model_seed),
                    family=family, variant=variant, model_seed=model_seed, split_seed=392005,
                    epochs=epochs, batch_size=batch, lr=lr, weight_decay=0,
                    optimizer=optimizer, stages=2 if family == 'ibrff' else 1,
                    schedule='ReduceLROnPlateau factor0.1 patience5 min1e-6; early_stop10 per stage' if family == 'ibrff' else 'constant_lr',
                    selection='source_validation_nll' if family in ('ibrff','sdg') else 'fixed_final_E100',
                    budget_source='author_public_WiSig_recipe' if family in ('ibrff','sdg') else 'user_E100_override_20261019',
                    initialization='scratch', U_s_use=False, satellite_train_augmentation=False,
                    default_head_joint_training=True, separate_head_training=False))
    dump(BASE/'matrix.json', dict(run_id=RUN, allowed_gpus=[0,1], own_workers_per_gpu=1,
        rows=rows, views=['clean','practical_high','practical_mid','practical_low_suburban',
                         'practical_high_urban','practical_mid_urban','practical_low_urban'],
        fixed_rows_no_target_feedback=True,
        riei=dict(version='SPAWC2023 project short_stem1d implementation', lambda_mi=1.2,
                  lambda_ie=1.2, alternating_fed_steps_per_batch=2,
                  original_epoch_budget='not disclosed; user sets100 instead of historical200',
                  epoch_schedule_rescale='100/200; no epoch-dependent LR milestones in constant recipe'),
        drift=dict(version='arXiv2510.09405v2', lambda_grl=1, lambda_center=.01,
                   lambda_mse=.02, negative_mse_reduction='sum_over_features_then_mean_batch',
                   mse_normalize=False, mse_cap=0, center_mode='batch',
                   original_epoch_budget='not disclosed; user sets100 instead of historical200',
                   epoch_schedule_rescale='100/200; no epoch-dependent LR milestones in constant recipe'),
        ibrff=dict(upstream_commit='5a03a04da86c4bbf5cf90f41a55c268bf2152203',
                   stage1_alpha=0, stage2_alpha=.04, alpha_source='first fixed public ManySig CRD recipe, not selected using current target',
                   epochs_max_per_stage=200, same_run_source_selected_stage1_parent=True),
        sdg=dict(upstream_commit='0090f4ba1d5cf76a4200a5be4830a9b08eb39699',
                 size='S', main_aug_depth=[2], aux_aug_depth=[1], lambda_con=[1,100],
                 author_augmentation_retained=True, pooled_source_RX_adaptation=True)))
    report_root = ROOT/'automation_reports/CV-SincNet'/RUN
    template = BASE/'experiment_spec.json'
    subprocess.run([sys.executable, str(ROOT/'tools/experiment_registry.py'), 'template', '--output', str(template)], check=True)
    spec = json.loads(template.read_text(encoding='utf-8-sig'))
    # Existing schema is retained; values below describe this actual matrix.
    spec.update(run_id=RUN, group_id='phase1-pamf-four-dg-nativebudget',
        display_name='PAMF Phase1 四种域泛化方法与自身CE：两GPU三种子',
        description='固定当前弱标签物理划分，四方法按各自配方及用户E100覆盖，从零训练与七视图truth-last评分。',
        kind='comparison', stage='Phase1', tags=['PAMF','RIEI','DRIFT','IB-RFF','SDG','three-seed','two-GPU'],
        authorization='用户2026-10-19要求设计并运行，仅两个GPU、三种子；RIEI/DRIFT明确E100并按比例调整训练安排。',
        status='PLANNED')
    spec['code'] = dict(commit='filled_after_code_commit', checkout='github_publish/CVS-RFFI-repo',
        environment='existing N607 SDG-SEI + existing task-local NumPy1.23.5; no shared environment mutation',
        cwd='local_artifacts/phase1_dg_nativebudget_20261019/runtime.json#/release_root')
    spec['data'] = dict(dataset='ManySig', version='existing validated physical roles', representation='equalized1 center256 unitRMS',
        contract_ref='runs/phase1_daot_rc4_pure_game_m3_20260917_r2/source_contract.json',
        physical_ids_ref='same source_contract.json#/role_ids', label_map_ref='matrix/source cache metadata; six fixed TX labels',
        source_receivers=[1,3,4,6,8], target_receivers=[0,2,5,7,9,10,11], source_days=[1,2,3], target_days=[0,1,2,3],
        tx_sets_ref='all6 old TX; same as PAMF', roles={'L_s':6300,'U_s':'56700 hidden and unused','V':27000},
        train_ratio=.07, capsule_id=None, split_id=None, validation_ref='existing Phase1 VALIDATED_ONCE capsule and sixscene observations',
        support_query_ref=None, leo_config_ref='clean plus existing six practical fixed observations')
    spec['permissions'] = dict(regime='matched_weak_label_source_only', query_use='frozen predictions only; no fit/select/feedback',
        external_method_exception=None, claim_scope='current fixed proxy benchmark; method recipe and cost differ; no new independent blind claim')
    spec['checkpoint'] = dict(initialization='scratch', sources=['none; IB stage2 only its own same-contract stage1'],
        contract_check_ref='pipeline.py prepare_source/load/frozen_model', provenance_verdict='scratch or same-run same-seed stage1 only',
        selection_rule='RIEI/DRIFT E100 final; IB-RFF/SDG minimum source V NLL per registered recipe')
    spec['execution'] = dict(host='N607 ordinary account', launch_owner='codex-root-phase1-dg-nativebudget-20261019',
        gpu_policy='fixed physicalGPU0,1 only; one own row worker per GPU; wait for foreign workloads, never displace',
        remote_run_root='local_artifacts/phase1_dg_nativebudget_20261019/runtime.json#/run_root',
        remote_log_root='local_artifacts/phase1_dg_nativebudget_20261019/runtime.json#/log_root',
        local_artifact_root='local_artifacts/phase1_dg_nativebudget_20261019',
        runtime_ref='local_artifacts/phase1_dg_nativebudget_20261019/runtime.json',
        launch_command='python -u pipeline.py dispatch --config runtime.json',
        stop_rule='wrong source/target roles, provenance, output collision, nonfinite/unlaunchable/no legal predictions or wrong truth linkage; two technical row failures pause unstarted rows; healthy rows continue; low scores never stop')
    spec['expected_artifacts']=['source_cache/metadata.json','pipeline_state.json','ROW/source/resolved_config.json',
        'ROW/source/stageN/epochs.jsonl','ROW/source/freeze.json','ROW/prediction/predictions.npz',
        'ROW/prediction/complete.json','ROW/scores/scores.json','completion.json']
    # Registry accepts a matrix list; seeds have separate roles on every row.
    spec['rows']=[dict(row_id=r['row_id'], method=r['family'], variant=r['variant'],
        seeds=dict(model=r['model_seed'],split=392005,data=None,augmentation=r['model_seed'] if r['family']=='sdg' and r['variant']=='full' else None,support=None,evaluation=392005),
        training=dict(epochs=r['epochs'],stages=r['stages'],batch_size=r['batch_size'],optimizer=r['optimizer'],lr=r['lr'],weight_decay=0,schedule=r['schedule']),
        config_ref='experiments/phase1_dg_nativebudget/matrix.json#/rows/'+str(i),
        seed_notes='model controls initialization and loader RNG; SDG full also controls native augmentation; data not resampled; support N/A; evaluation fixed shared views',
        epochs=r['epochs'],optimizer=r['optimizer'],lr=r['lr'],
        budget_ref='matrix.json; IB stages2 x max200, user RIEI/DRIFT100, SDG200',
        resolved_config_ref='runtime.json#/run_root/'+r['row_id']+'/source/resolved_config.json',
        output_root='runtime.json#/run_root/'+r['row_id'],
        log_path='runtime.json#/log_root/'+r['row_id']+'.log',
        command='python -u pipeline.py worker --config RUN_ROOT/'+r['row_id']+'.json') for i,r in enumerate(rows)]
    template.write_text(json.dumps(spec,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    subprocess.run([sys.executable, str(ROOT/'tools/experiment_registry.py'),'new','--spec',str(template)],check=True)
    print('REGISTERED',report_root)


if __name__=='__main__':main()
