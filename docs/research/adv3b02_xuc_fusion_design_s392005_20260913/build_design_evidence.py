"""Build a design manifest and recompute published counts; never launch training."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

SCENES = ['clean', 'leo_clear_weak', 'leo_low_elev_weak', 'leo_rain_weak']


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def write(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8', newline='\n')


def build(root, out):
    v2path = root / 'automation_reports/CV-SincNet/core90_cross_response_v2_s392005_20260911_r1/completed_detail_20260912/completed_summary.json'
    gamepath = root / 'code/snapshots/core90_game_20260911_wt/docs/evidence/core90_target25_392005_20260911.json'
    deeppath = root / 'code/snapshots/core90_game_20260911_wt/docs/evidence/core90_c2_s4_full_analysis.json'
    xpath = root / 'automation_reports/CV-SincNet/mechanism_activation_audit_20260910/ecrs_effects.csv'
    recipepath = root / 'code/snapshots/core90_cross_response_v2_20260911_wt/code/configs/phase1_core90_cross_response_v2.json'
    v2, game, deep = read(v2path), read(gamepath)['scores'], read(deeppath)
    history = []
    with xpath.open(encoding='utf-8-sig', newline='') as stream:
        for row in csv.DictReader(stream):
            if row['row'] in ('X0_A1_RUNTIME', 'X1_CROSS_RX_CLEAN', 'X2_CROSS_RX_VIEWS'):
                history.append(dict(family='A1_X', row=row['row'], scope='target_RX_only', n_per_scene=168000,
                                    clean=float(row['target_clean']), leo_mean=float(row['target_leo_mean']),
                                    source_file=xpath.relative_to(root).as_posix(), evidence_kind='published_aggregate'))
    for row in ('U0', 'U1', 'Ux', 'Ux_normalized'):
        for target_only in (False, True):
            values, counts = {}, {}
            for scene in SCENES:
                group = v2[row]['groups'][scene]
                cells = ([group['named'][key] for key in ('test_seen_day_unseen_rx', 'test_unseen_day_unseen_rx')]
                         if target_only else [group['aggregate']])
                total, correct = sum(c['tx_total'] for c in cells), sum(c['tx_correct'] for c in cells)
                assert total == (168000 if target_only else 198000)
                values[scene] = correct / total * 100
                counts[scene] = dict(correct=correct, total=total)
            history.append(dict(family='CROSS_RESPONSE_V2', row=row,
                scope='target_RX_only' if target_only else 'mixed_RX_and_day', n_per_scene=total,
                clean=values['clean'], leo_mean=sum(values[s] for s in SCENES[1:])/3,
                scenes=values, counts=counts, source_file=v2path.relative_to(root).as_posix(),
                evidence_kind='recomputed_disjoint_group_counts'))
    for row in ('B0', 'B1', 'C1', 'C2', 'S4'):
        values = {s: game[row]['scenes'][s]['accuracy'] * 100 for s in SCENES}
        assert game[row]['prediction_count'] == 672000 and game[row]['complete']
        history.append(dict(family='GAME_LEGACY', row=row, scope='target_RX_only', n_per_scene=168000,
            clean=values['clean'], leo_mean=sum(values[s] for s in SCENES[1:])/3,
            scenes=values, source_file=gamepath.relative_to(root).as_posix(), evidence_kind='published_scorer_fractions'))
    lookup = {(r['family'], r['row'], r['scope']): r for r in history}
    contrasts = []
    for family, a, b, scope in [('A1_X','X1_CROSS_RX_CLEAN','X0_A1_RUNTIME','target_RX_only'),
                               ('CROSS_RESPONSE_V2','Ux_normalized','U1','target_RX_only'),
                               ('CROSS_RESPONSE_V2','Ux_normalized','U0','target_RX_only'),
                               ('CROSS_RESPONSE_V2','Ux_normalized','Ux','target_RX_only'),
                               ('GAME_LEGACY','C2','B0','target_RX_only'),
                               ('GAME_LEGACY','C2','S4','target_RX_only')]:
        x, y = lookup[family,a,scope], lookup[family,b,scope]
        contrasts.append(dict(family=family, comparison=a+' - '+b, scope=scope,
                              clean_pp=x['clean']-y['clean'], leo_pp=x['leo_mean']-y['leo_mean']))
    c2 = deep['C2']
    assert c2['epochs'] == 200 and c2['steps'] == 9800 and c2['head_steps'] == 0
    assert c2['actions']['CORRECT'] == 167 and c2['audit_count'] == 31
    evidence = dict(status='VERIFIED_AGGREGATE_RECOMPUTATION', primary_files=[str(p.relative_to(root)).replace('\\','/') for p in [v2path,gamepath,deeppath,xpath]],
        limitation='Existing complete-log analyses and saved scoring counts were inspected; this task did not reparse remote raw predictions or every original stdout.',
        history=history, contrasts=contrasts,
        c2_activation={k:c2[k] for k in ['epochs','steps','actions','head_steps','audit_count','recovery_improved','stages']})
    write(out/'historical_evidence.json', evidence)
    with (out/'historical_comparison.csv').open('w',encoding='utf-8',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=['family','row','scope','n_per_scene','clean','leo_mean','evidence_kind','source_file'],extrasaction='ignore')
        writer.writeheader();writer.writerows(history)

    rows = [
        ('M00','ADV3B02_CORE90','ordinary',0,0,'off','fixed','core90', 'CORE90方法基线，标准loader、原BN/MixStyle'),
        ('M01','CORE90_GRID','v2_grid',0,0,'off','fixed','core90', '共同执行载体；隔离网格、BN和MixStyle角色策略'),
        ('M02','GRID_X','v2_grid',1,0,'off','fixed','core90', 'X1判别约束的CORE90移植'),
        ('M03','GRID_U','v2_grid',0,1,'off','fixed','core90', 'Ux_normalized机制复现，统一数值环境'),
        ('M04','GRID_C2','v2_grid',0,0,'legacy_both','legacy_capability','core90', 'C2原控制/课程移植到共同载体'),
        ('M05','GRID_XU','v2_grid',1,1,'off','fixed','core90', '几何双模块；第二个2x2的无控制无课程基点'),
        ('M06','GRID_XC2','v2_grid',1,0,'legacy_both','legacy_capability','core90', 'X与原C2联合'),
        ('M07','GRID_UC2','v2_grid',0,1,'legacy_both','legacy_capability','core90', 'U与原C2联合'),
        ('M08','GRID_XUC2','v2_grid',1,1,'legacy_both','legacy_capability','core90', '直接三模块组合，检验叠加是否已经足够'),
        ('M09','A1_X0_REFERENCE','a1_native',0,0,'off','a1_native','a1_fasttrust_daot', 'X0训练体系桥接；保留原FT/DAOT及步数'),
        ('M10','A1_X1_REFERENCE','a1_native',1,0,'off','a1_native','a1_fasttrust_daot', 'X1训练体系桥接，只与M09隔离X效应'),
        ('M11','CORE90_C2_REFERENCE','ordinary',0,0,'legacy_both','legacy_capability','core90', 'C2方法桥接；同一新数据契约，不加载旧权重'),
        ('M12','XUC_INTEGRATED','v2_grid',1,1,'reliable_both','exposure_matched_capability','core90', '预先指定的完整融合主候选'),
        ('M13','XUC_CONTROL_ONLY','v2_grid',1,1,'reliable_both','fixed','core90', '保留可靠动作控制，移除能力课程重排'),
        ('M14','XUC_CURRICULUM_ONLY','v2_grid',1,1,'off','exposure_matched_capability','core90', '保留能力课程重排，移除动作控制'),
    ]
    runs = []
    for rid,name,carrier,x,u,control,curriculum,pipeline,purpose in rows:
        runs.append(dict(id=rid,name=name,seed=392005,data_seed=392005,evaluation_seed=392005,
            epochs=200,initialization='scratch',checkpoint_selection='fixed_final_E200',
            carrier=carrier,pipeline=pipeline,x_enabled=bool(x),u_enabled=bool(u),
            lambda_x=.05 if x else 0.,lambda_u_interaction=.01 if u else 0.,x_margin=.2,
            control=control,curriculum=curriculum,amp=False,primary=(rid=='M12'),
            steps_per_epoch=222 if pipeline=='a1_fasttrust_daot' else 49,
            labeled_batch=128,unlabeled_batch=256 if pipeline=='a1_fasttrust_daot' else 128,
            purpose=purpose))
    common = dict(dataset='ManySig.pkl',equalized=1,source_receivers=[1,3,4,6,8],source_days=[1,2,3],
        target_receivers=[0,2,5,7,9,10,11],target_days=[0,1,2,3],old_class_count=6,
        source_sizes=dict(L=6300,U=56700,V=27000),source_ratios=[.07,.63,.30],
        labels_hidden_in_U=True,shared_physical_role_indices=True,
        model_seed=392005,data_seed=392005,evaluation_seed=392005,
        training_initialization='scratch',historical_checkpoints_allowed=False,
        source_V='one read-only source validation set',training_target_access=False,
        source_development_probes='L_s only, fit/monitor capture groups disjoint',
        scenes=SCENES,test_count_per_scene=168000,test_predictions_per_run=672000,
        formal_target_scope='receiver-disjoint existing research benchmark; not a fresh blind confirmation set',
        model_size='M',model_variant='lite_d',identity_dim=160,
        label_epochs=130,pseudo_epochs=70,optimizer='AdamW',lr=.0002,weight_decay=.0001,
        max_grad_norm=5.,amp=False,lambda_adv=.35,lambda_sat_cls=.68,lambda_sat_cons=0.,sat_ce_start_epoch=80,
        grid=dict(P=4,Q=4,K=2,blocks_per_batch=4,within_block_same_day=True,within_block_same_condition=True,
                  scheduler='uniform',role_policy='balanced_partitions_v2',mixstyle_role_policy='donor_only',
                  normalization='freeze BN running-state in labeled clean+sat forward, as V2; preserve other baseline forwards',
                  loss_view='clean_only',near_zero_norm_threshold=1e-8))
    cstar = dict(status='NEW_DESIGN_NOT_IMPLEMENTED',geometry_losses_unchanged=True,
        audit_every_main_steps=250,audit_max_age_steps=10,calibration_steps=[0,250,500],
        correction_max_fraction=.20,max_head_catchup_steps_per_action=3,
        recovery_steps=40,recovery_lr=.002,recovery_fresh_optimizer=True,
        recovery_validation='paired capture-group monitor CE upper95(recovered-online)<=0 AND fit CE decreases; otherwise UNKNOWN',
        recovery_bootstrap_resamples=500,recovery_bootstrap_seed=392005,
        invalid_recovery_action='ordinary main update, no head catchup or recovery-direction EG; curriculum can use independent valid identity evidence',
        head_readable_min=.1,lag_enter=.01,lag_exit=.005,imbalance_enter=.5,imbalance_exit=.3,
        action_confirmation=2,action_cooldown_steps=250,
        geometry_thresholds='frozen per-row quantiles from L_s observations at steps 0/250/500; report actual values',
        geometry_rules={'identity':'current >= calibration q10','margin':'current >= calibration q10',
                        'tx_main_energy':'current >= calibration q10','unit_interaction':'current <= calibration q90',
                        'leo_cosine':'current >= calibration q10'},
        geometry_confirmation=3,
        curriculum_window_epochs=10,window_boundaries_split_at_epochs=[41,80,91,131],
        curriculum_identity='reorder immutable full-batch tickets inside each window, preserve exact source IDs, applied masks and LEO scene exposure multiset',
        curriculum_choice='when capability confirmed, choose hardest remaining ticket; otherwise easiest; tie by canonical ticket_id; every ticket consumed once',
        difficulty_score='source-clean L_s audit per-scene TX CE excess times applied_mask_fraction; no target score; canonical scene order tie-break',
        window_end='consume remaining tickets in canonical order; log coverage-driven consumption separately from capability progression',
        fixed_arm='canonical tickets, same set; no capability-based reordering',
        coupled_EG='full current CORE90+X+U field evaluated twice on same ticket, RNG and frozen pseudo labels; one persistent commit')
    matrix = dict(schema='adv3b02_xuc_design_v1',artifact_kind='design_specification_not_executable_launcher_config',
        status='DESIGN_COMPLETE_NOT_IMPLEMENTED_NOT_LAUNCHED',launch=False,experiment_count=15,
        common=common,cstar=cstar,runs=runs,
        primary_contrasts=['M12-M00','M12-M01','M12-M05','M12-M08'],
        factorial_rows=['M01','M02','M03','M04','M05','M06','M07','M08'],
        control_curriculum_factorial={'neither':'M05','control_only':'M13','curriculum_only':'M14','both':'M12'},
        interaction_contrasts={'XU_no_C':{'M05':1,'M02':-1,'M03':-1,'M01':1},
            'XU_with_C':{'M08':1,'M06':-1,'M07':-1,'M04':1},
            'XUC_three_way':{'M08':1,'M06':-1,'M07':-1,'M05':-1,'M02':1,'M03':1,'M04':1,'M01':-1},
            'revised_control_curriculum':{'M12':1,'M13':-1,'M14':-1,'M05':1}},
        interpretation='Factorial C denotes original C2 control+curriculum package. Repaired controller package is separately identified only as a package. Native bridges are not equal-compute comparisons.',
        future_implementation=['shared source-index adapter','complete objective step transaction','C2 legacy-compatible adapter',
                               'reliable source evidence classification','exposure-matched ticket curriculum','single truth-last scorer'],
        expected_all_predictions=15*672000,nominal_main_update_opportunities=sum(r['epochs']*r['steps_per_epoch'] for r in runs))
    write(out/'matrix15.json',matrix)
    with (out/'matrix15.csv').open('w',encoding='utf-8',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(runs[0]));writer.writeheader();writer.writerows(runs)
    recipe = read(recipepath)['baseline_args']
    overrides=dict(seed=392005,amp=False,baseline_ckpt='',from_scratch=True,best_metric='clean_val_tx',
        test_eval_start_epoch=201,test_eval_interval=0,test_eval_final_window=0,enable_joint_safe_guard=False,
        phase2_export_prototypes=False)
    recipe.update(overrides)
    write(out/'core90_recipe_reference.json',dict(artifact_kind='reference_arguments_not_executable_config',
        source=recipepath.relative_to(root).as_posix(),overrides=overrides,baseline_args=recipe,
        mandatory_semantics='final E200 only; never build/access target loader during training; Phase2 exports disabled; native bridge overrides explicitly recorded in matrix15.json'))
    legacy = {k:v for k,v in c2['config'].items() if k.startswith('game_') and not any(s in k for s in ['path','resume','config','row','replay'])}
    write(out/'c2_legacy_reference.json',dict(source=deeppath.relative_to(root).as_posix(),training_commit='300ff4a9',
        historical_game_settings=legacy,required_overrides=dict(game_split_seed=392005,seed=392005,game_resume=''),
        historical_implicit_settings=dict(recovery_lr=.02,recovery_steps=40,
            recovery_source='online adv_head copy with fresh AdamW state',
            recovered_monitor_degradation_was_not_a_validity_rejection=True,
            lag_negative_difference_clamped_to_zero=True),
        note='Historical mechanism settings, not permissions. Runtime, data adapter and output paths must be explicitly integrated. No historical calibration thresholds or state are inherited.'))
    print(json.dumps({'history_rows':len(history),'contrasts':contrasts,'matrix_rows':len(runs),
                      'main_update_opportunities':matrix['nominal_main_update_opportunities'],
                      'predictions':matrix['expected_all_predictions']},ensure_ascii=False))


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--workspace',type=Path,required=True)
    parser.add_argument('--output',type=Path,default=Path(__file__).resolve().parent)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    build(args.workspace,args.output)
