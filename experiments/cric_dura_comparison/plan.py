"""Prospective matrix only. Importing or running this file never launches training."""
from copy import deepcopy
from experiments.phase1_external_dg.plan import make_plan as external_plan, SEEDS, VIEWS
from experiments.phase1_dg_sourcecount6.plan import counts, SOURCE_ORDER, TARGET

RUN = '20261021-phase1-T1-DURA-comparison-m153-r01'
ANALYSIS_RUN = '20261021-phase1-CRIC-DURA-F0-compare-m54-r01'
F_RUN = '20261021-phase1-T1-AsKL0-e30plus20-m3-r01'
G_RUN = '20261021-phase1-restoration-R65-four-route-s2026092701-m4-r69'
REGIMES = ('native_cric', 'matched_sat_supervised', 'matched_sat_plain_pl')


def make_plan():
    old = external_plan()['protocols']
    protocols = {}
    role_counts = {'F0': (6300, 56700, 27000), 'F1': (6300, 56700, 27000),
        'F2': (6300, 56700, 27000), 'chrono': (6300, 56700, 27000),
        'ManyRx': (6533, 58828, 28023), 'label01': (630, 62370, 27000),
        'label05': (3150, 59850, 27000)}
    for name, roles in role_counts.items():
        p = deepcopy(old[name])
        p['expected_counts'] = dict(zip(('L_s', 'U_s', 'V'), roles))
        p['target_count'] = 123021 if name == 'ManyRx' else 168000
        p['primary_endpoint'] = 'target_D3_42000' if name == 'chrono' else 'all_target_dates'
        protocols[name] = p
    for m in (1, 2, 3, 4, 6):
        for budget in ('natural', 'fixed'):
            pid = 'm%d-%s' % (m, budget)
            protocols[pid] = dict(protocol_id=pid, dataset='ManySig', source_count=m,
                source_rxs=list(SOURCE_ORDER[:m]), target_rxs=list(TARGET),
                source_days=[1, 2, 3], target_days=[0, 1, 2, 3], split_seed=392005,
                expected_counts=counts(m, budget), target_count=144000,
                primary_endpoint='all_target_dates', label_budget=budget,
                equalized=1, out_len=256, normalize=True)
    rows = []
    for pid in protocols:
        for regime in REGIMES:
            for seed in SEEDS:
                native = regime == 'native_cric'
                sup = regime == 'matched_sat_supervised'
                reuse = pid == 'F0' and native
                rows.append(dict(row_id='%s-%s-s%d' % (pid, regime, seed),
                    protocol_id=pid, regime=regime, model_seed=seed,
                    display_method='CRIC+DURA' if native else
                        ('PAMF-Sup+DURA-L' if sup else 'PAMF-PL+DURA-LU'),
                    status='FROZEN_RESULT_REUSE' if reuse else 'PLANNED_NOT_LAUNCHED',
                    epochs=50, steps_per_logical_epoch=222, optimizer_updates=11100,method_id='pamf_t1_e50',
                    trajectory=[20,10,20],supervised_only_epochs=13 if native else (50 if sup else 0),
                    logical_epoch_note='Update-defined epochs; not guaranteed dataset passes on changed contracts',
                    labeled_batch=128, unlabeled_batch=256 if native else (0 if sup else 128),
                    U_use=not sup, U_start_epoch=14 if native else (None if sup else 1),
                    satellite_concat=not native, satellite_CE_weight=0.0 if native else .68,
                    satellite_consistency_weight=0.0,
                    own_EMA=native, native_U_gate=native,
                    plain_PL_threshold=.95 if regime == 'matched_sat_plain_pl' else None,
                    plain_PL_weight=1.0 if regime == 'matched_sat_plain_pl' else None,
                    G_source_roles='L_s' if sup else 'L_s+U_s_TX_hidden',
                    G_model_seed=2026092701, G_shared_across_F_seeds=True,
                    G_inference='dura_v1; reuse=True, graph=True, native=False; FP32,TF32off',
                    default_head_joint=True, posthoc_head_fit=False,
                    initialization='existing exact-contract F0' if reuse else 'scratch matching contract',
                    F_parent_run=F_RUN if reuse else None,
                    no_cross_contract_checkpoint=True, views=list(VIEWS)))
    g_jobs = []
    for pid in protocols:
        for role in ('L_s', 'L_s+U_s_TX_hidden'):
            reused = pid == 'F0' and role != 'L_s'
            g_jobs.append(dict(job_id='%s-G-%s-s2026092701' % (pid, 'L' if role == 'L_s' else 'LU'),
                protocol_id=pid, source_roles=role, model_seed=2026092701,
                status='EXACT_CONTRACT_FROZEN_REUSE' if reused else 'PLANNED_ANCESTRY_PORT_REQUIRED',
                TX_labels=False, F_training_access=False, target_access=False,
                final_student_updates=4940, final_logical_epochs=20,
                batch=256, partial_last_batch_policy='Original F0 batches include partial final slot; preserve actual exposures',
                source_passes_note='Keep original 247-update logical epoch across contracts; disclose repeats and unique IQ',
                ancestry='Same-contract full original teacher/compression path required; not just final R69 stage',
                upstream_budget_status='Resolve actual inherited path before implementation; 100 inherited component epochs is not total compute',
                no_cross_contract_teacher=True))
    for seed in SEEDS[1:]:
        g_jobs.append(dict(job_id='m3-natural-G-LU-s%d' % seed, protocol_id='m3-natural',
            source_roles='L_s+U_s_TX_hidden', model_seed=seed,
            status='PLANNED_ANCESTRY_PORT_REQUIRED',
            purpose='Independent G-seed robustness: cross all three G with existing three F seeds; no new F',
            no_cross_contract_teacher=True))
    value = dict(schema='cric_dura_prospective_comparison_v1', run_id=RUN,
        status='DESIGN_COMPLETE_NOT_LAUNCHED', protocols=protocols, rows=rows,
        G_training_pipelines=g_jobs, views=list(VIEWS), excluded_source_counts=[5],
        note='Exclusion applies to new source-count sweep; existing F0 five-source results remain historical main comparison',
        expected_F_models=153, reused_F_models=3, new_F_training=150,new_F_ablation_training=63,
        G_pipeline_count=36, reused_G_pipelines=1, new_G_pipelines=35,
        core_native_F_models=51, additional_matched_control_F_models=102,
        planned_logical_view_cells=2541, reused_logical_view_cells=42,
        readouts='Each F raw and its regime G; additionally F-Sup with G-LU; extra two m3 G x three native F seeds',
        effective_anchor_rule='Physical RX only; m1 cross-RX contrast returns finite zero and coverage=0',
        uncertainty='Three F seeds conditional on one fixed G per contract; only m3-natural has 3x3 crossed G/F robustness',
        resource_policy='Independent new run; never alter active 360-row comparator controller or healthy workers',
        launch_status='No training/prediction command implemented by this design module',
        no_target_feedback=True)
    validate(value)
    return value


def validate(p):
    assert len(p['protocols']) == 17
    assert len(p['rows']) == 153 and len({r['row_id'] for r in p['rows']}) == 153
    assert sum(r['status'] == 'FROZEN_RESULT_REUSE' for r in p['rows']) == 3
    assert len(p['G_training_pipelines']) == 36
    assert sum(r['status'] == 'EXACT_CONTRACT_FROZEN_REUSE' for r in p['G_training_pipelines']) == 1
    for pid, c in p['protocols'].items():
        assert not set(c['source_rxs']) & set(c['target_rxs']), pid
        assert c.get('source_count') != 5, pid
    for r in p['rows']:
        assert r['optimizer_updates'] == 11100 and r['default_head_joint']
        assert not r['posthoc_head_fit'] and r['no_cross_contract_checkpoint']
        if r['regime'] == 'native_cric':
            assert r['own_EMA'] and not r['satellite_concat'] and r['U_start_epoch'] == 14
        if r['regime'] == 'matched_sat_supervised':
            assert not r['U_use'] and r['G_source_roles'] == 'L_s'
        if r['regime'] == 'matched_sat_plain_pl':
            assert r['satellite_concat'] and r['plain_PL_threshold'] == .95 and not r['own_EMA']


if __name__ == '__main__':
    import json
    print(json.dumps(make_plan(), ensure_ascii=False, indent=2))
