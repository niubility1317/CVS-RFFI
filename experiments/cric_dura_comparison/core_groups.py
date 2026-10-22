"""Three functional objective blocks, grouped without using target outcomes."""
from itertools import product

RUN = '20261022-phase1-PAMF-core-mechanisms-m24-r01'
CONTROL = '20261022-phase1-PAMF-core-control-r01'
READOUT = '20261022-phase1-PAMF-core-G-L-readouts-m9-r01'
ANALYSIS = '20261022-phase1-PAMF-core-factorial-analysis-r01'
REPAIR = '20261022-phase1-T1-DURA-F-ablation-binding-m63-r01'
MAIN = '20261021-phase1-T1-DURA-comparison-m153-r01'
L_READOUT = '20261022-phase1-PAMF-supervised-Lonly-readouts-m6-r01'
SEEDS = (2026092701, 2026092702, 2026092703)
NAMES = ('cross_RX_identity', 'paired_prediction_consistency', 'selective_SSL')
REUSE = {'111': 'T1', '011': 'F16', '101': 'F04', '110': 'F20'}


def switches(bits):
    if len(bits) != 3 or any(v not in '01' for v in bits):
        raise ValueError('Three declared core switches required')
    return dict(zip(NAMES, (v == '1' for v in bits)))


def validate_config(c):
    value = c.get('core_mechanisms')
    if value is None:
        return None
    if not isinstance(value, dict) or set(value) != set(NAMES) or any(type(v) is not bool for v in value.values()):
        raise ValueError('Exact three boolean mechanism groups required')
    a, b, use_u = (value[k] for k in NAMES)
    if (c.get('method_id') != 'pamf_t1_e50' or c.get('regime') != 'native_cric'
            or c.get('ablation') != (None if use_u else 'F20')
            or c.get('satellite_concat') is not False
            or c.get('optimizer_updates') != 11100
            or c.get('U_use') is not use_u or c.get('own_EMA') is not use_u
            or c.get('native_U_gate') is not use_u):
        raise ValueError('Core factorial must preserve final T1 scaffold and exact U-off execution')
    if c.get('supervised_architecture'):
        raise ValueError('Do not mix network changes with the objective factorial')
    return a, b, use_u


def check_parent(c, parent):
    if c.get('core_mechanisms') is not None or parent.get('core_mechanisms') is not None:
        validate_config(c)
        validate_config(parent)
        if c.get('core_mechanisms') != parent.get('core_mechanisms'):
            raise ValueError('Core row must inherit its own mechanism configuration')


def plan():
    rows = []
    for values in product('01', repeat=3):
        bits = ''.join(values)
        for seed in SEEDS:
            rows.append(dict(row_id='ABC' + bits + '-s' + str(seed), bits=bits,
                model_seed=seed, core_mechanisms=switches(bits),
                source='REUSE_' + REUSE[bits] if bits in REUSE else 'NEW_MATCHED_T1',
                new_F=bits not in REUSE, epochs=50, optimizer_updates=11100,
                trajectory=[20, 10, 20], backbone='Unchanged SIG_TIME_X3 PAMF/default joint head',
                CE_scaffold='0.5CEpair ordinary128 +0.25CEpair ordinary64 +0.25CEpair As64 +0.25CEpair raw128; LS.01; As KL0',
                paired_KL='0.5ordinary128 +0.25ordinary64 +0.25raw128; coefficient1',
                source_roles='L+hiddenTX_U' if bits[2] == '1' else 'L only, U/EMA/entropy fully off',
                readouts=['raw', 'DURA-L', 'DURA-LU'], no_new_G=True))
    return dict(schema='pamf_three_core_factorial_v1', rows=rows,
        main_table=['111', '011', '101', '110'], full_factorial_cells=8,
        model_seeds=list(SEEDS), logical_models=24, new_F=12, reused_F=12,
        new_F_epochs=600, new_F_updates=133200, new_G=0,
        new_prediction_rows=21, new_score_rows=21, aggregate_CPU_jobs=1,
        new_stage_jobs=55, target_feedback_used=False,
        grouping_basis='Domain identity objective / stochastic prediction consistency / selective unlabeled self-training; no result-dependent regrouping',
        claim='Factorial effects of F training; fixed G-L/G-LU readouts reported separately; C-off+G-LU is not a fully L-only system')
