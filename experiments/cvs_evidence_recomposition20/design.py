"""Fixed source-only PA evidence recomposition assay, before new training."""
from pathlib import Path
from experiments.cvs_crossrx_sharpness20 import design as parent

ROOT = Path(__file__).resolve().parents[2]
PROJECT, read, write = parent.PROJECT, parent.read, parent.write
RUN = '20261015-phase1-SIGTIME-PA-evidence-recomposition-sourceLV-s2026092701-r01'
RELEASE = 'cvs_SIGTIME_PA_evidence_recomposition_sourceLV_20261015_r01'
SOURCE_RELEASE = PROJECT + '/releases/' + parent.RELEASE
OUTPUT = PROJECT + '/runs/' + RUN + '/source_diagnostic'
ORDINALS = (0, 17, 35, 52)
STATES = ('INITIAL', 'FINAL20')
MODES = ('ZERO', 'SELF', 'SWAP_PA')
GATE = dict(minimum_qualified_cells=2, auxiliary_gradient_min=1e-8,
    SWAP_SELF_gradient_relative_difference_min=.05,
    SWAP_SELF_V_identity_margin_gain_min=1e-5,
    SWAP_SELF_V_preW_CE_reduction_min=1e-6,
    SWAP_SELF_V_feature_RMS_min=1e-7, minimum_auxiliary_changed_Sinc_kernels=1,
    interpretation='Source signal and single virtual-step direction only; not training efficacy, target selection or novelty proof')


def request():
    return dict(run_id=RUN, output_root=OUTPUT, source_release=SOURCE_RELEASE,
        source_row='BASE_CE-s2026092701', model_seed=2026092701,
        states=list(STATES), modes=list(MODES), anchor_ordinals=list(ORDINALS),
        V_cell_ordinals=[0, 1], coefficient=.1, source_falsification=GATE,
        source_roles=['L_s', 'V_s'], gradient_role='L_s', V_gradient=False,
        analysis_code_commit='release_commit.txt')


def validate(c):
    expected = request(); expected['analysis_code_commit'] = c.get('analysis_code_commit')
    if c != expected or not isinstance(c['analysis_code_commit'], str) or len(c['analysis_code_commit']) != 40:
        raise ValueError('Fixed source evidence assay request or actual commit differs')
    return c


def source_gate(records):
    if [(r['state'], r['ordinal']) for r in records] != [(s, o) for s in STATES for o in ORDINALS]:
        raise ValueError('All eight fixed source cells are required')
    qualified = []
    for r in records:
        swap, self_ = r['modes']['SWAP_PA'], r['modes']['SELF']
        if (swap['auxiliary_gradient_norm'] > GATE['auxiliary_gradient_min']
            and r['SWAP_SELF_gradient_relative_difference'] > GATE['SWAP_SELF_gradient_relative_difference_min']
            and r['SWAP_SELF_V_identity_margin_gain'] > GATE['SWAP_SELF_V_identity_margin_gain_min']
            and r['SWAP_SELF_V_preW_CE_reduction'] > GATE['SWAP_SELF_V_preW_CE_reduction_min']
            and r['SWAP_SELF_V_feature_RMS'] > GATE['SWAP_SELF_V_feature_RMS_min']
            and swap['physical_Sinc']['auxiliary_changed_kernels'] >= GATE['minimum_auxiliary_changed_Sinc_kernels']
            and swap['protected_auxiliary_gradients_None'] and self_['protected_auxiliary_gradients_None']):
            qualified.append(dict(state=r['state'], ordinal=r['ordinal']))
    return dict(passed=len(qualified) >= GATE['minimum_qualified_cells'], qualified_cells=qualified,
                qualified_by_state={s:sum(q['state']==s for q in qualified) for s in STATES})
