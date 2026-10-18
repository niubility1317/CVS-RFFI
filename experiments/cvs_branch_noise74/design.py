"""Four true additive contributions under fixed receiver-state noise removal."""
from pathlib import Path
import json

ROOT = Path(__file__).resolve().parents[2]
PROJECT = Path('/home/szu2070436088/2510044040/CV-SincNet')
RUN = '20261019-diagnostic-FcrossRX-branchNoise-sourceV-s2026092701-m2-r01'
RELEASE = 'cvs_FcrossRX_branch_noise_sourceV_20261019_r01'
OWNER = 'codex/root/branch-noise74'
BASE = PROJECT / 'runs' / RUN
PARENT_RUN = '20261018-phase1-FcrossRX-impulse-factorial-e30-s2026092701-m6-r01'
PANEL = PROJECT / 'runs/20261017-diagnostic-FcrossRX-fullsource-residual-s2026092701-m3-r01/diagnostic'
BATCH, N = 90, 1800
SCENES = ('practical_mid_urban', 'practical_low_urban')
BRANCHES = ('TF', 'PA', 'REF', 'SIG')
OPERATIONS = ('zero', 'swap_nonoise', 'swap_clean')
ROWS = ('NoAs', 'As')
ARMS = dict(NoAs='B00', As='B01')
# Native model examples versus additional default-head-only examples.
NN = 32404
EXTRA_HEAD = 219652

def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))

def write(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n', encoding='utf8')

def config():
    return dict(schema='source_branch_noise74_v1', run_id=RUN, owner=OWNER,
        rows=list(ROWS), checkpoint_arm=ARMS, checkpoint_run=PARENT_RUN,
        source_role='V', unique_source_samples=N, draws=2, scenes=list(SCENES),
        branches=list(BRANCHES), operations=list(OPERATIONS), batch=BATCH,
        source_RX=[1,3,4,6,8], source_day=[1,2,3], source_TX=list(range(6)),
        model_seed=2026092701, split_id=392005, U_truth_read=False,
        target_access=False, training_updates=0, head_fit_updates=0,
        new_model_parameters=0, CUDA_VISIBLE_DEVICES='', threads=2,
        full_model_examples=NN, additional_head_only_examples=EXTRA_HEAD,
        new_physical_simulations=0, diagnostic_fc_hz=2450000000.,
        parent_training_fc_hz=2462000000., formal_test_changed=False,
        noise_control='Existing no_noise/fixed with full receiver state/compensation; not total effect',
        classification='Re-sum actual postLN+gain contributions in native order; original head with final L2 and scale30',
        scope='Source explanatory oracle counterfactuals; no deployment, trained improvement or attainable upper-bound claim',
        source_rules=dict(local_noise=dict(swap_gain_min_pp=2., whole_gain_fraction_min=.5,
            conditions='One fixed branch must satisfy both in each of mid/low x two draws; whole gain positive'),
            misleading=dict(zero_gain_min_pp=1., clean_zero_max_cost_pp=.1,
            conditions='One fixed branch must satisfy in all four conditions and clean control'),
            auto_training=False, E50=False))

def validate_config(c):
    if c != config():
        raise ValueError('Only fixed two frozen source-model contribution diagnostics are authorized')
    return c

def validate_physics(engine):
    expected = dict(fs_hz=25000000., fc_hz=2450000000., reference_snr_db=30.,
        mode='post_sync', processing_route='residual', equalization_enabled=False,
        boundary='reflect', quality_adaptation_enabled=True,
        phase_tracking_enabled=True, iq_compensation_enabled=True)
    if not engine.get('EQ_off_G_equals_one') or engine.get('formal_test_changed'):
        raise ValueError('Existing full residual with EQoff G1 required')
    for scene in SCENES:
        c = engine['actual_configs'][scene]
        if any(c.get(k) != v for k,v in expected.items()):
            raise ValueError('Existing fixed source observation physics differ')
    return engine
