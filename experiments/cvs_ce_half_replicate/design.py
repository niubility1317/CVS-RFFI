"""User-requested CE_HALF seed completion; original E20 parents plus E10."""
from copy import deepcopy
from pathlib import Path
from experiments.cvs_As_replicate49 import design as parents

ROOT = Path(__file__).resolve().parents[2]
PROJECT, SOURCE = parents.PROJECT, parents.SOURCE
RUN = '20261019-phase1-FcrossRX-CEhalf-e30-s0203-m2-r01'
RELEASE = 'cvs_FcrossRX_CEhalf_E30_s0203_20261019_r01'
OWNER = 'codex/root/ce-half-three-seeds'
BASE = Path(PROJECT) / 'runs' / RUN
SEEDS = (2026092702, 2026092703)
STEPS, UPDATES = 222, 2220
RX = FIT = parents.RX
FULL_FP32_POLICY, CLASSES = parents.FULL_FP32_POLICY, parents.CLASSES
SCENES, FORMAL_VIEWS, TRUTH = parents.SCENES, parents.FORMAL_VIEWS, parents.TRUTH
PANEL, SOURCE_CACHE = parents.PANEL, parents.SOURCE_CACHE
VIEWS = ('clean',)
GPU_IDS = (2, 3)
read, write = parents.read, parents.write
SEED01_RUN = '20261018-phase1-FcrossRX-satelliteCE-temperature-e30-s2026092701-m2-r01'
SEED01 = Path(PROJECT) / 'runs' / SEED01_RUN / 'CE_HALF-s2026092701'
REFERENCE_RUN = parents.RUN


def rows():
    return [dict(arm='CE_HALF', row_id='CE_HALF-s' + str(s), model_seed=s) for s in SEEDS]


def parent_row(c):
    return next(r for r in parents.rows() if r['arm'] == 'B01' and r['model_seed'] == c['model_seed'])


def parent_config(c, segment=30):
    return parents.config(parent_row(c), segment)


def config(row, segment=30):
    if row not in rows() or segment != 30:
        raise ValueError('Only fixed CE_HALF seeds02/03 own20+10 permitted')
    inherited = parent_config(row)
    c = deepcopy(inherited)
    c.update(schema='CE_HALF_seed_completion_v1', run_id=RUN, **row,
             launch_owner=OWNER, output_root=str(BASE / row['row_id'] / 'source'),
             method='replicate_original_CE_HALF', stage='CE_HALF_E30_seed_completion',
             model_initialization='Own matching seed B01 scratch20 student and EMA; reset AdamW/cos10',
             test_views=list(VIEWS),
             ce_objective=dict(temperature=1., weight=.5, smoothing=.01),
             ce_objective_notes=dict(T_squared=False, deployment_head_scale=30.,
                 formula='delta=.25*(mean_two_pass_new_CE64-mean_two_pass_original_CE64)',
                 unchanged='ordinary/raw CE, all symmetric RDrop KL, original U/EMA and physical channel',
                 all_labeled_As_retained=True, extra_NN=0, extra_parameters=0),
             source_selection='Fixed FINAL10 cumulative E30; no selection by clean or satellite scores',
             baseline_reference='Reuse corresponding seed original NoAs30/As30; never retrain or repredict',
             budget_authorization='User explicitly requests complete three CE_HALF seeds; seed01 reused, each new seed adds10',
             benchmark_context='User-requested replication after exposed seed01 clean result; not independent blind confirmation',
             goal=dict(user_requested_fixed_seed_completion=True, no_performance_early_stop=True,
                       no_automatic_E50=True, no_temperature_or_weight_sweep=True,
                       test_scope='clean_only_current_chat_instruction'))
    return c


def validate(c):
    row = next((r for r in rows() if r['row_id'] == c.get('row_id')), None)
    if row is None or c != config(row):
        raise ValueError('Fixed CE_HALF parent/seed/recipe/clean scope differs')
    return c


def make_args(c, device='cuda:0'):
    validate(c)
    a = parents.make_args(parent_config(c), device)
    a.run_id, a.candidate_id, a.output_dir = RUN, c['row_id'], c['output_root']
    a.baseline_ckpt = c['parent_source'] + '/final_ssdg.pth'
    if a.seed != c['model_seed'] or a.epochs != 10 or a.label_epochs != 0 or a.pseudo_epochs != 10:
        raise ValueError('Actual native seed or original E10 tail differs')
    return a
