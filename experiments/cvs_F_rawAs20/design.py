"""One whole-input hypothesis on fixed pure F; matched scratch E20 controls."""
from copy import deepcopy
from pathlib import Path
from experiments.cvs_F_nontrue20 import design as base

read, write = base.read, base.write
ROOT = Path(__file__).resolve().parents[2]
RUN = '20261019-phase1-F-rawAs-e20-s2026092701-m2-r01'
RELEASE = 'cvs_F_rawAs_E20_20261019_r01'
LOCAL = 'local_artifacts/F_rawAs_E20_20261019'
OWNER = 'codex/root/F-rawAs20'
PROJECT = read(ROOT/'runtime.json')['project'] if (ROOT/'runtime.json').is_file() else 'PROJECT'
BASE = Path(PROJECT)/'runs'/RUN
REFERENCE_RUN = base.RUN
REFERENCE_RELEASE = 'cvs_F_sceneRisk_E20_20261019_r01'
SOURCE, PANEL, TRUTH = base.SOURCE, base.PANEL, base.TRUTH
FORMAL_VIEWS, SCENES, VIEWS, CLASSES = base.FORMAL_VIEWS, base.SCENES, base.VIEWS, base.CLASSES
FIT, FULL_FP32_POLICY = base.FIT, base.FULL_FP32_POLICY
SEED, STEPS, UPDATES = 2026092701, 222, 4440
GPU_IDS = (4, 5)
TRAIN_SCENES = ('practical_low_suburban', 'practical_mid_urban', 'practical_low_urban')
model_config = base.model_config


def rows():
    return [dict(row_id=a+'-s'+str(SEED), arm=a, model_seed=SEED)
            for a in ('RAW_NOAS', 'RAW_AS')]


def reference(arm):
    if arm not in ('B0', 'B1', 'C0'):
        raise ValueError('Only complete original pure-F references may be reused')
    return Path(PROJECT)/'runs'/REFERENCE_RUN/(arm+'-s'+str(SEED))


def config(row):
    if row not in rows():
        raise ValueError('Unregistered fixed whole-raw row')
    c = deepcopy(base.config(next(r for r in base.rows() if r['arm'] == 'B1')))
    c.update(schema='F_rawAs20_v1', run_id=RUN, **row,
             output_root=str(BASE/row['row_id']/'source'), launch_owner=OWNER,
             method='pure_F_rawAs20', stage='source_scratch20',
             As=row['arm'] == 'RAW_AS', training_recipe='pure F ORIGINAL; scratch20=13L+7U; '
             '.5 raw CE + .5 '+('full practical(raw)' if row['arm'] == 'RAW_AS' else 'raw')+' CE')
    c['training_mechanism'] = dict(mode='ORIGINAL', alpha=0., CROSS_RX=False, RDrop=False,
        L_forwards_per_sample=2, teacher_packets_per_step=0, raw_CE_weight=.5,
        extra_CE_weight=.5, first_input='raw original L after Dataset normalization',
        extra_input='complete practical(raw)' if row['arm'] == 'RAW_AS' else 'same raw L',
        CE_temperature=1., CE_smoothing=.01, normalization='mean over all128 in each slot',
        all_labeled_As_true_CE=True, ordinary_draw_shadow=True,
        ordinary_shadow_description='Generate original ordinary once in the native L context, '
        'discard its IQ; preserve random-stream consumption; first identity input is raw',
        native_use_aug=True, new_U_augmentation=False, U='original hidden TX gate+ownEMA unchanged',
        external_teacher=False, teacher_shadow=False, risk_q=False, new_deploy_parameters=0,
        extra_optimizer_steps=0, head_fit_updates=0)
    c['source_promotion'] = dict(RAW_AS_minus_B1_midlow_min_pp=.5,
        RAW_AS_minus_B0_fullVclean_min_pp=-.2, RAW_AS_minus_RAW_NOAS_fullVclean_min_pp=-.2,
        automatic_extension=False, max_cumulative_epochs=50, decision_before_formal_scores=True,
        on_fail='close fixed whole-raw recipe; no ratio/seed/schedule rescue')
    c.pop('source_falsification', None)
    c['baseline_definition'] = ('Fixed pure F ORIGINAL/default-head/U recipe; native L ordinary drawn '
        'only as shadow; raw first input; matched raw/raw and raw/As second slots')
    c['baseline_reference'] = ('Complete pure-F B0 ordinary-NoAs and B1 ordinary/As E20 reused '
        'as readouts only; no inherited checkpoint; RAW_NOAS is the new raw/raw matched control')
    c['goal'] = dict(primary='RAW_AS', low_urban_above=.60, clean_net_vs_matched_NoAs=True,
        suburban_gap_exploratory_pp=2., model_seeds_this_matrix=1, target_feedback=False,
        goal_completed=False)
    c['budget_authorization'] = ('Two new same-seed scratch E20 rows; 40 model-epochs/8880 updates '
        'total; no parent weights. E20+30 only after fixed source PASS and matched continuation design.')
    c['cost_claim'] = ('Per row: 1136640 student L packets plus original396900 U and396900 ownEMA '
        'eval packets;568320 ordinary shadow packets;0 external CSIL packets. RAW_AS adds568320 '
        'full practical generations. Reused B1 historical teacher shadow cost remains separate.')
    return c


def validate(c):
    row = next((r for r in rows() if r['row_id'] == c.get('row_id')), None)
    if row is None or c != config(row):
        raise ValueError('Fixed pure-F/source/default-head/scratchE20/raw recipe differs')
    return c


def make_args(c, device='cuda:0'):
    validate(c)
    a = base.make_args(base.config(next(r for r in base.rows() if r['arm'] == 'B1')), device)
    # The scoped ordinary factory changes only the actual first L input. Native
    # augmentation, U, optimizer, sampler and architecture arguments stay intact.
    a.run_id, a.candidate_id, a.output_dir = RUN, c['row_id'], c['output_root']
    return a
