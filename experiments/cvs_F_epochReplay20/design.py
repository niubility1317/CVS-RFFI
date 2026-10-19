"""Pure F scratch E20; retain first As per physical L within each epoch."""
from copy import deepcopy
from pathlib import Path
from experiments.cvs_F_nontrue20 import design as base
read, write = base.read, base.write
ROOT = Path(__file__).resolve().parents[2]
RUN = '20261019-phase1-F-epochReplay-e20-s2026092701-m1-r01'
RELEASE = 'cvs_F_epochReplay_E20_20261019_r01'
LOCAL = 'local_artifacts/F_epochReplay_E20_20261019'
OWNER = 'codex/root/F-epochReplay20'
PROJECT = read(ROOT/'runtime.json')['project'] if (ROOT/'runtime.json').is_file() else 'PROJECT'
BASE = Path(PROJECT)/'runs'/RUN
REFERENCE_RUN = base.RUN
REFERENCE_RELEASE = 'cvs_F_rawAs_E20_20261019_r01'
SOURCE, PANEL, TRUTH = base.SOURCE, base.PANEL, base.TRUTH
FORMAL_VIEWS, SCENES, VIEWS, CLASSES = base.FORMAL_VIEWS, base.SCENES, base.VIEWS, base.CLASSES
FIT, FULL_FP32_POLICY = base.FIT, base.FULL_FP32_POLICY
SEED, STEPS, UPDATES = 2026092701, 222, 4440
GPU_IDS = (4,)
TRAIN_SCENES = ('practical_low_suburban','practical_mid_urban','practical_low_urban')
model_config = base.model_config

def rows():
    return [dict(row_id='EPOCH_REPLAY-s'+str(SEED), arm='EPOCH_REPLAY', model_seed=SEED)]

def reference(arm):
    if arm not in ('B0','B1'):
        raise ValueError('Only complete original pure-F references are reused')
    return Path(PROJECT)/'runs'/REFERENCE_RUN/(arm+'-s'+str(SEED))

def config(row):
    if row not in rows():
        raise ValueError('Unregistered epoch replay row')
    c = deepcopy(base.config(next(r for r in base.rows() if r['arm']=='B1')))
    c.update(schema='F_epochReplay20_v1', run_id=RUN, **row,
             output_root=str(BASE/row['row_id']/'source'), launch_owner=OWNER,
             method='pure_F_epochReplay20', stage='source_scratch20', As=True,
             training_recipe='pure F ORIGINAL scratch20=13L+7U; .5 ordinary CE + .5 epoch-cached fullPractical(raw) CE')
    c['training_mechanism'] = dict(mode='ORIGINAL',alpha=0.,CROSS_RX=False,RDrop=False,
        L_forwards_per_sample=2,teacher_packets_per_step=0,ordinary_CE_weight=.5,
        extra_CE_weight=.5,first_input='unchanged original ordinary L',
        extra_input='first full practical(raw) per physicalID in current epoch',
        CE_temperature=1.,CE_smoothing=.01,normalization='all128 mean in each slot',
        all_labeled_As_true_CE=True,cache_scope='same epoch across optimizer updates',
        cache_key='physical (eq_i,rx_i,day_i,sig_i,base_index), never TX label',
        cache_reset='before first L step of every epoch, including1',
        fresh_generation='full original128/step; used on first visit, shadow on replay hit',
        changed_observation='within-epoch correlation and effective diversity; scene and noise held jointly',
        cross_epoch_refresh=True,ordinary_output_used_by_student=True,
        native_use_aug=True,new_U_augmentation=False,U='original hiddenTX gate+ownEMA unchanged',
        external_teacher=False,teacher_shadow=False,risk_q=False,new_deploy_parameters=0,
        extra_optimizer_steps=0,head_fit_updates=0)
    c['source_promotion'] = dict(EPOCH_REPLAY_minus_B1_midlow_min_pp=.5,
        EPOCH_REPLAY_minus_B0_fullVclean_min_pp=-.2,automatic_extension=False,
        max_cumulative_epochs=50,decision_before_formal_scores=True,
        on_fail='close fixed epoch-first-As replay recipe; no refresh-interval/seed/schedule rescue')
    c.pop('source_falsification',None)
    c['baseline_definition']='Original pure F B0/B1 unchanged and reused; no new NoAs input/objective/trajectory change'
    c['baseline_reference']='Completed pure-F B0 and B1 own scratch20; no inherited checkpoint or new reference predictions'
    c['goal']=dict(primary='EPOCH_REPLAY',low_urban_above=.60,clean_net_vs_B0=True,
        suburban_gap_exploratory_pp=2.,model_seeds_this_matrix=1,target_feedback=False,goal_completed=False)
    c['budget_authorization']='One new scratch20 model/4440 updates; originalB0/B1 readouts reused. OwnE20+30 only if fixed source PASS and matched continuation rules.'
    c['cost_claim']='1136640 L studentNN+396900 U studentNN+396900 ownEMAeval;568320 full fresh practical generations (hit generations discarded as shadow),568320 used As NN packets,0 extraNN/optimizer,0 externalCSIL. Unique actual As/hits measured separately. B1 historical568320 teacher shadowNN separately retained.'
    return c

def validate(c):
    row=next((r for r in rows() if r['row_id']==c.get('row_id')),None)
    if row is None or c!=config(row):
        raise ValueError('Fixed pureF/source/default-head/scratchE20/epoch replay differs')
    return c

def make_args(c,device='cuda:0'):
    validate(c)
    a=base.make_args(base.config(next(r for r in base.rows() if r['arm']=='B1')),device)
    a.run_id,a.candidate_id,a.output_dir=RUN,c['row_id'],c['output_root']
    return a
