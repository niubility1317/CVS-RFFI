"""Formal scratch20 default-head evaluation of one source-qualified noise route."""
from pathlib import Path
from experiments.cvs_evidence_responsibility_train20 import design as parent
from experiments.cvs_sigtime_cleanpath20 import design as screen
ROOT=Path(__file__).resolve().parents[2]
PROJECT,SOURCE,CAPSULE,TRUTH,SEEDS,FULL_FP32_POLICY,read,write=(getattr(parent,k) for k in ('PROJECT','SOURCE','CAPSULE','TRUTH','SEEDS','FULL_FP32_POLICY','read','write'))
RUN='20261015-phase1-SIGTIME-branch-noise-e20-clean-manysig-m6-r01'
RELEASE='cvs_SIGTIME_branch_noise_train20_20261015_r01'
DEPENDENCY_RELEASE=screen.DEPENDENCY_RELEASE
MECHANISM_RELEASE=PROJECT+'/releases/'+screen.RELEASE
FOLD_RELEASE=screen.FOLD_RELEASE
BASE=Path(PROJECT)/'runs'/RUN
SOURCE_DECISION=Path(PROJECT)/'runs'/screen.RUN/'matrix_result.json'
# Finalize this code/config identity only after the completed source decision.
ARMS=('NONE', 'CLEAN_SIG')
EPOCHS=20;UPDATES=4440;VIEWS=('clean',);READOUTS=('NATIVE_HEAD',)
SOURCE_GATE=dict(final_training_L_CE_max=.5,first_seed_source_V_drop_max=.01,first_seed_worst_source_RX_drop_max=.02,
    first_seed_only=True,stop_before_target_if_failed=True,later_seed_health_report_only=True)
def parent_config(seed):return parent.config(next(r for r in parent.rows() if r['arm']=='BASE_SSL' and r['model_seed']==seed))
def rows():return [dict(row_id=mode+'-s'+str(seed),arm=mode,model_seed=seed) for seed in SEEDS for mode in ARMS]
def config(row):
    if ARMS not in (('NONE',),('NONE','CLEAN_SIG')) or row not in rows():raise ValueError('Fixed source-qualified route/control and all three registered seeds required')
    c=parent_config(row['model_seed']);mode=row['arm']
    return dict(c,schema='phase1_SIGTIME_source_selected_noise_scratch20_v1',run_id=RUN,**row,
        launch_owner='codex/root/SIGTIME-branch-noise-train20-20261015',output_root=(BASE/row['row_id']/'source').as_posix(),
        training_mechanism=dict(mode=mode,signature_tap='t2.drop input after pool' if mode=='CLEAN_SIG' else 'original t2 output',
            main_t2_dropout_p=.1 if mode=='CLEAN_SIG' else 0.,new_parameters=0,extra_IQ_forwards=0,extra_losses=False,
            signature_gradient_detached=False,default_head_joint_training=True,head_fit_updates=0,
            unchanged_own_EMA_teacher=True,tail_average=False,EMA_export=False),
        source_falsification=SOURCE_GATE,baseline_reuse=False,
        baseline_reference=dict(run_id=parent.RUN,row_id=parent_config(row['model_seed'])['row_id'],reuse='Original actualLS0 BASE_SSL E20 FINAL predictions and scores only; no weight inheritance'),
        training_recipe='SIG_TIME+CENTER25K/defaultcos; full five source RX; scratch20 actualLS0;13L+7U oldgate/ownEMA; original L/U sampler and Gaussian U',
        source_selection=dict(reference=SOURCE_DECISION.as_posix(),rule='CLEAN_SIG primary passes: evaluate both NONE and CLEAN_SIG to isolate its increment; otherwise only independently confirmed NONE; neither closes without training',
            modes=list(ARMS),no_target_selection=True,no_source_fold_weight_inheritance=True),
        budget_authorization='Active Phase1 goal: source-qualified route plus required positive NONE control if CLEAN_SIG qualifies, all3 fixedseeds scratch20, automatic clean default-head truth-last; existing matched baseline reused',
        cost_claim=f'{len(rows())} new scratch20 backbones/{len(rows())*20}epochs/{len(rows())*4440}steps and{len(rows())*168000} default-head clean decisions; original3x20/4440 baseline cost separate, no head fit or extra training IQ forward',
        goal=dict(clean_three_seed_mean=.84,goal_completed=False,readout='NATIVE_HEAD',target_feedback=False))
def validate(c):
    row=next((r for r in rows() if r['row_id']==c.get('row_id')),None)
    if row is None or c!=config(row):raise ValueError('Fixed formal noise-training configuration differs')
    return c
def make_args(c,device='cuda:0'):
    validate(c);a=parent.make_args(parent_config(c['model_seed']),device)
    a.run_id=RUN;a.candidate_id=c['row_id'];a.output_dir=c['output_root'];return a
def source_selection(value):
    if value.get('status')!='SOURCE_MATRIX_COMPLETE' or value.get('actual_optimizer_steps')!=13320 or value.get('target_access') is not False:
        raise ValueError('Complete legal source-fold matrix required before formal training')
    q=screen.qualify(value['results'])
    if q!=value.get('qualification'):raise ValueError('Source qualification differs from its fixed rule')
    selected=('NONE','CLEAN_SIG') if q['CLEAN_SIG_primary_qualified'] else ('NONE',) if q['NONE_independent_confirmation_passed'] else ()
    if not selected or selected!=ARMS:raise ValueError('No matching source-qualified formal route/control matrix')
    return dict(modes=list(selected),primary='CLEAN_SIG' if 'CLEAN_SIG' in selected else 'NONE',source_run=screen.RUN,qualification=q,source_fold_checkpoint_inherited=False)
