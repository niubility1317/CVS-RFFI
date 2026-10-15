"""Fixed one-seed, three-mode actual source-fold training, no target access."""
from pathlib import Path
from experiments.cvs_rx_u_increment20 import design as previous
from .mechanism import MODES,TAIL_EPOCHS
ROOT=Path(__file__).resolve().parents[2]
PROJECT,SOURCE,FULL_FP32_POLICY,read,write=(getattr(previous,k) for k in ('PROJECT','SOURCE','FULL_FP32_POLICY','read','write'))
RUN='20261015-phase1-SIGTIME-t2-locked-tailavg-sourcefold-s2026092701-m3-r01'
RELEASE='cvs_SIGTIME_t2_locked_tailavg_sourcefold_20261015_r01'
DEPENDENCY_RELEASE=previous.DEPENDENCY_RELEASE
SEED=2026092701;HELD=1;FIT=(3,4,6,8);EPOCHS=20;UPDATES=4440
GATE=dict(primary='LOCKED_FINAL vs ELEMENT_FINAL and NONE_FINAL',held_RX=1,
    errors_strictly_fewer_than_both_controls=True,mean_CE_not_higher_than_either_control=True,
    final_training_L_CE_max=.5,qualification='Only separately preregistered formal three-seed training, never reuse source-fold weights',
    secondary_exports=['EMA','TAIL_AVG'],secondary_exports_do_not_rescue_failed_primary=True)

def config(mode):
    if mode not in MODES:raise ValueError('Fixed three dropout modes required')
    c=previous.config();row=mode+'-RX1-s'+str(SEED)
    return dict(c,schema='phase1_SIGTIME_t2_dropout_tail_export_sourcefold_v1',run_id=RUN,row_id=row,
        arm=mode,method='cvs_sigtime_t2_dropout',stage='source_RX_fold_full_training',
        launch_owner='codex/root/SIGTIME-t2-locked-20261015',output_root=PROJECT+'/runs/'+RUN+'/'+row+'/source',
        readouts=['FINAL','EMA','TAIL_AVG'],source_falsification=GATE,
        training_mechanism=dict(t2_mode=mode,p=.0 if mode=='NONE' else .1,other_layers_unchanged=True,
            loss='Original L CE plus original oldgate U; no new loss',extra_training_IQ_forwards=0,
            teacher='Own original EMA, eval dropout behavior unchanged; average never fed back',
            tail_epochs=list(TAIL_EPOCHS),average='FP64 all188 parameters including default W/gains/norm;12 constant buffers copied; same trajectory only',
            retained_tail_snapshots=True,head_fit_updates=0,extra_optimizer_steps=0,
            held_L_readout_forwards=7560,additional_L_IQ_reads=1404),
        budget_authorization='User supplied report: actual ELEMENT/NONE/LOCKED controlled training; one fixed source RX1/seed first; tail averaging accompanies those trajectories, no averaging-only rerun',
        baseline_reference='Fresh matched ELEMENT trajectory within this three-condition matrix',
        training_recipe='SIG_TIME+CENTER25K/default cosine; scratch20 LS0;13L+7U; same four-RX222-step sampler and original Gaussian-U',
        cost_claim='Per row20 epochs/4440 actual steps,396368 U observations; total3 rows60 epochs/13320 steps; six1260-packet eval readouts per row; no extra training forwards',
        goal=dict(clean_three_seed_mean=.84,goal_completed=False,source_diagnostic_only=True))

def validate(c):
    if c!=config(c.get('arm')):raise ValueError('Fixed SIGTIME dropout source recipe differs')
    return c

def make_args(c,device='cuda:0'):
    validate(c);a=previous.make_args(previous.config(),device)
    a.run_id=RUN;a.candidate_id=c['row_id'];a.output_dir=c['output_root']
    return a

source_validation=previous.source_validation
