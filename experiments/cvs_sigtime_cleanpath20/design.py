"""One new seed: signature-specific noise separation and a NONE confirmation."""
from pathlib import Path
import math
from experiments.cvs_sigtime_locked20 import design as previous
from experiments.cvs_evidence_responsibility_train20 import design as parent
from .mechanism import MODES
ROOT=Path(__file__).resolve().parents[2]
PROJECT,SOURCE,FULL_FP32_POLICY,read,write=(getattr(previous,k) for k in ('PROJECT','SOURCE','FULL_FP32_POLICY','read','write'))
RUN='20261015-phase1-SIGTIME-clean-signature-sourcefold-s2026092702-m3-r01'
RELEASE='cvs_SIGTIME_clean_signature_sourcefold_20261015_r01'
DEPENDENCY_RELEASE=previous.DEPENDENCY_RELEASE
FOLD_RELEASE=PROJECT+'/releases/'+previous.RELEASE
SEED=2026092702;HELD=1;FIT=(3,4,6,8);EPOCHS=20;UPDATES=4440
GATE=dict(primary='CLEAN_SIG_FINAL strictly fewer errors than both ELEMENT_FINAL and NONE_FINAL; CE no higher than either',
    independent_positive_route_confirmation='NONE_FINAL strictly fewer errors than ELEMENT_FINAL, CE no higher',
    all_final_training_L_CE_max=.5,readout='FINAL only; own EMA teacher unchanged; no export window selection',
    qualification='Separate all-five-RX three-seed scratch20 preregistration only; never reuse four-RX weights',
    no_fallback_into_failed_primary=True,no_target_feedback=True)

def base_config():return parent.config(next(r for r in parent.rows() if r['model_seed']==SEED and r['arm']=='BASE_SSL'))
def config(mode):
    if mode not in MODES:raise ValueError('Fixed signature-noise three-mode matrix required')
    c=base_config();row=mode+'-RX1-s'+str(SEED)
    return dict(c,schema='phase1_SIGTIME_clean_signature_sourcefold_v1',run_id=RUN,row_id=row,arm=mode,
        method='cvs_sigtime_branch_noise_placement',stage='source_RX_fold_full_training',
        test_views=[],readouts=['FINAL'],head_fit_updates=0,
        launch_owner='codex/root/SIGTIME-clean-signature-20261015',output_root=PROJECT+'/runs/'+RUN+'/'+row+'/source',
        fold=previous.config('ELEMENT')['fold'],source_falsification=GATE,
        baseline_reference='New seed2026092702 matched ELEMENT; previousseed01 is discovery, not a reused control',
        training_mechanism=dict(mode=mode,main_t2_dropout_p=0. if mode=='NONE' else .1,
            signature_tap='t2.drop input after pool' if mode=='CLEAN_SIG' else 't2 output',
            other_layers_unchanged=True,additional_loss=False,extra_parameters=0,extra_training_IQ_forwards=0,
            extra_rng_draws=0,signature_gradient_detached=False,own_EMA_teacher_unchanged=True,
            default_head_joint_training=True,head_fit_updates=0,tail_average=False),
        training_recipe='SIG_TIME+CENTER25K/default cosine; scratch20 LS0;13L+7U; same RX1-excluded source loaders and222-step budget',
        budget_authorization='Continue user training-mechanism goal: second predeclared seed confirms strong NONE source finding and isolates signature-only pre-drop read; three conditions only, no R-Drop/EMA combination',
        cost_claim='3x20 epochs/13320 optimizer steps;396368 U observations per row;1260 held-L final reads per row,1404 additional L reads incl144 discarded anchors, zero extra training forwards',
        goal=dict(clean_three_seed_mean=.84,goal_completed=False,source_diagnostic_only=True,
            novelty='Candidate branch-specific path preservation in CVS; no claim that dropout removal or signature pooling is globally novel'))
def validate(c):
    if c!=config(c.get('arm')):raise ValueError('Fixed new-seed clean signature source recipe differs')
    return c
def make_args(c,device='cuda:0'):
    validate(c);a=parent.make_args(base_config(),device)
    a.run_id=RUN;a.candidate_id=c['row_id'];a.output_dir=c['output_root']
    a.source_fold_held_rx=HELD;a.source_fold_training_receivers=','.join(map(str,FIT))
    return a
source_validation=previous.source_validation
def qualify(results):
    if set(results)!=set(MODES):raise ValueError('All three source conditions required')
    for row in results.values():
        metric=row['exports']['FINAL']
        if metric['count']!=1260 or not math.isfinite(metric['mean_CE']):raise ValueError('Complete finite held source results required')
    healthy=all(row['source_health'] for row in results.values())
    def better(a,b):
        a,b=results[a]['exports']['FINAL'],results[b]['exports']['FINAL']
        return a['errors']<b['errors'] and a['mean_CE']<=b['mean_CE']
    return dict(CLEAN_SIG_primary_qualified=healthy and all(better('CLEAN_SIG',control) for control in ('ELEMENT','NONE')),
        NONE_independent_confirmation_passed=healthy and better('NONE','ELEMENT'),all_source_healthy=healthy,
        main_run_launched=False,target_access=False)
