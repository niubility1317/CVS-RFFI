"""Source-qualified responsibility routing: one fair scratch20 matrix."""
from pathlib import Path
from experiments.cvs_crossrx_sharpness20 import design as parent
ROOT=Path(__file__).resolve().parents[2]
PROJECT,SOURCE,CAPSULE,TRUTH,CLASSES,SEEDS,FULL_FP32_POLICY,read,write=(getattr(parent,k) for k in ('PROJECT','SOURCE','CAPSULE','TRUTH','CLASSES','SEEDS','FULL_FP32_POLICY','read','write'))
RUN='20261015-phase1-SIGTIME-evidence-responsibility-e20-clean-manysig-m12-r01'
RELEASE='cvs_SIGTIME_evidence_responsibility_train20_20261015_r01'
OWNER='codex/root/evidence-responsibility-train20-20261015'
BASE=Path(PROJECT)/'runs'/RUN
ARMS=('BASE_CE','BASE_SSL','JOINT_GAP','ROUTED_GAP');PRIMARY=('ROUTED_GAP',)
VIEWS=('clean',);READOUTS=('NATIVE_HEAD',)
EPOCHS=20;STEPS=222;UPDATES=4440;FREQUENCY=16;AUX_EPOCHS=13;SCHEDULED_CALLS=180
PROBE_STEPS=(16,896,1776,2880)
TITLE='E20真实证据责任：等损失梯度路由与原默认头匹配矩阵'
PAIRS=[('ROUTED_GAP','BASE_SSL'),('ROUTED_GAP','JOINT_GAP'),('JOINT_GAP','BASE_SSL'),('BASE_SSL','BASE_CE')]
SOURCE_GATE=dict(final_L_CE_max=.5,V_mean_drop_max=.01,V_worst_RX_drop_max=.02,
    eligible_fraction_min=.95,minimum_active_gradient_probes=2,minimum_positive_gap_call_fraction=.01,
    selection='All firstseed final20 source health; active JOINT/ROUTED actual signal; no target feedback')
def rows():return [dict(row_id=f'{arm}-s{seed}',arm=arm,model_seed=seed) for seed in SEEDS for arm in ARMS]
def baseline_config(seed):return parent.baseline_config(seed)
def structural_config(seed):return parent.structural_config(seed)
def active(step):return 1<=step<=AUX_EPOCHS*STEPS and step%FREQUENCY==0
def scheduled(c):return SCHEDULED_CALLS if c['arm'] in ('JOINT_GAP','ROUTED_GAP') else 0
def objective_mode(arm):return 'ZERO' if arm in ('BASE_CE','BASE_SSL') else arm
def config(row):
    if row not in rows():raise ValueError('Unregistered responsibility E20 row')
    control=next(r for r in parent.rows() if r['model_seed']==row['model_seed'] and r['arm']==('BASE_CE' if row['arm']=='BASE_CE' else 'BASE_SSL'))
    base=parent.config(control)
    return dict(base,schema='phase1_evidence_responsibility_train20_v1',run_id=RUN,**row,launch_owner=OWNER,
        output_root=(BASE/row['row_id']/'source').as_posix(),
        training_mechanism=dict(mode=objective_mode(row['arm']),coefficient=.1,tolerance=1e-6,
            frequency=FREQUENCY,auxiliary_epochs=[1,13],scheduled_calls=scheduled(row),probe_steps=list(PROBE_STEPS),
            selected='actual encoder rawcomponent paths excluding defaultW/3gain',selected_tensors=68,
            bank_role='L_s',bank_current_ordinals=[2,19,37,54],bank_held_ordinals=[66,67],
            bank_unique_IQ_reads=900 if scheduled(row) else 0,bank_training_unique_packets=720 if scheduled(row) else 0,
            bank_count_per_call=180,fixed_loss_denominator=720,
            loss='.1/720 sum_b,i relu(CE_full-CE_without_b.detach-1e-6)',
            routing='responsible component only' if row['arm']=='ROUTED_GAP' else 'all4 components' if row['arm']=='JOINT_GAP' else 'zero',
            default_W_three_gains_stopped_in_aux=True,main_default_head_joint=True,
            auxiliary_IQ_forward_packets=32400 if scheduled(row) else 0,
            original_sampler_unchanged=True,current_main_and_auxiliary_ID_overlap_allowed=True,
            previously_trained_source_L=True,never_trained_holdout=False,
            extra_actual_optimizer_steps=0,head_fit_updates=0),
        training_recipe=base['training_recipe']+'; actual label_smoothing=0 explicitly set in make_args',
        baseline_definition='Fresh matched SIG_TIME+CENTER25K scratch20 actualLS0; historical sharpness20 actualLS.01 is separate context, no result reuse.',
        source_falsification=dict(SOURCE_GATE),goal=dict(primary='ROUTED_GAP',clean_three_seed_mean=.84,feedback_to_training=False),
        budget_authorization='Source INITIAL3/4 qualifies scratch20 only. Fourfirstseed then8 after fixedsource health/signal. 0weightinheritance.',
        cost_claim='12scratch20/240modelepoch/53280realupdates; firstseed4=80epoch/17760updates. JOINT/ROUTED180calls32400extraL/model; baseline0extraIQ. 8temporaryAdamW probes/model countedseparate; no newparameters/heads.')
def validate(c):
    row=next((r for r in rows() if r['row_id']==c.get('row_id')),None)
    if row is None or c!=config(row):raise ValueError('Fixed responsibility E20 configuration differs')
    return c
def make_args(c,device='cuda:0'):
    validate(c)
    row=next(r for r in parent.rows() if r['model_seed']==c['model_seed'] and r['arm']==('BASE_CE' if c['arm']=='BASE_CE' else 'BASE_SSL'))
    a=parent.make_args(parent.config(row),device)
    a.run_id=RUN;a.candidate_id=c['row_id'];a.output_dir=c['output_root'];a.label_smoothing=0.;return a
def source_reference_metrics(seed):return read(BASE/f'BASE_SSL-s{seed}'/'source/completion.json')['final_source_metrics']
def result_metadata():return dict(rows=12,new_backbones=12,existing_backbone_references=0,readouts=12,
    prediction_count=2016000,reused_native_decisions=0,new_prediction_decisions=2016000,
    metric_records=216,backbone_updates_new=53280,historical_baseline_updates=0)
