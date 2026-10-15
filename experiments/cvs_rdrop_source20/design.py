from pathlib import Path
from experiments.cvs_evidence_responsibility_train20 import design as parent

ROOT=Path(__file__).resolve().parents[2]
PROJECT,SOURCE,CAPSULE,SEEDS,FULL_FP32_POLICY,read,write=(getattr(parent,key) for key in ('PROJECT','SOURCE','CAPSULE','SEEDS','FULL_FP32_POLICY','read','write'))
RUN='20261015-phase1-SIGTIME-rdrop-source-e20-clean-manysig-m3-r01'
RELEASE='cvs_SIGTIME_rdrop_source20_20261015_r01'
OWNER='codex/root/SIGTIME-rdrop-source20-20261015'
DEPENDENCY_RELEASE=PROJECT+'/releases/cvs_SIGTIME_evidence_responsibility_train20_20261015_r02'
DEPENDENCY_COMMIT='3a5cff49091deef46cad5d787e5daea7b4c0dfb9'
BASE=Path(PROJECT)/'runs'/RUN
RX=(1,3,4,6,8);DAYS=(1,2,3);COHORT_SEED=2026210712;DROPOUT_SEED=2026101519
PARAMETERS=('id_backbone.encoder.core.id_backbone.t1.pw.weight',
    'id_backbone.encoder.core.id_backbone.t2.pw.weight',
    'id_backbone.encoder.sequence_pool.local_projection.weight',
    'id_backbone.encoder.core.id_backbone.cls_head.weight')
PARAMETER_SIZES=(3456,6912,576,960)
QUALIFICATION=dict(source_only=True,cohorts=5,KL_positive_each_cohort=True,
    t1_t2_SIG_KL_nonzero_each_cohort=True,aggregate_gradient_cosine_with_paired_CE_min=0.,
    parameters_buffers_unchanged=True,head_fit_updates=0,optimizer_steps=0,
    first_seed_fail_stops_before_other_parents=True,all_three_required_for_source_confirmation=True,
    on_pass='Source compatibility only; separately preregister a genuine held-RX matched two-CE/RDrop training falsification',
    on_fail='Preserve negative result; no coefficient, dropout, cohort, layer or parent scan')

def parent_config(seed):
    return parent.config(next(row for row in parent.rows() if row['arm']=='BASE_SSL' and row['model_seed']==seed))

def rows():return [dict(row_id='SOURCE_RDROP-s'+str(seed),arm='SOURCE_RDROP',model_seed=seed) for seed in SEEDS]

def config(row):
    if row not in rows():raise ValueError('Unregistered E20 R-Drop source row')
    old=parent_config(row['model_seed'])
    return dict(schema='SIGTIME_rdrop_E20_clean_source_v1',run_id=RUN,**row,launch_owner=OWNER,
        parent_run=parent.RUN,parent_row=old['row_id'],parent_checkpoint=old['output_root']+'/final_ssdg.pth',
        parent_rule='All original fixed BASE_SSL final20 students, never target-best, NONE or CLEAN_SIG',
        checkpoint_epoch=20,parent_training_epochs=20,parent_optimizer_steps=4440,
        feature='Actual deployed joint160, original SIG_TIME/CENTER25K and original default cosine6x160 scale30',
        source_roles='Original full5RX L6300/U56700hidden/V27000; only L forwarded; existing L/V cache loaded once, V discarded',
        RX=list(RX),days=list(DAYS),TX=list(range(6)),source_pool_count=6300,
        cohort_seed=COHORT_SEED,cohorts=5,cohort_packets=180,source_packets=900,
        cohort_rule='Original seeded L6300 permutation,70 per TX/RX/day cell; pair first10 balanced L90 ranks into5 L180',
        input='Same clean L IQ for both stochastic students and eval reference; no Gaussian or satellite view',
        stochastic='Original native dropout train mode, no new mask rule or dropout probability',
        objective='Mean of two default CE (LS0); symmetric KL of two differentiable default predictions',
        label_smoothing=0.,coefficient=1.,dropout_rng_seed=DROPOUT_SEED,parameters=list(PARAMETERS),
        model_forward_packets=2700,real_checkpoint_smoke_synthetic_packets=2,
        source_only=True,query_read=False,U_IQ_read=False,U_truth_read=False,V_forward=False,V_gradient=False,
        teacher_forward=False,training_epochs=0,optimizer_steps=0,head_fit_updates=0,
        numerical_policy=FULL_FP32_POLICY,device='cpu',CPU_threads=2,CUDA_VISIBLE_DEVICES='',
        qualification=QUALIFICATION.copy(),output_root=(BASE/row['row_id']).as_posix(),
        differences_from_legacy='E20/LS0/cleanL/original single SIG projection vs legacy E50/LS.01/strongL/multiview projection; fresh source evidence, not unchanged replication',
        limitation='Frozen-source gradient compatibility, not early trajectory, U-label precision, training efficacy, DG, newTX, Phase2 or84')

def validate(c):
    row=next((row for row in rows() if row['row_id']==c.get('row_id')),None)
    if row is None or c!=config(row):raise ValueError('Fixed E20 R-Drop source recipe differs')
    return c

def request():return dict(run_id=RUN,dependency_release=DEPENDENCY_RELEASE,rows=[config(row) for row in rows()],source_only=True)

def expand(value):
    if isinstance(value,str):return PROJECT+value[len('PROJECT'):] if value=='PROJECT' or value.startswith('PROJECT/') else value
    if isinstance(value,list):return [expand(item) for item in value]
    if isinstance(value,dict):return {key:expand(item) for key,item in value.items()}
    return value
