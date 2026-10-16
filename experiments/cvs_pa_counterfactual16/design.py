"""One fixed source counterfactual;no PA parameter search or model updates."""
from pathlib import Path
from experiments.cvs_commonphase_source import design as parent
ROOT=Path(__file__).resolve().parents[2]
PROJECT,BACKEND,BACKEND_COMMIT,MODEL_SOURCE,PARENT_RUN,SOURCE,PREPARATION,FIT,read,write=(getattr(parent,k) for k in ('PROJECT','BACKEND','BACKEND_COMMIT','MODEL_SOURCE','PARENT_RUN','SOURCE','PREPARATION','FIT','read','write'))
RUN='20261016-diagnostic-F-crossRX-ordinaryPA-fitL-s2026092701-m1-r01'
RELEASE='cvs_F_crossRX_ordinaryPA_fitL_20261016_r01'
OWNER='codex/root/F-CROSS_RX-ordinaryPA-20261016'
LOCAL='local_artifacts/F_ordinary_PA16_20261016'
BASE=Path(PROJECT)/'runs'/RUN
SEED=20261016241
EPOCHS=tuple(range(1,21))
SUBCHAINS=('_pa_memory_polynomial','_pa_apply_fir_memory','_pa_rapp','_pa_am_pm','_iq_imbalance')
def config():
 return dict(run_id=RUN,owner=OWNER,backend=BACKEND,backend_commit=BACKEND_COMMIT,checkpoint_source=MODEL_SOURCE,model='Frozen own scratch F+CROSS_RX B00/final20/seed01;default jointly trained cosine head,no new fitting',fit_RX=list(FIT),L=5040,U_used=False,V_used=False,held_RX1_used=False,target_access=False,epochs=list(EPOCHS),augmentation_seed=SEED,draws='Seed Python and Torch once at each epoch with SEED+epoch;fixed fit-cache order,batches128/last48;new diagnostic draws,not a reconstruction of historical training draws',augmentation='Exact native build_aug_base_cfg/make_augmentor/configure_augmentor_for_epoch using original checkpoint args;one actual normal call per batch',views='Raw once;each of20 actual schedules:fullordinary and PA-only-bypassed ordinary from captured post-general-channel prefix of same call',capture='PA is last;DAC probability must be0,class_signature false,no extra input RMS;record real strengths,subchains and changed packets;no independent redraw or forced trigger',source_NN_packets=206640,total_NN_packets=206642,smoke_synthetic_packets=2,ordinary_generated_packets=100800,training_updates=0,head_fit_updates=0,teacher_forwards=0,As_generations=0,device='CPU',threads=2,criteria=dict(min_PA_changed_packets=1000,pooled_changed_accuracy_recovery_pp_min=2.,RX_positive_min=3,RX_recovery_pp_min=-1.,each_day_recovery_pp_min=0.),on_fail='Close PA-harm training rationale,no seed/intensity/RX/threshold rescue;not proof PA identity preserving',on_pass='Only motivates matched separately registered scratch E20 ordinary-recipe ablation;not Ag cross-RX or joint efficacy',parent_cost_separate='E20/4440updates/4686678NN;frozen diagnosis only')
def validate(c):
 if c!=config():raise ValueError('Fixed PA diagnostic roles/budget/draws/criteria differ')
