"""One fixed source-only diagnostic; native L-stage schedule, zero updates."""
from pathlib import Path
from experiments.cvs_commonphase_source import design as parent
ROOT=Path(__file__).resolve().parents[2]
PROJECT,BACKEND,BACKEND_COMMIT,MODEL_SOURCE,PARENT_RUN,SOURCE,PREPARATION,FIT,read,write=(getattr(parent,k) for k in ('PROJECT','BACKEND','BACKEND_COMMIT','MODEL_SOURCE','PARENT_RUN','SOURCE','PREPARATION','FIT','read','write'))
RUN='20261016-diagnostic-F-crossRX-ordinaryshift-fitL-s2026092701-m1-r01'
RELEASE='cvs_F_crossRX_ordinaryshift_fitL_20261016_r01'
OWNER='codex/root/F-CROSS_RX-ordinaryshift-20261016'
LOCAL='local_artifacts/F_ordinary_shift16_20261016'
BASE=Path(PROJECT)/'runs'/RUN
SEED=20261016511
EPOCHS=tuple(range(1,14))
def config():
 return dict(run_id=RUN,owner=OWNER,backend=BACKEND,backend_commit=BACKEND_COMMIT,checkpoint_source=MODEL_SOURCE,model='Frozen own scratch F+CROSS_RX B00/final20/seed01;default jointly trained cosine head,no new fitting',fit_RX=list(FIT),L=5040,U_used=False,V_used=False,held_RX1_used=False,target_access=False,epochs=list(EPOCHS),augmentation_seed=SEED,draws='Python/Torch SEED+epoch;fixed fit-cache batches128/last48;new diagnostic draws,not historical training reconstruction',augmentation='Actual checkpoint args/native base and configure;only E1..13 because native label epoch is capped at13',views='Raw once;13 schedules full ordinary and same-draw shift bypass;replay complete pipeline including PA',capture='Module shift wrapper preserves original shift draws;whole replay preserves Python/Torch final state and PA strengths;downstream signal-dependent noise power naturally recalculates;canonical unwrapped output/RNG checked on first batch of every epoch',source_NN_packets=136080,total_NN_packets=136082,smoke_synthetic_packets=2,ordinary_generated_packets=131040,canonical_check_generated_packets=1664,total_augmentation_generated_packets=132704,shift_draw_inspection='Each invoked shift draw is replayed once per paired path for observation then native RNG restored;extra scalar draws counted separately,not model forwards',training_updates=0,head_fit_updates=0,teacher_forwards=0,As_generations=0,device='CPU',threads=2,criteria=dict(nonzero_shift_accuracy_recovery_pp_min=2.,RX_positive_min=3,all_RX_nonzero_coverage=True),on_fail='Close fixed shift-correction hypothesis;no seed/draw/RX/threshold rescue',on_pass='Only supports separately registered matched scratch E20 ordinary-recipe correction;not new Ag cross-RX or joint efficacy',parent_cost_separate='E20/4440updates/4686678NN;frozen diagnosis only')
def validate(c):
 if c!=config():raise ValueError('Fixed shift diagnostic roles/budget/draws/criteria differ')
