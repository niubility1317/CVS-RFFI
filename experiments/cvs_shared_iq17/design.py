"""One frozen source question,not a new training recipe or target evaluation."""
from pathlib import Path
from experiments.cvs_commonphase_source import design as prior
ROOT=Path(__file__).resolve().parents[2]
PROJECT,BACKEND,BACKEND_COMMIT,MODEL_SOURCE,PARENT_RUN,SOURCE,PREPARATION,FIT,read,write=(getattr(prior,k) for k in ('PROJECT','BACKEND','BACKEND_COMMIT','MODEL_SOURCE','PARENT_RUN','SOURCE','PREPARATION','FIT','read','write'))
RUN='20261017-diagnostic-F-crossRX-sharedDirection-fitL-s2026092701-m1-r01'
RELEASE='cvs_F_sharedDirection_fitL_20261017_r01'
OWNER='codex/root/F-sharedDirection-source-20261017'
LOCAL='local_artifacts/F_shared_direction_20261017'
BASE=Path(PROJECT)/'runs'/RUN
VIEWS=('raw','direction','permuted')

def config():
 from .parameters import EPS,XI,PEAK
 from .question import EXPECTED
 return dict(run_id=RUN,owner=OWNER,output_root=BASE.as_posix(),backend=BACKEND,backend_commit=BACKEND_COMMIT,checkpoint_source=MODEL_SOURCE,parent_run=PARENT_RUN,source_contract=SOURCE,preparation=PREPARATION,model_seed=2026092701,split=392005,model='F/X3/unifiedSinc+CROSS_RX,NoExplicitCFO,joint160,originalcosine6x160scale30',checkpoint_role='Frozen own scratch20/final20/no ancestors/no target;never new training initialization',data_role='Only original fitL5040/6TX/4RX/3day/70percell;opaque original role IDs matched exactly',folds='12 heldRX/day cells;reference excludes heldRX AND heldday;3RXx2dayx6TXx70=2520;held1RXx1dayx6TXx70=420',views=list(VIEWS),operator=dict(epsilon_absolute_complex_RMS=EPS,xi_absolute_complex_RMS=XI,peak_absolute=PEAK,FP32_relative_tolerance=1e-6,private_seed=202610178001,control_seed=202610178002,mean_zero_each_IQ=True,one_shared_vector_per_RXday_allTX=True,label_input=False,source_RXday_only_select_reference=True,KL_only_input_direction_no_training_loss=True,steps=1,zero_gradient='record_unidentifiable_zero_no_fallback',normalization='Original native RMS cache used unchanged,no postoperator RMS;record actual signalpower/perturbationratio',NoExplicitCFO='Original internal packet compensation retained;no CFO feature or offset modification'),expected=dict(EXPECTED),scientific_rule='Fixed question.qualify:raw>=98%,permuted-direction>=2pp,>=3RX gap>=1pp,direction rawcorrect retention overall/eachRX/eachTX>=95%,permuted overall>=99%/eachRX>=98%,all12 identifiable,fullbudget/no target',on_fail='Close fixed shared direction,no epsilon/xi/peak/iterations/model/RX/day/threshold/seed rescue',on_pass='Sensitivity evidence only;separately preregister any matched scratchE20 later',threads=2,device='CPU',CUDA_VISIBLE_DEVICES='',U_used=False,V_used=False,held_RX1_used=False,target_access=False,new_teacher_models=0,new_training_loss_terms=0,optimizer_updates=0,head_fit_updates=0,As_generations=0,parent_cost='Own originalB00 E20/4440updates/4686678NN historical cost,not repeated or inherited into training')
def validate(c):
 if c!=config():raise ValueError('Fixed source roles/model/operator/cost/question changed')
 return c
