"""One fixed source question; no amplitude search or automatic training."""
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
PROJECT='/home/szu2070436088/2510044040/CV-SincNet'
RUN='20261016-diagnostic-F-crossRX-commonphase-sourceL-s2026092701-m1-r01'
RELEASE='cvs_F_crossRX_commonphase_sourceL_20261016_r01'
OWNER='codex/root/F-crossRX-commonphase-20261016'
LOCAL='local_artifacts/F_crossrx_commonphase_source_20261016'
BACKEND=PROJECT+'/releases/cvs_F_crossRX_satmix25_source_e20_20261016_r01'
BACKEND_COMMIT='0ae066adb10a001e834c8c679c5b7b81c20b3ac8'
PARENT_RUN='20261016-phase1-F-crossRX-satmix25-sourceRX1-e20-s2026092701-m2-r01'
MODEL_SOURCE=PROJECT+'/runs/'+PARENT_RUN+'/B00-RX1-s2026092701/source'
SOURCE=PROJECT+'/runs/phase1_daot_rc4_pure_game_m3_20260917_r2/source_contract.json'
PREPARATION=PROJECT+'/runs/20261011-phase1-stage2-geometry-source-manysig-m3-r01/s2026092701'
BASE=Path(PROJECT)/'runs'/RUN
VIEWS={'identity':0.,'phase_plus':.37,'phase_minus':-.37}
FIT=(3,4,6,8)

def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def write(p,value):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')
def config():
    return dict(run_id=RUN,launch_owner=OWNER,output_root=BASE.as_posix(),backend=BACKEND,backend_commit=BACKEND_COMMIT,checkpoint_source=MODEL_SOURCE,source_contract=SOURCE,preparation=PREPARATION,model_seed=2026092701,model='F/X3/unifiedSinc+CROSS_RX,NoExplicitCFO,joint160,existingcosinehead',checkpoint_role='Frozen own scratch20/fixedfinal20/B00;not a new training initialization',data_role='Only original fit L5040;heldRX1 excluded before IQ conversion or inference',fit_RX=list(FIT),days=[1,2,3],TX=list(range(6)),samples_per_cell=70,views=VIEWS,phase_operation='constant unit complex multiplication on received IQ;no frequency shift,no RMS rerun,no channel',source_predictions=15120,smoke_synthetic_packets=2,total_new_NN_packets=15122,training_updates=0,head_fit_updates=0,teacher_forwards=0,As_generations=0,U_used=False,V_used=False,held_RX1_used=False,target_access=False,criteria=dict(each_sign_min_accuracy_drop_pp=2.,each_sign_min_RX_with_drop=3,each_RX_min_drop_pp=1.),on_fail='Close fixed bounded common-phase view direction;no amplitude/seed/RX rescue or automatic training',on_pass='Only motivates separately registered matched B00/B10 scratchE20;does not prove Ag efficacy or joint goal',device='CPU',threads=2,parent_cost_separate='ParentE20/4440updates/4686678actualNNpackets,not repeated or inherited into new training')
def validate(c):
    if c!=config():raise ValueError('Fixed source common-phase experiment differs')
