"""One fixed source gain-envelope hypothesis, no training or strength search."""
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
PROJECT='/home/szu2070436088/2510044040/CV-SincNet'
RUN='20261016-diagnostic-F-gainenvelope-source-crossfit-manysig-m1-r01'
RELEASE='cvs_F_gain_envelope_source_20261016_r01'
OWNER='codex/root/F-gain-envelope-20261016'
SOURCE=PROJECT+'/runs/phase1_daot_rc4_pure_game_m3_20260917_r2/source_contract.json'
PREPARATION=PROJECT+'/runs/20261011-phase1-stage2-geometry-source-manysig-m3-r01/s2026092701'
BASE=PROJECT+'/runs/'+RUN


def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def write(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')


def config():
    return dict(run_id=RUN,launch_owner=OWNER,source_contract=SOURCE,preparation=PREPARATION,output_root=BASE,
                count=5040,RX=[3,4,6,8],TX=list(range(6)),days=[1,2,3],samples_per_cell=70,
                observable='Fixed IQ[80:160], four20sample powers; dB-centered OLS slope on q=(-1,-1/3,1/3,1); no IQ modification',
                primary='Per TX/RX/day median absolute slope,18folds jointly excluding heldTX and heldday from all fitted means',
                controls='Pooled common mean,3cyclic wrongRX mappings,TXhalf centered cosine,perpacket linearR2,analytical wholepacketRMS cancellation',
                criteria=dict(min_mean_risk_gain=.05,min_positive_folds=14,min_per_RX_gain=0.,min_gain_over_best_wrong_RX=.02,min_TX_half_cosine=.8,min_median_packet_R2=.8,min_median_abs_slope_db=.01,max_RMS_slope_difference=1e-10),
                on_fail='Close fixed gain-envelope hypothesis; no strength/window/seed/RX rescue and no automatic training',
                on_pass='Only source rationale for separately preregistered bounded positive commonIQ gain view and matched F/E20 defaulthead recognition test',
                source_only=True,held_RX1_used=False,U_used=False,U_truth_read=False,V_used=False,target_access=False,
                checkpoint_loads=0,network_forwards=0,optimizer_steps=0,automatic_training=False,device='CPU',threads=2,
                model_boundary='Future fixed F/SIG_TIME3/unifiedSinc/joint160/defaultcos/NoExplicitCFO; no model constructed now',
                reference='https://wiki.analog.com/resources/eval/user-guides/ad9361_agc_tuning',
                limitation='Observed gain slope may be TX transient, waveform or propagation; statistical pass cannot identify AGC hardware or label preservation')


def validate(c):
    if c!=config():raise ValueError('Fixed source gain-envelope question differs')
