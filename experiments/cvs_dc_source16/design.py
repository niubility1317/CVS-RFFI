"""One DC nuisance hypothesis, fixed before any new source statistic."""
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
PROJECT='/home/szu2070436088/2510044040/CV-SincNet'
RUN='20261016-diagnostic-F-DC-source-crossfit-manysig-m1-r01'
RELEASE='cvs_F_DC_source_crossfit_20261016_r01'
OWNER='codex/root/F-DC-source-20261016'
SOURCE=PROJECT+'/runs/phase1_daot_rc4_pure_game_m3_20260917_r2/source_contract.json'
PREPARATION=PROJECT+'/runs/20261011-phase1-stage2-geometry-source-manysig-m3-r01/s2026092701'
BASE=PROJECT+'/runs/'+RUN


def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def write(p,value):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')


def config():
    return dict(run_id=RUN,launch_owner=OWNER,source_contract=SOURCE,preparation=PREPARATION,output_root=BASE,
                role='original L_s, then fixed fit RX3/4/6/8 only',count=5040,held_RX1_used=False,
                RX=[3,4,6,8],days=[1,2,3],TX=list(range(6)),samples_per_cell=70,source_only=True,target_access=False,
                observable='Complex DC mean of fixed native IQ[80:160], unmodified native packet RMS',
                cycle_check='Four20-sample cycles; odd/even cycle DC means; no CFO estimation, phase unwrap or compensation changes',
                crossfit='18 folds: exclude held TX AND held day from both RX and pooled training means; evaluate4RX cell means with equal weights',
                controls='Pooled common mean and all3 cyclic wrong-RX assignments, no classifier or TX prediction',
                criteria=dict(min_total_risk_gain=.05,min_positive_folds=14,min_RX_gain=0.,min_gain_above_best_wrong_RX=.02,min_TX_half_center_cosine=.8,min_DC_to_odd_even_difference_power_ratio=2.),
                on_fail='Close source-DC nuisance hypothesis; no new DC strength/seed/window/RX rescue or automatic training',
                on_pass='Only supports a separately fixed bounded input/default-head check and matched F/E20 Ag training; never proves Ag or permits target feedback',
                U_used=False,U_truth_read=False,V_used=False,checkpoint_loads=0,network_forwards=0,optimizer_steps=0,
                automatic_training=False,device='CPU',threads=2,model_boundary='Future training fixed F/X3/unifiedSinc/joint160/defaultcos; current diagnostic constructs no network',
                mechanism_source='https://www.analog.com/en/resources/design-notes/optimizing-performance-wideband-direct-conversion-receivers.html',
                limitation='DC may also contain TX LO leakage, finite waveform mean or preprocessing effects. Crossfit must establish RX/day transport before treating it as a candidate observation nuisance.')


def validate(c):
    if c!=config():raise ValueError('Fixed DC source-only question differs')
