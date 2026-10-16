"""One fixed source RX-associated circularity question, not an augmentation claim."""
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
PROJECT='/home/szu2070436088/2510044040/CV-SincNet'
RUN='20261016-diagnostic-F-IQ-circularity-source-crossfit-manysig-m1-r01'
RELEASE='cvs_F_IQ_circularity_source_crossfit_20261016_r01'
OWNER='codex/root/F-IQ-circularity-source-20261016'
SOURCE=PROJECT+'/runs/phase1_daot_rc4_pure_game_m3_20260917_r2/source_contract.json'
PREPARATION=PROJECT+'/runs/20261011-phase1-stage2-geometry-source-manysig-m3-r01/s2026092701'
BASE=PROJECT+'/runs/'+RUN
LOCAL=ROOT/'local_artifacts/F_IQ_circularity_source_20261016'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def write(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')
def config():
 return dict(run_id=RUN,launch_owner=OWNER,source_contract=SOURCE,preparation=PREPARATION,output_root=BASE,
 role='Original visible L_s only; fitRX3/4/6/8;excludeRX1 before any observation',count=5040,RX=[3,4,6,8],days=[1,2,3],TX=list(range(6)),samples_per_cell=70,
 representation='Native equalized1,256 samples,25Msps,packetRMS;never modified as training/test input',window=[80,160],period_samples=20,epsilon=1e-6,
 primary_observable='Centered circularity magnitude abs(mean((w-mean(w))**2))/(mean(abs(w-mean(w))**2)+eps**2),after unchanged reference lag20 CFO compensation',
 secondary_observable='Same raw-window circularity,odd/even corrected cycles,coherence;descriptive only,never fallback winner',
 cfo_scope='Reference ExcitationResponse.components lag20 atan2(real+eps**2) rule;no saved CFO feature;no deployment compensation changes',
 crossfit='18folds jointly exclude oneTX and oneday;equal70packet means in72cells;RXmeans versus pooledmean and all9wrongRX derangements',
 TX_split_control='All10 unordered3TX-versus3TX partitions;minimum centeredRX cosine;invariant to class label permutation',
 criteria=dict(min_total_risk_gain=.05,min_positive_folds=14,min_RX_gain=0.,min_gain_above_best_wrong_RX=.02,min_all_TX_split_center_cosine=.8,min_RX_center_to_odd_even_difference_power_ratio=2.),
 on_fail='Close this fixed circularity RX-nuisance hypothesis;no alternate coordinate/window/compensation/threshold/RX/seed rescue;no automatic training',
 on_pass='Only repeatable RX-associated observation evidence;requires separate identity-preservation/defaulthead check and matched E20 Ag training;not RXhardware or Ag efficacy',
 limitations=['TX,RX,channel and preprocessing can affect circularity;no unique IQ parameter recovery','Removing arbitrary RXimage may also erase TXidentity;no augmentation implemented','Original ordinary PA route can already draw image terms;not a wholly new perturbation by naming it RX','No classifier/regression head,model,checkpoint,teacher,training or target;source scalar mean predictor only'],
 U_used=False,U_truth_read=False,V_used=False,held_RX1_used=False,checkpoint_loads=0,network_forwards=0,optimizer_steps=0,automatic_training=False,
 source_only=True,target_access=False,device='CPU',threads=2,
 mechanism_source='https://arxiv.org/html/1402.6083',
 source_cache_scope='Existing source carrier deserialized;onlyoriginalL5040 selected forstatistics;unusedU/V/featuresand1260heldL never enterobservations;no weights consumed')
def validate(c):
 if c!=config():raise ValueError('Fixed source circularity question differs')
