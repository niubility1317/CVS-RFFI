"""One fixed source question, distinct from cycle-difference residual correlations."""
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
RUN='20261018-diagnostic-F-signalCoherence-sourceLtoV-manysig-m1-r01'
RELEASE='cvs_F_signal_coherence_source_20261018_r01'
OWNER='codex/root/F-signal-coherence-source-20261018'
LOCAL=ROOT/'local_artifacts/F_signal_coherence_source_20261018'
RX=(1,3,4,6,8)
SOURCE='PROJECT/runs/phase1_daot_rc4_pure_game_m3_20260917_r2/source_contract.json'
PREPARATION='PROJECT/runs/20261011-phase1-stage2-geometry-source-manysig-m3-r01/s2026092701'
ROW='PERIODIC_SIGNAL_PROJECTOR_L_TO_V'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def write(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')
def resolve(alias,project):
 if not alias.startswith('PROJECT/'):raise ValueError('Registered project-relative handle required')
 return Path(project)/alias[len('PROJECT/'):]
def config():
 return dict(run_id=RUN,launch_owner=OWNER,output_root='PROJECT/runs/'+RUN,source_contract=SOURCE,preparation=PREPARATION,RX=list(RX),days=[1,2,3],TX=list(range(6)),source_RX_holdout=None,L=6300,V=27000,L_per_cell=70,V_per_cell=300,IQ_points=256,window=[80,160],period=20,cycles=4,relative_epsilon=1e-12,CPU_threads=2,GPU='none',batch=256,
 observable='Blind lag20 phase cancellation,mean of4observed cycles m;u=m/sqrt(norm(m)^2+1e-12*mean_cycle_energy),zero if denominator0;P=u*u^H;all20x20 complex entries,no carrier thresholds or sample removal',
 quality='Lag20 coherence/coherent power fraction/zero mean only descriptive;no quality filtering or alternate window',
 fit='All5RX cellmeans;L-only sharedRX contrast from other5TX and other2days;18heldTX/day statistical folds,allRX always retained',
 evaluation='All original V27000;center RX within each evaluation TX/day only for contrast readout,never fit/update',
 control='ZeroRX contrast and sameTX/other2day L-only RX contrast;all coefficients1,not fitted classifier',
 criteria=dict(min_baseline_Frobenius_RMS=.02,min_relative_risk_gain=.20,min_positive_folds=15,max_each_RX_SSE_increase=0.,max_shared_to_TXconditional_SSE_ratio=1.05,numerical_SSE_tolerance=1e-20),
 on_fail='Close fixed periodic-signal-projector source hypothesis;no offdiagonal/lag/window/eigenvector/RX/threshold rescue or automatic training',
 on_pass='Source-associated periodic shape evidence only;no causal receiver identification,IQ operator,label preservation,training or candidate promotion',
 distinction='Signal retained in cycle mean,not old cycle-mean-removed residual lag1to4 covariance;not reopening failed spectral-phase observer,FIR,unitary phase or RXMIX recipes',
 source_only=True,IQ_modified_for_training=False,IQ_views_generated=0,network_constructed=False,checkpoint_loads=0,NN=0,teacher_NN=0,optimizer_steps=0,head_fit_updates=0,U_used=False,U_truth_read=False,target_access=False,automatic_training=False,NoExplicitCFO_preserved=True,CFO_estimate_discarded=True,
 future_training='F+CROSS_RX/full5RX/L6300/U56700hidden/V27000/256IQ/defaultcos joint/E30;no model in this row',
 limitations=['Mean-cycle shape mixes TX,channel,RX,equalization and finite noise;not hardware disentanglement','P gain/globalphase invariance does not establish any label-preserving IQ transform','All L/V cache roles matched;unused U/features deserialize but are never consumed','V is existing source validation,not a new independent hardware blind test'],
 references=['https://arxiv.org/pdf/2112.15363','https://www.mathworks.com/help/wlan/gs/non-ht-ppdu-structure.html'])
def validate(c):
 if c!=config():raise ValueError('Fixed source question/config differs')
