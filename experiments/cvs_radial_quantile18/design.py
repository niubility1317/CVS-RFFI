"""Fixed class-shared radial power-law source question;no training or IQ views."""
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
RUN='20261018-diagnostic-F-radialQuantile-sourceLtoV-manysig-m1-r01'
RELEASE='cvs_F_radial_quantile_source_20261018_r01'
OWNER='codex/root/F-radial-quantile-source-20261018'
LOCAL=ROOT/'local_artifacts/F_radial_quantile_source_20261018'
RX=(1,3,4,6,8)
SOURCE='PROJECT/runs/phase1_daot_rc4_pure_game_m3_20260917_r2/source_contract.json'
PREPARATION='PROJECT/runs/20261011-phase1-stage2-geometry-source-manysig-m3-r01/s2026092701'
ROW='RADIAL_POWERLAW_L_TO_V'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def write(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')
def resolve(alias,project):
 if not alias.startswith('PROJECT/'):raise ValueError('Project-relative source handle required')
 return Path(project)/alias[len('PROJECT/'):]
def config():
 return dict(run_id=RUN,launch_owner=OWNER,output_root='PROJECT/runs/'+RUN,source_contract=SOURCE,preparation=PREPARATION,RX=list(RX),reference_RX=1,days=[1,2,3],TX=list(range(6)),source_RX_holdout=None,L=6300,V=27000,L_per_cell=70,V_per_cell=300,IQ_points=256,CPU_threads=2,GPU='none',batch=256,quantiles=[.1,.2,.3,.7,.8,.9],center_quantile=.5,relative_amplitude_floor=1e-6,
 observable='Perpacket log(max(abs(z)/RMS,1e-6)) quantiles minus median,all256samples;mean of70/300 packets perTX/RX/day;no slicing,phase estimation,filtering or sample selection',
 mathematical_probe='For positive powerlaw g_a(z)=abs(z)^(a-1)*z followed by RMS,the unfloored centeredlogquantiles scale exactly by a;this is a probe family,not identified RX hardware or assumed labelpreservation',
 fit='L-only scalar a_RX throughorigin from other5TX/other2days,all5RX retained;18heldTX/day statistical folds;referenceRX1',
 evaluation='Predict V targetRX shape from sameTX/day referenceRX1 V shape;V only input for readout,never coefficient fit or persistent update',
 controls=['identitya1','L-only classshared sixdim additive offset','sameTX/other2days scalar slope'],
 criteria=dict(min_baseline_RMS=.01,min_shared_vs_identity_SSE_reduction=.20,min_shared_vs_additive_SSE_reduction=.10,min_positive_folds=15,max_each_nonreference_RX_SSE_increase=0.,max_shared_to_TXconditional_SSE_ratio=1.05,min_slope=.9,max_slope=1.1,max_floor_fraction_each_cell=.01,max_zero_RMS_packets=0,SSE_tolerance=1e-20),
 on_fail='Close fixed all256/quantiles/reference/powerlaw/mildrange/source question;no exponent/window/quantile/reference/RX/threshold rescue and no automatic training',
 on_pass='Only evidence for a common mild radial observation relation;separate actualboundedIQ identityguard/defaulthead and matched E30 recognition still required',
 distinction='Time-amplitude orderstatistics and nonlinear radial companding,not old spectralPSD affine/FIR/complexresponse,temporalenvelopeDCT or sameTX donor interpolation',
 source_only=True,IQ_modified_for_training=False,IQ_views_generated=0,network_constructed=False,checkpoint_loads=0,NN=0,teacher_NN=0,optimizer_steps=0,head_fit_updates=0,U_used=False,U_truth_read=False,target_access=False,V_updates=0,automatic_training=False,
 future_training='UnchangedF+CROSS_RX/all5RX/L6300/U56700TXhidden/V27000/256IQ/defaultcos joint/E30;no trained model in this row',
 limitations=['Unpaired distribution transfer cannot identify physical RX nonlinearity or separate TX/channel/equalization/noise','Powerlaw can alter PA identity;no label-preserving augmentation claim or training before separate realIQ guard','Unused U/features deserialize in existing cache but are never consumed','V is existing source validation;no new independent blind hardware test'],REJECTED_EXTRA_GATE='Generic designapproval is superseded by explicit existing research authorization and minimalprojectworkflow;no extra approval or seal')
def validate(c):
 if c!=config():raise ValueError('Fixed source configuration differs')
