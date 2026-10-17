"""No-refit source attribution on already saved factors and fixed responses."""
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
RUN='20261018-diagnostic-F-RXresponse-order-sourceCells-manysig-m1-r01'
RELEASE='cvs_F_RX_response_order_source_20261018_r01'
OWNER='codex/root/F-RX-response-order-source-20261018'
LOCAL=ROOT/'local_artifacts/F_RX_response_order_source_20261018'
RX=(1,3,4,6,8)
SOURCE='PROJECT/runs/phase1_daot_rc4_pure_game_m3_20260917_r2/source_contract.json'
DERIVED='PROJECT/runs/20261018-diagnostic-F-signalCoherence-sourceLtoV-manysig-m1-r01'
RESPONSE='PROJECT/runs/20261018-diagnostic-F-RXresponse-sourceCells-manysig-m1-r01'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def write(p,value):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')
def resolve(p,project):
 if not p.startswith('PROJECT/'):raise ValueError('Registered source handle required')
 return Path(project)/p[len('PROJECT/'):]
def config():
 return dict(run_id=RUN,launch_owner=OWNER,output_root='PROJECT/runs/'+RUN,source_contract=SOURCE,derived_input=DERIVED,response_input=RESPONSE,source_L=6300,source_V=27000,RX=list(RX),source_RX_holdout=None,original_IQ_points=256,window=[80,160],period=20,reference_RX=1,phase_anchor_bin=1,epsilon=1e-12,CPU_threads=2,GPU='none',
  hypothesis='Does exact mean of packetwise trace-normalized fixed response account for failed response-of-population-mean proxy?',
  response='Read the exact saved18x5x20 complex h;0refit;allbins/window/reference/phase/floor unchanged',
  prediction='u is saved periodic factor,not originalIQ;z=F*u;v=h*z*sqrt(norm(z)^2/norm(h*z)^2),zero input maps tozero;mean inverseDFT(v)*inverseDFT(v)^H over all300Vref packets',
  controls='Original shared additive L contrast,sameTX/otherday L contrast,and original population-mean response;all5RX centered as before',
  criteria=dict(min_additive_SSE_reduction=.20,min_positive_folds=15,each_RX_not_worse=True,max_SSE_to_TXconditional=1.05,old_phase_products_nonzero=True,numerical_SSE_tolerance=1e-20),
  on_fail='Close normalization-order explanation for these exact h;no refit/window/bin/anchor/reference/amplitude/threshold rescue or IQ training',
  on_pass='Source mathematical attribution only;original mean-proxy FAIL stays closed;no identitypreservingIQ claim,promotion or automatic training',
  inputs='Original L/V180cellmeans,V27000saved20complexfactors,Vrx/day/visibleTX metadata,and savedresponse completion/resolved/result/h;no rawIQ/cache/U/CP/target',
  no_sample_removal=True,source_only=True,raw_IQ_reads=0,closed_form_fits=0,CP=0,NN=0,teacher_NN=0,updates=0,U_used=False,U_truth_read=False,target_access=False,IQ_views_generated=0,automatic_training=False,joint_goal_complete=False,
  future_model_contract='F+CROSS_RX,SIG_TIME/unifiedSinc/joint160/defaultcos jointlytrained,full5RX,L6300/U56700hidden/V27000,256IQ,E30')
def validate(c):
 if c!=config():raise ValueError('Fixed no-refit source attribution differs')
