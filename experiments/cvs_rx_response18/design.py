"""Fixed source-only response hypothesis on existing complete cell matrices."""
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
RUN='20261018-diagnostic-F-RXresponse-sourceCells-manysig-m1-r01'
RELEASE='cvs_F_RX_response_source_20261018_r01'
OWNER='codex/root/F-RX-response-source-20261018'
LOCAL=ROOT/'local_artifacts/F_RX_response_source_20261018'
RX=(1,3,4,6,8);REFERENCE=1
SOURCE='PROJECT/runs/phase1_daot_rc4_pure_game_m3_20260917_r2/source_contract.json'
DERIVED='PROJECT/runs/20261018-diagnostic-F-signalCoherence-sourceLtoV-manysig-m1-r01'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def write(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')
def resolve(alias,project):
 if not alias.startswith('PROJECT/'):raise ValueError('Registered project-relative handle required')
 return Path(project)/alias[len('PROJECT/'):]
def config():
 return dict(run_id=RUN,launch_owner=OWNER,output_root='PROJECT/runs/'+RUN,source_contract=SOURCE,derived_input=DERIVED,RX=list(RX),reference_RX=REFERENCE,source_RX_holdout=None,source_L=6300,source_V=27000,original_IQ_points=256,original_window=[80,160],period=20,
  model='20point unitaryDFT-diagonal sharedcomplex response D_r;Phi_D(G)=D*G*D^H*trace(G)/trace(D*G*D^H);all20bins retained',
  fit='L only:mean centered log diagonal ratios for amplitude;phase of mean G_r[k,1]*conj(G_ref[k,1]) for phase;unitRMS response,phase gauge harmonic1;no iterativeoptimizer or classifier',
  epsilon=1e-12,phase_anchor_bin=1,
  folds='18leaveTXanddayout statistical folds;other5TX/2day/all5RX always retained;72nonreference fits;referenceRX fixed by index,not selected',
  readout='Held V referenceRX matrix is input only;one shared fitted response predicts otherRX matrices;center5RX for error comparison;no V fit/update',
  controls='Original L-only shared additive contrast and sameTX/other2day L conditional contrast,unchanged physical V evaluation',
  criteria=dict(min_additive_SSE_reduction=.20,min_positive_folds=15,each_RX_not_worse_than_additive=True,max_SSE_to_TXconditional=1.05,all_phase_products_nonzero=True,numerical_SSE_tolerance=1e-20),
  on_fail='Close fixed DFT20/logratio/anchor1/shared-response hypothesis;no bin/window/anchor/floor/referenceRX/model-order/threshold rescue or IQ training',
  on_pass='Shared source response prediction only;not pure receiver calibration,physical RXreplacement,label-preserving IQ or Ag efficacy;no automatic training',
  scope='New input-acting response predictor on population mean projectors,not altered old projector statistic or reopened fixed FIR/phase/RXMIX candidate',
  limitations=['Mean of normalized packet projectors need not equal normalized transform of the mean;this is a fixed population prediction approximation','RX/TX cellmeans are not simultaneously paired packets;channel,equalization and samplingnoise remain mixed','Passing source prediction is not physical RXreplacement or safe full256IQ filtering;failing closes this proxy,not every possible shared receiver response'],
  source_only=True,raw_IQ_reads=0,CP=0,NN=0,teacher_NN=0,optimizer_steps=0,head_fit_updates=0,U_used=False,U_truth_read=False,target_access=False,IQ_operator_for_training=False,automatic_training=False,CPU_threads=2,GPU='none',
  future_training='F+CROSS_RX/full5RX/L6300/U56700hidden/V27000/256IQ/defaultcos joint/E30 unchanged;no model trained in this row')
def validate(c):
 if c!=config():raise ValueError('Fixed source response definition differs')
