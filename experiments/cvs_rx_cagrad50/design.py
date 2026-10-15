from copy import deepcopy
from pathlib import Path
from experiments.cvs_receiver_vrex50 import design as parent
ROOT=Path(__file__).resolve().parents[2]
for name in ('PROJECT','SOURCE','CAPSULE','TRUTH','CLASSES','FULL_FP32_POLICY','write','READOUTS','VIEWS','QUERY_COUNT','EPOCHS','LABEL_EPOCHS','PSEUDO_EPOCHS','STEPS','UPDATES','BASELINE_EVAL_RUN','STYLE_LAYERS','PARAMETERS','RX','DIAGNOSTIC_EPOCHS','FEATURE_PARAMETER_COUNT','VERIFY_INITIAL_EVAL_EQUAL','baseline_config'):
 globals()[name]=getattr(parent,name)
RUN='20261015-phase1-rx-cagrad-oldgate-cos-e50-clean-manysig-s01-m3-r01'
RELEASE='cvs_rx_cagrad50_20261015_r01';OWNER='codex/root/rx-cagrad50-20261015';BASE=Path(PROJECT)/'runs'/RUN
SEEDS=(2026092701,);ARMS=('OLD_GATE_COS50','NORM_CONTROL','RX_CAGRAD');BASELINE_ARM=ARMS[0];FEATURE_CONTROL=INITIAL_CONTROL=ARMS[1];PRIMARY=ARMS[2]
PAIRS=[(PRIMARY,FEATURE_CONTROL),(PRIMARY,BASELINE_ARM),(FEATURE_CONTROL,BASELINE_ARM)]
CLASS_W='id_backbone.encoder.core.id_backbone.cls_head.weight'
def rows():return [dict(row_id=f'{a}-s{s}',arm=a,model_seed=s) for s in SEEDS for a in ARMS]
def training_rows():return [r for r in rows() if r['arm']!=BASELINE_ARM]
def config(row):
 if row not in rows():raise ValueError('Unregistered RXgradient row')
 old=next(r for r in parent.rows() if r['arm']==(BASELINE_ARM if row['arm']==BASELINE_ARM else parent.FEATURE_CONTROL));c=deepcopy(parent.config(old));c.pop('receiver_risk');c.pop('source_qualification_run');c.pop('source_decision_commit');c.update(run_id=RUN,**row,output_root=(BASE/row['row_id']/'source').as_posix(),launch_owner=OWNER)
 c['receiver_gradient']=dict(mode=row['arm'],role='L_s',groups=list(RX),coefficient=.5,reference='Original native sample-mean CE smoothing.01; RX mean gradients weighted by actual counts',scope='All195041 deployedf_theta parameters except exact classW alone; keep norm/gain/reference',formula='Solve simplex min_w g0.Gw+c*norm(g0)*norm(Gw); d=g0+c*norm(g0)*Gw/norm(Gw), no rescaling',control='Same solver/gradients, retain original direction scaled to norm(d); direction-only matched norm at own state',solver='SciPySLSQP analyticjac double normalizedGram; maxiter100 ftol1e-10; relative primal-dual gap<=1e-4',fallback='On solver failure/nonfinite/invalid simplex/gap, use originalCEdirection; log explicitly; sourcequalification fallback<=1%',absent_RX='Only presentRX; originalcount-weighted CE reference;oneRX analytic',default_W='Original native CE gradient alone; no receiverweight/mask/correction; normaljointtraining and originalglobalclip',original_U='Unchanged oldbatch_neighbor ownEMA .16UCE .01entropy/strong.015; f_theta correction only labelledCE',enable='Every11100 Lstep fromepoch1 on existingoriginalforward,5orless VJPs; no newmodelFF orparams',parameters=list(PARAMETERS),diagnostic_epochs=list(DIAGNOSTIC_EPOCHS),reference_paper='https://proceedings.neurips.cc/paper/2021/file/9d27fdf2477ffbff837d73ef7ae23db9-Paper.pdf',adaptation='Receiver groups as objectives, original sample-mean reference instead of paper uniformmean; ordinary AdamW preconditioning; no paper convergence/DG guarantee')
 c['source_falsification']=dict(final5_original_L_CE_max=.5,native_steps=11100,actual_U_packets=963900,feature_difference_min=1e-6,solver_fallback_fraction_max=.01,minimum_direction_change_steps=1,minimum_relative_worst_local_gain_vs_norm_control=-1e-4,actual_selected_gradient_formula_rtol=1e-5,default_W_gradient_correction_zero=True,required_gradient_epochs=[1,17,33,50],source_only=True,expansion='If exactmechanism/defaultfeature/budget/sourcehealthPASS, freezeunchangedbotharms02/03 before reading ownnewtargetsummary; no targetfeedback or V selection')
 c['hypothesis_basis']='Prior RXsign mask sourceprobe retained77.51%norm but aggregateRX4cos-.02688 vsrandom+.01731. Coordinate majority doesnotguaranteevector common descent. Direct maxmin trustregion usesfull5RX Gram, preserving originalCE reference; norm-onlycontrol isolates direction. General theory, nottarget scores, fixes c.5.'
 c['goal']=dict(primary=PRIMARY,clean_three_seed_mean=.84,first_seed_is_not_goal_completion=True,feedback_to_training=False)
 return c
def validate(c):
 r=next((r for r in rows() if r['row_id']==c.get('row_id')),None)
 if r is None or c!=config(r):raise ValueError('Fixed RXgradient recipe differs')
def make_args(c,device='cuda:0'):
 validate(c)
 if c['baseline_reuse']:raise ValueError('Historicalbaseline cannotretrain')
 a=parent.make_args(parent.config(next(r for r in parent.training_rows() if r['arm']==parent.FEATURE_CONTROL)),device);a.run_id,a.candidate_id,a.output_dir=RUN,c['row_id'],c['output_root'];return a
def read(path):return parent.read(path,root=ROOT)
