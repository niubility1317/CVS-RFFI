"""One source-only identity-protection question;not a model-training candidate yet."""
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
RUN='20261018-diagnostic-F-identitySensitivity-sourceLtoV-s2026092701-m1-r01';RELEASE='cvs_F_identity_sensitivity_source_20261018_r01';LOCAL='local_artifacts/F_identity_sensitivity_source_20261018';OWNER='codex/root/F-identity-sensitivity-source'
BASE='PROJECT/runs/'+RUN
SOURCE='PROJECT/runs/phase1_daot_rc4_pure_game_m3_20260917_r2/source_contract.json'
PREPARATION='PROJECT/runs/20261011-phase1-stage2-geometry-source-manysig-m3-r01/s2026092701'
REFERENCE_RUN='20261017-phase1-F-crossRX-fullsource-concat-e20-s2026092701-m3-r02'
MODELS=('REPEAT','CONCAT');FIT=(1,3,4,6,8);PER_CELL=20;FIT_PER_CELL=2
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def write(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')
def resolve(alias):
 if not alias.startswith('PROJECT/'):raise ValueError('Registered project alias required')
 return Path(read(ROOT/'runtime.json')['project'])/alias[len('PROJECT/'):]
def config():
 return dict(run_id=RUN,launch_owner=OWNER,source_contract=SOURCE,preparation=PREPARATION,models=[dict(arm=a,row=a+'-s2026092701',source='PROJECT/runs/'+REFERENCE_RUN+'/'+a+'-s2026092701/source',epoch=20,selection='original fixed source-final20;not target-ranked',parent_updates=4440) for a in MODELS],source_RX=list(FIT),source_RX_holdout=None,L6300_unchanged=True,U_used=False,U_TX_hidden=True,L_fit=180,L_fit_per_cell=2,V_role_count=27000,V_selected=1800,per_cell=20,model_seed=2026092701,split_seed=392005,selection='first2L andfirst20V opaqueIDs per90TX/RX/daycells;all5trainingRX retained',source_gradient=dict(readout='REPEAT final20 defaulthead only',function='trueclass minus highest other cosine logit,dividedby30',object='input IQ only;all modelparametersrequiresgradFalse',descriptor='abs(inputgradient dot taperedblockIQ / blockL2norm);perpacketnormalizeover16,meanperLcell,maxover90cells',low_blocks=4,high_blocks=4,ties='ascendingblockindex thenstableargsort;highlast4',no_fitted_classifier=True,no_target=True),operator=dict(kind='same I/Q positive localgain field,identity-directed observation variation;not physical RXreplacement',grid=16,block_length=16,taper='sin(pi*t/15)^2,exact0atbothends',gain_bound=.1,nominal_relative_IQ_step=.01,energy_control='perpacket delta=min(.01,.1*min_block_tapered_IQL2/fullIQL2);everyblock/sign same relativeL2 step;no packetfiltering',signs=[-1,1],extra_RMS=False,phase_or_CFO_changed=False,new_noise=False,PA_or_quantization=False),criteria=dict(each_low_view_accuracy_delta_pp_min=-.5,each_low_view_correct_retention_min=.995,each_RX_or_TX_accuracy_delta_pp_min=-1.,uniform_new_errors_each_readout_min=20,low_error_rate_over_uniform_max=.75,low_error_rate_over_high_max=.5,source_high_score_over_low_min=1.5,numerical_absolute_step_error_max=1e-6,gain_bound=.1),planned_NN=118984,real_CP_loads=2,smoke_NN=4,fit_L_forward_NN=180,source_V_forward_NN=118800,source_cache_loads=1,operator_applications=57600,input_gradient_packets_max=182,input_backward_tensor_calls_max=3,training_updates=0,parameter_gradient_updates=0,teacher_NN=0,head_fit_updates=0,ordinary_generations=0,target_access=False,automatic_training=False,on_fail='Close fixedgradientguide/maxcellscore/grid16/gain.1/step.01/parents/seed;no location/grid/gain/score/seed/parent rescue',on_pass='Source perturbability ranking only;not DG,Ag/As efficacy or automaticE30training/targetpromotion',device='one shared-capacity GPU',threads=2)

def validate(c):
 if c!=config():raise ValueError('Fixed source sensitivity experiment differs')
 return c
