"""Fixed source-only affine-grid premise,not an augmentation efficacy claim."""
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
PROJECT='/home/szu2070436088/2510044040/CV-SincNet'
RUN='20261017-diagnostic-source-quantgrid-fitL-manysig-m1-r01'
RELEASE='cvs_source_quantgrid_fitL_20261017_r01'
OWNER='codex/root/source-quantgrid-20261017'
BASE=Path(PROJECT)/'runs'/RUN
SOURCE=PROJECT+'/runs/phase1_daot_rc4_pure_game_m3_20260917_r2/source_contract.json'
REFERENCE=PROJECT+'/runs/20261017-phase1-FcrossRX-AsTiming-sourceRX1-e20-s2026092701-m4-r01/ALL-RX1-s2026092701/source/source_contract.json'
PREPARATION=PROJECT+'/runs/20261011-phase1-stage2-geometry-source-manysig-m3-r01/s2026092701'
FIT=(3,4,6,8);DAYS=(1,2,3);TX=tuple(range(6));SEED=202610171801

def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def write(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')
def config():
 return dict(schema='source_affine_quantgrid_premise_v1',run_id=RUN,owner=OWNER,source_only=True,target_access=False,fit_RX=list(FIT),days=list(DAYS),TX=list(TX),L=5040,U=0,V=0,held_RX1=0,source_contract=SOURCE,effective_contract=REFERENCE,preparation=PREPARATION,model_epochs=0,optimizer_updates=0,network_forwards=0,checkpoint_reads=0,head_fit_updates=0,new_IQ_transforms=0,baseline='FixedF+CROSS_RX/SIG_TIME_X3/unifiedSinc/joint160/NoExplicitCFO/defaultcos30 normallyjointtrainedE20;not loaded here',representation='Original cached source eq1 IQ256 after native input RMS normalization;no original ADC/raw capture claim',estimator=dict(min_relative_spacing=2.**-16,max_relative_spacing=.5,float32_resolution_multiplier=64,min_unique_values=32,affine_residual_p99_max=.02,candidate_gaps='16smallestplus17quantiles',integer_divisors=list(range(1,33)),largest_fitting_grid=True,axis_aligned_IQ_components=True),criterion=dict(each_TX_RX_day_observable_fraction_min=.95,crossfit_TX_day_positive_folds_min=15,crossfit_squared_risk_reduction_min=.20,median_equivalent_uniform_dither_SNR_db_max=50.),on_fail='Close fixed observable-and-material axis-aligned quantgrid premise;no grid/tolerance/window/RX/seed/threshold rescue,no automatic dither or E20',on_pass='Only separately register source identity-safe IQ experiment;not ADC bitdepth/identitysafety/Ag/As/targetgain',automatic_training=False,formal_test_changed=False,stop_rule='Missing or illegal source handles/role mismatch/nonfinite input/output collision => preserve outputs andstop;no automaticretry/no otherowner changes')
def validate(c):
 if c!=config():raise ValueError('One fixed source-only premise/configuration')
 return c

def verify_contract(contract,carrier):
 if contract['held_RX']!=1 or contract['source_receivers']!=list(FIT) or contract['native_role_comparison']!='EXACT_MATCH_CARRIER_THEN_SOURCE_FOLD' or {k:len(v) for k,v in contract['role_ids'].items()}!={'L_s':5040,'U_s':45360,'V':21600}:raise ValueError('Exact existing fit4RX roles only')
 rr=contract['role_ids']
 if any(len(v)!=len(set(v)) or not set(v)<=set(carrier['role_ids'][k]) for k,v in rr.items()) or set(rr['L_s'])&(set(rr['U_s'])|set(rr['V'])) or set(rr['U_s'])&set(rr['V']):raise ValueError('Original disjoint VALIDATED_ONCE source handles required')
 return contract
