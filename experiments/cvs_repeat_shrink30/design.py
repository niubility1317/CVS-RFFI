"""One IQ residual-contraction independent-value question,common NoAs20 parent,cumulativeE30."""
from copy import deepcopy
from pathlib import Path
from experiments.cvs_clock_factorial30 import design as previous
ROOT=Path(__file__).resolve().parents[2]
RUN='20261018-phase1-F-crossRX-repeatShrink-e20plus10-s2026092701-m2-r01';RELEASE='cvs_F_repeat_shrink_E30_20261018_r01';LOCAL='local_artifacts/F_repeat_shrink_E30_20261018';OWNER='codex/root/F-repeat-shrink-E30'
read,write=previous.read,previous.write
PROJECT,SOURCE,CAPSULE,TRUTH,VIEWS_ROOT=previous.PROJECT,previous.SOURCE,previous.CAPSULE,previous.TRUTH,previous.VIEWS_ROOT
AF_RELEASE,SEED,FIT,SCENES,VIEWS,CLASSES,FULL_FP32_POLICY=previous.AF_RELEASE,previous.SEED,previous.FIT,previous.SCENES,previous.VIEWS,previous.CLASSES,previous.FULL_FP32_POLICY
TRAIN_SCENES,CHANNEL_SEEDS=previous.TRAIN_SCENES,previous.CHANNEL_SEEDS
STEPS,UPDATES=222,2220;BASE=Path(PROJECT)/'runs'/RUN;SAMPLING_SEED=20261018104
base_model_config=previous.base_model_config
def runtime():
 p=ROOT/'runtime.json';return read(p if p.is_file() else ROOT/LOCAL/'runtime.json')
def rows():return [dict(row_id=name+'-s'+str(SEED),arm=name,new_Ag=enabled,model_seed=SEED) for name,enabled in [('B00',False),('B10',True)]]
def train_rows():return rows()
def parent_config(arm=None):return previous.parent_config('B00')
def parent_source(arm=None):return Path(parent_config()['output_root'])
def parent_proof(arm=None):return previous.parent_proof('B00')
def validate_parent_checkpoint(cp,c):return previous.validate_parent_checkpoint(cp,previous.config(next(r for r in previous.rows() if r['arm']=='B00')))
def model_config(c):return previous.config(next(r for r in previous.rows() if r['arm']=='B00'))
def config(row):
 if row not in rows():raise ValueError('Only fixed B00/B10 IQ-contraction rows')
 c=deepcopy(previous.config(next(r for r in previous.rows() if r['arm']=='B00')))
 c.update(run_id=RUN,**row,launch_owner=OWNER,output_root=(BASE/row['row_id']/'source').as_posix(),schema='F_REPEAT_SHRINK_E20PLUS10_V1',parent_source=parent_source().as_posix(),parent_selection='One original fixed fullsource NoAs REPEAT final20 student and ownEMA;not targetranked',model_initialization='Strict common NoAs20;resetAdamW/cos10;allsourceRX',training_recipe='ExistingF+CROSS_RX/defaultcos/RDrop1/LS.01/originalU;fixed independently supervised raw/Ag slot',ordinary_policy='Exact native parentE13 probabilities/ranges/RNG/order in bothrows',source_RX=list(FIT),source_RX_holdout=None,source_data_contract=dict(L=6300,U=56700,V=27000,IQ_points=256,days=[1,2,3],split_id=392005,U_TX_hidden=True,internal_L_split=False),As=False,new_losses=0,new_deployment_parameters=0,new_neural_forwards=2,new_U_rules=0,IQ_operator=dict(source_guard_run='20261018-diagnostic-F-repeatShrink-sourceV-s2026092701-m1-r01',equation='x_W-lambda*w*(x_W-mean_cycle(x_W))',window=[80,160],period=20,cycles=4,taper='same sin(pi*t/19)^2 eachcycle;ends0',lambda_distribution='perpacket independent Uniform[0,.1];same private draws inB00 control,appliedstrength0',sampling_seed=SAMPLING_SEED,labels_or_quality_used=False,extra_RMS=False,CFO_estimated=False,no_new_noise=True,only_labeled=True,As_composition=False),supervised_views=dict(primary='.5 RD(onegenerated ordinary)',anchor='.25 RD(raw)',newslot='.25 RD(raw inB00 / ownpacket contraction(raw) inB10)',physical_L_each_step=128,L_NN_forwards=6,RDrop='same generated IQ within eachpair;three independent ordinary/raw/slot dropout pairs',loss_scale='RD .5+.25+.25=1 meanperphysical;originalSupCon fromprimary and originalU unchanged',stable_original_supervision='same .5ordinary+.25raw anchor inbothrows'),scientific_falsification=dict(source='Full originalV27000 clean finaldefaulthead;reuses native finalsourceevaluation;all5RX',minimum_fullV_clean_gain_pp=.5,minimum_each_RX_gain_pp=-.25,on_fail='Close fixedcontraction/window/period/taper/Uniform0to.1/parent/seed/E30;no amplitude/window/seed/parent/budget rescue',on_pass='Only independent Ag sourcequalification;formal frozenB00/B10 still required,no targetfeedback;Ag+As separate later',exploratory_each_suburban_clean_gap_pp=2.),source_selection='Both fixedfinal10/cumulative30;both automaticclean+sixpractical;source decision frozen beforeprimary formalmetricread',cost_claim='Matched6Lforwards/step vsold4;commonfixed20+2x10newE;2220newupdates each;originalU/teacher unchanged;CPUoperator+logging explicit',budget_authorization='EachcumulativeE30;onecommon20+20newmodelE/4440newupdates;noRXholdout',REJECTED_EXTRA_GATE='No extra designapproval,seals,datarevalidation or unchangedoperator rereview')
 c['method']='fullsource_repeat_residual_contraction_independent_test'
 c['checkpoint_sources']=[parent_source().as_posix()+'/final_ssdg.pth']
 c['training_mechanism'].update(objective='.5RD(ordinary)+.25RD(rawanchor)+.25RD(rawcontrol_or_Ag)+originalSupCon(primary)',extra_L_identity_forwards=1420800,L_forwards_per_sample=6,expected_L_forward_packets=1704960,pair_weights=[.5,.25,.25],second_pair='unchanged raw anchor',third_pair='rawcontrol_or_ownpacketAg',all_satellite_L_supervised='NA,noAs training')
 c['channel']['training_enabled']=False
 c['channel']['scope']='No As in B00/B10;unchanged formal sixpractical observations only'
 return c

def validate(c):
 row=next((r for r in rows() if r['row_id']==c.get('row_id')),None)
 if row is None or c!=config(row):raise ValueError('Fixed Ag/roles/parent/E30 matrix differs')
 return c

def make_args(c,device='cuda:0'):
 validate(c);a=previous.make_args(model_config(c),device);a.run_id=RUN;a.candidate_id=c['row_id'];a.output_dir=c['output_root'];a.baseline_ckpt=parent_source().as_posix()+'/final_ssdg.pth';return a
