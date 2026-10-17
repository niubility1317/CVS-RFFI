"""Fixed carrier-selector organization x practical As,one common NoAs20 parent."""
from copy import deepcopy
from pathlib import Path
from experiments.cvs_clock_factorial30 import design as previous
ROOT=Path(__file__).resolve().parents[2]
RUN='20261018-phase1-F-crossRX-carrierGate-e20plus10-s2026092701-m4-r01';RELEASE='cvs_F_carrierGate_E30_20261018_r01';LOCAL='local_artifacts/F_carrier_batch_E30_20261018';OWNER='codex/root/F-carrierGate-E30'
read,write=previous.read,previous.write
PROJECT,SOURCE,CAPSULE,TRUTH,VIEWS_ROOT=previous.PROJECT,previous.SOURCE,previous.CAPSULE,previous.TRUTH,previous.VIEWS_ROOT
AF_RELEASE,SEED,FIT,SCENES,VIEWS,CLASSES,FULL_FP32_POLICY=previous.AF_RELEASE,previous.SEED,previous.FIT,previous.SCENES,previous.VIEWS,previous.CLASSES,previous.FULL_FP32_POLICY
TRAIN_SCENES,CHANNEL_SEEDS=previous.TRAIN_SCENES,previous.CHANNEL_SEEDS
STEPS,UPDATES=222,2220;BASE=Path(PROJECT)/'runs'/RUN;CARRIER_SEED=2026101891
base_model_config=previous.base_model_config
def runtime():
 p=ROOT/'runtime.json';return read(p if p.is_file() else ROOT/LOCAL/'runtime.json')
def rows():return [dict(row_id=name+'-s'+str(SEED),arm=arm,carrier_mode=mode,model_seed=SEED) for name,arm,mode in [('NOAS_BATCH','B00','BATCH'),('NOAS_PACKET','B00','PACKET'),('AS_BATCH','FULL_AS','BATCH'),('AS_PACKET','FULL_AS','PACKET')]]
def train_rows():return rows()
def parent_config(arm=None):return previous.parent_config('B00')
def parent_source(arm=None):return Path(parent_config()['output_root'])
def parent_proof(arm=None):return previous.parent_proof('B00')
def validate_parent_checkpoint(cp,c):return previous.validate_parent_checkpoint(cp,previous.config(next(r for r in previous.rows() if r['arm']=='B00')))
def model_config(c):return previous.config(next(r for r in previous.rows() if r['arm']==c['arm']))
def config(row):
 if row not in rows():raise ValueError('Only four fixed carrier organization/As rows')
 c=deepcopy(previous.config(next(r for r in previous.rows() if r['arm']=='B00')))
 c.update(run_id=RUN,**row,launch_owner=OWNER,output_root=(BASE/row['row_id']/'source').as_posix(),schema='F_CARRIER_CORRELATION_E20PLUS10_V1',parent_source=parent_source().as_posix(),parent_selection='One original fixed fullsource NoAs REPEAT final20 for allfour;never target-ranked',model_initialization='Strict same NoAs20 student and ownEMA;resetAdamW/cos10;all4 start before any As treatment',training_recipe='ExistingF+CROSS_RX/defaultcos/RDrop1/LS.01/originalU;only L-carrier selector dependence x complete ordinaryAs',ordinary_policy='Same configured parentE13 probabilities and ranges;BATCH bitwise native;PACKET iid phase/CFO selectors only;other global RNG preserved',source_RX=list(FIT),source_RX_holdout=None,source_data_contract=dict(L=6300,U=56700,V=27000,IQ_points=256,days=[1,2,3],split_id=392005,U_TX_hidden=True,internal_L_split=False),new_Ag=False,As=row['arm']=='FULL_AS',new_losses=0,new_deployment_parameters=0,new_neural_forwards=0,new_U_rules=0,carrier_policy=dict(mode=row['carrier_mode'],seed=CARRIER_SEED,scope='Single native L augmentor only;not U/teacher/practical',components=['existing phase rotation','existing CFO'],marginal='Same perpacket component probabilities/parameter laws as legacy;no stronger damage or new nuisance family',coupling='BATCH all/none selector versus independent perpacket selector',native_RNG='Consume original batchselector and conditional parameter draws in both;extra masks/missing parameters use private generators',ordering='Keep original native_general_channel order including multipath/noise before optional originalPA',SupCon='Unchanged existing original L pair only,two levels,batchstandardization;no added contrast or features',RDrop='Both forwards of each pair share exactly one alreadygeneratedIQ',not_new_IQ_Ag=True),supervised_views=dict(NoAs='.5RD(ordinary)+.5RD(sameordinary)',As='.5RD(ordinary)+.5RD(completepractical(raw))',all_satellite_L_supervised=True,physical_L_each_step=128,NN_L_forwards=4,total_L_scale='1 times mean perphysicalpacket before unchanged SupCon/U'),scientific_falsification=dict(type='Source organization x As;not IQAg x As',source='OriginalV fixed1800/full5RX,clean plus two practicaldraws before firstprimarytargetmetricread;fullV27000 separately reported',min_NOAS_PACKET_clean_gain_pp=.25,min_AS_PACKET_clean_vs_AS_BATCH_gain_pp=0.,min_AS_PACKET_clean_vs_NOAS_BATCH_gain_pp=0.,min_each_draw_lowurban_gain_pp=.5,min_each_draw_midurban_gain_pp=0.,min_each_suburban_gain_pp=-.25,exploratory_suburban_clean_gap_pp=2.,on_fail='Close fixed phase/CFO selectorcorrelation recipe;no amplitude/probability/component/parent/seed/budget rescue',on_pass='Source qualification only;not newIQAg or joint completion;no target-guided seed expansion'),source_selection='Every fixed final10/cumulative30;all4 formal sevenviews mandatory,sourcecrossrow verdict before metricread',cost_claim='4L forwards perrow/noextraNN/VJP;private selector generation counted separately;one shared parent20 counted once plus4x10newmodelE',budget_authorization='Each cumulativeE30;one shared20parent+40newmodelE,8880newupdates;noRXholdout',REJECTED_EXTRA_GATE='No extra design approval,source gate,seal or repeatedreview')
 return c
def validate(c):
 row=next((r for r in rows() if r['row_id']==c.get('row_id')),None)
 if row is None or c!=config(row):raise ValueError('Fixed carrier organization/data/parent/E30 matrix differs')
 return c
def make_args(c,device='cuda:0'):
 validate(c);a=previous.make_args(previous.config(next(r for r in previous.rows() if r['arm']=='B00')),device);a.run_id=RUN;a.candidate_id=c['row_id'];a.output_dir=c['output_root'];a.baseline_ckpt=str(parent_source()/'final_ssdg.pth');return a
