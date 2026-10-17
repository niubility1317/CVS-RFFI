"""Full original source, three fixed E20 references; no source-RX holdout."""
from copy import deepcopy
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
RUN='20261017-phase1-F-crossRX-fullsource-concat-e20-s2026092701-m3-r02'
RELEASE='cvs_F_fullsource_concat_e20_20261017_r02'
LOCAL='local_artifacts/F_full_source_concat_20261017'
OWNER='codex/root/F-fullsource-concat-20261017'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def write(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')
def runtime():
 p=ROOT/'runtime.json'
 q=p if p.is_file() else ROOT/LOCAL/'runtime.json'
 return read(q) if q.is_file() else dict(project='PROJECT')
PROJECT=runtime()['project']
SOURCE=PROJECT+'/runs/phase1_daot_rc4_pure_game_m3_20260917_r2/source_contract.json'
CAPSULE=PROJECT+'/runs/20260927-phase1-baselines-final-clean-satellite-m5-r01/data/capsule'
TRUTH=PROJECT+'/runs/20260927-phase1-baselines-final-clean-satellite-m5-r01/data/truth.json'
VIEWS_ROOT=Path(PROJECT)/'runs/20261002-phase1-cvs-residual-sixscene-manysig-m8-r01/received_views'
AF_RELEASE=PROJECT+'/releases/cvs_SIGTIME_RDrop_AF50_20261015_r02'
SMOKE_SOURCE=Path(PROJECT)/'runs/20261016-phase1-F-supcon-e50-clean-manysig-m9-r02/CROSS_RX-s2026092701/source'
BASE=Path(PROJECT)/'runs'/RUN
SEED=2026092701;UPDATES=4440;STEPS=222;FIT=(1,3,4,6,8)
CLASSES=['14-10','14-7','20-15','20-19','6-15','8-20']
FULL_FP32_POLICY=dict(cudnn_allow_tf32=False,cuda_matmul_allow_tf32=False,cudnn_benchmark=False,cudnn_deterministic=False,matmul_precision='highest')
ARMS=('NATIVE','REPEAT','CONCAT')
SCENES=('practical_high','practical_mid','practical_low_suburban','practical_high_urban','practical_mid_urban','practical_low_urban')
VIEWS=('clean',*SCENES);TRAIN_SCENES=('practical_low_suburban','practical_mid_urban','practical_low_urban')
CHANNEL_SEEDS=dict(train=20261016421,validation=20261016422,receiver=20261016423)
def rows():return [dict(row_id=a+'-s'+str(SEED),arm=a,model_seed=SEED) for a in ARMS]
def expand(v):
 if isinstance(v,str):return v.replace('PROJECT',PROJECT)
 if isinstance(v,list):return [expand(x) for x in v]
 if isinstance(v,dict):return {k:expand(x) for k,x in v.items()}
 return v
def config(row):
 if row not in rows():raise ValueError('Only full-source fixed references allowed')
 c=expand(read(Path(__file__).parent/'reference_recipe.json'));arm=row['arm'];nf=2 if arm=='NATIVE' else 4
 c.update(schema='F_fullsource_plain_concat20_v1',run_id=RUN,**row,launch_owner=OWNER,output_root=(BASE/row['row_id']/'source').as_posix(),epochs=20,label_epochs=13,pseudo_epochs=7,updates=UPDATES,source_falsification=None,source_selection='Fixedfinal20/allregisteredrows;Vhealthonly,no source-RX holdout or performance gate',test_views=list(VIEWS),baseline_reuse=False,baseline_reference='Current matched full-source references only;no historical fold scores or weights',training_recipe='F+CROSS_RX/ordinary/RDrop1/LS.01/fullSinc/defaultcos;fullL6300U56700V27000;scratch20=13L+7U;'+arm,method='fullsource_plain_concat_reference',stage='fullsource_reference',new_Ag=False,As=arm=='CONCAT')
 c['training_mechanism'].update(sinc_coordinates='CENTER_25K',L_forwards_per_sample=nf,expected_L_forward_packets=nf*128*UPDATES,pair_weights=[1.] if arm=='NATIVE' else [.5,.5],second_pair='none' if arm=='NATIVE' else 'samegeneratedordinary' if arm=='REPEAT' else 'completepractical(raw)',same_IQ_within_RDpair=True,ordinary_before_satellite=False,all_satellite_L_supervised=True,new_U_augmentation=False,new_crossview_loss=False)
 c['training_mechanism'].update(original_L_identity_forwards=128*UPDATES,extra_L_identity_forwards=(nf-1)*128*UPDATES,objective='RD(ordinary)+SupCon' if arm=='NATIVE' else '.5RD(ordinary)+.5RD(extra)+SupCon(sharedordinary)',hook_correction='Actual registered total L objective minus native smoothedCE1')
 c['sinc_optimizer_units']['diagnostic_epochs']=[]
 c['goal']=dict(clean_net_positive=True,low_urban_above=.60,each_suburban_gap_exploratory_pp=2.,target_feedback=False,joint_goal_complete=False)
 c['data_contract']=dict(RX=list(FIT),days=[1,2,3],L=6300,U=56700,V=27000,split=392005,source_RX_holdout=None,internal_L_split=False,U_TX_truth_hidden=True)
 c['channel']=dict(train_scenes=list(TRAIN_SCENES),seeds=dict(CHANNEL_SEEDS),route='existingresidual/post_sync',EQ=False,reference_snr_db=30.,input='originalLbeforeordinary',no_rejection=True,test_IQ_unchanged=True)
 c['budget_authorization']='Latestuserrequiresfulloriginalsource;3scratchE20/60modelE/13320updates;0parentweights/continuation'
 c['cost_claim']='NATIVE2L/1136640packets;REPEATandCONCAT4L/2273280each;CONCAT568320practical;originalU/teacher/eval/smokeandwallmeasuredseparately'
 return c
def validate(c):
 row=next((r for r in rows() if r['row_id']==c.get('row_id')),None)
 if row is None or c!=config(row):raise ValueError('Full-source fixed data/budget/weights/defaulthead differs')
 return c
def base_model_config():
 from experiments.cvs_rdrop_sinc_units import design as af
 return af.config(next(r for r in af.rows() if r['arm']=='F' and r['model_seed']==SEED))
def make_args(c,device='cuda:0'):
 validate(c)
 from experiments.cvs_sigtime_rdrop_legacy import design as native
 ref=native.config(next(r for r in native.rows() if r['epochs']==20 and r['mode']=='RDROP' and r['model_seed']==SEED))
 a=native.make_args(ref,device);a.run_id=RUN;a.candidate_id=c['row_id'];a.output_dir=c['output_root']
 if getattr(a,'source_fold_held_rx',None) is not None or getattr(a,'source_fold_training_receivers',None):raise ValueError('Source holdout forbidden')
 if (a.epochs,a.label_epochs,a.pseudo_epochs,a.seed,a.label_smoothing,a.batch_size)!=(20,13,7,SEED,.01,128) or a.baseline_ckpt or a.teacher_ckpt or a.use_concat_sat_channel_aug or not a.use_ema_teacher:raise ValueError('Original full-source scratch20/U recipe required')
 return a
