"""One seed, four scratch E20 rows; no CROSS_RX, RDrop or inherited weights."""
import json
from copy import deepcopy
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
RUN='20261019-phase1-F-nontrue-CSIL-e20-s2026092701-m4-r01'
RELEASE='cvs_F_nontrue_CSIL_E20_20261019_r01'
LOCAL='local_artifacts/F_nontrue_E20_20261019';OWNER='codex/root/F-nontrue'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def write(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')
PROJECT=read(ROOT/'runtime.json')['project'] if (ROOT/'runtime.json').is_file() else 'PROJECT'
BASE=Path(PROJECT)/'runs'/RUN;SEED=2026092701;STEPS=222;UPDATES=4440;FIT=(1,3,4,6,8)
SOURCE=Path(PROJECT)/'runs/phase1_daot_rc4_pure_game_m3_20260917_r2/source_contract.json'
FORMAL_VIEWS=Path(PROJECT)/'runs/20261002-phase1-cvs-residual-sixscene-manysig-m8-r01/received_views'
TRUTH=Path(PROJECT)/'runs/20260927-phase1-baselines-final-clean-satellite-m5-r01/data/truth.json'
PANEL=Path(PROJECT)/'runs/20261017-diagnostic-FcrossRX-fullsource-residual-s2026092701-m3-r01/diagnostic'
SOURCE_CACHE=Path(PROJECT)/'runs/20261011-phase1-stage2-geometry-source-manysig-m3-r01/s2026092701'
TEACHER=Path(PROJECT)/'paper_reproduction/runs/20260930-phase1-native-baselines-practical-manysig-m5-r01/csil-s2026092701'
TEACHER_QUAL=Path(PROJECT)/'runs/20261019-diagnostic-FcrossRX-sameIQ-experts-s2026092701-m2-r02/csil/comparison.json'
CLASSES=['14-10','14-7','20-15','20-19','6-15','8-20']
SCENES=('practical_high','practical_mid','practical_low_suburban','practical_high_urban','practical_mid_urban','practical_low_urban');VIEWS=('clean',*SCENES)
FULL_FP32_POLICY=dict(cudnn_allow_tf32=False,cuda_matmul_allow_tf32=False,cudnn_benchmark=False,cudnn_deterministic=False,matmul_precision='highest')
def rows():return [dict(row_id=a+'-s'+str(SEED),arm=a,model_seed=SEED) for a in ('B0','C0','B1','M')]
def native_config():
 from experiments.cvs_sigtime_rdrop_legacy import design as old
 return old.config(next(r for r in old.rows() if r['epochs']==20 and r['mode']=='ORIGINAL' and r['model_seed']==SEED))
def model_config():
 c=deepcopy(native_config());c.update(arm='F',combination=dict(SIG_TIME='SIG_TIME_X3',Sinc_coordinates='CENTER_25K',Sinc_optimizer='CENTER_UNITS'));return c
def config(row):
 if row not in rows():raise ValueError('Unregistered fixed F row')
 c=deepcopy(native_config());arm=row['arm'];nf=1 if arm=='B0' else 2
 c.update(schema='F_nontrue20_v1',run_id=RUN,**row,output_root=str(BASE/row['row_id']/'source'),launch_owner=OWNER,baseline_reuse=False,epochs=20,label_epochs=13,pseudo_epochs=7,updates=4440,checkpoint_sources=[],native_scratch_only=True,model_initialization='same-seed scratch F',test_views=list(VIEWS),checkpoint_selection='final_only',training_recipe='F=SIG_TIME_X3+CENTER_25K+CENTER_UNITS; ORIGINAL CE.01/defaultcos30; fullL6300U56700V27000; scratch20=13L+7U; '+arm,As=arm in ('B1','M'),head_fit_updates=0)
 c['training_mechanism']=dict(mode='ORIGINAL',alpha=0.,CROSS_RX=False,RDrop=False,L_forwards_per_sample=nf,ordinary_CE_weight=1. if arm=='B0' else .5,extra_CE_weight=0. if arm=='B0' else .5,extra_input='none' if arm=='B0' else 'raw' if arm=='C0' else 'complete practical(raw)',CE_temperature=1.,CE_smoothing=.01,nontrue_weight=.5 if arm=='M' else 0.,KD_multiplier=6.,student_KD_temperature=6.,teacher_target='native logits restricted to five nontrue classes',teacher_correct_mask=True,normalization='all128; no accepted-only denominator',teacher_packets_per_step=0 if arm=='B0' else 128,teacher_shadow=arm in ('C0','B1'),physical_shadow=arm=='C0',U='original hidden TX gate+ownEMA unchanged',new_deploy_parameters=0)
 c['source_promotion']=dict(M_minus_B1_midlow_min_pp=.5,M_minus_B0_fullVclean_min_pp=-.2,automatic_extension=False,max_cumulative_epochs=50,decision_before_formal_scores=True,on_fail='close fixed nontrue-only recipe; no coefficient/temperature rescue')
 c['data_contract']=dict(L=6300,U=56700,V=27000,RX=list(FIT),days=[1,2,3],split=392005,U_TX_hidden=True,source_RX_holdout=None)
 return c
def validate(c):
 r=next((r for r in rows() if r['row_id']==c.get('row_id')),None)
 if r is None or c!=config(r):raise ValueError('Fixed F/default head/E20/source data matrix differs')
 return c
def make_args(c,device='cuda:0'):
 validate(c)
 from experiments.cvs_sigtime_rdrop_legacy import design as old
 a=old.make_args(native_config(),device);a.run_id=RUN;a.candidate_id=c['row_id'];a.output_dir=c['output_root']
 if (a.epochs,a.label_epochs,a.pseudo_epochs,a.batch_size,a.seed,a.label_smoothing)!=(20,13,7,128,SEED,.01) or a.baseline_ckpt or a.teacher_ckpt or not a.use_unlabeled or not a.use_ema_teacher or not a.use_aug or a.use_concat_sat_channel_aug or getattr(a,'source_fold_held_rx',None) is not None:raise ValueError('Full source native scratch/oldU recipe differs')
 return a
