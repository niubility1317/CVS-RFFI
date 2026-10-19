from copy import deepcopy
from pathlib import Path
from experiments.cvs_feature_head_joint30 import design as joint
read,write=joint.read,joint.write
ROOT=Path(__file__).resolve().parents[2]
RUN='20261019-phase1-F-rawCoAdapt-e20plus10-s2026092701-m4-r01';RELEASE='cvs_F_rawCoAdapt_E30_20261019_r01';LOCAL='local_artifacts/F_rawCoAdapt_E30_20261019';OWNER='codex/root/F-raw-coadapt'
PROJECT=read(ROOT/'runtime.json')['project'] if (ROOT/'runtime.json').is_file() else 'PROJECT';BASE=Path(PROJECT)/'runs'/RUN
SEED,EPOCHS,STEPS,UPDATES,BATCH=2026092701,10,222,2220,128
ADAPTER_SEED=20261019901
SOURCE,FORMAL_VIEWS,TRUTH,PANEL=joint.SOURCE,joint.FORMAL_VIEWS,joint.TRUTH,joint.PANEL
FIT,SCENES,VIEWS,CLASSES,FULL_FP32_POLICY=joint.FIT,joint.SCENES,joint.VIEWS,joint.CLASSES,joint.FULL_FP32_POLICY
REFERENCE_RUN,PARENT_ROW,HEAD,parent_config=joint.REFERENCE_RUN,joint.PARENT_ROW,joint.HEAD,joint.parent_config
REFERENCE_RELEASE=joint.RELEASE;REFERENCE_COMMIT='931af701470e781122494398dc58c3e027597550'
BANK_RUN=joint.FROZEN_RUN;BANK_BASE=Path(PROJECT)/'runs'/BANK_RUN
CONTROL_RUN=joint.RUN;CONTROL_BASE=Path(PROJECT)/'runs'/CONTROL_RUN
def rows():return [dict(row_id=s+'_'+name+'-s'+str(SEED),style=s,As=flag,arm=s+'_'+name,model_seed=SEED) for s in ('RAW','PAIR') for name,flag in (('NoAs',False),('As',True))]
def config(row):
 if row not in rows():raise ValueError('Unregistered raw/paired input four-row matrix')
 ref=joint.config(next(r for r in joint.rows() if r['style']=='TEMP' and r['As']==row['As']));c=deepcopy(ref)
 c.update(schema='F_raw_coadapt30_v1',run_id=RUN,**row,launch_owner=OWNER,output_root=str(BASE/row['row_id']/'source'),matched_control_run=CONTROL_RUN,observation_bank=str(BANK_BASE/'observation_bank'))
 c.pop('fixed_head_control_run',None)
 c['cumulative_backbone_exposure']='shared legal parent20 + own10;all original identity encoder,original W and2252 supplemental parameters jointly train;parent cost retained'
 c['feature_adapter']=dict(style=row['style'],raw_IQ_bypass=True,input='same unmodified core IQ' if row['style']=='RAW' else 'pair-average-repeat of same IQ for supplemental path only;native F still full IQ',encoder='2to8 k5 s2 p2 / SiLU / 8to12 k1 / DW12 k5 p2 / SiLU',heads='12to72 k1 at t1 preGN;12to96 k1 after AvgPool2 at t2 preGN',shared_encoder_once_per_forward=True,adapter_seed=ADAPTER_SEED,parameters=2252,convolution_MAC_per_packet=214528,initialization='encoder normal;B1/B2 zero;no zero gate',original_F_frozen=False,original_F_joint=True,default_W_joint=True,original_blocks_and_SIG_TIME_hooks=True)
 c['optimizer']['Sinc']='original encoder/Sinc/raw features/original W joint;CENTER_UNITS physical AdamW/globalclip5 exactly as FULL'
 c['source_promotion']=dict(per_style_urban_minus_FULL_joint_As_min_pp=.5,per_style_fullVclean_minus_FULL_joint_As_min_pp=-.2,per_style_As_minus_own_NoAs_fullVclean_min_pp=-.2,RAW_PAIR_urban_difference='secondary causal contrast,not eligibility',selection='highest eligible source urban;then clean;then RAW',automatic_extension=False,max_cumulative_epochs=50,on_fail='close fixed originalF+2252 joint k5/width12/E10 RAW/PAIR;no kernel/rank/lambda/seed rescue')
 return c
def validate(c):
 row=next((r for r in rows() if r['row_id']==c.get('row_id')),None)
 if row is None or c!=config(row):raise ValueError('Fixed raw source/parent/input/budget recipe differs')
 return c
