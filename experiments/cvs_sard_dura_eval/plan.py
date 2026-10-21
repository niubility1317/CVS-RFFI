from pathlib import Path

RUN='20261021-phase1-SARD-NoAs-DURA-e50-m6-r02'
RELEASE='cvs_SARD_NoAs_DURA_E50_m6_20261021_r02'
SEEDS=(2026092701,2026092702,2026092703)
VIEWS=('clean','practical_high','practical_mid','practical_low_suburban','practical_high_urban','practical_mid_urban','practical_low_urban')
PROJECT=Path('/home/szu2070436088/2510044040/CV-SincNet')
BASE=PROJECT/'runs'/RUN
PREVIOUS_BASE=PROJECT/'runs/20261021-phase1-SARD-NoAs-DURA-e50-m6-r01'
F50=PROJECT/'runs/20261016-phase1-F-supcon-e50-clean-manysig-m9-r02'
SARD01=PROJECT/'runs/20261020-phase1-T1-AsKL0-e50-s2026092701-m2-r01'
SARD23=PROJECT/'runs/20261021-phase1-T1-AsKL0-e30plus20-m3-r01'
DURA_RUN=PROJECT/'runs/20261021-phase1-restoration-stage2-graph-s2026092701-r01'
DURA_ROW=DURA_RUN/'stage2_residual_graph'
DURA_PRED=DURA_ROW/'target_confirmation/native_map_prior/prediction'
DURA_RELEASE=PROJECT/'releases/pamf_stage2_inference_20261021_r01'
RAW_F50=PROJECT/'runs/20261019-phase1-restoration-formal-FcrossRX50-s2026092701-r09'
SOURCE_CACHE=PROJECT/'data_caches/phase1_manysig_reference_20261019/source_cache'
DURA_CHECKPOINT=PROJECT/'runs/20261021-phase1-restoration-R65-four-route-s2026092701-m4-r69/stage2/native_map_prior/final.pt'
DURA_COMMIT='47c264286ca689f060a3f36aafb763fe209a6f06'
COUNT=168000

def rows():return [dict(arm=a,seed=s,row_id=a+'-s'+str(s)) for a in ('NOAS','SARD') for s in SEEDS]
def row(rid):
 found=next((r for r in rows() if r['row_id']==rid),None)
 if found is None:raise ValueError('Only registered six fixed reader rows')
 return found
def row_root(r):
 row(r['row_id'])
 return (PREVIOUS_BASE if r['arm']=='NOAS' else BASE)/r['row_id']
def pending_rows():return [r for r in rows() if r['arm']=='SARD']
def source(r):
 row(r['row_id'])
 if r['arm']=='NOAS':return F50/('CROSS_RX-s'+str(r['seed']))/'source'
 return (SARD01 if r['seed']==SEEDS[0] else SARD23)/('E30PLUS20-s'+str(r['seed']))/'source'
def reused(r,mode,view):
 row(r['row_id'])
 if mode not in ('raw','dura') or view not in VIEWS:raise ValueError('Only registered readout/views')
 return (r['arm']=='NOAS' and r['seed']==SEEDS[0]) or (mode=='raw' and (r['arm']=='SARD' or view=='clean'))
def policy():return dict(rows=rows(),views=list(VIEWS),count=COUNT,new_training_updates=0,head_fit_updates=0,target_fit=False,
 restorer='dura_v1',restorer_runtime_commit=DURA_COMMIT,restorer_runtime=dict(reuse=True,graph=True,native=False),
 restored_IQ='Reuse original complete fixed DURA v1 FP32 graph/RR outputs;no new G inference',
 common_restorer_weights=True,new_prediction_cells=47,reused_prediction_cells=37,std_ddof=1,
 recovery_complete_NOAS_root=str(PREVIOUS_BASE),new_prediction_cells_this_run=21,new_source_F_NN_this_run=12)
def validate(value):
 if value!=policy():raise ValueError('Fixed frozen matrix/runtime/zero-update policy differs')
 return value
