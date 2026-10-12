"""Report9.3: independent source prototypes directly constrain TX relations."""
from pathlib import Path
from copy import deepcopy
from experiments.cvs_nocfo_engineering import design as e
ROOT=Path(__file__).resolve().parents[2]
PROJECT,SOURCE,CAPSULE,TRUTH,CLASSES,SEEDS,FULL_FP32_POLICY,TEACHER_EXECUTION_COMMIT=(getattr(e,k) for k in ('PROJECT','SOURCE','CAPSULE','TRUTH','CLASSES','SEEDS','FULL_FP32_POLICY','TEACHER_EXECUTION_COMMIT'))
read,write,baseline=e.read,e.write,e.baseline
RUN='20261012-phase1-crossrx-relation-manysig-m15-r01';RELEASE='cvs_crossrx_relation_20261012_r01';OWNER='codex/root/crossrx-relation-20261012';BASE=Path(PROJECT)/'runs'/RUN
ARMS=('L30_CE','REL_FINAL_MERGED','REL_FINAL_SEPARATE','REL_MULTI_MERGED','REL_MULTI_SEPARATE');PRIMARY='REL_MULTI_SEPARATE'
PAIRS=[(a,'SSL') for a in ARMS]+[(a,'L30_CE') for a in ARMS if a!='L30_CE']+[('REL_FINAL_SEPARATE','REL_FINAL_MERGED'),('REL_MULTI_SEPARATE','REL_MULTI_MERGED'),('REL_MULTI_MERGED','REL_FINAL_MERGED'),('REL_MULTI_SEPARATE','REL_FINAL_SEPARATE')]
def rows():return [dict(arm=a,model_seed=s,row_id=f'{a}-s{s}') for s in SEEDS for a in ARMS]
def config(row):
    if row not in rows():raise ValueError('Unregistered relation row')
    c=deepcopy(e.config(next(r for r in e.rows() if r['arm']=='SSL' and r['model_seed']==row['model_seed'])))
    c.update(schema='crossrx_relation_v1',run_id=RUN,**row,method='cvs_crossrx_relation',stage='phase1_crossrx_relation',output_root=(BASE/row['row_id']/'source').as_posix(),launch_owner=OWNER)
    c['teacher']['cache_root']=(e.BASE/'peer_cache').as_posix();c['evaluation']['primary_arm']=PRIMARY
    arm=row['arm'];c['relation']=dict(mode='extra_CE' if arm=='L30_CE' else 'separate_RX' if arm.endswith('SEPARATE') else 'merged_RX',levels=[] if arm=='L30_CE' else ['time_down','t2.pool','id_feat_joint'] if 'MULTI' in arm else ['id_feat_joint'],interval=4,support_batch=30,reference_per_TX_RX=1,reference_day='cycle1/2/3 per active auxiliary step; same chosen day across balanced6TXx5RX',source_receivers=[1,3,4,6,8],source_days=[1,2,3],support_query_PID_disjoint=True,prototype_reference_RX='exclude query RX',prototype_grad=False,reference_mode='eval/no_grad for relation; ordinary train/backward for extraCE',centering='mean of independent 24 reference observations from other4RX',temperature=.1,weight=1.,weight_schedule='base semi ramp; starts on step448 after444warmup;999 calls',level_average='mean of enabled levels',query_features='actual clean L forward; shallow/middle mean+population std;final joint160',identity_CE_retained=True,U_relation=False,no_auxiliary_heads=True)
    c['SSL_control_prediction']=(e.BASE/f"SSL-s{row['model_seed']}"/'prediction/predictions.npz').as_posix()
    return c
def validate(c):
    row=next((r for r in rows() if r['row_id']==c.get('row_id')),None)
    if row is None or c!=config(row):raise ValueError('Fixed relation config differs')
    return c
learning_rate,learning_rate_step=e.learning_rate,e.learning_rate_step
