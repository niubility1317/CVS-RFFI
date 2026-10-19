"""As-only compensated temperature from common scratch; full 20+10 schedule."""
from copy import deepcopy
from pathlib import Path
from experiments.cvs_noise_shape_factorial43 import design as old
ROOT=Path(__file__).resolve().parents[2]
RUN='20261019-phase1-FcrossRX-AsTemperature-e30-s2026092701-m2-r02';RELEASE='cvs_FcrossRX_AsTemperature_E30_20261019_r02';OWNER='codex/root/temperature84'
PROJECT=old.PROJECT;BASE=Path(PROJECT)/'runs'/RUN
SEED=2026092701;STEPS=222;UPDATES=6660;RX=FIT=old.RX
for _k in ('SOURCE','FULL_FP32_POLICY','CLASSES','SCENES','VIEWS','FORMAL_VIEWS','TRUTH','PANEL','SOURCE_CACHE'):globals()[_k]=deepcopy(getattr(old,_k))
read,write=old.read,old.write
ORIGINAL_RUN='20261019-phase1-FcrossRX-AsTemperature-e30-s2026092701-m2-r01'
def row_root(row):return Path(PROJECT)/'runs'/(ORIGINAL_RUN if row.startswith('T6-') else RUN)/row
def rows():return [dict(arm=a,row_id=a+'-s'+str(SEED),model_seed=SEED) for a in ('T1','T6')]
def reference_config(segment):return old.config(next(r for r in old.rows() if r['arm']=='B01'),segment)
def config(row,segment=30):
 if row not in rows() or segment not in (20,30):raise ValueError('Only matched scratch20+own10 two temperatures')
 c=deepcopy(reference_config(segment));b=row_root(row['row_id']);run=ORIGINAL_RUN if row['arm']=='T6' else RUN;parent=b/'stage20';tail=segment==30
 c.update(schema='AsTemperature84_v1',run_id=run,**row,launch_owner=OWNER,output_root=str(b/('source' if tail else 'stage20')),parent_source=str(parent) if tail else None,checkpoint_sources=[str(parent/'final_ssdg.pth')] if tail else [],method='As_only_compensated_CE_temperature',training_recipe='Original P43 sixL, own U/EMA, ORD128 SupCon;only As slots T*CE(z/T), no As KL in BOTH arms;common scratch20+own10',budget_authorization='Two full matched scratch E30 paths;60 model epochs,13320updates;no extra neural forwards',baseline_reference='T1 is exact CE-only As control;old P43 As/NoAs descriptive only,not initialization',benchmark_context='Human-exposed benchmark;temperature proposed in supplied analysis;no target-based temperature/candidate selection')
 c['temperature']=dict(T=1. if row['arm']=='T1' else 6.,compensated=True,scope='64 actual As slots only',As_KL=0.,ordinary_KL=1.,raw_KL=1.,label_smoothing=.01,default_head_scale=30.,deployment_temperature=1.,start='first As update from scratch',source_min_midlow_gain_pp=.5,source_clean_max_drop_vs_T1_pp=.1,automatic_extension=False)
 c['training_mechanism'].update(L_forwards_per_sample=6,objective='.5RDordinary+.5[.5RDordinary+.5TCE_As]+.25RDraw+originalSupCon',pair_weights=[.5,.5,.25])
 c['source_selection']='Fixed FINAL30;source criteria only future extension,all rows automatically all7'
 return c
def validate(c):
 row=next((r for r in rows() if r['row_id']==c.get('row_id')),None)
 if row is None or c!=config(row,c.get('segment')):raise ValueError('Fixed temperature/default head/source/schedule/budget differs')
 return c
def make_args(c,device='cuda:0'):
 validate(c);a=old.make_args(reference_config(c['segment']),device);a.run_id=RUN;a.candidate_id=c['row_id'];a.output_dir=c['output_root'];a.baseline_ckpt='' if c['segment']==20 else c['parent_source']+'/final_ssdg.pth';a.teacher_ckpt='';return a
