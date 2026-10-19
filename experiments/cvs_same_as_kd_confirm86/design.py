"""Source-qualified fixed three-student-seed confirmation; reuse completed seed01."""
from copy import deepcopy
from pathlib import Path
from experiments.cvs_same_as_kd85 import design as old
ROOT=Path(__file__).resolve().parents[2];PROJECT=old.PROJECT
RUN='20261019-phase1-FcrossRX-sameAsKD-confirm-e30-m6-r01';RELEASE='cvs_FcrossRX_sameAsKD_confirm_E30_20261019_r01';OWNER='codex/root/sameAsKDconfirm86';BASE=Path(PROJECT)/'runs'/RUN
RX=FIT=old.RX;STEPS=222;UPDATES=6660;SEEDS=(2026092701,2026092702,2026092703)
for _k in ('SOURCE','FULL_FP32_POLICY','CLASSES','SCENES','VIEWS','FORMAL_VIEWS','TRUTH','PANEL','SOURCE_CACHE'):globals()[_k]=deepcopy(getattr(old,_k))
read,write=old.read,old.write
def rows():return [dict(arm=a,row_id=a+'-s'+str(s),model_seed=s) for s in SEEDS for a in ('T1','KD')]
def is_reused(row):return row['model_seed']==2026092701
def reference_config(row,segment):return old.config(next(r for r in old.rows() if r['arm']==row['arm']),segment)
def row_root(row):
 r=next(r for r in rows() if r['row_id']==row)
 return old.row_root(row) if is_reused(r) else BASE/row
def config(row,segment=30):
 if row not in rows() or segment not in (20,30):raise ValueError('Only fixed3seeds/2arms/full20+own10')
 c=deepcopy(reference_config(row,segment))
 if is_reused(row):return c
 b=BASE/row['row_id'];parent=b/'stage20';tail=segment==30
 c.update(schema='sameAsKD_confirm86_v1',run_id=RUN,**row,launch_owner=OWNER,output_root=str(b/('source' if tail else 'stage20')),parent_source=str(parent) if tail else None,checkpoint_sources=[str(parent/'final_ssdg.pth')] if tail else [],budget_authorization='Two further modelseeds2702/2703 bothT1/KD fullE30;4newmodels120modelEpoch/26640updates;seed01 reused',confirmation='Candidate fixed from seed01 SOURCE pass +.777778midlow/-.074074clean before any formal metrics read;no recipe/teacher/beta/seed selection')
 return c
def validate(c):
 row=next((r for r in rows() if r['row_id']==c.get('row_id')),None)
 if row is None or c!=config(row,c.get('segment')):raise ValueError('Fixed confirmation seed/arm/budget/source/head differs')
 return c
def make_args(c,device='cuda:0'):
 validate(c);row=next(r for r in rows() if r['row_id']==c['row_id']);a=old.make_args(reference_config(row,c['segment']),device);a.seed=c['model_seed'];a.run_id=c['run_id'];a.candidate_id=c['row_id'];a.output_dir=c['output_root'];a.baseline_ckpt='' if c['segment']==20 else c['parent_source']+'/final_ssdg.pth';a.teacher_ckpt='';return a
