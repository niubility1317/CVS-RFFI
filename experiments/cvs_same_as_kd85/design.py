"""Frozen same-As CSIL distribution transfer; original scratch T1 control reused."""
from copy import deepcopy
from pathlib import Path
from experiments.cvs_as_temperature84 import design as old
ROOT=Path(__file__).resolve().parents[2];PROJECT=old.PROJECT
RUN='20261019-phase1-FcrossRX-sameAsKD-e30-s2026092701-m2-r01';RELEASE='cvs_FcrossRX_sameAsKD_E30_20261019_r01';OWNER='codex/root/sameAsKD85';BASE=Path(PROJECT)/'runs'/RUN
RX=FIT=old.RX;SEED=old.SEED;STEPS=222;UPDATES=6660
for _k in ('SOURCE','FULL_FP32_POLICY','CLASSES','SCENES','VIEWS','FORMAL_VIEWS','TRUTH','PANEL','SOURCE_CACHE'):globals()[_k]=deepcopy(getattr(old,_k))
read,write=old.read,old.write
def rows():return [dict(arm=a,row_id=a+'-s'+str(SEED),model_seed=SEED) for a in ('T1','KD')]
def reference_config(segment):return old.config(next(r for r in old.rows() if r['arm']=='T1'),segment)
def row_root(row):return old.row_root(row) if row.startswith('T1-') else BASE/row
def config(row,segment=30):
 if row not in rows() or segment not in (20,30):raise ValueError('Fixed scratch20+own10,one new KD row and reused T1')
 c=deepcopy(reference_config(segment))
 if row['arm']=='T1':return c
 b=row_root(row['row_id']);tail=segment==30;parent=b/'stage20'
 c.update(schema='sameAsKD85_v1',run_id=RUN,**row,launch_owner=OWNER,output_root=str(b/('source' if tail else 'stage20')),parent_source=str(parent) if tail else None,checkpoint_sources=[str(parent/'final_ssdg.pth')] if tail else [],method='same_damaged_IQ_frozen_CSIL_output_transfer',training_recipe='OriginalT1 scratch20+own10;AsCEscale30/AsKL0;addbeta1 teacher-correct same-As KL at common scale5;allotherL/Uunchanged',budget_authorization='One new matched fullE30;reuse existing T1;no extra student forward;one frozenCSIL As64forward per step',baseline_reference='Exact fixed T1 from current temperature experiment,not oldAs20',benchmark_context='Chosen from supplied research priority and frozen source same-IQ complement;not target result')
 c['knowledge_transfer']=dict(teacher='CSIL s392005 scratchFINAL200 original head',teacher_checkpoint='PROJECT/paper_reproduction/runs/20260930-phase1-native-baselines-practical-manysig-m5-r01/csil-s392005/last.pt',teacher_scale=5.,teacher_offset=5.,student_native_scale=30.,common_scale=5.,beta=1.,teacher_correct_only=True,student_confidence_gate=False,all_As_true_CE=True,reduction='all64 mean;zero wrong-teacher terms',student_extra_NN=0,teacher_NN_per_step=64,feature_alignment=False,new_parameters=0,new_head=False,V_training=False)
 return c
def validate(c):
 row=next((r for r in rows() if r['row_id']==c.get('row_id')),None)
 if row is None or c!=config(row,c.get('segment')):raise ValueError('Fixed sameAsKD source/budget/head/teacher contract differs')
 return c
def make_args(c,device='cuda:0'):
 validate(c);a=old.make_args(reference_config(c['segment']),device);a.run_id=c['run_id'];a.candidate_id=c['row_id'];a.output_dir=c['output_root'];a.baseline_ckpt='' if c['segment']==20 else c['parent_source']+'/final_ssdg.pth';a.teacher_ckpt='';return a
