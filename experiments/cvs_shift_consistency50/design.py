from pathlib import Path
from experiments.cvs_scratch30_baseline import design as parent
ROOT=Path(__file__).resolve().parents[2]
PROJECT,SOURCE,CAPSULE,TRUTH,CLASSES,SEEDS,FULL_FP32_POLICY,read,write=(getattr(parent,k) for k in ('PROJECT','SOURCE','CAPSULE','TRUTH','CLASSES','SEEDS','FULL_FP32_POLICY','read','write'))
RUN='20261013-phase1-shift-consistency-e50-clean-manysig-m12-r01'
RELEASE='cvs_shift_consistency50_20261013_r01'
OWNER='codex/root/shift-consistency50-20261013'
BASE=Path(PROJECT)/'runs'/RUN
ARMS=('CE50','SHIFT_CE50','SHIFT_STATS50','SHIFT_STATS_SSL50')
VIEWS=('clean',);READOUTS=('NATIVE_HEAD',);EPOCHS=50;STEPS=222;UPDATES=11100;WARMUP=1110
U_CACHE=parent.U_CACHE
LEVELS=('time_down','t2.pool')

def rows():return [dict(row_id=f'{a}-s{s}',arm=a,model_seed=s) for s in SEEDS for a in ARMS]

def config(row):
 if row not in rows():raise ValueError('Unregistered E50 source row')
 base=parent.config(next(r for r in parent.rows() if r['arm']=='CE30' and r['model_seed']==row['model_seed']))
 semi=row['arm']=='SHIFT_STATS_SSL50'
 return dict(base,run_id=RUN,**row,output=(BASE/row['row_id']/'source').as_posix(),launch_owner=OWNER,epochs=EPOCHS,updates=UPDATES,source_U=semi,U_cache=U_CACHE.as_posix() if semi else None,source_validation='Vrolecontract only; noforward/fit/selection',test_views=list(VIEWS),readouts=list(READOUTS),
             checkpoint_selection='fixed_final_epoch50',
             source_shift=dict(enabled=row['arm']!='CE50',deltas=[-2,-1,1,2],noncircular=True,start_after_update=WARMUP,pairCE_weights=[.5,.5],physical_sample_count_rule='two computed source views stillone physical packet'),
             shift_statistics=dict(enabled=row['arm'] in ('SHIFT_STATS50','SHIFT_STATS_SSL50'),levels=list(LEVELS),features='normalized common-valid-region perchannelmean/std; no pointwisefractionalstride equality',weight=.05,warmup_updates=WARMUP,ramp_updates=2220),
             semi=dict(enabled=semi,weight=.25,warmup_updates=WARMUP,ramp_updates=2220,confidence=.95,min_streak=2,weak_views='clean andsamephysical noncircular shift; agreement/minconfidence',strong_noise_std=.015,target='own currentstudent detached clean softprobability',EMA=False,teacher_pretrained=False),
             goal=dict(primary='SHIFT_STATS_SSL50',clean_three_seed_mean=.84,feedback_to_training=False),budget_authorization='user: Agentchooseepochs normally<=50/model; fixedscratch50 allcontrols; referencesource_loader E20duration notconsumed fortraining',cost_claim='same11100updates/Lphysicalexposure; extraL/U/teacherforwards separatelyreported, notsamecompute')

def validate(c):
 row=next((r for r in rows() if r['row_id']==c.get('row_id')),None)
 if row is None or c!=config(row) or c['epochs']>50:raise ValueError('Fixed E50 source config differs')
 return c

def ramp(step):return min(1.,max(0.,(step-WARMUP)/2220))
def delta(step):return (-2,-1,1,2)[(step-1)%4]
