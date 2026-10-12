from copy import deepcopy
from pathlib import Path
from experiments.cvs_complement_cnn import design as old
ROOT=Path(__file__).resolve().parents[2]
PROJECT,SOURCE,FULL_FP32_POLICY,read,write,SEED,RX,REFERENCE=(getattr(old,k) for k in ('PROJECT','SOURCE','FULL_FP32_POLICY','read','write','SEED','RX','REFERENCE'))
RUN='20261012-phase1-rx-heads-source-manysig-s2026092701-r01';RELEASE='cvs_rx_heads_20261012_r01';OWNER='codex/root/rx-heads-20261012';BASE=Path(PROJECT)/'runs'/RUN;ARMS=('RANDOM_HEAD','RX_HEAD')
def rows():return [dict(row_id=f'{a}-RX{rx}-s{SEED}',arm=a,held_rx=rx,model_seed=SEED) for rx in RX for a in ARMS]
def config(row):
 if row not in rows():raise ValueError('Unregistered RXheads source row')
 c=deepcopy(old.config(next(r for r in old.rows() if r['held_rx']==row['held_rx'])));c.pop('cnn');c.update(run_id=RUN,**row,output=(BASE/row['row_id']).as_posix(),launch_owner=OWNER,
  heads=dict(count=4,classes=6,feature=160,scale=16.,aux_weight=.25,assignment='true sourcefitRX' if row['arm']=='RX_HEAD' else 'fixedbalancedrandompartition fitL indices withoutusingTX/RX/day,each1260',assignment_seed=SEED+713,initializer_seed=SEED+161803,
   formula='0.5globalnative softmaxT1+0.5uniform mean fourexpert softmaxT1;no inferenceRX/metadata',training='originalglobalLCE +0.25oneassignedexpertLCE,same joint160forward'),
  goal=dict(primary='RX_HEAD+native3',scope='source_only_not84',feedback_to_training=False),
  scientific_rule=dict(mean_gain_vs_native3=.01,mean_gain_vs_RANDOM=.005,nonnegative_RX_each=4,worst_RX_noninferiority_each=.01),
  initialization='native+fourrandomcosheads fromscratch;nocheckpoint initialization,teacher or AUX inherits oldstate',
  budget_authorization='scratch30,10models66600updates;matchedsameheads/readoutbudget,actualgroups/dataset roles fixed')
 c['ensemble'].update(formula='0.5*newnative4headclassifier_probability+0.5*fixednative3probabilitymean;no fittedweights')
 return c
def validate(c):
 row=next((r for r in rows() if r['row_id']==c.get('row_id')),None)
 if row is None or c!=config(row):raise ValueError('Fixed RXhead source matrix differs')
 return c
