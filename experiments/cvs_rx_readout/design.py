from copy import deepcopy
from pathlib import Path
from experiments.cvs_rx_heads import design as parent
ROOT=Path(__file__).resolve().parents[2]
PROJECT,SOURCE,FULL_FP32_POLICY,read,write,SEED,RX,REFERENCE=(getattr(parent,k) for k in ('PROJECT','SOURCE','FULL_FP32_POLICY','read','write','SEED','RX','REFERENCE'))
RUN='20261012-phase1-rx-head-readout-source-manysig-s2026092701-r01';RELEASE='cvs_rx_readout_20261012_r01';OWNER='codex/root/rx-readout-20261012';BASE=Path(PROJECT)/'runs'/RUN
ARMS=('GLOBAL_ONLY','ENTROPY_GATE','CONFIDENT_HEAD')
def rows():return [dict(row_id=f'{a}-RX{rx}-s{SEED}',arm=a,held_rx=rx,model_seed=SEED) for rx in RX for a in ARMS]
def config(row):
 if row not in rows():raise ValueError('Unregistered frozenheadreadout')
 cfg=[parent.config(next(r for r in parent.rows() if r['held_rx']==row['held_rx'] and r['arm']==arm)) for arm in parent.ARMS]
 return dict(run_id=RUN,**row,output=(BASE/row['row_id']).as_posix(),launch_owner=OWNER,parents=cfg,source_loader_config=cfg[0]['source_loader_config'],source_only=True,source_U=False,training_target_access=False,new_training_updates=0,
  readout=row['arm'],formula=dict(GLOBAL_ONLY='originalnativeglobal softmaxT1',ENTROPY_GATE='0.5global+0.5sum softmax(-4*expertentropy)*expertprobability',CONFIDENT_HEAD='0.5global+0.5expertwithhighestmaxprobability;lowestindexbreaktie')[row['arm']],
  mixture='0.5newreadout+0.5fixednative3probabilitymeanT1,allclasses/packetsidentical',scientific_rule=dict(mean_gain_native3=.01,mean_gain_RANDOM=.005,nonnegative_RX_each=4,worst_RX_noninferiority_each=.01),
  selection='all15rowscomplete;onlyqualified readouts,highestRX_HEADmixture source mean,tieGLOBALthenentropythenconfident;no targetfeedback',test_views=[],budget='0training/0fit/calibration;10fixedsource parents30each,historicalcostreported')
def validate(c):
 row=next((r for r in rows() if r['row_id']==c.get('row_id')),None)
 if row is None or c!=config(row):raise ValueError('Fixed zero-trainingreadout differs')
 return c
