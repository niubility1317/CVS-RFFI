"""One CPU-only existing-observation reanalysis;zero checkpoint or network."""
import os,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from experiments.cvs_envelope_variance18 import design as d
if __name__=='__main__':
 if sys.argv[1:]==['--checks']:
  from experiments.cvs_envelope_variance18.checks import run
  d.write(d.ROOT/'checks.json',run())
 elif sys.argv[1:]:raise ValueError('Only fixed analysis or checks')
 else:
  if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('CPU-only observation reanalysis')
  import numpy as np
  from experiments.cvs_envelope_variance18.core import cells,fit
  out=d.resolve(d.BASE);out.mkdir(parents=True,exist_ok=False);start=time.perf_counter()
  try:
   registered=d.read(d.ROOT/'registration/experiment.json')
   if registered['run_id']!=d.RUN or registered['rows'][0]['config']!=d.config():raise ValueError('Registered fixed statistic differs')
   old=d.read(d.resolve(d.OBSERVATIONS) / 'result.json');done=d.read(d.resolve(d.OBSERVATIONS) / 'completion.json');contract=d.read(d.resolve(d.SOURCE))
   if done['status']!='ANALYZED_SOURCE_STATISTICS_ONLY' or old['source_IQ_unique']!=6300 or old['source_cells']!=90 or old['source_RX_holdout'] is not None or old['target_access'] or old['NN'] or old['CP'] or old['config']['source_contract']!=d.SOURCE:raise ValueError('Existing completed legal allsource observations required')
   if {k:len(v) for k,v in contract['role_ids'].items()}!={'L_s':6300,'U_s':56700,'V':27000}:raise ValueError('Source role handles differ')
   with np.load(d.resolve(d.OBSERVATIONS)/'source_observations.npz',allow_pickle=False) as p:
    ids=p['ids'];coef=p['coefficients'];y=p['labels'];rx=p['RX'];day=p['day']
    if ids.shape!=(6300,) or len(set(ids.tolist()))!=6300 or set(ids.tolist())!=set(contract['role_ids']['L_s']):raise ValueError('All and only original L physical handles required')
   scales,cell_records,decomposition=cells(coef,y,rx,day);result=fit(scales)
   result.update(status='ANALYZED_SOURCE_DERIVED_ONLY',config=d.config(),cells=cell_records,decomposition=decomposition,previous_mean_basis_decision=old['decision'],source_observation_loads=1,coefficient_samples_used=6300,source_cells=90,source_RX_holdout=None,new_IQ_reads=0,CP=0,NN=0,updates=0,teacher=0,new_IQ_views=0,U_used=False,U_truth_read=False,V_used=False,target_access=False,AGC_causality_proved=False,identity_preservation_proved=False,Ag_efficacy_proved=False,joint_goal_complete=False,automatic_training=False,elapsed_seconds=time.perf_counter()-start,execution_commit=(d.ROOT/'release_commit.txt').read_text().strip())
   d.write(out/'result.json',result);d.write(out/'completion.json',{k:result[k] for k in ('status','decision','coefficient_samples_used','source_cells','new_IQ_reads','CP','NN','updates','target_access','automatic_training')});print('COMPLETE '+result['decision'],flush=True)
  except Exception as e:d.write(out/'failure.json',dict(status='FAILED',error=repr(e),automatic_retry=False));raise
