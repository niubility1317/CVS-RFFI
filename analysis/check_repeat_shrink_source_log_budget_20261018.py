"""Six-forward parser guard and unchanged four-forward default;no real artifacts."""
import gzip,json,tempfile
from pathlib import Path
from analysis import cvs_view_pcgrad_source_logs_20261018 as logs

def run():
 oldrun,oldmodes=logs.RUN,logs.MODES
 with tempfile.TemporaryDirectory() as tmp:
  try:
   logs.RUN='SYNTHETIC';logs.MODES=('B00',);B=Path(tmp)/'runs/SYNTHETIC';S=B/'B00-s2026092701/source';S.mkdir(parents=True);(B/'logs').mkdir();(B/'launch.json').write_text('{}',encoding='utf8');(S/'actual_native_args.json').write_text('{}',encoding='utf8');(B/'logs/B00-s2026092701.stdout.log').write_text('synthetic only\n',encoding='utf8')
   for name in ('epoch_metrics.jsonl',): (S/name).write_text('',encoding='utf8')
   def write(n):
    with gzip.open(S/'step_metrics.jsonl.gz','wt',encoding='utf8') as f:f.write(json.dumps(dict(step=1,metrics={'concat/As_packets':0,'concat/L_forwards':n,'train/pseudo_truth_available':0}))+'\n')
   write(6);a=logs._base_run(tmp,False,expected_As_packets=0,expected_L_forwards=6);assert a['rows']['B00']['step_log']['records']==1
   try:logs._base_run(tmp,False,expected_As_packets=0)
   except ValueError:pass
   else:raise AssertionError('Sixforward run incorrectly accepted asdefault4')
   write(4);logs._base_run(tmp,False,expected_As_packets=0)
   try:logs._base_run(tmp,False,expected_As_packets=0,expected_L_forwards=6)
   except ValueError:pass
   else:raise AssertionError('Fourforward run incorrectly accepted asexpected6')
  finally:logs.RUN,logs.MODES=oldrun,oldmodes
 return dict(status='PASS',expected6_accepts6=True,default4_accepts4=True,mismatched_forward_counts_rejected=True,real_logs=0,IQ=0,CP=0,NN=0,updates=0,target_access=False)
if __name__=='__main__':print(json.dumps(run()))
