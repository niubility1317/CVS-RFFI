"""One immutable local aggregate-only run;no model/no automatic training."""
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[key]='2'
os.environ['CUDA_VISIBLE_DEVICES']=''
import gzip,json,subprocess,time
from . import design as d
from .core import analyze
from .data import load

def run():
 if d.BASE.exists():raise FileExistsError('Immutable local result exists;reuse terminal artifacts,no retry/overwrite')
 d.BASE.mkdir();start=time.perf_counter();d.write(d.BASE/'resolved_config.json',d.config())
 try:
  commit=subprocess.check_output(['C:/Program Files/Git/cmd/git.exe','rev-parse','HEAD'],cwd=d.ROOT,text=True).strip();d.write(d.BASE/'execution.json',dict(commit=commit,source_only=True,environment='Verified native Fpython/NumPy;CPU2',cwd=str(d.ROOT),command='Fpython -X utf8 -m experiments.cvs_psd_affine17.run',CUDA_visible_devices='',new_IQ=0,new_CP=0,new_NN=0,new_updates=0,target_access=False))
  psd,origin=load();s,folds=analyze(psd)
  with gzip.open(d.BASE/'folds.jsonl.gz','xt',encoding='utf8') as stream:
   for row in folds:stream.write(json.dumps(row,allow_nan=False)+'\n')
  s.update(run_id=d.RUN,execution_commit=commit,origin=origin,elapsed_CPU_wall_seconds=time.perf_counter()-start,aggregate_NPZ_reads=1,all_144_cell_spectra_used=True,all_256_bins_used=True,source_prediction_accuracy_available=False,automatic_training=False)
  d.write(d.BASE/'summary.json',s);d.write(d.BASE/'completion.json',dict(status='ANALYZED',verdict=s['verdict'],qualified=s['qualified'],all_fits=len(folds),source_only=True,target_access=False,NN=0,optimizer_updates=0,goal_complete=False))
  print(json.dumps(dict(status=s['status'],verdict=s['verdict'],fit_count=len(folds),passed_unordered_RX_pairs=sum(v['qualified'] for v in s['unordered_pairs']),total_unordered_RX_pairs=6,wall_seconds=s['elapsed_CPU_wall_seconds'],NN=0,target_access=False)))
 except Exception as error:
  d.write(d.BASE/'failure.json',dict(status='FAILED_PRESERVED',reason=type(error).__name__+': '+str(error),new_NN=0,target_access=False));raise
if __name__=='__main__':run()
