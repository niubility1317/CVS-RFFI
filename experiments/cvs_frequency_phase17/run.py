"""One immutable local CPU matrix;no Torch/dataset/model/target imports."""
import argparse,gzip,json,os,subprocess,sys,time
from pathlib import Path
import numpy as np
from . import design as d
from .estimator import estimate
from .synthetic import generate,score

def main():
 p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args();out=Path(a.output).resolve()
 if out!=(d.ROOT/d.LOCAL/'run').resolve():raise ValueError('Only own fixed immutable outputroot allowed')
 c=d.validate(d.config());out.mkdir(parents=True,exist_ok=False);start=time.perf_counter();commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=d.ROOT,text=True).strip()
 d.write(out/'resolved_config.json',dict(c,execution_commit=commit,pid=os.getpid(),cwd=str(Path.cwd()),python=sys.version,numpy=np.__version__,started_unix=time.time(),threads={k:os.environ.get(k) for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS')}))
 count=packets=points=0
 with gzip.open(out/'cells.jsonl.gz','xt',encoding='utf8') as f:
  for ti,name in enumerate(d.TEMPLATES):
   for si,snr in enumerate(d.SNRS):
    for case in d.CASES:
     for draw in range(d.DRAWS):
      seed=d.DATA_SEED+ti*100000+si*1000+draw;x,truth=generate(name,snr,case,seed);result,quality=estimate(x);error=quality['CFO_estimate_Hz']-truth.pop('CFO_Hz')
      row=dict(cell=count,template=name,SNR_dB=snr,case=case,draw=draw,data_seed=seed,estimate=result,truth=truth,quality=dict(min_carrier_ratio=float(quality['min_carrier_ratio'].min()),CFO_error_RMSE_Hz=float(np.sqrt(np.mean(error**2))),CFO_error_max_abs_Hz=float(np.max(np.abs(error)))),metrics=score(result,truth,case))
      f.write(json.dumps(row,allow_nan=False)+'\n');count+=1;packets+=len(x);points+=x.size
     f.flush();d.write(out/'progress.json',dict(cells_complete=count,expected_cells=c['cells'],synthetic_packets=packets,synthetic_complex_points=points,elapsed_seconds=time.perf_counter()-start,real_IQ=0,NN=0,updates=0,target_access=False));print(json.dumps(dict(cells=count,template=name,SNR_dB=snr,case=case)),flush=True)
 if (count,packets,points)!=(252,1270080,325140480):raise ValueError('Actual fixed synthetic budget differs')
 d.write(out/'completion.json',dict(status='ARTIFACTS_COMPLETE_NOT_SCIENTIFIC_PASS',cells=count,synthetic_packets=packets,synthetic_complex_points=points,wall_seconds=time.perf_counter()-start,execution_commit=commit,real_IQ=0,checkpoint=0,NN=0,updates=0,teacher=0,head_fit=0,target_access=False,all_cells_retained=True));print('ARTIFACTS_COMPLETE independentrecount pending',flush=True)
if __name__=='__main__':main()
