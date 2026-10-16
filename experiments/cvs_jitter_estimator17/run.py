"""One immutable local diagnostic;does not import Torch or any dataset loader."""
import argparse,gzip,json,os,subprocess,time
from pathlib import Path
import numpy as np
from . import design as d
from .estimator import estimate
from .synthetic import generate,score

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--output',required=True);a=parser.parse_args();out=Path(a.output).resolve()
 if out!=(d.ROOT/d.LOCAL/'run').resolve():raise ValueError('Only registered own immutable output root allowed')
 c=d.validate(d.config());out.mkdir(parents=True,exist_ok=False)
 start=time.perf_counter();commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=d.ROOT,text=True).strip()
 d.write(out/'resolved_config.json',dict(c,execution_commit=commit,pid=os.getpid(),cwd=str(Path.cwd()),python_version=__import__('sys').version,numpy_version=np.__version__,started_unix=time.time(),threads={k:os.environ.get(k) for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS')}))
 count=0
 with gzip.open(out/'cells.jsonl.gz','xt',encoding='utf8') as f:
  for ti,name in enumerate(d.TEMPLATES):
   for si,SNR in enumerate(d.SNRS):
    for case in d.CASES:
     for draw in range(d.DRAWS):
      seed=d.DATA_SEED+ti*10000+si*100+draw
      observed,truth=generate(name,SNR,case,seed)
      result=estimate(observed)
      row=dict(cell=count,template=name,SNR_dB=SNR,case=case[0],draw=draw,data_seed=seed,estimate=result,truth=truth,metrics=score(result,truth))
      f.write(json.dumps(row,allow_nan=False)+'\n');count+=1
     f.flush();d.write(out/'progress.json',dict(cells_complete=count,expected_cells=c['cells'],elapsed_seconds=time.perf_counter()-start,real_IQ_reads=0,NN=0,updates=0,target_access=False));print(json.dumps(dict(cells=count,template=name,SNR_dB=SNR,case=case[0])),flush=True)
 if count!=c['cells']:raise ValueError('Incomplete frozen matrix')
 d.write(out/'completion.json',dict(status='ARTIFACTS_COMPLETE_NOT_SCIENTIFIC_PASS',cells=count,synthetic_packets=count*d.PACKETS,synthetic_complex_points=count*d.PACKETS*d.CYCLES*d.PERIOD,wall_seconds=time.perf_counter()-start,execution_commit=commit,real_IQ_reads=0,checkpoint_loads=0,NN=0,updates=0,teacher_forwards=0,head_fit_updates=0,target_access=False,all_cells_retained=True))
 print('ARTIFACTS_COMPLETE synthetic-only;independent scoring pending',flush=True)
if __name__=='__main__':main()
