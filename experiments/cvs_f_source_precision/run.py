"""Native CPU-only immutable run; mathematical checks first, then cached statistics."""
import os
for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[name]='2'
import argparse,json,time
from pathlib import Path
from . import design as d

def write(path,value):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
 with path.open('x',encoding='utf8') as f:json.dump(value,f,indent=2,allow_nan=False);f.write('\n')

def run(a):
 root=Path(__file__).resolve().parents[2];base=Path(a.base);source=Path(a.source_cache)
 if os.name!='nt' or Path.cwd().resolve()!=root or base.name!=d.RUN or source.name!=d.SOURCE_RUN or (root/'release_commit.txt').read_text().strip()!=a.commit:raise ValueError('Fixed native CPU release and fresh output required')
 base.mkdir(parents=True,exist_ok=False);write(base/'execution.json',dict(commit=a.commit,pid=os.getpid(),cwd=str(root),argv=list(__import__('sys').argv),source_cache=str(source),CPU_threads=2,GPU_workers=0))
 from .checks import run as check
 from .analysis import analyze
 checks=check();write(base/'checks.json',checks);print('CHECKS_PASS',flush=True)
 result=analyze(source,lambda row:write(base/('BASE-s'+str(row['model_seed']))/'summary.json',row));result['analysis_code_commit']=a.commit;write(base/'result.json',result);write(base/'completion.json',dict(status='ARTIFACTS_COMPLETE',run_id=d.RUN,analysis_code_commit=a.commit,models=3,source_feature_rows=81000,IQ_reads=0,model_forwards=0,head_fit_updates=0,optimizer_updates=0,target_reads=0,classifier_decisions=0));print('SOURCE_PRECISION_COMPLETE',flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--base',required=True);p.add_argument('--source-cache',required=True);p.add_argument('--commit',required=True);run(p.parse_args())
