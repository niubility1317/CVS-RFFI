"""Wait on this existing P60 dispatcher, then run zero-NN frozen-result analysis."""
import argparse,json,os,sys,time
from pathlib import Path
RUN='20261018-phase1-FcrossRX-PAwaveform-tail-e30-s2026092701-m2-r01'
def main():
    p=argparse.ArgumentParser()
    for key in ('project','watch-root','out','analysis-script','truth','code-commit'):
        p.add_argument('--'+key,required=True)
    p.add_argument('--dispatcher-pid',type=int,required=True)
    a=p.parse_args()
    project=Path(a.project);base=project/'runs'/RUN
    watch=Path(a.watch_root);out=Path(a.out);script=Path(a.analysis_script)
    if watch.exists() or out.exists():raise FileExistsError('Reconcile existing analysis owner/output; never duplicate')
    if not script.is_file():raise FileNotFoundError(script)
    watch.mkdir()
    def state(status,**extra):
        value=dict(status=status,pid=os.getpid(),dispatcher_pid=a.dispatcher_pid,run_id=RUN,
                   NN=0,IQ_reads=0,checkpoint_reads=0,optimizer_updates=0,
                   prediction_launches=0,scorer_launches=0,training_launches=0,**extra)
        tmp=watch/'state.tmp'
        tmp.write_text(json.dumps(value,indent=2)+chr(10),encoding='utf8')
        os.replace(tmp,watch/'state.json')
    while not (base/'complete.json').is_file():
        alive=Path('/proc/'+str(a.dispatcher_pid)).exists()
        state('WAIT_EXISTING_RUN',dispatcher_alive=alive,truth_opened=False)
        if not alive:raise RuntimeError('Original dispatcher terminal without complete artifact; preserve outputs, no restart')
        time.sleep(30)
    state('RUN_FROZEN_RESULT_ANALYSIS',truth_opened=None)
    os.environ['CUDA_VISIBLE_DEVICES']=''
    for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[key]='2'
    cmd=[sys.executable,'-u',str(script),'--project',str(project),'--out',str(out),
         '--truth',a.truth,'--code-commit',a.code_commit]
    os.execv(sys.executable,cmd)
if __name__=='__main__':main()
