"""One sequential source-only owner; no training, target prediction or scoring."""
import os,subprocess,sys,time,json
from . import design as d
from experiments.cvs_receiver_residual_capacity4.control import lock
def run():
 with lock(d.ROOT/'owner.lock',nonblocking=True):
  spec=d.read(d.ROOT/'registration/experiment.json')
  if spec['run_id']!=d.RUN or [r['config'] for r in spec['rows']]!=[d.config(r) for r in d.rows()]:raise ValueError('Source proxy preregistration differs')
  d.BASE.mkdir(parents=True,exist_ok=False);(d.BASE/'logs').mkdir();launch=[]
  d.write(d.BASE/'dispatcher.json',dict(pid=os.getpid(),cwd=os.getcwd(),argv=sys.argv,owner=d.OWNER));d.write(d.BASE/'experiment.json',spec)
  for row in d.rows():
   cmd=[sys.executable,'-u','-m','experiments.cvs_source_label_gate.audit','--row',row['row_id']];log=d.BASE/'logs'/f"{row['row_id']}.log"
   with log.open('x') as f:
    child=subprocess.Popen(cmd,cwd=d.ROOT,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True);launch.append(dict(row_id=row['row_id'],pid=child.pid,cwd=str(d.ROOT),argv=cmd,log=str(log)));d.write(d.BASE/'launch.json',dict(rows=launch));code=child.wait()
   summary=d.BASE/row['row_id']/'summary.json'
   if code or not summary.is_file() or d.read(summary)['status']!='SOURCE_GATE_DIAGNOSTIC_COMPLETE':
    d.write(d.BASE/'failure.json',dict(status='FAILED',row_id=row['row_id'],exit_code=code,no_automatic_retry=True));raise RuntimeError('Source proxy failed; preserve all outputs')
  d.write(d.BASE/'completion.json',dict(status='SOURCE_DIAGNOSTICS_COMPLETE',rows=3,source_L_packets=18900,forward_packets=56700,backbone_updates=0,head_fit_updates=0,EMA_updates=0,target_access=False,U_TX_truth_read=False,complete_at=time.time()))
if __name__=='__main__':run()
