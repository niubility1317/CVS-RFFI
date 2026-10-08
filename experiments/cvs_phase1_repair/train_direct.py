"""Visible pre-CUDA reservation, then exec immutable worker with the same PID."""
import argparse,json,os,sys,time
from pathlib import Path
from direct_controller import visible_reservations


def capacity_ready(caps,reservations,gpu,pid):
    c=caps[gpu];pids=set(c['pids'])|set(reservations.get(gpu,()))|{pid}
    return len(pids)<=2 and c['free_mb']>=12000,pids


def main():
    p=argparse.ArgumentParser();p.add_argument('--worker-root',type=Path,required=True)
    p.add_argument('--lease',type=Path,required=True);p.add_argument('--entry',type=Path,required=True)
    p.add_argument('args',nargs=argparse.REMAINDER);a=p.parse_args()
    allowed={a.worker_root/'experiments/cvs_phase1_repair'/n for n in ('train_repair.py','predict_repair.py')}
    if a.entry not in allowed:raise ValueError('Unknown immutable worker entry')
    gpu=int(os.environ['CUDA_VISIBLE_DEVICES']);pid=os.getpid()
    sys.path.insert(0,str(a.worker_root/'experiments/adv3b02_xuc/code'))
    from scripts.dispatch_xuc_full import occupancy
    a.lease.parent.mkdir(parents=True,exist_ok=True);stable=0
    while stable<3:
        ready,pids=capacity_ready(occupancy({}),visible_reservations(),gpu,pid)
        stable=stable+1 if ready else 0
        value=dict(pid=pid,gpu=gpu,observed_pids=sorted(pids),stable_checks=stable,
            status='CAPACITY_READY' if stable==3 else 'WAITING_CAPACITY',updated_at=time.time(),cuda_initialized=False)
        tmp=a.lease.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(a.lease)
        if stable<3:time.sleep(1 if ready else 5)
    args=a.args[1:] if a.args and a.args[0]=='--' else a.args
    print('DIRECT_CAPACITY_READY '+json.dumps(value),flush=True)
    os.execv(sys.executable,[sys.executable,'-u',str(a.entry),*args])


if __name__=='__main__':main()
