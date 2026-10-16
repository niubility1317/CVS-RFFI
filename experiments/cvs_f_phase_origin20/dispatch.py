import os,subprocess,sys,time
from pathlib import Path
from . import design as d
from experiments.cvs_receiver_residual_capacity4.control import lock
from experiments.cvs_multi_state_action.dispatch import capacity

def dispatch():
    with lock(d.ROOT/'F_phase_origin.owner.lock',nonblocking=True):
        if d.BASE.exists():raise FileExistsError('Preserve immutable source factor output')
        spec=d.read(d.ROOT/'experiments/cvs_f_phase_origin20/registration.json')
        if [r['config'] for r in spec['rows']]!=[d.config(r) for r in d.rows()]:raise ValueError('Registered factor matrix differs')
        d.prerequisite()
        from experiments.cvs_f_practical_source20.provenance import check
        for r in d.rows():check(d.prior.config(r))
        d.BASE.mkdir();(d.BASE/'logs').mkdir();d.write(d.BASE/'dispatcher.json',dict(pid=os.getpid(),cwd=os.getcwd(),argv=sys.argv,owner=d.OWNER))
        child=None
        while child is None:
            with lock(Path(d.PROJECT)/'runs/receiver_residual_capacity4.lock'):
                caps=capacity();choices=[g for g,v in caps.items() if len(v['pids'])<3 and v['free_mb']>=12000]
                if choices:
                    gpu=min(choices,key=lambda g:(len(caps[g]['pids']),-caps[g]['free_mb'],g))
                    cmd=[sys.executable,'-u',str(d.ROOT/'experiments/cvs_f_phase_origin20/runner.py'),'--worker'];log=d.BASE/'logs/worker.log'
                    with log.open('x') as f:child=subprocess.Popen(cmd,cwd=d.ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu)),stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
                    d.write(d.BASE/'launch.json',dict(rows=[r['row_id'] for r in d.rows()],pid=child.pid,gpu=gpu,cwd=str(d.ROOT),argv=cmd,log=str(log)))
            if child is None:time.sleep(3)
        code=child.wait()
        if code or not (d.BASE/'completion.json').is_file():
            d.write(d.BASE/'dispatcher_failure.json',dict(status='FAILED',exit_code=code,no_automatic_retry=True));raise RuntimeError('Own factor failure preserved')
