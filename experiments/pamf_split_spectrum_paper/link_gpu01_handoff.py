"""Record the verified successor alongside the previous controller state."""
import argparse
from pathlib import Path
from experiments.cric_dura_comparison.io import read, write
from experiments.cric_dura_comparison.pipeline import alive


def link(previous, current):
    previous,current=Path(previous),Path(current)
    old=read(previous/'pipeline_state.json');new=read(current/'pipeline_state.json')
    if (new.get('status')!='RUNNING' or not alive(new.get('pid',-1)) or
        alive(old.get('pid',-1)) or new.get('previous_pid')!=old.get('pid') or
        new.get('dispatch_gpus')!=[2,3,4,5,6,7] or
        any(v.get('status')=='RUNNING' and v.get('gpu') in (0,1) for v in new['jobs'].values())):
        raise ValueError('Cannot verify unique successor and reserved GPUs')
    record=dict(status='HANDOFF',previous_pid=old['pid'],new_pid=new['pid'],
                control_root=str(current),dispatch_gpus=[2,3,4,5,6,7])
    path=previous/'handoff.json'
    if path.exists():
        if read(path)!=record:raise ValueError('Different handoff already exists')
        return record
    write(path,record)
    if read(path)!=record:raise RuntimeError('Handoff readback differs')
    return record


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--previous',required=True);p.add_argument('--current',required=True)
    a=p.parse_args();print(link(a.previous,a.current))
