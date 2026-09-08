"""Bounded user-requested cancellation of B2-V1/B2, preserving all artifacts."""
import argparse
import datetime
import json
import os
from pathlib import Path
import signal
import time


def current(pid):
    p=Path('/proc')/str(pid)
    try:
        stat=(p/'stat').read_text().rsplit(')',1)[1].split()
        if stat[0]=='Z': return None
        return {'pid':pid,'ppid':int(stat[1]),'start_ticks':int(stat[19]),
                'cwd':str((p/'cwd').resolve()),
                'argv':[v.decode() for v in (p/'cmdline').read_bytes().split(bytes([0])) if v]}
    except FileNotFoundError:
        return None


def matches(before,now):
    return now is not None and all(before[k]==now[k] for k in ('pid','start_ticks','cwd','argv'))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--inventory',type=Path,required=True)
    p.add_argument('--apply',action='store_true')
    args=p.parse_args()
    data=json.loads(args.inventory.read_text())
    assert data['queue']['pending']==[]
    expected={3515808:'B2-V1',3515809:'B2'}
    parents={x['pid']:x for x in data['processes'] if x['pid'] in expected}
    assert set(parents)==set(expected)
    targets=[x for x in data['processes'] if x['pid'] in expected or x['ppid'] in expected]
    for pid,row in expected.items():
        old=parents[pid]
        assert old['ppid']==3515639
        assert old['argv'][old['argv'].index('--run_name')+1]==row
        assert any(v.endswith('/phase1_ecrs_v1r_firstbatch_s392005_e200_20260908_r1/'+row) for v in old['argv'])
    for old in targets:
        now=current(old['pid'])
        if now is not None and not matches(old,now): raise RuntimeError('process identity changed')
    print(json.dumps({'time':datetime.datetime.now().isoformat(),'apply':args.apply,'targets':[x['pid'] for x in targets]}),flush=True)
    if not args.apply: return
    sent=[]
    # Terminate only the two verified training parents, then clean up their
    # already-verified data-loader children if those did not exit with them.
    for old in parents.values():
        if matches(old,current(old['pid'])):
            os.kill(old['pid'],signal.SIGTERM);sent.append(old['pid'])
    time.sleep(5)
    for old in targets:
        if old['pid'] not in parents and matches(old,current(old['pid'])):
            os.kill(old['pid'],signal.SIGTERM);sent.append(old['pid'])
    time.sleep(3)
    remaining=[x['pid'] for x in targets if matches(x,current(x['pid']))]
    print(json.dumps({'time':datetime.datetime.now().isoformat(),'signal':'SIGTERM','sent':sent,'remaining':remaining,'status':'STOPPED' if not remaining else 'UNKNOWN'}),flush=True)


if __name__=='__main__':
    main()
