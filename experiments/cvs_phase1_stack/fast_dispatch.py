"""Adopt live workers; use the execution adapter only for future source rows."""
import argparse
from pathlib import Path
import capacity16 as controller

RELEASE='cvs_reference_stack_fast_20261005_r01'

def run(worker_root,handoff):
    controller.CONTROL_RELEASE=RELEASE
    controller.OWNER_MARKER=RELEASE+'_owner.json'
    controller.SOURCE_ENTRY=Path(__file__).resolve().parent/'fast_source.py'
    controller.run(worker_root,handoff)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--worker-root',type=Path,required=True);p.add_argument('--handoff',type=Path,required=True)
    a=p.parse_args();run(a.worker_root,a.handoff)
