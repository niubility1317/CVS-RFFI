"""Visible pre-CUDA reservation for both current and legacy dispatchers."""
import argparse
from experiments.cvs_dual_evidence.dispatch import worker

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--kind',choices=['source','predict'],required=True);p.add_argument('--row',required=True)
    a=p.parse_args();worker(a.kind,a.row)
