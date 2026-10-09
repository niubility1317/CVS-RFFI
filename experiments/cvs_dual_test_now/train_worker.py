"""Visible pre-CUDA reservation for the immediate evaluation queue."""
import argparse
from experiments.cvs_dual_test_now.driver import worker
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--row',required=True);a=p.parse_args();worker(a.row)
