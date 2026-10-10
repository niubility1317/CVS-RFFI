import argparse
from experiments.cvs_legacy_no_sat.dispatch import worker
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--row',required=True)
    worker(p.parse_args().row)
