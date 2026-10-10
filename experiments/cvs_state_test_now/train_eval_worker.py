import argparse
from experiments.cvs_state_test_now.dispatch import worker
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--row',required=True)
    worker(parser.parse_args().row)
