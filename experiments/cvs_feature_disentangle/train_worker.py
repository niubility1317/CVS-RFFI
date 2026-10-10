import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT),str(ROOT/'code')]
if __name__=='__main__':
 import argparse
 p=argparse.ArgumentParser();p.add_argument('--kind',choices=['train','test'],required=True);p.add_argument('--row',required=True);a=p.parse_args()
 from experiments.cvs_feature_disentangle.dispatch import worker
 worker(a.kind,a.row)
