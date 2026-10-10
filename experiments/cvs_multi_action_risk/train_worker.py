from pathlib import Path
import sys,argparse
root=Path(__file__).resolve().parents[2];sys.path[:0]=[str(root),str(root/'code')]
from experiments.cvs_multi_action_risk.dispatch import worker
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--kind',choices=['diagnostic','identity'],required=True);p.add_argument('--row',required=True);a=p.parse_args();worker(a.kind,a.row)
