"""Run pure synthetic checks with an explicit existing dependency checkout."""
import argparse,json,sys,types
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--dependency-root',type=Path,required=True);args=p.parse_args()
root=Path(__file__).resolve().parents[2];dep=args.dependency_root
pkg=types.ModuleType('experiments');pkg.__path__=[str(root/'experiments'),str(dep/'experiments')];sys.modules['experiments']=pkg
sys.path.insert(0,str(dep/'experiments/adv3b02_xuc/code'))
from experiments.cvs_as_prefix81.checks import run
print(json.dumps(run()),flush=True)
