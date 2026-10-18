"""Local-only explicit dependency checkout; no remote bootstrap or data."""
import argparse,sys,types,runpy
from pathlib import Path
parser=argparse.ArgumentParser();parser.add_argument('--dependencies',required=True,type=Path);parser.add_argument('--flow-only',action='store_true');args=parser.parse_args()
w=Path(__file__).resolve().parents[2];ow=args.dependencies
sys.path[:0]=[str(w),str(w/'code')]
p=types.ModuleType('experiments');p.__path__=[str(ow/'experiments'),str(w/'experiments')];sys.modules['experiments']=p
p=types.ModuleType('experiments.cvs_satellite_budget62');p.__path__=[str(w/'experiments/cvs_satellite_budget62')];sys.modules[p.__name__]=p
runpy.run_module('experiments.cvs_satellite_budget62.'+('flow_checks' if args.flow_only else 'synthetic'),run_name='__main__')
