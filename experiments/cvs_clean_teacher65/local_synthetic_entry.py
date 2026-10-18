"""Explicit local dependency checkout; zero data/query/checkpoint access."""
import argparse,sys,types,runpy
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--dependencies',required=True,type=Path);p.add_argument('--synthetic',action='store_true');p.add_argument('--flow-only',action='store_true');a=p.parse_args();w=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(w),str(w/'code')];m=types.ModuleType('experiments');m.__path__=[str(a.dependencies/'experiments'),str(w/'experiments')];sys.modules['experiments']=m;m=types.ModuleType('experiments.cvs_clean_teacher65');m.__path__=[str(w/'experiments/cvs_clean_teacher65')];sys.modules[m.__name__]=m
runpy.run_module('experiments.cvs_clean_teacher65.'+('synthetic' if a.synthetic else 'flow_checks' if a.flow_only else 'checks'),run_name='__main__')
