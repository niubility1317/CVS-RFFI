"""Use verified local dependencies; isolate each family row's module bindings."""
import runpy,sys,types
from pathlib import Path
root=Path(__file__).resolve().parents[2]
deps=Path('E:/type10-7/code/worktrees/cvs_iq_timing_20261016')
urban=Path('E:/type10-7/code/worktrees/phase1_urban_sigtime_20261014')
sys.path[:0]=[str(root),str(root/'code'),str(deps/'code'),str(urban/'code')]
p=types.ModuleType('experiments');p.__path__=[str(root/'experiments'),str(urban/'experiments'),str(deps/'experiments')];sys.modules['experiments']=p
for base in (root,urban,deps):
 for folder in (base/'experiments').iterdir():
  if folder.is_dir():
   name='experiments.'+folder.name
   if name not in sys.modules:
    q=types.ModuleType(name);q.__path__=[];sys.modules[name]=q
   sys.modules[name].__path__.append(str(folder))
runpy.run_module('experiments.cvs_selected_seed_completion.checks',run_name='__main__')
