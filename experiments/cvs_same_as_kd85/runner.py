"""Reuse released original runtime; register only this committed candidate package."""
import importlib.util,json,sys,types
from pathlib import Path
def bootstrap():
 here=Path(__file__).resolve().parents[2];rt=json.loads((here/'runtime.json').read_text())
 ref=Path(rt['reference_release']);s=importlib.util.spec_from_file_location('kd85_backend',ref/'experiments/cvs_as_temperature84/runner.py');m=importlib.util.module_from_spec(s);s.loader.exec_module(m);m.bootstrap()
 name='experiments.cvs_same_as_kd85';p=types.ModuleType(name);p.__path__=[str(here/'experiments/cvs_same_as_kd85')];sys.modules[name]=p
 return here
if __name__=='__main__':
 bootstrap()
 from experiments.cvs_same_as_kd85.dispatch import main
 main()
