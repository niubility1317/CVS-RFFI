"""Use each frozen model's original immutable runtime; never switch profiles in-process."""
import importlib.util,json,sys,types
from pathlib import Path

def bootstrap(seed):
 here=Path(__file__).resolve().parents[2];runtime=json.loads((here/'runtime.json').read_text());backend=Path(runtime['backends'][str(seed)])
 spec=importlib.util.spec_from_file_location('iqmix_model_bootstrap',backend/'experiments/cvs_crossrx_satmix20/runner.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);m.bootstrap()
 name='experiments.cvs_same_tx_iq16';package=types.ModuleType(name);package.__path__=[str(here/'experiments/cvs_same_tx_iq16')];sys.modules[name]=package
 return here
if __name__=='__main__':
 if len(sys.argv)!=3 or sys.argv[1]!='--seed' or sys.argv[2] not in ('2026092701','2026092702'):raise ValueError('Only fixed two source-model seeds permitted')
 seed=int(sys.argv[2]);bootstrap(seed)
 from experiments.cvs_same_tx_iq16.worker import run
 run(seed)
