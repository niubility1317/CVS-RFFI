import importlib.util,json,sys,types
from pathlib import Path
def bootstrap():
 here=Path(__file__).resolve().parents[2];rt=json.loads((here/'runtime.json').read_text());ref=Path(rt['reference_release'])
 if (ref/'release_commit.txt').read_text().strip()!=rt['reference_commit']:raise ValueError('Pinned source runtime differs')
 spec=importlib.util.spec_from_file_location('_raw_temporal_runtime',ref/'experiments/cvs_feature_head_joint30/runner.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);m.bootstrap();namespace=types.ModuleType('experiments.cvs_raw_temporal30');namespace.__path__=[str(here/'experiments/cvs_raw_temporal30')];sys.modules[namespace.__name__]=namespace
 return here
if __name__=='__main__':
 bootstrap()
 from experiments.cvs_raw_temporal30.dispatch import main
 main()
