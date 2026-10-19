"""Reuse immutable original F runtime; no training or target entry point."""
import importlib.util,sys,types
from pathlib import Path
HERE=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('requested_source_audit',HERE/'experiments/cvs_f_txrisk/design.py');d=importlib.util.module_from_spec(spec);spec.loader.exec_module(d)
def bootstrap():
 if (Path(d.ORIGINAL)/'release_commit.txt').read_text().strip()!=d.ORIGINAL_COMMIT:raise ValueError('Wrong immutable original F execution')
 p=Path(d.ORIGINAL)/'experiments/cvs_f_supcon50/runner.py';spec=importlib.util.spec_from_file_location('fixed_F_bootstrap',p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);m.bootstrap()
 name='experiments.cvs_f_txrisk';package=types.ModuleType(name);package.__path__=[str(HERE/'experiments/cvs_f_txrisk')];sys.modules[name]=package
 return HERE
if __name__=='__main__':
 if len(sys.argv)!=1:raise ValueError('Source-only audit has no arbitrary path, target, training or fitted-head mode')
 bootstrap()
 from experiments.cvs_f_txrisk.audit import run
 run()
