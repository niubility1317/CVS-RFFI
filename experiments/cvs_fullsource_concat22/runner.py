"""Reuse immutable code dependencies, never their held-RX data machinery or weights."""
import importlib.util,sys,types
from pathlib import Path
HERE=Path(__file__).resolve().parents[2]
def bootstrap():
 backend=Path('/home/szu2070436088/2510044040/CV-SincNet/releases/cvs_FcrossRX_additiveAs_RX6_source_E20_20261016_r01')
 if (backend/'release_commit.txt').read_text().strip()!='367b2f945ef65488e6d97eddd2357388d2477100':raise ValueError('Immutable dependency code identity differs')
 spec=importlib.util.spec_from_file_location('fullsource_code_dependencies',backend/'experiments/cvs_f_additive_rx6/runner.py');mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);mod.bootstrap()
 package=types.ModuleType('experiments.cvs_fullsource_concat22');package.__path__=[str(HERE/'experiments/cvs_fullsource_concat22')];sys.modules[package.__name__]=package
 # The new explicit full-source source.py is not the backend's rewritten fold trainer.
if __name__=='__main__':
 bootstrap()
 from experiments.cvs_fullsource_concat22.dispatch import main
 main()
