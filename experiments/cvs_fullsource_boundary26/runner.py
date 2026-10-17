"""Reuse immutable full-source E30 dependencies; never inherit their weights."""
import importlib.util,sys,types
from pathlib import Path
HERE=Path(__file__).resolve().parents[2]
def bootstrap():
 source=Path('/home/szu2070436088/2510044040/CV-SincNet/releases/cvs_FcrossRX_fullsource_concat_E30_20261017_r01')
 if (source/'release_commit.txt').read_text().strip()!='bafba40e365d0914a2e86f861fb55a49f1cd3997':raise ValueError('Immutable full-source dependency differs')
 spec=importlib.util.spec_from_file_location('boundary_fullsource_dependencies',source/'experiments/cvs_fullsource_concat30_25/runner.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);m.bootstrap()
 parent=types.ModuleType('experiments.cvs_fullsource_concat30_25');parent.__path__=[str(source/'experiments/cvs_fullsource_concat30_25')];sys.modules[parent.__name__]=parent
 own=types.ModuleType('experiments.cvs_fullsource_boundary26');own.__path__=[str(HERE/'experiments/cvs_fullsource_boundary26')];sys.modules[own.__name__]=own
if __name__=='__main__':
 bootstrap()
 from experiments.cvs_fullsource_boundary26.dispatch import main
 main()
