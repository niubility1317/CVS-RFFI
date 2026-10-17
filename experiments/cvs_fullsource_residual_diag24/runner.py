"""One frozen-source diagnostic job. Reuse immutable P22 code, not its test path."""
import importlib.util,sys,types
from pathlib import Path
HERE=Path(__file__).resolve().parents[2]
def bootstrap():
 source=Path('/home/szu2070436088/2510044040/CV-SincNet/releases/cvs_FcrossRX_fullsource_concat_E20_20261017_r01')
 if (source/'release_commit.txt').read_text().strip()!='dc4e47fe0c2b3e0dcf73dda104cdbd4a7c814f5a':raise ValueError('Immutable deployed-model dependency differs')
 spec=importlib.util.spec_from_file_location('diagnostic_fullsource_dependencies',source/'experiments/cvs_fullsource_concat22/runner.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);m.bootstrap()
 own=types.ModuleType('experiments.cvs_fullsource_residual_diag24');own.__path__=[str(HERE/'experiments/cvs_fullsource_residual_diag24')];sys.modules[own.__name__]=own
if __name__=='__main__':
 bootstrap()
 from experiments.cvs_fullsource_residual_diag24.dispatch import main
 main()
