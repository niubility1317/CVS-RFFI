"""Load immutable full-source code dependencies into one new isolated process."""
import importlib.util,sys,types
from pathlib import Path
HERE=Path(__file__).resolve().parents[2]
def bootstrap():
 backend=Path('/home/szu2070436088/2510044040/CV-SincNet/releases/cvs_FcrossRX_fullsource_concat_E20_20261017_r01')
 if (backend/'release_commit.txt').read_text().strip()!='dc4e47fe0c2b3e0dcf73dda104cdbd4a7c814f5a':raise ValueError('Full-source dependency code differs')
 spec=importlib.util.spec_from_file_location('fullsource22_dependency',backend/'experiments/cvs_fullsource_concat22/runner.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);m.bootstrap()
 name='experiments.cvs_fullsource_rawpractical23';p=types.ModuleType(name);p.__path__=[str(HERE/'experiments/cvs_fullsource_rawpractical23')];sys.modules[name]=p
if __name__=='__main__':
 bootstrap()
 from experiments.cvs_fullsource_rawpractical23.dispatch import main
 main()
