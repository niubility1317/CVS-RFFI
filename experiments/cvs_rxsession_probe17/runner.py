"""Use immutable verified F/As defaulthead backend;never train."""
import importlib.util,sys,types
from pathlib import Path
HERE=Path(__file__).resolve().parents[2]
def bootstrap():
 backend=Path('/home/szu2070436088/2510044040/CV-SincNet/releases/cvs_FcrossRX_AsTiming_sourceRX1_E20_20261017_r01')
 if (backend/'release_commit.txt').read_text().strip()!='f788309f5f9e27689ec1e98ae99a6bd83237e1f9':raise ValueError('Wrong immutable actual ALL backend')
 spec=importlib.util.spec_from_file_location('frozen_session_backend',backend/'experiments/cvs_as_timing20/runner.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);m.bootstrap()
 package=types.ModuleType('experiments.cvs_rxsession_probe17');package.__path__=[str(HERE/'experiments/cvs_rxsession_probe17')];sys.modules['experiments.cvs_rxsession_probe17']=package
if __name__=='__main__':
 if sys.argv[1:]:raise ValueError('Fixed single frozen source probe,no overrides')
 bootstrap()
 from experiments.cvs_rxsession_probe17.run import run
 run()
