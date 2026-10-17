"""Pinned original runtime; new view coverage only, no deployed change."""
import importlib.util,sys,types
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
def bootstrap():
 back=Path('/home/szu2070436088/2510044040/CV-SincNet/releases/cvs_FcrossRX_cleanurban_U_E30_20261018_r01')
 if (back/'release_commit.txt').read_text().strip()!='91bbfcdc026b48abfeaa576e14bc4b7f44d3d64a':raise ValueError('Pinned original runtime differs')
 s=importlib.util.spec_from_file_location('crossview41_backend',back/'experiments/cvs_Uurban39/runner.py');m=importlib.util.module_from_spec(s);s.loader.exec_module(m);m.bootstrap()
 p=types.ModuleType('experiments.cvs_crossview_supcon41');p.__path__=[str(ROOT/'experiments/cvs_crossview_supcon41')];sys.modules[p.__name__]=p
if __name__=='__main__':
 bootstrap()
 from experiments.cvs_crossview_supcon41.dispatch import main
 main()
