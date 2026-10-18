"""Pin the verified P75 runtime; own package and outputs are new."""
import importlib.util
from pathlib import Path
import sys,types
ROOT=Path(__file__).resolve().parents[2]
REFERENCE=ROOT.parent/'cvs_FcrossRX_noise_resample6_E30_20261019_r01'
REFERENCE_COMMIT='ecdefc9bb562535bcc8087b5d59e38af25ab817a'
def bootstrap():
 if (REFERENCE/'release_commit.txt').read_text().strip()!=REFERENCE_COMMIT:raise ValueError('Pinned P75 runtime differs')
 spec=importlib.util.spec_from_file_location('PROPCONTRAST76_runtime',REFERENCE/'experiments/cvs_noise_resample75/runner.py');runtime=importlib.util.module_from_spec(spec);spec.loader.exec_module(runtime);runtime.bootstrap()
 package=types.ModuleType('experiments.cvs_prop_contrast76');package.__path__=[str(ROOT/'experiments/cvs_prop_contrast76')];sys.modules[package.__name__]=package
if __name__=='__main__':
 bootstrap()
 from experiments.cvs_prop_contrast76.dispatch import main
 main()
