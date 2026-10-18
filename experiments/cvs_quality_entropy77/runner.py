"""Pin the verified P76 runtime; own package and outputs are new."""
import importlib.util
from pathlib import Path
import sys,types
ROOT=Path(__file__).resolve().parents[2]
REFERENCE=ROOT.parent/'cvs_FcrossRX_propContrast_E30_20261019_r01'
REFERENCE_COMMIT='32355ff0e6d73a7305e2fb8ce332651c9d1a7cbe'
def bootstrap():
 if (REFERENCE/'release_commit.txt').read_text().strip()!=REFERENCE_COMMIT:raise ValueError('Pinned P76 runtime differs')
 spec=importlib.util.spec_from_file_location('QUALITYENTROPY77_runtime',REFERENCE/'experiments/cvs_prop_contrast76/runner.py');runtime=importlib.util.module_from_spec(spec);spec.loader.exec_module(runtime);runtime.bootstrap()
 package=types.ModuleType('experiments.cvs_quality_entropy77');package.__path__=[str(ROOT/'experiments/cvs_quality_entropy77')];sys.modules[package.__name__]=package
if __name__=='__main__':
 bootstrap()
 from experiments.cvs_quality_entropy77.dispatch import main
 main()
