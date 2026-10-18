"""Bootstrap the published P43 runtime and the four unchanged P24 helpers."""
import argparse,importlib.util,sys,types
from pathlib import Path

HERE=Path(__file__).resolve().parents[2]
P43_RELEASE='cvs_FcrossRX_impulse_factorial_E30_20261018_r01'
P43_COMMIT='19aedc3604fed9c193237dd92366164178af0983'

def namespace(name,path):
 module=types.ModuleType(name);module.__path__=[str(path)];sys.modules[name]=module

def bootstrap(project):
 reference=Path(project)/'releases'/P43_RELEASE
 if (reference/'release_commit.txt').read_text().strip()!=P43_COMMIT:raise ValueError('Only the fixed published P43 runtime')
 spec=importlib.util.spec_from_file_location('_current_controls79_P43',reference/'experiments/cvs_noise_shape_factorial43/runner.py');original=importlib.util.module_from_spec(spec);spec.loader.exec_module(original);original.bootstrap()
 namespace('experiments.cvs_noise_shape_factorial43',reference/'experiments/cvs_noise_shape_factorial43')
 namespace('experiments.cvs_fullsource_residual_diag24',HERE/'experiments/cvs_fullsource_residual_diag24')
 namespace('experiments.cvs_controlled_revisit79',HERE/'experiments/cvs_controlled_revisit79')

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--project',type=Path,default=HERE.parent.parent);parser.add_argument('--device',choices=('cpu','cuda:0'),default='cuda:0');args=parser.parse_args();bootstrap(args.project)
 from experiments.cvs_controlled_revisit79.probe import run
 run(args.project,args.device)

if __name__=='__main__':main()
