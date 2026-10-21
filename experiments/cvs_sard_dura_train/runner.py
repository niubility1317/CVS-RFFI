import importlib.util, sys, types
from pathlib import Path

def bootstrap():
    root = Path(__file__).resolve().parents[2]
    project = Path('/home/szu2070436088/2510044040/CV-SincNet')
    prior = project/'releases/cvs_SARD_NoAs_DURA_E50_m6_20261021_r02'
    if (prior/'release_commit.txt').read_text().strip()!='4f85eee2444c31f83981a736e45d6022be34cbe9':
        raise ValueError('Original SARD/DURA runtime differs')
    spec = importlib.util.spec_from_file_location('sard_dura_training_backend', prior/'experiments/cvs_sard_dura_eval/runner.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.bootstrap()
    # Resolve the original full SARD trainer before restoration dependencies
    # can import their inference-only SSDG shim under the same module name.
    from experiments.cvs_phase1_overlay.model import native_modules
    native_modules()
    import experiments
    experiments.__path__.append(str(prior/'experiments'))
    name = 'experiments.cvs_sard_dura_train'
    package = types.ModuleType(name)
    package.__path__ = [str(root/name.replace('.', '/'))]
    sys.modules[name] = package

if __name__=='__main__':
    bootstrap()
    from experiments.cvs_sard_dura_train.execution import main
    main()
