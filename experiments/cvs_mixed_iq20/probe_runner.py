"""Use the original RDrop constructor/provenance with its real release imports."""
import importlib.util
from pathlib import Path
import sys
import types


def bootstrap():
    here = Path(__file__).resolve().parents[2]
    project = Path('/home/szu2070436088/2510044040/CV-SincNet')
    dependency = project / 'releases/cvs_SIGTIME_evidence_responsibility_train20_20261015_r02'
    legacy = project / 'releases/cvs_originalSIGTIME_rdrop20_50_20261015_r01'
    fold = project / 'releases/cvs_SIGTIME_t2_locked_tailavg_sourcefold_20261015_r01'
    sys.path[:0] = [str(dependency), str(dependency / 'code'),
                    str(dependency / 'experiments/adv3b02_xuc/code')]
    for suffix, root in (('cvs_sigtime_rdrop_legacy', legacy), ('cvs_rdrop_fold20', legacy),
                         ('cvs_sigtime_locked20', fold), ('cvs_rx_u_increment20', fold),
                         ('cvs_mixed_iq20', here)):
        name = 'experiments.' + suffix
        module = types.ModuleType(name)
        module.__path__ = [str(root / 'experiments' / suffix)]
        sys.modules[name] = module
    name = 'experiments.cvs_feature_sequence50.source'
    spec = importlib.util.spec_from_file_location(name, fold / 'experiments/cvs_feature_sequence50/source.py')
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)


if __name__ == '__main__':
    bootstrap()
    from experiments.cvs_mixed_iq20.probe import run
    run()
