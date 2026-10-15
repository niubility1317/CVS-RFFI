"""Select audited native modules before loading the source adapters."""
import importlib.util
from pathlib import Path
import sys
import types


def bootstrap(roots=None):
    here=Path(__file__).resolve().parents[2];project=Path('/home/szu2070436088/2510044040/CV-SincNet')
    if roots is None:
        roots=[project/'releases/cvs_SIGTIME_evidence_responsibility_train20_20261015_r02',
               project/'releases/cvs_originalSIGTIME_rdrop20_50_20261015_r01',
               project/'releases/cvs_SIGTIME_t2_locked_tailavg_sourcefold_20261015_r01']
    dependency,legacy,fold=map(Path,roots)
    sys.path[:0]=[str(dependency/'experiments/adv3b02_xuc/code'),str(dependency),str(dependency/'code')]
    from SSDG import train_ssdg as native
    if Path(native.__file__).resolve()!=(dependency/'experiments/adv3b02_xuc/code/SSDG/train_ssdg.py').resolve():
        raise ValueError('Actual native trainer differs from audited adv3b02_xuc code')
    for suffix,root in (('cvs_sigtime_rdrop_legacy',legacy),('cvs_ordinary_fold20',here)):
        name='experiments.'+suffix;module=types.ModuleType(name)
        module.__path__=[str(root/'experiments'/suffix)];sys.modules[name]=module
    name='experiments.cvs_feature_sequence50.source'
    spec=importlib.util.spec_from_file_location(name,fold/'experiments/cvs_feature_sequence50/source.py')
    module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module)


if __name__=='__main__':
    bootstrap()
    from experiments.cvs_ordinary_fold20.dispatch import main
    main()
