"""Bootstrap the exact SIG_TIME/RDrop dependencies before any native imports."""
import importlib.util
from pathlib import Path
import sys
import types


def bootstrap(roots=None):
    here=Path(__file__).resolve().parents[2];project=Path('/home/szu2070436088/2510044040/CV-SincNet')
    if roots is None:
        dependency=project/'releases/cvs_SIGTIME_evidence_responsibility_train20_20261015_r02'
        legacy=project/'releases/cvs_originalSIGTIME_rdrop20_50_20261015_r01'
        fold=project/'releases/cvs_SIGTIME_t2_locked_tailavg_sourcefold_20261015_r01'
    else:dependency,legacy,fold=(Path(v) for v in roots)
    # The audited running implementation is this native tree. Resolve it before
    # the repository's retired code/SSDG implementation in fresh processes too.
    sys.path[:0]=[str(dependency/'experiments/adv3b02_xuc/code'),str(dependency),str(dependency/'code')]
    # Shared source.py later prepends code/ itself. Select/cache the audited
    # native package now, while its path is first, before that import occurs.
    from SSDG import train_ssdg as native
    if Path(native.__file__).resolve()!=(dependency/'experiments/adv3b02_xuc/code/SSDG/train_ssdg.py').resolve():
        raise ValueError('Actual native trainer is not the audited adv3b02_xuc implementation')
    for suffix,root in (('cvs_sigtime_rdrop_legacy',legacy),('cvs_rdrop_fold20',legacy),
                        ('cvs_sigtime_locked20',fold),('cvs_rx_u_increment20',fold),('cvs_mixed_iq20',here)):
        name='experiments.'+suffix;module=types.ModuleType(name)
        module.__path__=[str(root/'experiments'/suffix)];sys.modules[name]=module
    name='experiments.cvs_feature_sequence50.source'
    spec=importlib.util.spec_from_file_location(name,fold/'experiments/cvs_feature_sequence50/source.py')
    module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module)


if __name__=='__main__':
    bootstrap()
    from experiments.cvs_mixed_iq20.train_dispatch import main
    main()
