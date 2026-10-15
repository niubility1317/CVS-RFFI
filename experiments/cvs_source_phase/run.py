"""Read only existing legal L IQ from the validated raw-source cache."""
import os
from pathlib import Path
import sys
import time
import numpy as np
from . import design as d
from .core import observe, phase_audit


def verify_contract(actual, expected):
    if any(actual.get(k) != v for k, v in expected.items()) or actual.get('native_role_comparison') != 'EXACT_MATCH':
        raise ValueError('Existing source physical-role contract differs')


def run():
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise ValueError('Registered source diagnostic is CPU-only')
    import torch
    torch.set_num_threads(2)
    c = d.config(); d.validate(c)
    out = Path(c['output_root']); out.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter()
    d.write(out / 'active.json', dict(pid=os.getpid(), cwd=os.getcwd(), argv=sys.argv, owner=d.OWNER))
    try:
        prepared = d.read(Path(d.PREPARATION) / 'preparation_complete.json')
        if prepared.get('status') != 'SOURCE_GEOMETRY_AUDIT_COMPLETE' or prepared.get('target_access') is not False or prepared.get('identity_updates') != 0:
            raise ValueError('Existing source preparation provenance differs')
        packed = torch.load(Path(d.PREPARATION) / 'source_cache.pt', map_location='cpu', weights_only=False)
        contract = d.read(d.SOURCE)
        verify_contract(packed['contract'], contract); verify_contract(prepared['source_contract'], contract)
        L = {k: packed['labeled'][k] for k in ('x', 'y', 'rx', 'day', 'ids')}
        del packed
        ids = set(L['ids'])
        if len(ids) != 6300 or len(L['ids']) != 6300 or ids != set(contract['role_ids']['L_s']) or ids & (set(contract['role_ids']['V']) | set(contract['role_ids']['U_s'])):
            raise ValueError('Only full legal L6300 physical IDs allowed')
        x, tx, rx, day = (np.asarray(L[k].tolist()) for k in ('x', 'y', 'rx', 'day'))
        del L
        d.write(out / 'resolved_config.json', dict(config=c, pid=os.getpid(), cwd=os.getcwd(), python=sys.executable,
            numpy_version=np.__version__, torch_version=torch.__version__,
            commit=(d.ROOT / 'release_commit.txt').read_text().strip(),
            source_cache=d.PREPARATION + '/source_cache.pt',
            old_cache_other_roles_and_features_deserialized_but_never_used=True,
            only_L_IQ_and_allowed_metadata_used=True, source_roles_exact_match=True,
            checkpoint_access=False, network_constructed=False, target_constructed=False,
            module_files=dict(design=d.__file__, core=sys.modules[observe.__module__].__file__)))
        reports = {}
        for window in d.WINDOWS:
            name = f'w{window[0]}_{window[1]}'
            result = phase_audit(observe(x, window), tx, rx, day)
            reports[name] = result
            print(name, result['status'], 'gain', result['mean_risk_gain'], 'positive_folds', result['positive_folds'], flush=True)
        d.write(out / 'phase_crossfit.json', dict(config=c, windows=reports, source_only=True,
            target_access=False, V_fit=False, U_used=False, phase_unwrapping=False))
        completion = dict(status='SOURCE_DIAGNOSTIC_COMPLETE', source_decision=reports[c['primary']]['status'],
            L_packets=6300, window_observations=18900, cycle_observations=75600,
            network_forwards=0, identity_updates=0, checkpoint_loads=0, target_access=False,
            automatic_training=False, seconds=time.perf_counter() - start)
        d.write(out / 'completion.json', completion)
        print('COMPLETE', completion, flush=True)
    except Exception as error:
        d.write(out / 'failure.json', dict(status='FAILED', error=repr(error), no_automatic_retry=True))
        raise


if __name__ == '__main__':
    run()
