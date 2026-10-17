"""One immutable CPU source row, no model/checkpoint/classifier or target path."""
import os
import sys
import time
from pathlib import Path
import numpy as np
from . import design as d
from .core import cube, evaluate
from .permissions import select
from experiments.cvs_iq_circularity16.core import observe


def run():
    if len(sys.argv) != 1 or os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise ValueError('Fixed CPU-only entry; no overrides')
    config = d.config()
    d.validate(config)
    out = Path(d.BASE)
    out.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    d.write(out/'active.json', dict(pid=os.getpid(), cwd=os.getcwd(), argv=sys.argv, owner=d.OWNER))
    try:
        from .checks import run as checks
        d.write(out/'checks.json', checks())
        print('SMOKE PASS source roles and scalar controls;0NN;G0 has no model/CP', flush=True)
        import torch
        torch.set_num_threads(2)
        contract = d.read(d.SOURCE)
        ready = d.read(Path(d.PREPARATION)/'preparation_complete.json')
        packed = torch.load(Path(d.PREPARATION)/'source_cache.pt', map_location='cpu', weights_only=False)
        selected = select(packed, ready, contract)
        del packed
        arrays, metadata, cubes, cells = {}, {}, {}, {}
        for role, per_cell in (('L_s', 70), ('V', 300)):
            values = selected.pop(role)
            metadata[role] = {k: values[k].tolist() for k in ('y', 'rx', 'day')}
            metadata[role]['ids'] = list(values['ids'])
            x = np.asarray(values['x'].tolist(), dtype=np.float64)
            del values
            result = observe(x)
            del x
            arrays[role] = result
            m = metadata[role]
            cubes[role], cells[role] = cube(result['primary'], m['y'], m['rx'], m['day'], per_cell)
        result = evaluate(cubes['L_s'], cubes['V'])
        result.update(run_id=d.RUN, status='SOURCE_ASSOCIATION_READOUT_COMPLETE', cells=cells,
                      source_L_unique=6300, source_V_unique=27000, all_source_RX=list(d.RX),
                      physical_L_V_disjoint=True, U_used=False, U_truth_read=False,
                      V_model_updates=0, head_fit_updates=0, automatic_training=False)
        d.write(out/'resolved_config.json', dict(config=config, execution_commit=(d.ROOT/'release_commit.txt').read_text().strip(),
                observer_module=observe.__module__, observer_file=sys.modules[observe.__module__].__file__, source_roles_exact_match=True,
                existing_source_cache_deserialized_unused_features_not_consumed=True,
                network_constructed=False, checkpoint_access=False, target_access=False))
        d.write(out/'source_ids.json', metadata)
        np.savez_compressed(out/'observations.npz', **{role+'_'+key:value for role, obs in arrays.items() for key,value in obs.items()})
        d.write(out/'result.json', result)
        d.write(out/'completion.json', dict(status='ARTIFACTS_COMPLETE_SOURCE_ONLY', run_id=d.RUN,
                decision=result['decision'], L=6300, V=27000, unique_IQ_observations=33300,
                source_cells=90, folds=18, wrong_RX_controls=44, source_RX_holdout=None,
                IQ_transformed_for_training=0, CP=0, NN=0, teacher_NN=0, updates=0,
                head_fit_updates=0, U_used=False, U_truth_read=False, target_access=False,
                historical_verdict_changed=False, automatic_training=False, joint_goal_complete=False,
                wall_seconds=time.perf_counter()-started))
        print('SOURCE_CONFIRMATION_COMPLETE '+result['decision'], flush=True)
    except Exception as exc:
        d.write(out/'failure.json', dict(status='FAILED', error=repr(exc), automatic_retry=False))
        raise


if __name__ == '__main__':
    run()
