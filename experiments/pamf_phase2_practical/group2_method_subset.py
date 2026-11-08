"""Continue unstarted B tasks for two C cores, retaining B for later MoPC."""
import argparse
from pathlib import Path

from .io import read, write, require
from .group2_resume import dispatch


def smoke_saved_b(rt):
    """One real-checkpoint, source-only smoke including B-state serialization."""
    import numpy as np
    import torch
    from .group1 import row, source_scope
    from .group1_source import embedding, load_source, source_data
    from .group1_target import fit_B
    from .group2_cross_fit import prepare_b
    from .group2_b_state import save_b_state, load_b_state
    from .external_execution import seed_for
    from .group2_schedule import smoke

    result = smoke(rt)
    source = source_scope(rt)
    source['run_root'] = rt['source_run_root']
    labeled = source_data(source, 'L_s')
    labels = np.asarray(labeled.metadata['y'])
    indices = [int(np.flatnonzero(labels == i)[0]) for i in range(6)]
    x = labeled.x[indices].to('cuda:0')
    y = torch.arange(6, device='cuda:0', dtype=torch.long)
    root = Path(rt['run_root'])/'source_smoke_B'
    root.mkdir(exist_ok=False)
    checked = []
    for profile in ('protonet_cda', 'dadda_sda'):
        seed = rt['model_seeds'][0]
        bundle = load_source(source, profile, seed, 'cuda:0')
        b_seed = seed_for(seed, 'source-only-smoke', 1, 'B')
        state = fit_B(rt, bundle, x, y, b_seed, None)
        native = state.get('native_state')
        prototypes = getattr(native, 'prototypes', None) if profile == 'protonet_cda' else None
        extractor = lambda model, data: embedding(model, row(profile)['method'], data)
        prepared = prepare_b(state['model'], profile, extractor, x, y,
                             old_prototypes=prototypes, b_scores=state['score_fn'])
        job = dict(key=profile+'-source-only-smoke', profile=profile, seed=seed, capsule='source-only-smoke')
        path = root/(profile+'.pt')
        save_b_state(path, job, 1, 'source-L_s', b_seed, state['model'], prepared['mopc_hr'], prototypes)
        restored = load_b_state(path, job, 1, 'source-L_s', b_seed, 'cuda:0')['anchor']
        with torch.no_grad():
            before = prepared['mopc_hr'].model(x)['tx_logits']
            after = restored.model(x)['tx_logits']
            require(torch.allclose(before, after, rtol=1e-5, atol=1e-6), 'Reloaded B anchor scores differ')
        checked.append(profile)
        del restored, prepared, state, bundle
    result['saved_B_profiles'] = checked
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--runtime', required=True)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    runtime = read(args.runtime)
    if args.smoke:
        write(Path(runtime['run_root'])/'source_smoke.json', smoke_saved_b(runtime))
    else:
        dispatch(runtime, method_subset=True)
