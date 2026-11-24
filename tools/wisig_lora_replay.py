"""Source fitting and truth-blind WiSig replay for the LoRa recovery comparison."""

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

SEEDS = (2026092701, 2026092702, 2026092703)
COUNTS = {'cross_rx': 126000, 'cross_both': 42000}


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def readers(rt):
    from experiments.cric_dura_comparison.evaluation import reject_truth
    from experiments.pamf_repl_view_shuffle.plan import RUN, TASKS
    from experiments.pamf_repl_view_shuffle.source import load_reader
    configs, models = {}, {}
    for seed in SEEDS:
        row = f'main_split-repl-shuffled_time-3view-s{seed}'
        cfg = read(Path(rt['recognizer_run']) / 'configs' / f'predict__{row}.json')
        if (cfg['run_id'], cfg['row_id'], cfg['model_seed']) != (RUN, row, seed) or set(cfg['tests']) != set(TASKS):
            raise ValueError('Expected the registered WiSig Main split reader')
        reject_truth(cfg)
        model, proof = load_reader(cfg['source_output'], read(cfg['source_contract']), 'cuda:0')
        if (proof['run_id'], proof['row_id'], proof['model_seed']) != (RUN, row, seed):
            raise ValueError('Unexpected actual frozen WiSig reader')
        configs[seed] = cfg
        models[seed] = model.eval().requires_grad_(False)
    return configs, models


def bind(rt, arm):
    original = Path(rt[arm + '_release'])
    new_release = Path(rt['release_root'])
    sys.path.insert(0, str(original))
    import experiments.pamf_unrolled_restore as package
    package.__path__.append(str(new_release / 'experiments' / 'pamf_unrolled_restore'))
    from experiments.pamf_lora_main.runner import activate
    origin = read(original / 'runtime.json')
    activate(origin)
    return origin


def frozen(arm, origin):
    from experiments.pamf_lora_main.dura_repair import PacketRestorer
    from experiments.pamf_lora_main.full_packet_views import source_contract
    from experiments.pamf_unrolled_restore.wisig_lora_window_mapping import restore_expanded, restore_guard
    contract = source_contract(Path(origin['source_root']))
    root = Path(origin['run_root'])
    tracker = PacketRestorer('consensus').cuda().eval()
    if arm == 'expanded':
        from experiments.pamf_lora_main.dura_capacity_source import RECIPES, recover
        from experiments.pamf_lora_main.spectral_block_source import load_models
        models, threshold = load_models(root, contract, 'cuda:0', {'expanded': RECIPES['expanded']})
        return lambda iq: restore_expanded(iq, recover, tracker, models, threshold)
    from experiments.pamf_lora_main.weak_guard_source import parent_model, frozen_guard, recover
    model, threshold = parent_model(root, contract, 'cuda:0')
    gate, old_threshold = frozen_guard(root, contract, 'cuda:0')
    if threshold != old_threshold:
        raise ValueError('LoRa guard threshold differs from its frozen parent')
    return lambda iq: restore_guard(iq, recover, tracker, model, gate, threshold)


def retrained(root, arm):
    from experiments.pamf_unrolled_restore.wisig_lora_source_fit import make_model
    from experiments.pamf_unrolled_restore.wisig_lora_adaptation import expanded_readout, rank_readout
    from experiments.pamf_unrolled_restore.wisig_receiver_estimates import guard_score, receive_coordinates
    cp = torch.load(Path(root) / f'{arm}.pt', map_location='cuda:0', weights_only=True)
    metadata = read(Path(root) / f'{arm}-complete.json')
    if cp['metadata'] != metadata or metadata['arm'] != arm or metadata['steps'] != 40000:
        raise ValueError('Expected the final source-only WiSig checkpoint')
    if any(metadata[key] != 0 for key in ('target_reads', 'TX_reads', 'F_reads', 'simulator_parameter_reads')):
        raise ValueError('Unexpected source-model ancestry')
    model = make_model(arm).cuda().eval().requires_grad_(False)
    model.load_state_dict(cp['weights'], strict=True)
    diagonal, factors = cp['diagonal'], cp['factors']
    threshold = metadata['threshold']

    def restore(iq):
        active = receive_coordinates(iq)[-1].flatten() > threshold
        if arm == 'guard':
            weak = metadata['weak_guard']
            active |= guard_score(iq, weak) > weak['threshold']
        output = iq.clone()
        if active.any():
            result = (expanded_readout(model, iq[active]) if arm == 'expanded'
                      else rank_readout(model, iq[active], diagonal, factors))
            output[active] = result
        return output, int(active.sum())
    return restore


@torch.no_grad()
def smoke(rt, arm, mode, origin):
    from experiments.pamf_unrolled_restore.wisig_lora_source_fit import source_iq, perturb
    source, _, ids = source_iq(rt['source_root'])
    iq = torch.as_tensor(np.array(source[ids[:12]], copy=True), device='cuda:0')
    disturbed = perturb(iq, torch.Generator(device='cuda:0').manual_seed(3117))
    restore = frozen(arm, origin) if mode == 'frozen' else retrained(rt['run_root'], arm)
    _, heads = readers(rt)
    for sample in (iq, disturbed):
        output, active = restore(sample)
        if output.shape != sample.shape or not torch.isfinite(output).all():
            raise ValueError('Candidate returned invalid IQ')
        for head in heads.values():
            logits = head.classify_features(head.features(output))
            if logits.shape != (len(sample), 6) or not torch.isfinite(logits).all():
                raise ValueError('Frozen WiSig reader returned invalid scores')
    print('WISIG_LORA_SMOKE_PASS', arm, mode, 'target_reads=0', flush=True)


@torch.no_grad()
def predict(rt, arm, mode, origin, route, split):
    from experiments.cric_dura_comparison.evaluation import handle
    from experiments.pamf_repl_view_shuffle.plan import VIEWS
    originals, heads = readers(rt)
    cfg = originals[SEEDS[0]]
    views_root = (Path(rt['baseline_root']) / 'full_views' / split if route == 'full'
                  else Path(cfg['tests'][split]['views_root']))
    manifest, ids, positions = handle(views_root, cfg['classes'])
    if len(ids) != COUNTS[split] or manifest['channel'] != f'{route}/post_sync/noeq':
        raise ValueError('Unexpected WiSig Main split view')
    restore = frozen(arm, origin) if mode == 'frozen' else retrained(rt['run_root'], arm)
    results, activations = {}, {}
    for view in VIEWS:
        file = Path(manifest['clean_ref']) if view == 'clean' else Path(manifest['observations_root']) / f'{view}.npy'
        observed = np.load(file, mmap_mode='r', allow_pickle=False)
        if observed.shape[1:] != (2, 256) or observed.dtype != np.float32:
            raise ValueError('Unexpected received-IQ layout')
        decisions = np.empty((len(SEEDS), len(ids)), dtype=np.uint8)
        count = 0
        for start in range(0, len(ids), 24):
            selection = np.s_[start:min(start + 24, len(ids))]
            rows = observed[selection if positions is None else positions[selection]]
            iq = torch.as_tensor(np.array(rows, copy=True), device='cuda:0')
            output, active = restore(iq)
            if not torch.isfinite(output).all():
                raise ValueError('Nonfinite restored WiSig sample')
            count += active
            for index, head in enumerate(heads.values()):
                logits = head.classify_features(head.features(output))
                if logits.shape != (len(iq), 6) or not torch.isfinite(logits).all():
                    raise ValueError('Invalid frozen WiSig reader output')
                decisions[index, selection] = logits.argmax(1).cpu().numpy()
        results[view], activations[view] = decisions, count
        print('WISIG_LORA_VIEW', arm, mode, route, split, view, len(ids), count, flush=True)
    folder = Path(rt['run_root']) / mode / arm / route / split
    folder.mkdir(parents=True, exist_ok=False)
    with (folder / 'decisions.npz').open('xb') as stream:
        np.savez(stream, ids=ids, **results)
    (folder / 'complete.json').write_text(json.dumps(dict(status='PREDICTIONS_FROZEN', arm=arm, mode=mode,
        route=route, split=split, count=len(ids), seeds=list(SEEDS), views=list(VIEWS),
        active=activations, truth_read=False)), encoding='utf-8')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('fit-smoke', 'fit', 'smoke', 'predict'))
    parser.add_argument('runtime', type=Path)
    parser.add_argument('arm', choices=('expanded', 'guard'))
    parser.add_argument('mode', choices=('frozen', 'retrained'))
    parser.add_argument('--route', choices=('full', 'residual'))
    parser.add_argument('--split', choices=('cross_rx', 'cross_both'))
    args = parser.parse_args()
    rt = read(args.runtime)
    origin = bind(rt, args.arm)
    if args.action in ('fit-smoke', 'fit'):
        if args.mode != 'retrained':
            raise ValueError('Only the new WiSig arm can be trained')
        from experiments.pamf_unrolled_restore.wisig_lora_source_fit import fit
        fit(rt['run_root'], rt['source_root'], args.arm, smoke=args.action == 'fit-smoke')
    elif args.action == 'smoke':
        smoke(rt, args.arm, args.mode, origin)
    else:
        if not args.route or not args.split:
            raise ValueError('Prediction requires route and split')
        predict(rt, args.arm, args.mode, origin, args.route, args.split)


if __name__ == '__main__':
    main()
