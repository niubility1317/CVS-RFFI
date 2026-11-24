"""Compare fixed front-end and GRL changes on whole receivers held out of L_s."""
import argparse
import json
from pathlib import Path
from statistics import mean

import torch

from experiments.cric_dura_comparison.f_data import SourceDataset
from experiments.cric_dura_comparison.io import read, write
from .model import spectrogram
from .runner import (RECIPES, SOURCE_RXS, auxiliary_iq, create_model, labels,
                     make_optimizers, source_view, train_loss, validate)


SEED = 2026092701
HELD_RECEIVERS = (1, 8)
EPOCHS = 22
MATRICES = {
    'cis': ('original', 'rx', 'stable', 'wifi', 'joint', 'stable_rx'),
    'paper_pdp': ('wifi', 'paper_pdp_half', 'paper_pdp_all'),
}


def run(source, out, device, matrix='cis'):
    contract = read(source / 'source_contract.json')
    if contract['source_rxs'] != SOURCE_RXS or contract['counts']['L_s'] != 6300:
        raise ValueError('Require ManySig Main split source L_s')
    data = SourceDataset(source, 'L_s', contract)
    tx, rx = labels(data, device)
    iq = data.x.to(device)
    practical_iq = auxiliary_iq(data).to(device)
    out.mkdir(exist_ok=False, parents=True)
    recipes = MATRICES[matrix]
    results = {name: [] for name in recipes}
    for receiver in HELD_RECEIVERS:
        domain = SOURCE_RXS.index(receiver)
        train = (rx != domain).nonzero(as_tuple=True)[0]
        held = (rx == domain).nonzero(as_tuple=True)[0]
        rx_train = rx[train] - (rx[train] > domain).long()
        write(out / f'held-{receiver}.json',
              dict(receiver=receiver, train_ids=data.metadata['ids'][train.tolist()].tolist(),
                   held_ids=data.metadata['ids'][held.tolist()].tolist()))
        for recipe in recipes:
            front_end = RECIPES[recipe][0]
            torch.manual_seed(SEED)
            rng = torch.Generator(device=device).manual_seed(SEED)
            channel_rng = torch.Generator(device=device).manual_seed(SEED + 1237)
            model = create_model(device, recipe, receivers=len(SOURCE_RXS) - 1)
            optimizer, adversary = make_optimizers(model, recipe)
            held_iq = spectrogram(iq[held], front_end)
            held_sat = spectrogram(practical_iq[held], front_end)
            train_sat = spectrogram(practical_iq[train], front_end)
            with (out / f'rx{receiver}-{recipe}.jsonl').open('x', encoding='utf-8') as log:
                for epoch in range(1, EPOCHS + 1):
                    model.train()
                    for batch in torch.randperm(len(train), device=device, generator=rng).split(64):
                        use = train[batch]
                        optimizer.zero_grad(set_to_none=True)
                        clean = spectrogram(source_view(iq[use], recipe, rng, channel_rng), front_end)
                        loss, _ = train_loss(model, clean, train_sat[batch], tx[use],
                                             rx_train[batch], adversary, (epoch - 1) / (EPOCHS - 1))
                        loss.backward()
                        optimizer.step()
                    clean_acc = validate(model, held_iq, tx[held])[1]
                    practical_acc = validate(model, held_sat, tx[held])[1]
                    record = dict(receiver=receiver, recipe=recipe, epoch=epoch,
                                  clean=clean_acc, practical=practical_acc)
                    log.write(json.dumps(record, allow_nan=False) + '\n')
                    log.flush()
                results[recipe].append(record)
                print(json.dumps(record), flush=True)
    averages = {name: {key: mean(r[key] for r in folds) for key in ('clean', 'practical')}
                for name, folds in results.items()}
    baseline = recipes[0]
    base = averages[baseline]
    eligible = [name for name, m in averages.items() if name != baseline
                and mean(m.values()) - mean(base.values()) >= .01
                and all(m[k] >= base[k] - .02 for k in base)]
    winner = max(eligible, key=lambda name: mean(averages[name].values()), default=None)
    decision = dict(results=results, averages=averages, eligible=eligible, selection=winner,
                    target_reads=0, epochs=EPOCHS, seed=SEED, matrix=matrix, baseline=baseline)
    write(out / 'decision.json', decision)
    print(json.dumps(decision), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--matrix', choices=MATRICES, default='cis')
    args = parser.parse_args()
    torch.set_num_threads(2)
    run(args.source, args.output, args.device, args.matrix)
