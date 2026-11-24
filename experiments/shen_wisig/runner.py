"""Train and freeze on ManySig source roles, then predict and score separately."""
import argparse
import json
import os
from pathlib import Path
import tempfile

import torch
from torch.nn import functional as F

from experiments.cric_dura_comparison.f_data import SourceDataset
from experiments.cric_dura_comparison.io import read, write
from experiments.cric_dura_comparison.t1_practical import Practical
from experiments.pamf_repl_view_shuffle import evaluation
from .model import ReceiverAdversarialCNN, spectrogram


SOURCE_FIELDS = {
    'run_id', 'row_id', 'model_seed', 'code_commit', 'source_cache',
    'source_contract', 'output_root',
}
SOURCE_RXS = [1, 3, 4, 6, 8]
SOURCE_DAYS = [1, 2, 3]
CLASSES = ['14-10', '14-7', '20-15', '20-19', '6-15', '8-20']
EPOCHS = 50
RECIPES = {
    'original': ('original', False),
    'rx': ('original', True),
    'stable': ('stable', False),
    'wifi': ('wifi', False),
    'joint': ('wifi', True),
    'stable_rx': ('stable', True),
}


def source_contract(config):
    keys = set(config) - {'recipe'}
    if keys != SOURCE_FIELDS or config.get('recipe', 'original') not in RECIPES:
        raise ValueError('Training requires only registered source paths and row identity')
    contract = read(config['source_contract'])
    if (contract['source_rxs'] != SOURCE_RXS or contract['source_days'] != SOURCE_DAYS
            or contract['classes'] != CLASSES
            or contract['counts'] != {'L_s': 6300, 'U_s': 56700, 'V': 27000}):
        raise ValueError('Expected the fixed ManySig Main split')
    return contract


def create_model(device, recipe='original', receivers=len(SOURCE_RXS)):
    return ReceiverAdversarialCNN(len(CLASSES), receivers, RECIPES[recipe][0]).to(device)


def make_optimizers(model, recipe):
    adversary = None
    if RECIPES[recipe][1]:
        adversary = torch.optim.Adam(model.rx.parameters(), lr=0.002)
        params = [*model.backbone.parameters(), *model.tx.parameters()]
    else:
        params = model.parameters()
    return torch.optim.SGD(params, lr=0.001, momentum=0.9), adversary


def labels(dataset, device):
    rx_map = {source_rx: index for index, source_rx in enumerate(SOURCE_RXS)}
    tx = torch.tensor(dataset.metadata['y'].tolist(), device=device)
    rx = torch.tensor([rx_map[i] for i in dataset.metadata['rx_i'].tolist()], device=device)
    if any(day not in SOURCE_DAYS for day in dataset.metadata['day_i']) or set(dataset.metadata['eq_i']) != {1}:
        raise ValueError('Expected source dates and the equalized WiSig input')
    if set(tx.tolist()) != set(range(len(CLASSES))):
        raise ValueError('Unexpected transmitter labels in the source role')
    return tx, rx


def add_noise(iq, generator):
    snr_db = torch.empty(len(iq), 1, 1, device=iq.device).uniform_(0, 49, generator=generator)
    power = iq.square().mean(dim=(1, 2), keepdim=True)
    noise = torch.randn(iq.shape, device=iq.device, generator=generator)
    return iq + noise * (power * 10 ** (-snr_db / 10)).sqrt()


def auxiliary_iq(dataset):
    simulate = Practical({'source_rxs': SOURCE_RXS, 'source_days': SOURCE_DAYS})
    fields = ('eq_i', 'rx_i', 'day_i', 'sig_i', 'base_index')
    output = []
    for start in range(0, len(dataset), 256):
        indices = slice(start, start + 256)
        metadata = {key: torch.as_tensor(dataset.metadata[key][indices].tolist()) for key in fields}
        output.append(simulate(dataset.x[indices], metadata, epoch=1, step=start // 256))
    return torch.cat(output)


def train_loss(model, original, satellite, tx, rx, adversary=None, progress=0.):
    features = model.encode(torch.cat((original, satellite), dim=0))
    tx_logits = model.classify_features(features)
    ordinary, augmented = tx_logits.split(len(tx))
    source = features[:len(tx)]
    strength = 1.
    if adversary is not None:
        detached = source.detach()
        for _ in range(2):
            adversary.zero_grad(set_to_none=True)
            F.cross_entropy(model.rx(detached), rx).backward()
            adversary.step()
        # The independent optimizer learns an informative RX classifier first.
        # This branch then trains the shared extractor against that classifier.
        strength = 0.2 + 0.8 * min(1., 2. * progress)
        model.rx.requires_grad_(False)
    rx_logits = model.receiver_logits(source, strength)
    if adversary is not None:
        model.rx.requires_grad_(True)
    loss = F.cross_entropy(ordinary, tx) + F.cross_entropy(rx_logits, rx)
    return loss + 0.68 * F.cross_entropy(augmented, tx), ordinary


@torch.no_grad()
def validate(model, images, tx, rx=None):
    model.eval()
    loss = correct = rx_correct = 0
    for start in range(0, len(tx), 256):
        indices = slice(start, start + 256)
        x, y = images[indices], tx[indices]
        features = model.encode(x)
        logits = model.classify_features(features)
        loss += F.cross_entropy(logits, y, reduction='sum')
        if rx is not None:
            r = rx[indices]
            rx_logits = model.receiver_logits(features)
            loss += F.cross_entropy(rx_logits, r, reduction='sum')
            rx_correct += (rx_logits.argmax(1) == r).sum()
        correct += (logits.argmax(1) == y).sum()
    return float(loss) / len(tx), float(correct) / len(tx), float(rx_correct) / len(tx) if rx is not None else None


def load_reader(folder, contract, device):
    path = Path(folder)
    frozen = read(path / 'complete.json')
    if frozen['status'] != 'SOURCE_FROZEN' or frozen['target_access'] is not False:
        raise ValueError('Expected the frozen source-only CNN')
    payload = torch.load(path / 'model.pt', map_location='cpu', weights_only=True)
    if payload['source_contract'] != contract or any(
        payload['identity'][k] != frozen[k] for k in ('run_id', 'row_id', 'model_seed')
    ):
        raise ValueError('Frozen checkpoint contract or row differs')
    recipe = payload.get('recipe', 'original')
    if recipe != frozen.get('recipe', 'original'):
        raise ValueError('Checkpoint and frozen input representation differ')
    model = create_model(device, recipe)
    model.load_state_dict(payload['weights'], strict=True)
    return model.eval(), frozen


def freeze(folder, model, contract, config, **metrics):
    identity = {k: config[k] for k in ('run_id', 'row_id', 'model_seed', 'code_commit')}
    recipe = config.get('recipe', 'original')
    torch.save(dict(weights=model.state_dict(), source_contract=contract, identity=identity, recipe=recipe),
               folder / 'model.pt')
    write(folder / 'complete.json',
          dict(status='SOURCE_FROZEN', target_access=False, recipe=recipe, **identity, **metrics))


def smoke(config, device):
    contract = source_contract(config)
    labeled = SourceDataset(config['source_cache'], 'L_s', contract)
    tx, rx = labels(labeled, device)
    # Use real L packets and the same augmentation, loss, and checkpoint reader.
    selected = [next(i for i, value in enumerate(tx.tolist()) if value == cls) for cls in range(len(CLASSES))]
    sample = labeled.x[selected].to(device)
    simulate = Practical({'source_rxs': SOURCE_RXS, 'source_days': SOURCE_DAYS})
    physical = {k: torch.tensor(labeled.metadata[k][selected].tolist())
                for k in ('eq_i', 'rx_i', 'day_i', 'sig_i', 'base_index')}
    satellite = simulate(sample, physical, epoch=1, step=0)
    recipe = config.get('recipe', 'original')
    model = create_model(device, recipe)
    optimizer, adversary = make_optimizers(model, recipe)
    encode = model.front_end
    loss, _ = train_loss(model, spectrogram(sample, encode), spectrogram(satellite, encode),
                         tx[selected], rx[selected], adversary)
    loss.backward()
    if not torch.isfinite(loss) or any(p.grad is None or not torch.isfinite(p.grad).all() for p in model.parameters()):
        raise FloatingPointError('Source checkpoint smoke encountered a bad gradient')
    optimizer.step()
    out = Path(config['output_root']).parent
    with tempfile.TemporaryDirectory(dir=out) as name:
        freeze(Path(name), model, contract, config)
        loaded, _ = load_reader(name, contract, device)
        with torch.inference_mode():
            logits = loaded.classify_features(loaded.features(sample))
            torch.testing.assert_close(logits, model.classify_features(model.features(sample)))
    write(out / 'smoke.json', dict(status='PASS', source_packets=len(selected), target_reads=0))
    print('SHEN_WISIG_SOURCE_CHECKPOINT_SMOKE_PASS', flush=True)


def train(config, device):
    contract = source_contract(config)
    torch.manual_seed(config['model_seed'])
    seed = torch.Generator(device=device).manual_seed(config['model_seed'])
    labeled = SourceDataset(config['source_cache'], 'L_s', contract)
    val = SourceDataset(config['source_cache'], 'V', contract)
    tx, rx = labels(labeled, device)
    val_tx, val_rx = labels(val, device)
    out = Path(config['output_root'])
    out.mkdir(parents=False, exist_ok=False)
    print('SHEN_WISIG_SOURCE_ROLES', len(labeled), len(val), flush=True)
    recipe = config.get('recipe', 'original')
    front_end = RECIPES[recipe][0]
    sat = spectrogram(auxiliary_iq(labeled).to(device), front_end)
    val_images = spectrogram(val.x.to(device), front_end)
    iq = labeled.x.to(device)
    model = create_model(device, recipe)
    optimizer, adversary = make_optimizers(model, recipe)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.2, patience=9,
                                                           threshold=0)
    best_loss = float('inf')
    best_epoch = 0
    with (out / 'epochs.jsonl').open('x', encoding='utf-8') as log:
        for epoch in range(1, EPOCHS + 1):
            model.train()
            correct = loss_sum = 0
            for index in torch.randperm(len(iq), device=device, generator=seed).split(64):
                optimizer.zero_grad(set_to_none=True)
                ordinary = spectrogram(add_noise(iq[index], seed), front_end)
                loss, logits = train_loss(model, ordinary, sat[index], tx[index], rx[index],
                                          adversary, (epoch - 1) / (EPOCHS - 1))
                loss.backward()
                optimizer.step()
                correct += (logits.detach().argmax(1) == tx[index]).sum()
                loss_sum += loss.detach() * len(index)
            val_loss, val_accuracy, val_rx_accuracy = validate(model, val_images, val_tx, val_rx)
            stats = dict(epoch=epoch, loss=float(loss_sum) / len(iq), accuracy=float(correct) / len(iq),
                         val_loss=val_loss, val_accuracy=val_accuracy, val_rx_accuracy=val_rx_accuracy,
                         lr=optimizer.param_groups[0]['lr'])
            line = json.dumps(stats, allow_nan=False)
            log.write(line + '\n')
            log.flush()
            print(line, flush=True)
            if recipe != 'original':
                continue  # Fixed E50, matching the source-held comparison's fixed checkpoint rule.
            scheduler.step(val_loss)
            if val_loss < best_loss:
                best_loss, best_epoch, best_accuracy = val_loss, epoch, val_accuracy
                best_weights = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            if epoch - best_epoch >= 20:
                break
    if recipe == 'original':
        model.load_state_dict(best_weights, strict=True)
    else:
        best_epoch, best_loss, best_accuracy = epoch, val_loss, val_accuracy
    freeze(out, model, contract, config, epoch=best_epoch, epochs_trained=epoch,
           val_loss=best_loss, val_accuracy=best_accuracy)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('smoke', 'train', 'predict', 'score'))
    parser.add_argument('--config', required=True)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    config = read(args.config)
    limit = os.environ.get('SHEN_WISIG_CUDA_MEMORY_FRACTION')
    if limit is not None and torch.device(args.device).type == 'cuda':
        torch.cuda.set_per_process_memory_fraction(float(limit), device=args.device)
    if args.stage == 'predict':
        evaluation.predict(config, args.device, load_reader=load_reader)
    elif args.stage == 'score':
        evaluation.score(config)
    elif args.stage == 'train':
        train(config, args.device)
    else:
        smoke(config, args.device)


if __name__ == '__main__':
    main()
