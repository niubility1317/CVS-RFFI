"""Train and freeze on ManySig source roles, then predict and score separately."""
import argparse
import json
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


def source_contract(config):
    if set(config) != SOURCE_FIELDS:
        raise ValueError('Training requires only registered source paths and row identity')
    contract = read(config['source_contract'])
    if (contract['source_rxs'] != SOURCE_RXS or contract['source_days'] != SOURCE_DAYS
            or contract['classes'] != CLASSES
            or contract['counts'] != {'L_s': 6300, 'U_s': 56700, 'V': 27000}):
        raise ValueError('Expected the fixed ManySig Main split')
    return contract


def create_model(device):
    return ReceiverAdversarialCNN(len(CLASSES), len(SOURCE_RXS)).to(device)


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


def train_loss(model, original, satellite, tx, rx):
    features = model.encode(torch.cat((original, satellite), dim=0))
    tx_logits = model.classify_features(features)
    ordinary, augmented = tx_logits.split(len(tx))
    rx_logits = model.receiver_logits(features[:len(tx)])
    loss = F.cross_entropy(ordinary, tx) + F.cross_entropy(rx_logits, rx)
    return loss + 0.68 * F.cross_entropy(augmented, tx), ordinary


@torch.no_grad()
def validate(model, images, tx, rx):
    model.eval()
    loss = correct = 0
    for x, y, r in zip(images.split(256), tx.split(256), rx.split(256), strict=True):
        features = model.encode(x)
        logits = model.classify_features(features)
        loss += F.cross_entropy(logits, y, reduction='sum')
        loss += F.cross_entropy(model.receiver_logits(features), r, reduction='sum')
        correct += (logits.argmax(1) == y).sum()
    return float(loss) / len(tx), float(correct) / len(tx)


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
    model = create_model(device)
    model.load_state_dict(payload['weights'], strict=True)
    return model.eval(), frozen


def freeze(folder, model, contract, config, **metrics):
    identity = {k: config[k] for k in ('run_id', 'row_id', 'model_seed', 'code_commit')}
    torch.save(dict(weights=model.state_dict(), source_contract=contract, identity=identity),
               folder / 'model.pt')
    write(folder / 'complete.json',
          dict(status='SOURCE_FROZEN', target_access=False, **identity, **metrics))


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
    model = create_model(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    loss, _ = train_loss(model, spectrogram(sample), spectrogram(satellite), tx[selected], rx[selected])
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
    sat = spectrogram(auxiliary_iq(labeled).to(device))
    val_images = spectrogram(val.x.to(device))
    iq = labeled.x.to(device)
    model = create_model(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.2, patience=9,
                                                           threshold=0)
    best_loss = float('inf')
    best_epoch = 0
    with (out / 'epochs.jsonl').open('x', encoding='utf-8') as log:
        for epoch in range(1, EPOCHS + 1):
            model.train()
            correct = loss_sum = 0
            for index in torch.randperm(len(iq), device=device, generator=seed).split(64):
                ordinary = spectrogram(add_noise(iq[index], seed))
                loss, logits = train_loss(model, ordinary, sat[index], tx[index], rx[index])
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                correct += (logits.detach().argmax(1) == tx[index]).sum()
                loss_sum += loss.detach() * len(index)
            val_loss, val_accuracy = validate(model, val_images, val_tx, val_rx)
            stats = dict(epoch=epoch, loss=float(loss_sum) / len(iq), accuracy=float(correct) / len(iq),
                         val_loss=val_loss, val_accuracy=val_accuracy, lr=optimizer.param_groups[0]['lr'])
            line = json.dumps(stats, allow_nan=False)
            log.write(line + '\n')
            log.flush()
            print(line, flush=True)
            scheduler.step(val_loss)
            if val_loss < best_loss:
                best_loss, best_epoch, best_accuracy = val_loss, epoch, val_accuracy
                best_weights = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            if epoch - best_epoch >= 20:
                break
    model.load_state_dict(best_weights, strict=True)
    freeze(out, model, contract, config, epoch=best_epoch, epochs_trained=epoch,
           val_loss=best_loss, val_accuracy=best_accuracy)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('smoke', 'train', 'predict', 'score'))
    parser.add_argument('--config', required=True)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    config = read(args.config)
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
