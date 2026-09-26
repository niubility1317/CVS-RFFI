"""Inductive Phase2 baseline predictions. No truth file is an input.

Frozen DG, normalized nearest-class-mean registration, and the two authors'
supervised fine-tuning heads use one shared received-IQ capsule. Query data
never participates in fitting, normalization statistics, or model selection.
"""
import argparse
import copy
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'code')]
import numpy as np
import torch
from torch.nn import functional as F


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def features(model, method, x):
    if method == 'cvcnn_ce':
        return model.forward_features(x)
    if method in ('poster', 'radionet'):
        return model.embedding(x)
    output = model(x)
    return output['z_e' if method == 'riei_fd' else 'z_tx']


def logits(model, method, x):
    output = model(x)
    return output['emitter_logits' if method == 'riei_fd' else 'tx_logits'] if isinstance(output, dict) else output


def factory(method, classes, receivers):
    if method == 'cvcnn_ce':
        from baselines.cvcnn_ce.model import BasicCVCNN
        return BasicCVCNN(classes)
    if method in ('poster', 'radionet'):
        from baselines.common.li_backbones import PosterHomegrown, RadioNetDF
        return (PosterHomegrown if method == 'poster' else RadioNetDF)(classes)
    if method == 'riei_fd':
        from baselines.riei_fd.model import RIEIModel
        return RIEIModel(classes, receivers)
    if method == 'drift':
        from baselines.drift.model import DRIFTModel
        return DRIFTModel(classes, receivers)
    raise ValueError(method)


def source_provenance(source, expected, expected_epoch=200):
    initialization = read(source / 'initialization.json')
    completion = read(source / 'completion.json')
    actual = read(source / 'source_contract.json')
    if initialization.get('scratch_only') is not True or initialization.get('checkpoint_sources') != [] or initialization.get('target_contact') is not False:
        raise ValueError('CHECKPOINT_PROVENANCE_UNVERIFIED')
    if completion.get('target_evaluated') is not False or completion.get('epoch') != expected_epoch or completion.get('status') != 'SOURCE_TRAINED':
        raise ValueError('CHECKPOINT_TARGET_CONTAMINATED_OR_INCOMPLETE')
    for key in ('role_ids', 'source_rxs', 'source_days', 'ratios', 'split_seed', 'num_classes'):
        if actual.get(key) != expected.get(key):
            raise ValueError('CHECKPOINT_DATA_CONTRACT_MISMATCH: ' + key)
    if actual.get('physical_roles') != 'EXACT_MATCH':
        raise ValueError('CHECKPOINT_PROVENANCE_UNVERIFIED')
    return actual


def tensor(array, device):
    # N607 Torch2.1/NumPy2 bridge compatibility.
    return torch.tensor(array.tolist(), dtype=torch.float32, device=device)


def extract(model, method, iq, device, embedding=True):
    model.eval()
    pieces = []
    with torch.no_grad():
        for start in range(0, len(iq), 256):
            x = tensor(iq[start:start+256], device)
            value = features(model, method, x) if embedding else logits(model, method, x)
            pieces.append(value.cpu())
    return torch.cat(pieces)


def ncm(support, labels, query, classes):
    support = F.normalize(support, dim=1)
    means = torch.stack([support[labels == c].mean(0) for c in range(classes)])
    if not torch.isfinite(means).all():
        raise ValueError('Missing registered-class support')
    return F.normalize(query, dim=1) @ F.normalize(means, dim=1).T


def author_adapt(base, method, iq, labels, classes, seed, device):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    model = copy.deepcopy(base).to(device)
    model.author_finetune(classes)
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=.001, eps=1e-7)
    x, y = tensor(iq, device), labels.to(device)
    epochs, batch = (10, 256) if method == 'poster' else (30, 128)
    generator = torch.Generator().manual_seed(seed)
    model.train()
    for _ in range(epochs):
        for indices in torch.randperm(len(y), generator=generator).split(batch):
            indices = indices.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(x[indices]), y[indices])
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite support loss')
            loss.backward()
            optimizer.step()
    model.eval()
    return model


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', type=Path, required=True)
    a = p.parse_args()
    cfg = read(a.config)
    source, capsule, out = map(Path, (cfg['source_root'], cfg['capsule'], cfg['output_root']))
    manifest = read(capsule / 'manifest.json')
    if manifest['protocol_schema'] != 'p2_min_v1' or manifest['phase2_data_status'] != 'VALIDATED_ONCE':
        raise ValueError('Unvalidated capsule')
    expected = read(cfg['source_contract'])
    contract = source_provenance(source, expected)
    source_cfg = read(source / 'resolved_config.json')
    if source_cfg['method'] != cfg['method'] or source_cfg['model_seed'] != cfg['model_seed']:
        raise ValueError('Wrong source row')
    out.mkdir(parents=True, exist_ok=False)
    (out / 'resolved_config.json').write_text(json.dumps(cfg, indent=2), encoding='utf-8')
    (out / 'checkpoint_provenance.json').write_text(json.dumps(dict(verdict='MATCHED_SOURCE_ONLY_SCRATCH',
        checkpoint=str(source/'last.pt'), source_contract=str(source/'source_contract.json'),
        initialization=str(source/'initialization.json'), selection='fixed_epoch_200', capsule_id=manifest['capsule_id'])), encoding='utf-8')
    torch.set_num_threads(2)
    device = torch.device(cfg.get('device', 'cuda:0'))
    model = factory(cfg['method'], len(contract['classes']), len(contract['source_rxs'])).to(device)
    payload = torch.load(source / 'last.pt', map_location=device, weights_only=False)
    if payload['epoch'] != 200:
        raise ValueError('Not the registered final checkpoint')
    model.load_state_dict(payload['model'], strict=True)
    model.eval()
    data = np.load(capsule / 'received.npz', allow_pickle=False)
    iq, ids = data['iq'], data['ids']
    split_files = sorted((capsule / 'splits').glob('*.json'))
    # Smoke uses only explicitly labeled support, before any query inference.
    first = read(split_files[0])
    smoke = extract(model, cfg['method'], iq[first['support_indices'][:2]], device, embedding=False)
    if not torch.isfinite(smoke).all() or smoke.shape[1] != len(contract['classes']):
        raise ValueError('Real checkpoint support-only smoke failed')
    (out/'support_smoke.json').write_text(json.dumps({'passed': True, 'query_read': False}),encoding='utf-8')
    embedded = extract(model, cfg['method'], iq, device)
    dg_seen = set()
    count = 0
    with (out/'predictions.jsonl').open('x', encoding='utf-8') as f:
        def save(split, mode, scores):
            nonlocal count
            if scores.shape != (len(split['query_indices']), len(split['registered_classes'])) or not torch.isfinite(scores).all():
                raise ValueError('Invalid prediction dimensions/values')
            record = dict(split_id=split['split_id'], capsule_id=manifest['capsule_id'], mode=mode,
                receiver=split['receiver'], scenario=split['scenario'], k=split['k'], support_seed=split['support_seed'],
                classes=split['registered_classes'], query_ids=ids[split['query_indices']].tolist(),
                predicted_indices=scores.argmax(1).tolist(), scores=scores.tolist())
            f.write(json.dumps(record) + '\n')
            f.flush()
            count += 1
        for path in split_files:
            split = read(path)
            if split['capsule_id'] != manifest['capsule_id']:
                raise ValueError('Capsule mismatch')
            s, q = split['support_indices'], split['query_indices']
            y = torch.tensor(split['support_labels'])
            classes = len(split['registered_classes'])
            save(split, 'support_ncm', ncm(embedded[s], y, embedded[q], classes))
            key = (split['receiver'], split['scenario'])
            if split['registered_classes'] == contract['classes'] and key not in dg_seen:
                save(split, 'frozen_dg', extract(model, cfg['method'], iq[q], device, embedding=False))
                dg_seen.add(key)
            if cfg['method'] in ('poster', 'radionet'):
                adapted = author_adapt(model, cfg['method'], iq[s], y, classes, cfg['model_seed'], device)
                save(split, 'author_finetune', extract(adapted, cfg['method'], iq[q], device, embedding=False))
                del adapted
            print(json.dumps(dict(split_id=split['split_id'], predictions=count)), flush=True)
    (out/'predictions_complete.json').write_text(json.dumps(dict(status='PREDICTIONS_COMPLETE',
        predictions=count, split_count=len(split_files), capsule_id=manifest['capsule_id'], truth_read=False)), encoding='utf-8')


if __name__ == '__main__':
    main()
