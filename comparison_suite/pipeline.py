"""Final source checkpoint -> paired Phase1 and support-only Phase2 predictions.

This module has no truth-file input. The independent score module opens truth
only after prediction completion. C inherits B; neither receives query tensors.
"""
from __future__ import annotations
import argparse
import copy
import csv
import json
import random
import sys
import time
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'code'))
from comparison_suite.models import factory, classifier_logits
from comparison_suite.adaptation import fit_adaptation, features
from comparison_suite.registration import prepare_registration_state, register_new_classes

REGISTRATION = {'protonet', 'csil', 'mopc_hr', 'orthogonal'}
ADAPTER = {'mrior': 'mrior_sda', 'dadda': 'dadda_sda', 'twostage': 'twostage_sda',
           'feature_separation': 'feature_separation_ft', 'radionet_ada': 'radionet_ada_knn'}

def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))

def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')

def tensor(array, device):
    # Torch2.1/NumPy2 on N607 cannot share the NumPy C ABI.
    return torch.tensor(np.asarray(array).tolist(), device=device, dtype=torch.float32)

@torch.no_grad()
def infer(score, array, device, batch_size=256, expected_classes=None):
    result = []
    for start in range(0, len(array), batch_size):
        values = score(tensor(array[start:start+batch_size], device))
        if values.ndim != 2 or not bool(torch.isfinite(values).all()):
            raise FloatingPointError('Invalid query scores')
        if values.shape[0] != min(batch_size,len(array)-start) or (expected_classes is not None and values.shape[1] != expected_classes):
            raise ValueError('Scores must face all registered classes')
        result.extend(values.argmax(1).cpu().tolist())
    return result

def task_key(split):
    return (split['receiver'], split['scenario'], int(split['k']), int(split['support_seed']))

def group_splits(splits, source_classes, capsule_id):
    groups = {}
    for split in splits:
        classes = split['registered_classes']
        support, labels, query = split['support_indices'], split['support_labels'], split['query_indices']
        if (len(set(classes)) != len(classes) or len(support) != len(labels)
                or len(set(support)) != len(support) or len(set(query)) != len(query)
                or set(labels) != set(range(len(classes)))
                or any(labels.count(c) != split['k'] for c in range(len(classes)))):
            raise ValueError('Invalid support/class/query schema')
        if (split['capsule_id'] != capsule_id or split['protocol_schema'] != 'p2_min_v1'
                or split['phase2_data_status'] != 'VALIDATED_ONCE'):
            raise ValueError('Phase2 capsule/split contract mismatch')
        n = len(split['registered_classes']) - len(source_classes)
        if n not in (0, 2, 5, 10, 20) or set(split['registered_classes'][:len(source_classes)]) != set(source_classes):
            raise ValueError('Unexpected registered class set')
        key = task_key(split)
        if n in groups.setdefault(key, {}):
            raise ValueError('Repeated task/new-class count')
        groups[key][n] = split
    for rows in groups.values():
        if set(rows) != {0, 2, 5, 10, 20}:
            raise ValueError('Incomplete K x new-class matrix')
        old = rows[0]
        for n, split in rows.items():
            old_count = len(source_classes)
            olds = [i for i, y in zip(split['support_indices'], split['support_labels']) if y < old_count]
            if olds != old['support_indices'] or split['registered_classes'][:old_count] != old['registered_classes']:
                raise ValueError('B/C old support physical IDs or class order differ')
            if not set(old['query_indices']).issubset(split['query_indices']):
                raise ValueError('B/C old query physical IDs differ')
            if set(split['support_indices']) & set(split['query_indices']):
                raise ValueError('Support/query overlap')
    return groups

def provenance(cfg, source):
    completed = read(source/'completion.json')
    initial = read(source/'initialization.json')
    contract = read(source/'source_contract.json')
    actual = read(cfg['source_contract'])
    if completed['status'] != 'SOURCE_TRAINED' or completed['epoch'] != 200:
        raise ValueError('Not the fixed final-200 source checkpoint')
    if (not initial.get('scratch_only') or initial.get('checkpoint_sources') or initial.get('target_contact')
            or initial.get('status') != 'SCRATCH' or initial.get('checkpoint') is not None
            or initial.get('ancestors') or initial.get('target_access') is not False):
        raise ValueError('CHECKPOINT_PROVENANCE_UNVERIFIED or CHECKPOINT_TARGET_CONTAMINATED')
    for field in ('role_ids', 'source_rxs', 'source_days', 'ratios', 'split_seed'):
        if contract[field] != actual[field]:
            raise ValueError('CHECKPOINT_DATA_CONTRACT_MISMATCH: '+field)
    if contract.get('equalized') != 1 or contract.get('out_len') != 256 or contract.get('normalize') is not True or contract['num_classes'] != actual['num_classes']:
        raise ValueError('CHECKPOINT_DATA_CONTRACT_MISMATCH: input view/class count')
    if initial['model_seed'] != cfg['model_seed']:
        raise ValueError('Source seed mismatch')
    return contract, initial

def source_replay(cfg, contract):
    from dataset_wisig import load_wisig_compact_pkl, WiSigCompactDataset
    from baselines.common.practical_source import match_roles
    raw = load_wisig_compact_pkl(cfg['dataset'])
    if [str(c) for c in raw['tx_list']] != contract['classes']:
        raise ValueError('Source replay class mapping mismatch')
    base = WiSigCompactDataset(raw, out_len=256, crop_mode='center', normalize=True,
        equalized=1, rx_keep=contract['source_rxs'], day_keep=contract['source_days'], domain='rx')
    parts = match_roles(base, contract)
    data = [base[i] for i in parts['L_s']]
    return torch.stack([item[0] for item in data]), torch.tensor([item[1] for item in data], dtype=torch.long)

class FitLog:
    def __init__(self, out):
        self.file = (out/'fit_metrics.jsonl').open('x', encoding='utf-8')
        self.text = (out/'training.log').open('x', encoding='utf-8')
        self.context = {}
    def __call__(self, row):
        value = {**self.context, **row}
        self.file.write(json.dumps(value, allow_nan=False)+'\n'); self.file.flush()
        self.text.write('FIT '+json.dumps(value, allow_nan=False)+'\n'); self.text.flush()
        if row.get('epoch', row.get('iteration', 1)) % 10 == 0:
            print('FIT '+json.dumps(value, allow_nan=False), flush=True)
    def close(self):
        self.file.close(); self.text.close()
        # Union schema retains measured method-specific components; missing=N/A.
        path = self.file.name
        fields = set()
        with open(path, encoding='utf-8') as f:
            for line in f: fields.update(json.loads(line))
        with open(path, encoding='utf-8') as f, open(Path(path).with_suffix('.csv'), 'x', encoding='utf-8', newline='') as dst:
            writer = csv.DictWriter(dst, fieldnames=sorted(fields)); writer.writeheader()
            for line in f: writer.writerow(json.loads(line))

def run(cfg):
    out, source = Path(cfg['output_root']), Path(cfg['source_output'])
    out.mkdir(parents=True, exist_ok=True)
    if (out/'resolved_config.json').exists():
        raise FileExistsError('Existing pipeline may not be overwritten/restarted')
    device = torch.device(cfg.get('device', 'cuda:0'))
    torch.set_num_threads(2)
    random.seed(cfg['model_seed']); torch.manual_seed(cfg['model_seed'])
    contract, initial = provenance(cfg, source)
    payload = torch.load(source/'last.pt', map_location=device, weights_only=False)
    if (payload['epoch'] != 200 or payload['method'] != cfg['method']
            or payload['classes'] != contract['classes'] or payload['source_contract'] != contract
            or payload['initialization'] != initial or payload['selection'] != 'fixed_last_epoch'):
        raise ValueError('Checkpoint content/provenance mismatch')
    model = factory(cfg['method'], payload['num_classes'], payload['num_receivers'], **payload['model_kwargs']).to(device)
    model.load_state_dict(payload['model'], strict=True); model.eval()
    write(out/'provenance.json', dict(status='VERIFIED', checkpoint=str(source/'last.pt'),
        scratch_ancestors=[], source_roles='EXACT_MATCH', selection='fixed_last_epoch_200',
        target_contact_before_this_pipeline=False, contract=str(cfg['source_contract'])))
    write(out/'resolved_config.json', dict(cfg, initial=initial, classes=contract['classes'],
        query_fit=False, truth_input=False, source_replay_exception=cfg['method'] in ADAPTER,
        B_C_training='C inherits B; same old support/query', hardware=torch.cuda.get_device_name(device) if device.type=='cuda' else 'cpu'))
    print('RESOLVED_CONFIG '+json.dumps(read(out/'resolved_config.json')), flush=True)
    p2 = Path(cfg['p2_capsule']); manifest = read(p2/'manifest.json')
    if manifest['phase2_data_status'] != 'VALIDATED_ONCE' or manifest['channel'] != dict(route='residual', mode='post_sync', equalization_enabled=False, fs_hz=25000000.0):
        raise ValueError('Wrong received channel version')
    data = np.load(p2/'received.npz', allow_pickle=False)
    iq, ids = data['iq'], data['ids']
    splits = [read(path) for path in sorted((p2/'splits').glob('*.json'))]
    groups = group_splits(splits, contract['classes'], manifest['capsule_id'])
    first = next(iter(groups.values()))[0]
    with torch.no_grad():
        smoke = classifier_logits(model, cfg['method'], tensor(iq[first['support_indices'][:2]], device))
    if smoke.shape != (2, len(contract['classes'])) or not bool(torch.isfinite(smoke).all()):
        raise ValueError('Real checkpoint support-only smoke failed')
    write(out/'real_checkpoint_smoke.json', dict(passed=True, query_read=False, checkpoint=str(source/'last.pt')))
    # Phase1 completion survives any subsequent Phase2 technical failure.
    p1 = Path(cfg['p1_capsule']); p1m = read(p1/'manifest.json')
    if p1m['status'] != 'VALIDATED_ONCE' or p1m['classes'] != contract['classes'] or p1m['channel'] != 'residual/post_sync/noeq':
        raise ValueError('Phase1 class/channel mismatch')
    ix = np.load(p1/'index.npz', allow_pickle=False)
    if len(set(ix['ids'].tolist())) != len(ix['ids']): raise ValueError('Repeated Phase1 identity')
    predictions = {}
    for view in ('clean', 'satellite'):
        array = np.load(p1/(view+'.npy'), mmap_mode='r', allow_pickle=False)
        if array.shape != (len(ix['ids']),2,256): raise ValueError('Phase1 paired IQ shape mismatch')
        predictions[view] = np.asarray(infer(lambda x: classifier_logits(model, cfg['method'], x), array, device, cfg.get('batch_size',256), len(contract['classes'])), dtype=np.int64)
    np.savez(out/'phase1_predictions.npz', ids=ix['ids'], **predictions)
    write(out/'phase1_complete.json', dict(status='PREDICTIONS_COMPLETE', count=len(ix['ids']), truth_read=False))
    source_x = source_y = None
    if cfg['method'] in ADAPTER and cfg['method'] != 'feature_separation':
        source_x, source_y = source_replay(cfg, contract)
    base_metadata = torch.load(source/'base_state.pt', map_location=device, weights_only=False) if cfg['method'] in REGISTRATION else None
    log = FitLog(out); count = 0
    try:
        with (out/'phase2_predictions.jsonl').open('x', encoding='utf-8') as target, (out/'resources.jsonl').open('x', encoding='utf-8') as resource:
            def save(split, mode, scorer, state=None, paired_split=None):
                nonlocal count
                tic = time.perf_counter()
                predicted = infer(scorer, iq[split['query_indices']], device, cfg.get('batch_size',256), len(split['registered_classes']))
                if device.type == 'cuda': torch.cuda.synchronize(device)
                if any(i < 0 or i >= len(split['registered_classes']) for i in predicted):
                    raise ValueError('Prediction outside registered classes')
                value = dict(split_id=split['split_id'], capsule_id=manifest['capsule_id'], mode=mode,
                    receiver=split['receiver'], scenario=split['scenario'], k=split['k'], support_seed=split['support_seed'],
                    classes=split['registered_classes'], query_ids=ids[split['query_indices']].tolist(),
                    predicted_indices=predicted, paired_old_split_id=paired_split or split['split_id'],
                    C_inherits_B=mode.startswith('C'), query_fit=False)
                target.write(json.dumps(value)+'\n'); target.flush(); count += 1
                measured = dict(getattr(state, 'resources', getattr(state, 'resource', {}))) if state else {}
                resource.write(json.dumps(dict(measured, split_id=split['split_id'], mode=mode, inference_seconds=time.perf_counter()-tic,
                    hardware=torch.cuda.get_device_name(device) if device.type=='cuda' else 'cpu'), allow_nan=False)+'\n'); resource.flush()
            for key, rows in sorted(groups.items()):
                old = rows[0]; old_classes = [contract['classes'].index(c) for c in old['registered_classes']]
                sx = tensor(iq[old['support_indices']], device)
                sy = torch.tensor(old['support_labels'], device=device, dtype=torch.long)
                log.context = dict(receiver=key[0], scenario=key[1], k=key[2], support_seed=key[3], stage='B', split_id=old['split_id'])
                save(old, 'A_source_frozen', lambda x: classifier_logits(model,cfg['method'],x)[:,old_classes])
                # Every fitting random stream is fixed before prediction/scoring.
                seed = (int(cfg['model_seed']) + int(old['support_seed']) + int(old['k'])) % (2**31-1)
                torch.manual_seed(seed)
                if cfg['method'] in REGISTRATION:
                    rcfg = dict(cfg.get('registration', {}), seed=seed)
                    state = prepare_registration_state(cfg['method'], model, sx, sy, old_classes,
                        config=rcfg, base_metadata=base_metadata, logger=log,
                        metadata=dict(source_checkpoint=str(source/'last.pt'), source_contract=cfg['source_contract']))
                else:
                    acfg = dict(cfg.get('adaptation', {}), seed=seed, device=str(device), k=old['k'], logger=log)
                    state = fit_adaptation(ADAPTER[cfg['method']], model, sx,
                        torch.tensor([old_classes[y] for y in old['support_labels']], device=device), old_classes,
                        acfg, source_x=source_x, source_y=source_y,
                        metadata=dict(source_contract=cfg['source_contract'], source_checkpoint=str(source/'last.pt')))
                if tuple(state.classes) != tuple(old_classes): raise ValueError('B class order mismatch')
                save(old, 'B_old_support', state.score, state)
                for n in (2,5,10,20):
                    split = rows[n]
                    if cfg['method'] not in REGISTRATION and cfg['method'] != 'radionet_ada':
                        continue # Closed-set methods do not acquire invented native CIL mechanisms.
                    log.context = dict(log.context, stage='C', split_id=split['split_id'], new_classes=n)
                    selected = [i for i,y in zip(split['support_indices'],split['support_labels']) if y >= len(old_classes)]
                    ny = [y for y in split['support_labels'] if y >= len(old_classes)]
                    newids = list(range(len(contract['classes']),len(contract['classes'])+n))
                    torch.manual_seed(seed)
                    if cfg['method'] in REGISTRATION:
                        registered = register_new_classes(state, tensor(iq[selected],device),
                            torch.tensor(ny,device=device,dtype=torch.long), newids, config=rcfg, logger=log)
                        mode = 'C_native_registration' if cfg['method'] != 'protonet' else 'C_prototype_append_extension'
                    else:
                        if device.type=='cuda': torch.cuda.synchronize(device)
                        reg_start=time.perf_counter()
                        registered = copy.deepcopy(state)
                        with torch.no_grad():
                            nz = features(registered.model, tensor(iq[selected],device))
                        registered.classes += tuple(newids)
                        registered.support_features = torch.cat([registered.support_features,nz])
                        registered.support_labels = torch.cat([registered.support_labels,torch.tensor(ny,device=device)])
                        if device.type=='cuda': torch.cuda.synchronize(device)
                        registered.resources = dict(training_seconds=time.perf_counter()-reg_start, trainable_parameters=0,optimizer_steps=0,
                            resident_state_bytes=sum(v.numel()*v.element_size() for v in registered.model.state_dict().values())+
                                registered.support_features.numel()*registered.support_features.element_size()+registered.support_labels.numel()*registered.support_labels.element_size(),
                            extension='frozen ADA encoder, append new support to cosine distance-weighted kNN')
                        mode = 'C_knn_append_extension'
                    if tuple(registered.classes) != tuple(old_classes+newids): raise ValueError('C class order mismatch')
                    save(split, mode, registered.score, registered, old['split_id'])
                    del registered
                del state
                print(json.dumps(dict(task=key, predictions=count, status='TASK_COMPLETE')), flush=True)
        expected = len(groups)*(6 if cfg['method'] in REGISTRATION or cfg['method']=='radionet_ada' else 2)
        if count != expected: raise ValueError('Incomplete prediction matrix')
        write(out/'phase2_complete.json',dict(status='PREDICTIONS_COMPLETE', predictions=count,
            paired_old_tasks=len(groups), split_count=len(splits), capsule_id=manifest['capsule_id'], truth_read=False,
            native_C_supported=cfg['method'] in REGISTRATION, closed_set_C_NA=cfg['method'] not in REGISTRATION and cfg['method']!='radionet_ada'))
    finally:
        log.close()

def main():
    p=argparse.ArgumentParser(); p.add_argument('--config',required=True)
    run(read(p.parse_args().config))

if __name__ == '__main__': main()
