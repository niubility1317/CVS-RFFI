"""Explicit source contract and label-blind practical residual augmentation."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


def physical_id(item):
    raw = f'{item.tx_i}/{item.rx_i}/{item.day_i}/{item.eq_i}/{item.sig_i}'
    return hashlib.sha256(raw.encode()).hexdigest()[:24]


class ContractRoleDataset(Dataset):
    def __init__(self, base, indices, role):
        self.base, self.indices, self.role = base, list(indices), role

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        k = self.indices[index]
        x, y, d, meta = self.base[k]
        visible = self.role != 'U_s'
        return dict(iq=x, label=int(y) if visible else -1,
                    domain=int(d), receiver=int(meta['rx_i']), day=int(meta['day_i']),
                    sig_i=int(meta['sig_i']),
                    meta=dict(sample_id=physical_id(self.base.index[k]),
                              session_id=f"rx:{meta['rx']}/day:{meta['day']}",
                              rx_i=int(meta['rx_i']), day_i=int(meta['day_i'])))


def match_roles(base, contract):
    expected = contract['role_ids']
    if set(expected) != {'L_s', 'U_s', 'V'}:
        raise ValueError('Expected exactly L_s/U_s/V')
    owner = {}
    for role, ids in expected.items():
        for sid in ids:
            if sid in owner:
                raise ValueError('Repeated physical source ID')
            owner[sid] = role
    parts = {r: [] for r in expected}
    actual = set()
    for k, item in enumerate(base.index):
        sid = physical_id(item)
        if sid not in owner or sid in actual:
            raise ValueError('Source contract physical IDs differ from dataset')
        actual.add(sid)
        parts[owner[sid]].append(k)
    if actual != set(owner):
        raise ValueError('Source contract contains missing physical IDs')
    return parts


def build_contract_split(args):
    from dataset_wisig import load_wisig_compact_pkl, WiSigCompactDataset
    from baselines.common.cvs_data import CVSSplit, parse_csv_indices

    if not args.source_contract or not args.use_source_ssl_split:
        raise ValueError('source_only requires source_contract and use_source_ssl_split')
    contract = json.loads(Path(args.source_contract).read_text(encoding='utf-8'))
    rxs, days = contract['source_rxs'], contract['source_days']
    if parse_csv_indices(args.wisig_train_rxs) != rxs or parse_csv_indices(args.wisig_train_days) != days:
        raise ValueError('Requested source RX/day differs from contract')
    if set(rxs) & set(parse_csv_indices(args.wisig_test_rxs) or []):
        raise ValueError('Source and target receivers overlap')
    if [args.wisig_labeled_ratio, args.wisig_unlabeled_ratio, args.wisig_source_val_ratio] != contract['ratios']:
        raise ValueError('Requested role ratios differ from contract')
    if int(args.wisig_split_seed) != int(contract['split_seed']):
        raise ValueError('Requested split seed differs from contract')
    raw = load_wisig_compact_pkl(args.wisig_pkl)
    if len(raw['tx_list']) != contract['num_classes']:
        raise ValueError('Source class count differs from contract')
    base = WiSigCompactDataset(raw, out_len=args.wisig_out_len, crop_mode='center',
                               normalize=args.wisig_rms_normalize, equalized=int(args.wisig_equalized),
                               rx_keep=rxs, day_keep=days, domain=args.wisig_domain)
    parts = match_roles(base, contract)
    roles = {r: ContractRoleDataset(base, idx, r) for r, idx in parts.items()}
    info = dict(train_rxs_idx=rxs, train_rxs_label=[str(raw['rx_list'][i]) for i in rxs],
                train_days_label=[str(raw['capture_date_list'][i]) for i in days],
                counts={r: len(ds) for r, ds in roles.items()}, source_only=True,
                source_contract=str(args.source_contract), physical_roles='EXACT_MATCH')
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / 'source_contract.json').write_text(json.dumps(dict(contract,
        physical_roles='EXACT_MATCH', dataset_path=str(args.wisig_pkl),
        classes=[str(c) for c in raw['tx_list']], equalized=int(args.wisig_equalized),
        out_len=int(args.wisig_out_len), normalize=bool(args.wisig_rms_normalize)), indent=2), encoding='utf-8')
    # Empty targets are structural, not dummy source validation aliases.
    return CVSSplit(roles['L_s'], roles['U_s'], roles['V'], [], {}, {}, info,
                    len(raw['tx_list']), len(raw['rx_list']), args.wisig_out_len)


class PracticalResidualAugment:
    def __init__(self, *, fs_hz, seed=2027, receiver_seed=2027):
        from leo_practical.channel import Config
        self.seed, self.receiver_seed, self.epoch = int(seed), int(receiver_seed), 1
        self.configs = {name: Config(fs_hz=float(fs_hz), scenario=name,
            processing_route='residual', mode='post_sync', equalization_enabled=False,
            input_processing_state='WiSig_equalized1_center256_unit_rms')
            for name in ('practical_high', 'practical_mid', 'practical_low_urban')}

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def __call__(self, iq, *, metadata=None):
        from leo_practical.batch import apply_leo_practical_channel_batch
        from leo_practical.channel import stable_seed
        if metadata is None or len(metadata) != len(iq):
            raise ValueError('Practical augmentation requires sample and receiver-session metadata')
        ids = [m['sample_id'] for m in metadata]
        if len(set(ids)) != len(ids):
            raise ValueError('Repeated sample IDs in training batch')
        names, probability = (('practical_high',), .30) if self.epoch <= 40 else (
            (('practical_mid', 'practical_low_urban'), .60) if self.epoch <= 90 else
            (tuple(self.configs), .80))
        # List conversion avoids the Torch 2.1 / NumPy 2 ABI bridge on N607.
        array = np.asarray(iq.detach().cpu().tolist(), dtype=np.float32)
        result = array.copy()
        for name in names:
            chosen = []
            for i, sid in enumerate(ids):
                rng = np.random.default_rng(stable_seed(self.seed, 'train-selection', self.epoch, sid))
                if rng.random() < probability and names[int(rng.integers(len(names)))] == name:
                    chosen.append(i)
            if not chosen:
                continue
            y, _, _ = apply_leo_practical_channel_batch(
                array[chosen], self.configs[name], seed=self.seed,
                sample_ids=[ids[i] for i in chosen],
                session_ids=[metadata[i]['session_id'] for i in chosen],
                realization_namespace=f'phase1/train/epoch-{self.epoch}',
                receiver_seed=self.receiver_seed, return_meta=False)
            result[chosen] = y
        if not np.isfinite(result).all():
            raise ValueError('Nonfinite practical channel output')
        return torch.tensor(result.tolist(), device=iq.device, dtype=iq.dtype)


def apply_training_view(augment, iq, batch):
    if isinstance(augment, PracticalResidualAugment):
        return augment(iq, metadata=batch['meta'])
    return augment(iq)
