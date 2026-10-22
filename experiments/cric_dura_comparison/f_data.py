"""Contract-bound blind source loaders; no dataset or target path is accepted."""
from pathlib import Path
import numpy as np
import torch
from .io import read, tensor
from .source_iq import aligned_source_iq


class SourceDataset(torch.utils.data.Dataset):
    def __init__(self, root, role, contract):
        if role not in ('L_s', 'U_s', 'V'):
            raise ValueError('Only source roles allowed')
        root = Path(root)
        with np.load(root / (role + '_metadata.npz'), allow_pickle=False) as values:
            names = set(values.files)
            allowed = {'rx_i', 'day_i', 'eq_i', 'sig_i', 'base_index', 'domain', 'ids'}
            if role != 'U_s':
                allowed.add('y')
            if names != allowed:
                raise ValueError('U truth or missing physical metadata: ' + repr(names))
            self.metadata = {k: values[k].copy() for k in names}
        if list(self.metadata['ids']) != contract['role_ids'][role]:
            raise ValueError('Source role physical order differs')
        x,self.iq_binding = aligned_source_iq(root,role,self.metadata['ids'])
        if x.shape != (contract['counts'][role], 2, 256):
            raise ValueError('Source role count differs')
        self.x = tensor(x)
        self.role = role

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        meta = {k: int(self.metadata[k][i]) for k in ('rx_i', 'day_i', 'eq_i', 'sig_i', 'base_index')}
        meta['tx_label_visible'] = self.role != 'U_s'
        label = -1 if self.role == 'U_s' else int(self.metadata['y'][i])
        return self.x[i], label, int(self.metadata['domain'][i]), meta


class FixedEpochLoader:
    """Cycle native shuffled batches to 222 steps, retaining partial U batches."""
    def __init__(self, dataset, batch, seed, steps=222, drop_last=False):
        self.dataset = dataset
        self.batch_size = batch
        self.steps = steps
        self.actual_examples = 0
        self.loader = torch.utils.data.DataLoader(dataset, batch_size=batch, shuffle=True,
            generator=torch.Generator().manual_seed(seed), drop_last=drop_last,
            num_workers=0, pin_memory=False)
        if not len(self.loader):
            raise ValueError('Empty source loader')

    def __len__(self):
        return self.steps

    def __iter__(self):
        it = iter(self.loader)
        for _ in range(self.steps):
            try:
                value = next(it)
            except StopIteration:
                it = iter(self.loader)
                value = next(it)
            self.actual_examples += len(value[0])
            yield value


def context(c):
    if any(c.get(k) for k in ('target_truth', 'truth', 'dataset_path', 'target_inputs', 'views_root')):
        raise ValueError('Source worker rejects dataset/target inputs')
    contract = read(c['source_contract'])
    if contract['source_rxs'] != c['source_rxs'] or contract['source_days'] != c['source_days']:
        raise ValueError('Source contract mismatch')
    root = c['source_cache']
    labeled = SourceDataset(root, 'L_s', contract)
    validation = SourceDataset(root, 'V', contract)
    # Disabled U is never opened or iterated, including strict supervised runs.
    unlabeled = SourceDataset(root, 'U_s', contract) if c['regime'] != 'matched_sat_supervised' and c.get('ablation') not in ('F15','F20') else None
    train = FixedEpochLoader(labeled, 128, c['model_seed'], drop_last=True)
    unlab = FixedEpochLoader(unlabeled, 256 if c['regime'] == 'native_cric' else 128,
        c['model_seed'], drop_last=False) if unlabeled is not None else None
    val = torch.utils.data.DataLoader(validation, batch_size=256, shuffle=False, num_workers=0)
    count_l, count_u, count_v = (contract['counts'][k] for k in ('L_s', 'U_s', 'V'))
    domains = len(c['source_rxs']) * len(c['source_days'])
    return dict(train_loader=train, unlabeled_loader=unlab, val_loader=val,
        named_test_loaders={}, domain_label_map={}, num_domains=domains,
        input_len=256, class_id_to_tx=contract['classes'],
        split_info=dict(mode='contract_bound_existing_roles', labeled_size=count_l,
            unlabeled_size=count_u if unlabeled is not None else 0, source_val_size=count_v,
            rho_label=count_l / (count_l + count_u), named_test_meta={}), source_contract=contract)
