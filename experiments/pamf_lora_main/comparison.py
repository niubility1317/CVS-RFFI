"""Supervised ResNet control, scratch two-stage DURA, and blind prediction."""
import math
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from experiments.cric_dura_comparison.io import read, tensor, write
from .data import CLASSES, SOURCE, TARGET, VIEWS
from .physical import rms_normalize
from .recognition import STEPS, reader_scope


def seed_all(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def lora_resnet():
    from experiments.standard_ce_baselines.model import ResNet
    model = ResNet((2, 2, 2, 2))
    model.classifier = nn.Linear(model.classifier.in_features, len(CLASSES))
    return model


def train_resnet(c):
    root = Path(c['output_root'])
    root.mkdir(parents=True, exist_ok=False)
    seed_all(c['model_seed'])
    src = Path(c['source_cache'])
    x = rms_normalize(tensor(np.load(src / 'L_s_x.npy'), 'cuda:0'))
    with np.load(src / 'L_s_metadata.npz', allow_pickle=False) as m:
        y = tensor(m['y'], 'cuda:0')
    x_v = rms_normalize(tensor(np.load(src / 'V_x.npy'), 'cuda:0'))
    with np.load(src / 'V_metadata.npz', allow_pickle=False) as m:
        y_v = tensor(m['y'], 'cuda:0')
    views = tensor(np.load(c['source_practical']), 'cuda:0')
    model = lora_resnet().cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50*STEPS, eta_min=1e-6)
    for epoch in range(1, 51):
        model.train()
        epoch_loss = torch.zeros((), device='cuda:0')
        for step in range(STEPS):
            idx = torch.randint(len(x), (128,), device='cuda:0')
            batch = x[idx].clone()
            mask = (torch.arange(128, device='cuda:0') + step) % 2 == 0
            batch[mask] = views[(idx[mask] + epoch + step) % (len(VIEWS)-1), idx[mask]]
            loss = F.cross_entropy(model(batch), y[idx])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            schedule.step()
            epoch_loss += loss.detach()
        model.eval()
        with torch.no_grad():
            correct = sum(int((model(b).argmax(1) == y).sum()) for b,y in
                          zip(x_v.split(256), y_v.split(256), strict=True))
        avg = float(epoch_loss) / STEPS
        if not math.isfinite(avg):
            raise FloatingPointError('Nonfinite ResNet training')
        print(f'RESNET_EPOCH {epoch} CE {avg:.6f} SOURCE_V {correct}/{len(y_v)}', flush=True)
    torch.save(model.state_dict(), root / 'resnet.pt')
    write(root / 'complete.json', dict(status='SOURCE_FROZEN', config=c, epoch=50, updates=50*STEPS,
                                      target_access=False, U_truth_read=False))


def dura_model():
    from experiments.cric_dura_comparison.g_lineage import skeleton
    return skeleton('r69')


def train_dura(seed, source_cache, source_channel, binding, out):
    """Fit the DURA two-stage structure directly on source IQ pairs."""
    from experiments.pamf_r44_pruned_refine.models import spectral_error
    out.mkdir(parents=True, exist_ok=False)
    seed_all(seed)
    # Files here only contain IQ, not the L/U metadata or any TX labels.
    ground = rms_normalize(tensor(np.concatenate([
        np.load(source_cache / f'{role}_x.npy', allow_pickle=False) for role in ('L_s', 'U_s')]), 'cuda:0'))
    received = tensor(np.load(source_channel, allow_pickle=False), 'cuda:0')
    if received.shape != ground.shape:
        raise ValueError('DURA matched source pair count differs')
    model = dura_model().cuda().train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20*math.ceil(len(ground)/128),
                                                         eta_min=2e-5)
    for epoch in range(1, 21):
        progress = torch.zeros((), device='cuda:0')
        count = 0
        for indices in torch.randperm(len(ground), device='cuda:0').split(128):
            clean, noisy = ground[indices], received[indices]
            raw, final = model.stage_outputs(noisy)[0]
            loss = F.mse_loss(final, clean) + .25*F.mse_loss(raw, clean) + .05*spectral_error(final, clean).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5., error_if_nonfinite=True)
            optimizer.step()
            schedule.step()
            progress += loss.detach()*len(indices)
            count += len(indices)
        avg = float(progress) / count
        if not math.isfinite(avg):
            raise FloatingPointError('Nonfinite DURA training')
        print(f'DURA_LORA_EPOCH {epoch} LOSS {avg:.6f} SOURCE_IQ {count}', flush=True)
    torch.save(model.state_dict(), out / 'dura_lora.pt')
    write(out / 'complete.json', dict(status='DURA_LORA_SOURCE_FROZEN', model_seed=seed, epochs=20,
          training='two_stage_DURA_scratch_LoRa', source_contract=binding, TX_labels_read=0, F_reads=0,
          target_reads=0, architecture_parameters=sum(p.numel() for p in model.parameters())))


def load_reader(folder, contract):
    from experiments.cric_dura_comparison.frozen_inference import load_F
    state = read(folder / 'complete.json')
    c = state['config']
    if (state['status'] != 'SOURCE_FROZEN' or state['epoch'] != 50 or state['target_access']
            or read(c['source_contract']) != contract):
        raise ValueError('LoRa frozen same-contract F is required')
    if c['lora_method'] == 'resnet18':
        model = lora_resnet()
        model.load_state_dict(torch.load(folder / 'resnet.pt', map_location='cpu'), strict=True)
        return model.cuda().eval()
    with reader_scope(c):
        return load_F(folder, contract, 'cuda:0', require_no_unlabeled=not c['U_use'])[0]


@torch.no_grad()
def predict(folder, target, target_channel, out, contract_path, dura_folder=None, *, reader=None, seed=None):
    contract = read(contract_path)
    model = load_reader(folder, contract) if reader is None else reader
    modes = {'raw': None}
    if dura_folder:
        state = read(dura_folder / 'complete.json')
        model_seed = read(folder / 'complete.json')['config']['model_seed'] if seed is None else seed
        if (state['status'] != 'DURA_LORA_SOURCE_FROZEN' or state['source_contract'] != str(contract_path)
                or state['TX_labels_read'] or state['F_reads'] or state['target_reads']
                or state['model_seed'] != model_seed):
            raise ValueError('DURA-LoRa and PAMF source training bindings differ')
        restorer = dura_model()
        restorer.load_state_dict(torch.load(dura_folder/'dura_lora.pt', map_location='cpu'), strict=True)
        modes['dura_lora'] = restorer.cuda().eval()
    target_contract = read(target / 'contract.json')
    if (target_contract['receivers'] != list(TARGET) or target_contract['truth_read']
            or set(contract['receiver_names']) & set(TARGET)):
        raise ValueError('Target must be held-out blind IQ')
    out.mkdir(parents=True, exist_ok=False)
    counts = {}
    for receiver in TARGET:
        channel = np.load(target_channel / f'{receiver}.npy', mmap_mode='r', allow_pickle=False)
        for i, view in enumerate(VIEWS):
            if i == 0:
                raw = np.load(target / f'{receiver}.npy', allow_pickle=False)
            else:
                raw = channel[i-1]
            for mode, restorer in modes.items():
                result = []
                for inputs in tensor(raw).split(128):
                    x = inputs.cuda()
                    if i == 0:
                        x = rms_normalize(x)
                    if restorer is not None:
                        x = restorer.stage_outputs(x)[0][-1]
                    logits = model(x)
                    if not torch.isfinite(logits).all():
                        raise ValueError('Nonfinite blind target prediction')
                    result.extend(logits.argmax(dim=1).cpu().tolist())
                counts[receiver, view, mode] = len(result)
                np.save(out / f'{receiver}_{view}_{mode}.npy', np.asarray(result, dtype=np.int16))
        print('LORA_PREDICTED_RECEIVER', receiver, len(VIEWS), flush=True)
    if set(counts.values()) != {2000} or len(counts) != len(TARGET)*len(VIEWS)*len(modes):
        raise ValueError('Complete target prediction matrix required')
    write(out / 'complete.json', dict(status='PREDICTIONS_FROZEN', source=str(folder), modes=list(modes),
          views=list(VIEWS), receivers=list(TARGET), per_receiver=2000, target_truth_read=False))
