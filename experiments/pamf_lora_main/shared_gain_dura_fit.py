"""Fit and freeze the same recovery architecture on each lawful source U_s."""

from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from torch.nn import functional as F

from experiments.cric_dura_comparison.io import read, tensor, write
from experiments.pamf_lora_main import gain_dura_screen as screen
from experiments.pamf_lora_main.gain_dura_nuisance import add_packet_nuisance
from experiments.pamf_lora_main.gain_dura_stage_fit import paired_detail_loss, source_training_pair
from experiments.pamf_lora_main.shared_gain_dura import SharedGainDura
from experiments.pamf_lora_main.w24_native_dura import capture_scale, iq_error
from experiments.pamf_restoration.models import complex_iq


STEPS, BATCH, CALIBRATION_COUNT = 6400, 8, 512
ISOLATION = ('F_reads', 'V_reads', 'TX_reads', 'target_reads')


def source_data(rt, dataset, device):
    if dataset == 'lora':
        config, joint, _ = screen.lora_archives(rt)
        joint.activate(config)
        source = Path(config['source_root'])
        contract = joint.source_contract(source)
        data = joint.source_role(source, contract, 'U_s')[0]
        if len(data) != 25200:
            raise ValueError('Expected the registered LoRa source U_s')
        reserved = np.random.default_rng(screen.SHARED_SEED + 3).choice(len(data), CALIBRATION_COUNT + 2, replace=False)
        ids = np.setdiff1d(np.arange(len(data)), reserved)

        def sample(rows):
            return tensor(np.ascontiguousarray(data[rows]).view(np.float32), device).view(torch.complex64)

    elif dataset == 'wisig':
        original, config = screen.wifi_context(rt)
        data, ids, reserved = original.core.source_iq(Path(config['source_root']))
        if len(ids) != 52604 or len(np.intersect1d(ids, reserved)) or len(reserved) < CALIBRATION_COUNT + 2:
            raise ValueError('Expected registered WiSig source fit and holdout IDs')

        def sample(rows):
            return complex_iq(tensor(data[rows], device))

    else:
        raise ValueError(f'Unknown source data adapter {dataset!r}')
    return ids, reserved[:2], reserved[2:CALIBRATION_COUNT + 2], sample


def objective(model, clean, observed):
    stages, action, variance, quality = model.restoration(observed)
    first, second = stages[-1].chunk(2, dim=0)
    scale = capture_scale(observed)
    noise = (observed / scale - action(clean / scale)).abs().square().mean(dim=-1)
    log_noise = (noise.detach() / 2).clamp_min(1e-5).log()
    impairment = (observed - clean).abs().square().mean(dim=-1) > .02 * scale.flatten().square()
    return (.40 * iq_error(stages[-1], clean) + .15 * iq_error(stages[0], clean)
            + .45 * paired_detail_loss(first, second, clean.chunk(2, dim=0)[0])
            + .10 * noise.mean() + .02 * F.smooth_l1_loss(variance.log(), log_noise)
            + .10 * F.binary_cross_entropy_with_logits(quality, impairment.to(quality.dtype)))


def fit(rt, dataset, device):
    folder = Path(rt['run_root']) / ('fit-' + dataset)
    folder.mkdir()
    ids, smoke, calibration, sample = source_data(rt, dataset, device)
    torch.manual_seed(screen.SHARED_SEED)
    modes = screen.SHARED_MODES[3:]
    models = [SharedGainDura(packet_gain=packet_gain).to(device) for packet_gain in (False, True)]
    models[1].load_state_dict(models[0].state_dict(), strict=True)
    optimizers = [torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4) for model in models]
    schedules = [torch.optim.lr_scheduler.CosineAnnealingLR(opt, STEPS, eta_min=3e-5) for opt in optimizers]
    draws = np.random.default_rng(screen.SHARED_SEED)
    seeds = (screen.SHARED_SEED + 1, screen.SHARED_SEED + 2)
    base_rng, nuisance_rng = [torch.Generator(device=device).manual_seed(seed) for seed in seeds]

    def views(rows):
        raw = sample(rows).repeat(2, 1)
        return add_packet_nuisance(*source_training_pair(raw, base_rng), nuisance_rng)

    batch = views(smoke)
    for model in models:
        active = (model.coarse[-1].weight, model.channel.weight, model.local_phase.weight,
                  model.gain[-1].weight, *(prior[-1].weight for prior in model.priors))
        for gradient in torch.autograd.grad(objective(model, *batch), active):
            if not torch.isfinite(gradient).all() or not gradient.norm() > 0:
                raise ValueError('An active restoration head has no finite gradient on real source IQ')
    write(folder/'smoke.json', dict(status='PASS', **dict.fromkeys(ISOLATION, 0)))
    base_rng.manual_seed(seeds[0])
    nuisance_rng.manual_seed(seeds[1])
    torch.cuda.reset_peak_memory_stats()
    start = perf_counter()
    for step in range(1, STEPS + 1):
        batch = views(draws.choice(ids, BATCH, replace=False))
        losses = []
        for model, optimizer, schedule in zip(models, optimizers, schedules, strict=True):
            optimizer.zero_grad(set_to_none=True)
            loss = objective(model, *batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            optimizer.step()
            schedule.step()
            losses.append(loss.detach())
        if step == 1 or step % 100 == 0:
            print(dataset, step, *(float(loss) for loss in losses), flush=True)

    clean_cuts = {}
    with torch.no_grad():
        for name, model in zip(modes, models, strict=True):
            quality = torch.cat([model.restoration(sample(rows))[3] for rows in np.array_split(calibration, 64)])
            if not torch.isfinite(quality).all():
                raise ValueError('Nonfinite source clean quality')
            clean_cuts[name] = float(torch.quantile(quality, .995))
    with (folder/'model.pt').open('xb') as stream:
        torch.save({name: model.state_dict() for name, model in zip(modes, models, strict=True)}, stream)
    write(folder/'fit.json', dict(status='SOURCE_FROZEN', dataset=dataset, modes=list(modes),
          seed=screen.SHARED_SEED, steps=STEPS, batch=BATCH, views_per_source_packet=2, source_training_ids=len(ids),
          clean_calibration_ids=len(calibration), clean_cuts=clean_cuts, seconds=perf_counter()-start,
          peak_cuda_bytes=torch.cuda.max_memory_allocated(), parameters=sum(p.numel() for p in models[0].parameters()),
          **dict.fromkeys(ISOLATION, 0)))


def frozen(rt, dataset, device):
    folder = Path(rt['run_root']) / ('fit-' + dataset)
    modes = screen.SHARED_MODES[3:]
    proof = read(folder/'fit.json')
    if ((proof['status'], proof['dataset'], proof['seed'], proof['steps'], proof['clean_calibration_ids'], proof['modes'])
            != ('SOURCE_FROZEN', dataset, screen.SHARED_SEED, STEPS, CALIBRATION_COUNT, list(modes))
            or any(proof[key] for key in ISOLATION)):
        raise ValueError('Expected both lawful source-frozen arms')
    weights = torch.load(folder/'model.pt', map_location=device, weights_only=True)
    result = []
    for name, packet_gain in zip(modes, (False, True), strict=True):
        model = SharedGainDura(packet_gain=packet_gain).to(device).eval().requires_grad_(False)
        model.load_state_dict(weights[name], strict=True)
        cutoff = proof['clean_cuts'][name]
        if not np.isfinite(cutoff):
            raise ValueError('Expected fixed finite source clean cutoff')
        result.append((model, cutoff))
    return result
