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
from experiments.pamf_lora_main.phase_closure import phase_closure_loss
from experiments.pamf_lora_main.preamble_noise import repeat_dispersion
from experiments.pamf_lora_main.received_quality import capture_features, fit_clean_geometry
from experiments.pamf_lora_main.repeated_edit import PREAMBLE
from experiments.pamf_lora_main.shared_gain_dura import SharedGainDura
from experiments.pamf_lora_main.w24_native_dura import capture_scale, iq_error
from experiments.pamf_restoration.models import complex_iq


STEPS, BATCH, CALIBRATION_COUNT = 6400, 8, 512
CLOSURE_STEPS, CLOSURE_WEIGHT = 7200, .25
ISOLATION = ('F_reads', 'V_reads', 'TX_reads', 'target_reads')


def source_data(rt, dataset, device):
    reserved_count = (2 if rt.get('observed_anchor') else 1) * CALIBRATION_COUNT + 2
    if dataset == 'lora':
        config, joint, _ = screen.lora_archives(rt)
        joint.activate(config)
        source = Path(config['source_root'])
        contract = joint.source_contract(source)
        data = joint.source_role(source, contract, 'U_s')[0]
        if len(data) != 25200:
            raise ValueError('Expected the registered LoRa source U_s')
        seed = fitting_plan(rt)[0]
        reserved = np.random.default_rng(seed + 3).choice(len(data), reserved_count, replace=False)
        ids = np.setdiff1d(np.arange(len(data)), reserved)

        def sample(rows):
            return tensor(np.ascontiguousarray(data[rows]).view(np.float32), device).view(torch.complex64)

    elif dataset == 'wisig':
        original, config = screen.wifi_context(rt)
        data, ids, reserved = original.core.source_iq(Path(config['source_root']))
        if len(ids) != 52604 or len(np.intersect1d(ids, reserved)) or len(reserved) < reserved_count:
            raise ValueError('Expected registered WiSig source fit and holdout IDs')

        def sample(rows):
            return complex_iq(tensor(data[rows], device))

    else:
        raise ValueError(f'Unknown source data adapter {dataset!r}')
    return ids, reserved[:2], reserved[2:reserved_count], sample


def aligned_waveform_error(estimate, clean):
    phase = (clean * estimate.conj()).mean(-1, keepdim=True).angle().detach()
    return iq_error(estimate * torch.polar(torch.ones_like(phase), phase), clean)


def objective(model, clean, observed, *, align_phase=False, closure_weight=0.):
    stages, action, variance, quality = model.restoration(observed)
    first, second = stages[-1].chunk(2, dim=0)
    scale = capture_scale(observed)
    noise = (observed / scale - action(clean / scale)).abs().square().mean(dim=-1)
    log_noise = (noise.detach() / 2).clamp_min(1e-5).log()
    impairment = (observed - clean).abs().square().mean(dim=-1) > .02 * scale.flatten().square()
    reference = clean
    waveform = iq_error
    if align_phase:
        reference = clean * (scale / capture_scale(clean))
        waveform = aligned_waveform_error
    detail = (.45 - closure_weight) * paired_detail_loss(first, second, reference.chunk(2, dim=0)[0])
    if closure_weight:
        detail = detail + closure_weight * phase_closure_loss(stages[-1], reference)
    return (.40 * waveform(stages[-1], reference) + .15 * waveform(stages[0], reference)
            + detail
            + .10 * noise.mean() + .02 * F.smooth_l1_loss(variance.log(), log_noise)
            + .10 * F.binary_cross_entropy_with_logits(quality, impairment.to(quality.dtype)))


def fitting_plan(rt):
    if rt.get('phase_closure'):
        return screen.CLOSURE_SEED, screen.CLOSURE_MODES[3:], (
            dict(retain_received=True), dict(retain_received=True),
        )
    if rt.get('preamble_noise'):
        return screen.PREAMBLE_NOISE_SEED, screen.PREAMBLE_NOISE_MODES[3:], (
            dict(retain_received=True), dict(retain_received=True),
        )
    if rt.get('observed_anchor'):
        return screen.ANCHOR_SEED, screen.ANCHOR_MODES[3:], (
            dict(retain_received=True), dict(retain_received=False),
        )
    return screen.SHARED_SEED, screen.SHARED_MODES[3:], (dict(packet_gain=False), dict(packet_gain=True))


def fit(rt, dataset, device):
    folder = Path(rt['run_root']) / ('fit-' + dataset)
    folder.mkdir()
    ids, smoke, calibration, sample = source_data(rt, dataset, device)
    seed, modes, model_options = fitting_plan(rt)
    observed_anchor = bool(rt.get('observed_anchor'))
    phase_closure = bool(rt.get('phase_closure'))
    steps = CLOSURE_STEPS if phase_closure else STEPS
    closure_weights = (CLOSURE_WEIGHT, 0.) if phase_closure else (0., 0.)
    torch.manual_seed(seed)
    models = [SharedGainDura(**options).to(device) for options in model_options]
    models[1].load_state_dict(models[0].state_dict(), strict=True)
    extra = {}
    if rt.get('preamble_noise'):
        with torch.no_grad():
            clean_dispersion = torch.cat([repeat_dispersion(sample(rows))
                                          for rows in np.array_split(calibration[:CALIBRATION_COUNT], 64)])
            floor = clean_dispersion.median(dim=0).values
        if not torch.isfinite(floor).all():
            raise ValueError('Expected finite clean repeated-preamble variation')
        models[0].clean_noise_floor = floor
        if phase_closure:
            models[1].clean_noise_floor = floor
        extra['preamble_clean_floor'] = floor.tolist()
    if phase_closure:
        extra['phase_closure_weights'] = dict(zip(modes, closure_weights, strict=True))
    optimizers = [torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4) for model in models]
    schedules = [torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps, eta_min=3e-5) for opt in optimizers]
    draws = np.random.default_rng(seed)
    seeds = (seed + 1, seed + 2)
    base_rng, nuisance_rng = [torch.Generator(device=device).manual_seed(seed) for seed in seeds]

    def views(rows):
        raw = sample(rows).repeat(2, 1)
        return add_packet_nuisance(*source_training_pair(raw, base_rng), nuisance_rng)

    batch = views(smoke)
    for model, closure_weight in zip(models, closure_weights, strict=True):
        active = (model.coarse[-1].weight, model.channel.weight, model.local_phase.weight,
                  model.gain[-1].weight, *(prior[-1].weight for prior in model.priors))
        loss = objective(model, *batch, align_phase=observed_anchor, closure_weight=closure_weight)
        for gradient in torch.autograd.grad(loss, active):
            if not torch.isfinite(gradient).all() or not gradient.norm() > 0:
                raise ValueError('An active restoration head has no finite gradient on real source IQ')
    write(folder/'smoke.json', dict(status='PASS', **dict.fromkeys(ISOLATION, 0)))
    base_rng.manual_seed(seeds[0])
    nuisance_rng.manual_seed(seeds[1])
    torch.cuda.reset_peak_memory_stats()
    start = perf_counter()
    for step in range(1, steps + 1):
        batch = views(draws.choice(ids, BATCH, replace=False))
        losses = []
        for model, optimizer, schedule, closure_weight in zip(models, optimizers, schedules, closure_weights, strict=True):
            optimizer.zero_grad(set_to_none=True)
            loss = objective(model, *batch, align_phase=observed_anchor, closure_weight=closure_weight)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            optimizer.step()
            schedule.step()
            losses.append(loss.detach())
        if step == 1 or step % 100 == 0:
            print(dataset, step, *(float(loss) for loss in losses), flush=True)

    clean_cuts = {}
    with torch.no_grad():
        if observed_anchor:
            observations = torch.cat([capture_features(sample(rows)) for rows in np.array_split(calibration, 64)])
            geometry = fit_clean_geometry(*observations.chunk(2))
            clean_cuts = dict.fromkeys(modes, geometry)
        else:
            for name, model in zip(modes, models, strict=True):
                quality = torch.cat([model.restoration(sample(rows))[3] for rows in np.array_split(calibration, 64)])
                if not torch.isfinite(quality).all():
                    raise ValueError('Nonfinite source clean quality')
                clean_cuts[name] = float(torch.quantile(quality, .995))
    with (folder/'model.pt').open('xb') as stream:
        torch.save({name: model.state_dict() for name, model in zip(modes, models, strict=True)}, stream)
    write(folder/'fit.json', dict(status='SOURCE_FROZEN', dataset=dataset, modes=list(modes),
          seed=seed, steps=steps, batch=BATCH, views_per_source_packet=2, source_training_ids=len(ids),
          clean_calibration_ids=len(calibration), clean_cuts=clean_cuts, seconds=perf_counter()-start,
          peak_cuda_bytes=torch.cuda.max_memory_allocated(), parameters=sum(p.numel() for p in models[0].parameters()),
          **dict.fromkeys(ISOLATION, 0), **extra))


def frozen(rt, dataset, device):
    folder = Path(rt['run_root']) / ('fit-' + dataset)
    seed, modes, model_options = fitting_plan(rt)
    calibration_count = (2 if rt.get('observed_anchor') else 1) * CALIBRATION_COUNT
    steps = CLOSURE_STEPS if rt.get('phase_closure') else STEPS
    proof = read(folder/'fit.json')
    if ((proof['status'], proof['dataset'], proof['seed'], proof['steps'], proof['clean_calibration_ids'], proof['modes'])
            != ('SOURCE_FROZEN', dataset, seed, steps, calibration_count, list(modes))
            or any(proof[key] for key in ISOLATION)):
        raise ValueError('Expected both lawful source-frozen arms')
    if rt.get('phase_closure') and proof.get('phase_closure_weights') != dict(zip(modes,(CLOSURE_WEIGHT,0.),strict=True)):
        raise ValueError('Expected frozen predeclared source detail loss weights')
    weights = torch.load(folder/'model.pt', map_location=device, weights_only=True)
    if rt.get('preamble_noise'):
        length = {'lora':8192, 'wisig':256}[dataset]
        floor = np.asarray(proof['preamble_clean_floor'])
        if floor.shape != (PREAMBLE[length][2],) or not np.isfinite(floor).all() or (floor < 0).any():
            raise ValueError('Invalid frozen clean preamble-dispersion reference')
        model_options[0]['clean_noise_floor'] = torch.as_tensor(floor, dtype=torch.float32, device=device)
        if rt.get('phase_closure'):
            model_options[1]['clean_noise_floor'] = model_options[0]['clean_noise_floor']
    result = []
    for name, options in zip(modes, model_options, strict=True):
        model = SharedGainDura(**options).to(device).eval().requires_grad_(False)
        model.load_state_dict(weights[name], strict=True)
        cutoff = proof['clean_cuts'][name]
        if rt.get('observed_anchor'):
            if (np.asarray(cutoff['center']).shape != (10,) or np.asarray(cutoff['precision']).shape != (10, 10)
                    or not all(np.isfinite(value).all() for value in cutoff.values()) or cutoff['cutoff'] < 1):
                raise ValueError('Expected fixed finite clean observation geometry')
        elif not np.isfinite(cutoff):
            raise ValueError('Expected fixed finite source clean cutoff')
        result.append((model, cutoff))
    return result
