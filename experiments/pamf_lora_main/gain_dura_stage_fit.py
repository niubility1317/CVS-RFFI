"""Fit both matched in-stage fusion arms on lawful, unlabeled source IQ."""
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from experiments.cric_dura_comparison.io import read, tensor, write
from experiments.pamf_lora_main import gain_dura_screen as screen
from experiments.pamf_lora_main.gain_dura_stages import ComplexPriorResidual, StageGain
from experiments.pamf_lora_main.w24_native_dura import capture_scale, iq_error, source_pair
from experiments.pamf_restoration.models import complex_iq, pair


SEED, STEPS, BATCH = 2026113067, 3200, 8
COMPLEX_STEPS = 4800
ARMS = ('stage_frequency', 'stage_packet')


def source_training_pair(raw, rng):
    clean, observed, _, _ = source_pair(raw, rng)
    options = dict(device=observed.device, generator=rng, dtype=observed.real.dtype)
    draw = torch.rand((len(observed), 3), **options)
    # Additional generic colored, bursty receiver noise, independent of the test generator.
    n = torch.randn(observed.shape, device=observed.device, generator=rng, dtype=observed.dtype)
    filtered = n + .65 * torch.roll(n, 1, dims=-1) - .25 * torch.roll(n, 3, dims=-1)
    time = torch.linspace(0, 1, observed.shape[-1], device=observed.device)
    envelope = torch.exp(-.5 * ((time - draw[:, 1:2]) / (.02 + .12 * draw[:, 2:3])).square())
    stress = (draw[:, :1] < .35) * (.15 + .6 * draw[:, 2:3]) * envelope
    recorded = observed + stress * filtered / capture_scale(filtered)
    scale = capture_scale(recorded)
    return clean / scale, recorded / scale


def phase_sensitive_loss(stages, clean):
    out = stages[-1]
    error = torch.fft.fft(out - clean, norm='ortho').abs().square()
    denom = .03 + torch.fft.fft(clean, norm='ortho').abs().square()
    weak_spectrum = (error / denom).log1p().mean()
    return (iq_error(out, clean) + .2 * iq_error(out.diff(), clean.diff())
            + .15 * weak_spectrum + .25 * iq_error(stages[0], clean))


def packet_spectra(wave, reference):
    """Use one common phase gauge per packet so chirp/block differences survive."""
    wave = wave / capture_scale(wave)
    with torch.no_grad():
        phase = (reference * wave.conj()).sum(-1, keepdim=True).angle()
        align = torch.polar(torch.ones_like(phase), phase)
    return torch.fft.fft((wave * align).unflatten(-1, (8, -1)), norm='ortho')


def paired_detail_loss(first, second, clean):
    reference = clean / capture_scale(clean)
    clean_spectra = packet_spectra(reference, reference)
    scale = .03 + clean_spectra.abs().square().mean(dim=1, keepdim=True)
    differences = clean_spectra.diff(dim=1)
    diff_scale = .03 + differences.abs().square().mean(dim=1, keepdim=True)
    first = packet_spectra(first, reference)
    second = packet_spectra(second, reference)

    def distance(a, b):
        error = ((a - b).abs().square() / scale).log1p().mean()
        delta = ((a.diff(dim=1) - b.diff(dim=1)).abs().square() / diff_scale).log1p().mean()
        return .5 * (error + delta)

    anchored = .5 * (distance(first, clean_spectra) + distance(second, clean_spectra))
    return anchored + .35 * distance(first, second)


def fit_plan(rt):
    if rt.get('drift_from'):
        return screen.DRIFT_SEED, screen.DRIFT_MODES[3:]
    if rt.get('train_bound_from'):
        return screen.TRAIN_BOUND_SEED, screen.TRAIN_BOUND_MODES[3:]
    if rt.get('complex_from'):
        return screen.COMPLEX_SEED, screen.COMPLEX_MODES[3:]
    if rt.get('paired_from'):
        return screen.PAIRED_SEED, screen.PAIRED_MODES[3:]
    return SEED, ARMS


def make_branch(name):
    correction = None
    if name in (*screen.COMPLEX_MODES[3:], *screen.TRAIN_BOUND_MODES[3:], *screen.DRIFT_MODES[3:]):
        correction = ComplexPriorResidual(aligned_only=name == screen.COMPLEX_MODES[-1])
    return StageGain(packet_only=name == ARMS[1], correction=correction)


def training_components(rt, dataset, device):
    if dataset == 'lora':
        from experiments.pamf_lora_main.lora_dura_stage2_reference import load_frozen_stage2

        config, joint, native = screen.lora_archives(rt)
        joint.activate(config)
        source = Path(config['source_root'])
        contract = joint.source_contract(source)
        data = joint.source_role(source, contract, 'U_s')[0]
        if len(data) != 25200:
            raise ValueError('Expected approved LoRa source U_s')
        dura = load_frozen_stage2(Path(rt['lora_stage2']), device)
        joint.load_restorer(*joint.training_runtime(config), contract, device)
        gain = screen.load_lora_gain(config, native, contract, device)

        def sample(rows):
            packets = np.ascontiguousarray(data[rows]).view(np.float32)
            return tensor(packets, device).view(torch.complex64)

        return np.arange(2, len(data)), np.arange(2), sample, dura, gain

    original, config = screen.wifi_context(rt)
    data, fit_rows, reserved = original.core.source_iq(Path(config['source_root']))
    if len(fit_rows) != 52604 or len(np.intersect1d(fit_rows, reserved)):
        raise ValueError('Expected approved WiSig U_s fit mask and reserved source IQ')
    engine = original.tri.core.original_r69(Path(config['parent_root']))
    dura = engine.pipeline.model
    gain = screen.load_wifi_gain(rt, device)

    def sample(rows):
        return complex_iq(tensor(data[rows], device))

    return fit_rows, reserved, sample, dura, lambda raw: gain(raw)[-1]


def fit(rt, dataset, device):
    out = Path(rt['run_root']) / ('fit-' + dataset)
    out.mkdir()
    ids, reserved, sample, dura, gain = training_components(rt, dataset, device)
    dura.eval().requires_grad_(False)
    seed, arms = fit_plan(rt)
    paired_run = bool(rt.get('paired_from'))
    complex_run = bool(rt.get('complex_from'))
    train_bound = bool(rt.get('train_bound_from'))
    drift_run = bool(rt.get('drift_from'))
    curve = None
    if train_bound:
        from experiments.pamf_lora_main.gain_dura_variance import load_curve, project_reference_increment

        curve = load_curve(dict(rt, run_root=rt['train_bound_from']), dataset)
    steps = COMPLEX_STEPS if complex_run else STEPS
    torch.manual_seed(seed)
    branches = [make_branch(mode).to(device) for mode in arms]
    branches[1].load_state_dict(branches[0].state_dict())
    opts = [torch.optim.AdamW(item.parameters(), lr=3e-4, weight_decay=1e-4) for item in branches]
    schedules = [torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps, eta_min=3e-5) for opt in opts]
    rng = torch.Generator(device=device).manual_seed(seed + 1)
    if drift_run:
        from experiments.pamf_lora_main.gain_dura_nuisance import add_packet_nuisance

        nuisance_rng = torch.Generator(device=device).manual_seed(seed + 2)
    draws = np.random.default_rng(seed)

    def view(raw, source_pair_iq=None):
        with torch.no_grad():
            clean, observed = source_training_pair(raw, rng) if source_pair_iq is None else source_pair_iq
            native_scale = capture_scale(raw)
            raw_observed = observed * native_scale
            if dataset == 'lora':
                real_prior = gain(raw_observed) / native_scale
            else:
                # The archived WiSig DURA checkpoint uses the native cache units.
                observed, clean = raw_observed, clean * native_scale
                real_prior = gain(observed)
            received = pair(observed)
            context = dura.estimate_context(received)
        return received, real_prior, context, clean

    def objectives(raw):
        if complex_run:
            repeated = raw.repeat(2, 1)
            if drift_run:
                base = source_training_pair(repeated, rng)
                enhanced = add_packet_nuisance(*base, nuisance_rng)
                contexts = (view(repeated, enhanced), view(repeated, base))
            else:
                common = view(repeated)
                contexts = (common, common)
            losses = []
            for name, model, (received, prior, context, clean) in zip(arms, branches, contexts, strict=True):
                observed = complex_iq(received)
                reference = None
                if train_bound:
                    with torch.no_grad():
                        reference = prior if dataset == 'lora' else complex_iq(dura.stage_outputs(received, context)[0][-1])
                stages = model(received, prior, context, dura.prior, reference=reference)
                if train_bound and not drift_run and name == arms[0]:
                    stages = [project_reference_increment(observed, reference, z, context[2], curve) for z in stages]
                first, second = stages[-1].chunk(2, dim=0)
                losses.append(.20 * phase_sensitive_loss(stages, clean) + .80 * paired_detail_loss(first, second, raw))
            return losses
        views = [view(raw) for _ in range(2 if paired_run else 1)]
        result = []
        for name, item in zip(arms, branches, strict=True):
            outputs = [item(received, prior, context, dura.prior) for received, prior, context, _ in views]
            bulk = torch.stack([phase_sensitive_loss(stages, data[-1])
                                for stages, data in zip(outputs, views, strict=True)]).mean()
            if paired_run and name == arms[0]:
                bulk = .20 * bulk + .80 * paired_detail_loss(outputs[0][-1], outputs[1][-1], raw)
            result.append(bulk)
        return result

    for item, objective in zip(branches, objectives(sample(reserved[:2])), strict=True):
        heads = [item.network[-1].weight]
        if item.correction is not None:
            heads.extend((item.correction.temporal[-1].weight, item.correction.spectral[-1].weight))
        gradients = torch.autograd.grad(objective, heads)
        if not all(bool(torch.isfinite(g).all() and g.norm() > 0) for g in gradients):
            raise ValueError('Real source fit smoke did not reach every active head')
    write(out/'smoke.json', dict(status='PASS', target_reads=0, F_reads=0, V_reads=0, TX_reads=0))
    rng.manual_seed(seed + 1)
    if drift_run:
        nuisance_rng.manual_seed(seed + 2)
    torch.cuda.reset_peak_memory_stats()
    start = perf_counter()
    for step in range(1, steps + 1):
        raw = sample(draws.choice(ids, BATCH, replace=False))
        losses = objectives(raw)
        for item, optimizer, schedule, loss in zip(branches, opts, schedules, losses, strict=True):
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(item.parameters(), 1., error_if_nonfinite=True)
            optimizer.step()
            schedule.step()
        if step == 1 or step % 100 == 0:
            print(dataset, step, *(float(loss.detach()) for loss in losses), flush=True)

    torch.cuda.synchronize()
    with (out/'gates.pt').open('xb') as stream:
        torch.save({name: item.state_dict() for name, item in zip(arms, branches, strict=True)}, stream)
    write(out/'fit.json', dict(status='SOURCE_FROZEN', dataset=dataset, seed=seed, steps=steps,
          batch=BATCH, arms=list(arms), source_training_ids=len(ids), views_per_source_packet=2 if paired_run or complex_run else 1,
          parameters=sum(p.numel() for p in branches[0].parameters()), seconds=perf_counter()-start,
          peak_cuda_bytes=torch.cuda.max_memory_allocated(), F_reads=0, V_reads=0, TX_reads=0, target_reads=0))


def load_gates(rt, dataset, device):
    root = Path(rt['run_root']) / ('fit-' + dataset)
    proof = read(root/'fit.json')
    seed, arms = fit_plan(rt)
    steps = COMPLEX_STEPS if rt.get('complex_from') else STEPS
    if ((proof['status'], proof['dataset'], proof['seed'], proof['steps'], proof['arms'])
            != ('SOURCE_FROZEN', dataset, seed, steps, list(arms))
            or ((rt.get('paired_from') or rt.get('complex_from')) and proof.get('views_per_source_packet') != 2)
            or any(proof[k] for k in ('F_reads', 'V_reads', 'TX_reads', 'target_reads'))):
        raise ValueError('Expected complete source-only gates with fixed budget')
    weights = torch.load(root/'gates.pt', map_location=device, weights_only=True)
    branches = []
    for name in arms:
        item = make_branch(name).to(device).eval().requires_grad_(False)
        item.load_state_dict(weights[name], strict=True)
        branches.append(item)
    return branches
