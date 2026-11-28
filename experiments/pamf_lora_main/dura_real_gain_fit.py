"""Source-only fit for the shared real-gain physical-residual hypothesis."""

import argparse
import importlib
from pathlib import Path
from time import monotonic

import numpy as np
import torch

from experiments.cric_dura_comparison.io import read, tensor, write
from experiments.pamf_lora_main.dura_real_gain_fusion import PhysicsGain, RealSpectralGain
from experiments.pamf_restoration.models import complex_iq, pair


SEED = 2026112845
STEPS = 2000
BATCH = {'lora': 4, 'wisig': 24}
WIDTH = {'lora': 1024, 'wisig': 32}
DEVICE = 'cuda:0'
LORA_GAIN = '20261128-phase1-lora-raw8192-frequency-quadrature-s2026113037-r01'
LORA_ACTION = '20261127-phase1-lora-raw8192-r02-r07-r17-s2026113009-r01'
WISIG_CACHE = '20261019-phase1-reception-restoration-selfsup-manysig-m6-r02'
WISIG_PARENT = '20261020-phase1-restoration-unrolled-selfsup-s2026092701-m2-r17'
WISIG_PACKET = '20261127-phase1-lora-wisig-packet-dura-s2026113021-r02'


def lora_models(project):
    from experiments.pamf_lora_main import w24_joint_pamf as joint
    from experiments.pamf_lora_main import w24_native_training as native

    root = project / 'runs' / LORA_GAIN
    rt = read(root / 'frequency_inphase_native' / 'fit-inputs.json')
    if Path(rt['project']) != project:
        raise ValueError('Wrong LoRa source project')
    joint.activate(rt)
    contract = joint.source_contract(Path(rt['source_root']))
    # Registers the original tracker implementation from its lawful release.
    joint.load_restorer(*joint.training_runtime(rt), contract, DEVICE)
    prior = native.task_model(None, DEVICE)
    prior.core.innovation = native.spectral_innovation('frequency_inphase_native', DEVICE)
    native.load_stage(root, 'frequency_inphase_native', contract, prior.core, DEVICE)
    prior.eval().requires_grad_(False)
    action_model = native.dura_model(project / 'runs' / LORA_ACTION, contract, DEVICE,
                                     stage='raw_r07').eval().requires_grad_(False)
    source = joint.source_role(Path(rt['source_root']), contract, 'U_s')[0]

    def base(y):
        p = prior(y)
        _, action, _ = action_model.estimate(pair(y))
        return p, action

    base.model = prior
    return source, np.arange(len(source)), base


def activate_wisig_archive(reference):
    """Resolve historical helper modules from the release that supplied them."""
    experiments = importlib.import_module('experiments')
    root = str(Path(reference) / 'experiments')
    if root not in experiments.__path__:
        experiments.__path__.insert(0, root)
    for name in ('pamf_lora_main', 'cric_dura_comparison', 'pamf_unrolled_restore'):
        package = importlib.import_module('experiments.' + name)
        path = str(Path(reference) / 'experiments' / name)
        if path not in package.__path__:
            package.__path__.insert(0, path)


def wisig_models(project, frozen_sources):
    from experiments.cric_dura_comparison.bootstrap import activate
    activate({'frozen_sources': frozen_sources})
    reference = project / 'releases' / '20261125-phase1-wisig-native-fir-source-m4-r01-release'
    activate_wisig_archive(reference)
    original = importlib.import_module('experiments.pamf_unrolled_restore.wisig_anchored_window_source')
    cache = project / 'runs' / WISIG_CACHE / 'cache'
    source, indices, _ = original.core.source_iq(cache)
    from experiments.pamf_unrolled_restore.runtime import load_checkpoint
    path = project / 'runs' / WISIG_PARENT / 'unroll_unshared' / 'native_map_prior' / 'final.pt'
    model, evidence = load_checkpoint(path, DEVICE)
    if (evidence['candidate'], evidence['epoch'], evidence['updates']) != ('unroll_unshared', 20, 4940):
        raise ValueError('Expected the source-selected original R17 final checkpoint')
    model.eval().requires_grad_(False)

    def base(y):
        p, _, action, _ = model.training_outputs(pair(y))
        return complex_iq(p), action

    return source, indices, base


def as_complex(values):
    if np.iscomplexobj(values):
        return tensor(np.ascontiguousarray(values).view(np.float32), DEVICE).view(torch.complex64)
    return complex_iq(tensor(np.ascontiguousarray(values), DEVICE))


def wisig_real_prior(project):
    from experiments.pamf_lora_main.w24_packet_unet import PacketUNet

    folder = project / 'runs' / WISIG_PACKET / 'wisig'
    proof = read(folder / 'fit.json')
    required = dict(status='SOURCE_FROZEN', steps=1600, seed=2026113021,
                    U_count=52604, TX_reads=0, target_reads=0, F_reads=0)
    if any(proof.get(k) != v for k, v in required.items()):
        raise ValueError('WiSig packet prior must be the registered source-only fit')
    model = PacketUNet(across_chirps=True, center=32).to(DEVICE).eval().requires_grad_(False)
    model.load_state_dict(torch.load(folder / 'packet_unet.pt', map_location=DEVICE, weights_only=True), strict=True)

    def prior(y):
        groups = y.unflatten(-1, (8, 32))
        scale = groups.abs().square().mean(-1, keepdim=True).clamp_min(1e-12).sqrt()
        normalized = groups / scale
        quality = (normalized[..., 1:] * normalized[..., :-1].conj()).mean(-1, keepdim=True).abs().clamp_max(1.)
        edges = torch.cat((groups[..., :24], groups[..., -24:]), dim=-1)
        corrected = model(normalized, quality, scale, edges)[-1]
        estimate = normalized + .5 * (corrected - normalized)
        return normalized, estimate, scale

    return prior


def fit(rt, dataset, folder):
    from experiments.cric_dura_comparison.bootstrap import numerical_policy
    from experiments.pamf_lora_main.w24_native_dura import source_pair

    numerical_policy()
    project = Path(rt['project'])
    source, ids, base = (lora_models(project) if dataset == 'lora'
                         else wisig_models(project, rt['frozen_sources']))
    real_control = dataset == 'wisig_real_gain'
    physical_dataset = 'wisig' if real_control else dataset
    packet_prior = wisig_real_prior(project) if real_control else None
    torch.manual_seed(SEED)
    gain = (RealSpectralGain(32) if real_control else PhysicsGain(WIDTH[dataset])).to(DEVICE).train()
    rng = np.random.default_rng(SEED)
    generator = torch.Generator(device=DEVICE).manual_seed(SEED + 1)
    optimizer = torch.optim.AdamW(gain.parameters(), lr=3e-4, weight_decay=1e-4)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, STEPS, eta_min=3e-5)
    started = monotonic()
    for step in range(1, STEPS + 1):
        raw = as_complex(source[rng.choice(ids, BATCH[physical_dataset], replace=False)])
        clean, y, _, _ = source_pair(raw, generator)
        if real_control:
            with torch.no_grad():
                yn, pn, scale = packet_prior(y)
                cn, cpn, cscale = packet_prior(clean)
            prior = (pn * scale).flatten(-2)
            clean_prior = (cpn * cscale).flatten(-2)
            prediction = ((pn + gain(yn, pn)) * scale).flatten(-2)
            clean_prediction = ((cpn + gain(cn, cpn)) * cscale).flatten(-2)
        else:
            with torch.no_grad():
                prior, action = base(y)
                clean_prior, clean_action = base(clean)
            prediction = gain(y, prior, action)
            clean_prediction = gain(clean, clean_prior, clean_action)
        if step == 1:
            torch.testing.assert_close(prediction, prior, rtol=0, atol=0)
        error = prediction - clean
        spectrum = torch.fft.fft(clean, norm='ortho').abs().square()
        spectral = (torch.fft.fft(error, norm='ortho').abs().square() / (.03 + spectrum)).log1p().mean()
        fidelity = (clean_prediction - clean_prior).abs().square().mean()
        loss = error.abs().square().mean() + .3 * spectral + 1.5 * fidelity
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad = torch.nn.utils.clip_grad_norm_(gain.parameters(), 1., error_if_nonfinite=True)
        if step == 1 and (not torch.isfinite(loss) or gain.net[-1].weight.grad.norm() <= 0):
            raise ValueError('No finite source gradient for real physical gain')
        optimizer.step()
        schedule.step()
        if step == 1 or step % 50 == 0:
            print('PHYSICS_GAIN_FIT', dataset, step, float(loss.detach()), float(grad), flush=True)
    torch.cuda.synchronize()
    proof = dict(status='SOURCE_FROZEN', dataset=dataset, seed=SEED, updates=STEPS,
                 batch=BATCH[physical_dataset], source_u_pool=len(ids), source_u_exposures=STEPS * BATCH[physical_dataset],
                 newly_read_TX_labels=0, newly_read_F=0, V_reads=0, target_reads=0,
                 params=sum(p.numel() for p in gain.parameters()), seconds=monotonic()-started,
                 peak_cuda_bytes=torch.cuda.max_memory_allocated())
    with (folder / 'gain.pt').open('xb') as stream:
        torch.save(dict(weights=gain.state_dict(), proof=proof), stream)
    write(folder / 'fit.json', proof)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('runtime', type=Path)
    parser.add_argument('dataset', choices=(*WIDTH, 'wisig_real_gain'))
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    fit(read(args.runtime), args.dataset, args.output)
