"""Source-only IQ diagnosis of a fixed spectral mixture and privileged oracle.

The oracle uses the clean source waveform and is not a deployable method, nor
an upper bound on arbitrary restoration networks or identity accuracy.
"""
import argparse
import os
from pathlib import Path
import time
import numpy as np
import torch
from experiments.pamf_phase1_flow import experiment as core
from experiments.pamf_phase1_flow.anchor_source_smoke import frozen_dura
from experiments.phase1_dura_innovation_probe.source_probe import source_inputs
from experiments.phase1_dura_orthogonal.train_source import load_source_training
from experiments.phase1_dura_orthogonal.signal_gate_oof_train import select_oof_ids
from experiments.phase1_joint_particle.particle import JointParticleRecovery
from experiments.phase1_joint_particle.train import SCHEMA as JOINT_SCHEMA
from experiments.phase1_residual_diffusion.model import ResidualDiffusion
from experiments.phase1_residual_diffusion.train import SCHEMA as DIFFUSION_SCHEMA
from experiments.phase1_residual_diffusion.source_probe import (
    load_model, normalized_error, JOINT, JOINT_COMMIT, DIFFUSION, DIFFUSION_COMMIT)
from experiments.phase1_spectral_moe.model import SpectralMixture
from experiments.phase1_spectral_moe.train import SCHEMA, EPOCHS, WIDTH

MIXTURE = '20261108-phase1-dura-spectral-mixture-source-s2026092701-r01'
MIXTURE_COMMIT = '44c051a2490f874499ef7a845efbd446e41ac1f4'


def frequency_oracle(experts, clean):
    """Choose the closest expert independently at each complex FFT bin."""
    spectra = torch.stack([torch.fft.fft(torch.complex(v[:, 0], v[:, 1]), norm='ortho') for v in experts], 1)
    truth = torch.fft.fft(torch.complex(clean[:, 0], clean[:, 1]), norm='ortho')
    choices = (spectra-truth[:, None]).abs().square().argmin(1)
    spectrum = spectra.gather(1, choices[:, None]).squeeze(1)
    signal = torch.fft.ifft(spectrum, norm='ortho')
    return torch.stack((signal.real, signal.imag), 1), choices


@torch.no_grad()
def evaluate(dura, joint, diffusion, gate, received, clean):
    context = dura.estimate_context(received)
    anchor = dura.stage_outputs(received, context)[0][-1]
    action, variance = context[1], context[2]
    y, z = torch.complex(received[:, 0], received[:, 1]), torch.complex(anchor[:, 0], anchor[:, 1])
    residual = action.adjoint(y-action(z))
    innovation = torch.stack((residual.real, residual.imag), 1)
    proposal = joint(received, anchor, innovation, variance, action)
    diffusion_iq = diffusion(received, anchor, innovation, variance)
    mixture = gate(received, anchor, proposal, diffusion_iq, variance)
    oracle, choices = frequency_oracle((anchor, proposal, diffusion_iq), clean)
    signals = {'dura': anchor, 'joint': proposal, 'diffusion': diffusion_iq,
               'mixture': mixture, 'privileged_frequency_oracle': oracle}
    errors = torch.stack([normalized_error(v, clean) for v in signals.values()], 1)
    fractions = torch.stack([(choices == i).float().mean(1) for i in range(3)], 1)
    return errors, fractions


def run(root, out, commit):
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    if out.exists() or not torch.cuda.is_available():
        raise ValueError('New output and explicitly selected GPU required')
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    ids, rx, scene, _, received, clean = source_inputs(root)
    cache, _, _, parents = load_source_training(root, ids, fit_size=32768)
    roles = np.load(cache/'train/roles.npy', allow_pickle=False)
    fit = select_oof_ids(np.load(cache/'train/rx.npy', allow_pickle=False),
                         np.load(cache/'train/day.npy', allow_pickle=False), roles, parents, ids)
    if len(ids) != 672 or len(fit) != 17632 or np.intersect1d(ids, fit).size:
        raise ValueError('Fixed source pilot must be disjoint from expert and gate fit records')
    folder = root/'runs'/MIXTURE
    summary = core.read(folder/'source_summary.json')
    state = torch.load(folder/'final.pt', map_location='cpu', weights_only=True)
    required = dict(schema=SCHEMA, commit=MIXTURE_COMMIT, epochs=EPOCHS, fit_ids=17632,
        actual_fit_ids=17632, actual_source_view_count=52896, views_per_record=3, width=WIDTH,
        joint=JOINT, joint_commit=JOINT_COMMIT, diffusion=DIFFUSION, diffusion_commit=DIFFUSION_COMMIT,
        parent_fit_disjoint=True, pilot_disjoint=True, TX_labels_read=0, F_weights_read=0,
        F_outputs_read=0, V_reads=0, target_reads=0, DURA_frozen=True)
    if summary.get('status') != 'SOURCE_TRAINED' or any(
            obj.get(k) != v for obj in (summary, state) for k, v in required.items()):
        raise ValueError('Fixed source-only mixture checkpoint required')
    gate = SpectralMixture(WIDTH).to('cuda:0')
    gate.load_state_dict(state['model'], strict=True)
    gate.eval().requires_grad_(False)
    dura = frozen_dura(root, 'cuda:0')
    joint = load_model(root, JOINT, JOINT_SCHEMA, JOINT_COMMIT, 3, JointParticleRecovery)
    diffusion = load_model(root, DIFFUSION, DIFFUSION_SCHEMA, DIFFUSION_COMMIT, 8, ResidualDiffusion)
    evaluate(dura, joint, diffusion, gate, core.tensor(received[ids[:2]], 'cuda:0'), core.tensor(clean[ids[:2]], 'cuda:0'))
    print('SPECTRAL_DIAGNOSTIC_SMOKE_PASS 2', flush=True)
    out.mkdir(parents=True)
    start_time = time.monotonic()
    chunks, winners = [], []
    for start in range(0, len(ids), 8):
        chosen = ids[start:start+8]
        err, fraction = evaluate(dura, joint, diffusion, gate, core.tensor(received[chosen], 'cuda:0'),
                                 core.tensor(clean[chosen], 'cuda:0'))
        chunks.append(err.cpu().to(torch.float64))
        winners.append(fraction.cpu().to(torch.float64))
        if (start+len(chosen)) % 168 == 0:
            print('SPECTRAL_DIAGNOSTIC_PROGRESS', start+len(chosen), flush=True)
    errors = np.from_dlpack(torch.cat(chunks).contiguous()).copy()
    fractions = np.from_dlpack(torch.cat(winners).contiguous()).copy()
    names = ('dura', 'joint', 'diffusion', 'mixture', 'privileged_frequency_oracle')
    groups = {}
    for name, receivers in (('RX4_6', (4, 6)), ('RX8', (8,))):
        for view in core.VIEWS:
            mask = np.isin(rx, receivers) & (scene == view)
            if int(mask.sum()) != 48: raise ValueError('Complete source pilot cells required')
            values = errors[mask]
            groups[name+'/'+view] = dict(count=48, nmse={key:float(values[:, i].mean()) for i,key in enumerate(names)},
                mixture_lower_than_dura=int(np.sum(values[:, 3] < values[:, 0]-1e-9)),
                mixture_higher_than_dura=int(np.sum(values[:, 3] > values[:, 0]+1e-9)),
                oracle_frequency_fraction=[float(x) for x in fractions[mask].mean(0)])
    core.write(out/'result.json', dict(status='SOURCE_IQ_DIAGNOSTIC_COMPLETE', commit=commit,
        mixture_commit=MIXTURE_COMMIT, joint_commit=JOINT_COMMIT, diffusion_commit=DIFFUSION_COMMIT,
        records=len(ids), groups=groups, elapsed_seconds=time.monotonic()-start_time,
        TX_labels_read=0, F_weights_read=0, F_outputs_read=0, V_reads=0,
        target_IQ_reads=0, target_truth_reads=0, model_updates=0,
        caveat='Previously explored source pilot; privileged oracle uses clean source IQ, is undeployable and not an identity-accuracy bound.'))
    print('SPECTRAL_DIAGNOSTIC_COMPLETE', flush=True)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--commit', required=True)
    a=p.parse_args()
    run(a.root,a.out,a.commit)

if __name__ == '__main__': main()
