"""Fixed source checkpoint and seven-view frozen classifier evaluation."""
from pathlib import Path
import numpy as np
import torch

from experiments.pamf_phase1_flow import experiment as core
from experiments.pamf_phase1_flow.anchor_source_smoke import frozen_dura
from experiments.phase1_joint_particle import ground as joint_ground
from experiments.phase1_residual_diffusion.model import ResidualDiffusion, WIDTH, STEPS, PARTICLES, LATENT_SEED
from experiments.phase1_residual_diffusion.train import SCHEMA, EPOCHS, MODEL_SEED, ORDER_SEED, TIME_SEED, NOISE_SEED


def load_fixed(root, source, commit, device, *, preflight=False):
    folder = root/'runs'/(source+'-training-smoke' if preflight else source)
    summary = core.read(folder/('smoke.json' if preflight else 'source_summary.json'))
    state = torch.load(folder/('smoke.pt' if preflight else 'final.pt'), map_location='cpu', weights_only=True)
    expected = dict(schema=SCHEMA, commit=commit, epochs=1 if preflight else EPOCHS,
        fit_ids=32768, actual_source_view_count=64 if preflight else 98304, views_per_record=3,
        width=WIDTH, steps=STEPS, inference_particles=PARTICLES, latent_seed=LATENT_SEED,
        model_seed=MODEL_SEED, order_seed=ORDER_SEED, time_seed=TIME_SEED, noise_seed=NOISE_SEED,
        TX_labels_read=0, F_weights_read=0, F_outputs_read=0, V_reads=0, target_reads=0, DURA_frozen=True)
    if summary.get('status') != ('SOURCE_TRAINING_SMOKE_PASS' if preflight else 'SOURCE_TRAINED') or any(
            obj.get(k) != v for obj in (summary, state) for k, v in expected.items()):
        raise ValueError('Fixed source-trained diffusion checkpoint required')
    model = ResidualDiffusion(width=WIDTH).to(device)
    model.load_state_dict(state['model'], strict=True)
    model.eval().requires_grad_(False)
    dura = frozen_dura(root, device)
    reader, _ = core.teacher(root, 'cpu')
    return model, dura, reader.eval().requires_grad_(False)


@torch.no_grad()
def decisions(received_cpu, model, dura, reader, device):
    raw = reader.classify_features(reader.features(received_cpu))
    received = received_cpu.to(device)
    context = dura.estimate_context(received)
    anchor = dura.stage_outputs(received, context)[0][-1]
    action, variance = context[1], context[2]
    y = torch.complex(received[:, 0], received[:, 1])
    z = torch.complex(anchor[:, 0], anchor[:, 1])
    residual = action.adjoint(y-action(z))
    innovation = torch.stack((residual.real, residual.imag), 1)
    restored = model(received, anchor, innovation, variance)
    logits = [raw]
    for signal in (anchor, restored):
        logits.append(reader.classify_features(reader.features(signal.to('cpu'))))
    if any(x.shape != (len(received_cpu), 6) or not torch.isfinite(x).all() for x in logits):
        raise ValueError('Finite six-class frozen reader outputs required')
    return tuple(x.argmax(1) for x in logits)


def main():
    joint_ground.load_fixed = load_fixed
    joint_ground.decisions = decisions
    joint_ground.main()


if __name__ == '__main__':
    main()
