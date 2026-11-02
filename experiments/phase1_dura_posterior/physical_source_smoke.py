"""No-query integration check using actual DURA and source physical posteriors."""

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch
from torch.nn import functional as F

from experiments.pamf_phase1_flow.anchor_source_smoke import frozen_dura
from experiments.pamf_phase1_flow import experiment as core
from experiments.phase1_dura_posterior.integrated import PosteriorInDura
from experiments.phase1_dura_posterior.pipeline import GatedPosteriorDura, PhysicalPosterior

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'phase1_joint_mixture'))
from source_pilot import s, BlindActionNet, action_matrix, posterior, SOURCE, PRIOR_RUN


def load_source_models(root):
    blind_state = torch.load(root/'runs'/SOURCE/'checkpoint_epoch10.pt', map_location='cpu', weights_only=False)
    prior = torch.load(root/'runs'/PRIOR_RUN/'source_prior.pt', map_location='cpu', weights_only=False)
    gate_state = torch.load(root/'runs'/s.GATE_RUN/'source/selected.pt', map_location='cpu', weights_only=False)
    if (blind_state.get('commit') != '94693a342bb2daa684c0f21415d7b949f9fb2107'
            or blind_state.get('epoch') != 10 or prior.get('source_ids') != 8192
            or gate_state.get('commit') != s.GATE_COMMIT
            or gate_state.get('selected', {}).get('kind') != 'gated_direct_iq'):
        raise ValueError('The fixed source-only physical estimator, prior and gate are required')
    blind = BlindActionNet()
    blind.load_state_dict(blind_state['model'], strict=True)
    physical = PhysicalPosterior(blind, prior['means'], prior['covariances'], prior['priors'],
                                 action_matrix=action_matrix, posterior=posterior)
    gate = s.BlindGate()
    gate.load_state_dict(gate_state['gate'], strict=True)
    return physical, gate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.manual_seed(2026110217)
    reader, contract = core.teacher(args.root, 'cpu')
    cache, clean, roles = core.data(args.root, contract)
    received = np.load(cache/'epochs/01/received.npy', mmap_mode='r', allow_pickle=False)
    candidates = np.flatnonzero(roles == 'L_s')[:512]
    physical, gate = load_source_models(args.root)
    gate.eval().requires_grad_(False)
    with torch.no_grad():
        logits = gate(core.tensor(received[candidates], 'cpu')).reshape(-1)
    selected_active = candidates[(logits.sigmoid() > 0.8).tolist()][:2]
    selected_closed = candidates[(logits.sigmoid() <= 0.8).tolist()][:2]
    if len(selected_active) != 2 or len(selected_closed) != 2:
        raise ValueError('Two active and two closed real-source gate examples are required')
    rows = np.concatenate((selected_active, selected_closed))
    x = core.tensor(received[rows], 'cpu')
    target = core.tensor(clean[rows], 'cpu')
    dura = frozen_dura(args.root, 'cpu')
    system = GatedPosteriorDura(PosteriorInDura(dura), gate, physical).train()
    with torch.no_grad():
        expected = dura(x)
        actual, active = system(x)
        clean_features = reader.features(target)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    if active.tolist() != [True, True, False, False]:
        raise ValueError('Both real-source gate paths must be exercised')
    result, _ = system(x)
    features = reader.features(result)
    den = target.square().mean((1, 2)).clamp_min(1e-8)
    loss = ((result-target).square().mean((1, 2))/den).mean()
    loss += 0.2*(1-F.cosine_similarity(features, clean_features, dim=1)).mean()
    loss.backward()
    gradients = []
    for correction in system.integrated.corrections:
        grad = correction.coefficients.weight.grad
        if grad is None or not torch.isfinite(grad).all() or grad.abs().sum() <= 0:
            raise ValueError('Each inserted stage requires finite nonzero source gradients')
        gradients.append(float(grad.norm()))
    frozen = (dura, reader, system.gate, system.posterior)
    if any(m.training or any(p.grad is not None or p.requires_grad for p in m.parameters()) for m in frozen):
        raise ValueError('DURA, reader, gate and posterior must remain frozen')
    print(json.dumps(dict(status='REAL_PHYSICAL_DURA_INSERTION_PASS', source_rows=len(rows),
                          active=int(active.sum()), closed=int((~active).sum()),
                          initial_max_abs=float((actual-expected).abs().max()), gradient_norms=gradients,
                          optimizer_updates=0, TX_labels_read=0, U_s_truth_reads=0,
                          V_reads=0, test_reads=0)), flush=True)


if __name__ == '__main__':
    main()
