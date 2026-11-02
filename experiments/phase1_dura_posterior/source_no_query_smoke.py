"""Check the inserted stages on real source IQ and the exact frozen checkpoints."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from experiments.pamf_phase1_flow import experiment as core
from experiments.pamf_phase1_flow.anchor_source_smoke import frozen_dura
from experiments.phase1_dura_posterior.integrated import PosteriorInDura


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.manual_seed(2026110207)
    reader, contract = core.teacher(args.root, 'cpu')
    cache, clean, roles = core.data(args.root, contract)
    with np.load(cache/'epochs/01/groups.npz', allow_pickle=False) as groups:
        scene = groups['scene']
    rows = np.flatnonzero((roles == 'L_s') & (scene != 'clean'))[:2]
    if len(rows) != 2:
        raise ValueError('Two actual nonclean L_s source rows are required')
    received = np.load(cache/'epochs/01/received.npy', mmap_mode='r', allow_pickle=False)
    x = core.tensor(received[rows], 'cpu')
    clean_iq = core.tensor(clean[rows], 'cpu')
    dura = frozen_dura(args.root, 'cpu')
    system = PosteriorInDura(dura).train()
    # The received input is a placeholder for the as-yet unconnected posterior.
    # This verifies the DURA insertion, not the posterior estimator or accuracy.
    posterior, logvar = x.detach(), torch.zeros(len(x))
    with torch.no_grad():
        expected = dura(x)
        actual = system(x, posterior, logvar)
        clean_features = reader.features(clean_iq)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    result = system(x, posterior, logvar)
    features = reader.features(result)
    energy = clean_iq.square().mean((1, 2)).clamp_min(1e-8)
    loss = ((result-clean_iq).square().mean((1, 2))/energy).mean()
    loss = loss + 0.2*(1-F.cosine_similarity(features, clean_features, dim=1)).mean()
    loss.backward()
    norms = []
    for correction in system.corrections:
        grad = correction.coefficients.weight.grad
        if grad is None or not torch.isfinite(grad).all() or grad.abs().sum() <= 0:
            raise ValueError('Both inserted stages require finite nonzero source gradients')
        norms.append(float(grad.norm()))
    if dura.training or reader.training or any(p.grad is not None for p in dura.parameters()) or any(p.grad is not None for p in reader.parameters()):
        raise ValueError('Both original networks must remain frozen')
    print(json.dumps(dict(status='REAL_SOURCE_INSERTION_SMOKE_PASS', source_rows=len(rows),
                          initial_max_abs=float((actual-expected).abs().max()), gradient_norms=norms,
                          posterior='received_placeholder', F_frozen=True, DURA_frozen=True,
                          optimizer_updates=0, TX_labels_read=0, U_s_truth_reads=0,
                          V_reads=0, test_reads=0)), flush=True)


if __name__ == '__main__':
    main()
