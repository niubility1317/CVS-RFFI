"""Synthetic integration check using the native DURA stage implementation."""

import argparse
import json
from pathlib import Path

import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--native-root', type=Path, action='append', required=True)
    args = parser.parse_args()
    path = args.native_root[0]/'experiments/pamf_r65_four_route/stage_polyphase.py'
    if not path.is_file():
        raise FileNotFoundError('Native DURA stage code is required')
    import experiments
    experiments.__path__ = [*(str(root/'experiments') for root in args.native_root), *list(experiments.__path__)]
    from experiments.pamf_r65_four_route.stage_polyphase import StageDistilled
    from experiments.phase1_dura_posterior.integrated import PosteriorInDura
    from experiments.phase1_dura_posterior.source_objective import source_objective

    class SyntheticReader(torch.nn.Module):
        def features(self, value): return value.flatten(1)
        def classify_features(self, value): return value[:, :6]

    torch.set_num_threads(2)
    torch.manual_seed(19)
    dura = StageDistilled().eval().requires_grad_(False)
    model = PosteriorInDura(dura).train()
    x = torch.randn(2, 2, 256)
    posterior = torch.randn_like(x)
    clean = torch.randn_like(x)
    logvar = torch.zeros(2)
    reader = SyntheticReader().eval().requires_grad_(False)
    before = {key: value.detach().clone() for key, value in dura.state_dict().items()}
    with torch.no_grad():
        original = dura(x)
        initial = model(x, posterior, logvar)
    torch.testing.assert_close(initial, original, rtol=0, atol=0)
    output = model(x, posterior, logvar)
    target_features = reader.features(clean)
    target_logits = reader.classify_features(target_features)
    baseline_logits = reader.classify_features(reader.features(original))
    loss, _ = source_objective(reader, output, clean, original, target_features, target_logits,
                               baseline_logits, torch.tensor([4, 6]))
    optimizer = torch.optim.AdamW(model.corrections.parameters(), lr=0.0003, weight_decay=0.0001)
    loss.backward()
    norms = []
    for correction in model.corrections:
        grad = correction.coefficients.weight.grad
        if grad is None or not torch.isfinite(grad).all() or grad.abs().sum() <= 0:
            raise ValueError('Both native DURA stages must receive finite nonzero correction gradients')
        norms.append(float(grad.norm()))
    optimizer.step()
    if (dura.training or any(p.requires_grad or p.grad is not None for p in dura.parameters())
            or any(not torch.equal(value, before[key]) for key, value in dura.state_dict().items())):
        raise ValueError('The native DURA skeleton must remain frozen')
    with torch.no_grad():
        updated = model(x, posterior, logvar)
    if not torch.isfinite(updated).all() or torch.equal(updated, original):
        raise ValueError('The correction update must change a finite native output')
    print(json.dumps(dict(status='NATIVE_STRUCTURE_SYNTHETIC_PASS', native_roots=[str(x) for x in args.native_root],
                          examples=2, dura_parameters=sum(p.numel() for p in dura.parameters()),
                          initial_max_abs=float((initial-original).abs().max()), gradient_norms=norms,
                          synthetic_optimizer_steps=1, checkpoint_loaded=False,
                          source_IQ_read=False, test_IQ_read=False)), flush=True)


if __name__ == '__main__':
    main()
