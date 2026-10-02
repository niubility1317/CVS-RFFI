"""Data-free, four-seed CE/AdamW smoke for the two learned readout variants."""
import argparse
import json
from pathlib import Path

import torch
from torch.nn import functional as F

from experiments.cvs_equivariant_identity.model import rotate_pair
from experiments.cvs_neural_residual_identity.model import build as control
from experiments.cvs_neural_readout_identity.model import BASE_VARIANT, VARIANTS, build, readout_contract


SEEDS = (2026092701, 2026092702, 2026092703, 2026092704)


def public_synthetic_input():
    generator = torch.Generator().manual_seed(2026100300)
    x = torch.randn(4, 2, 256, generator=generator)
    x[0].zero_()
    x[1].fill_(1.)
    t = torch.arange(256, dtype=x.dtype)
    x[2] = torch.stack([torch.cos(.13 * t) + .15 * torch.cos(.39 * t),
                        torch.sin(.13 * t) + .15 * torch.sin(.39 * t)])
    return x, torch.tensor([0, 1, 2, 3])


def run(output):
    output = Path(output)
    if output.exists():
        raise FileExistsError('Preserve prior smoke evidence; choose a new output path')
    torch.set_num_threads(2)
    x, labels = public_synthetic_input()
    records, paired_initialization = [], {}
    for variant in VARIANTS:
        for seed in SEEDS:
            torch.manual_seed(seed)
            base = control(BASE_VARIANT).eval()
            expected_rng = torch.get_rng_state().clone()
            torch.manual_seed(seed)
            model = build(variant).eval()
            rng_preserved = torch.equal(torch.get_rng_state(), expected_rng)
            old_state_exact = all(torch.equal(value, model.state_dict()[name])
                                  for name, value in base.state_dict().items())
            paired = {name + '.' + key: value.detach().clone()
                      for name, block in model.readout_blocks()
                      for key, value in block.state_dict().items() if not key.startswith('mixer.')}
            if variant == VARIANTS[0]:
                paired_initialization[seed] = paired
                paired_exact = True
            else:
                paired_exact = all(torch.equal(value, paired_initialization[seed][key])
                                   for key, value in paired.items())
            with torch.no_grad():
                initial_exact = torch.equal(model(x), base(x))
            assert model.contract() == readout_contract(variant)
            assert initial_exact and old_state_exact and rng_preserved and paired_exact
            optimizer = torch.optim.AdamW(model.parameters(), lr=.0002, weight_decay=.0001)
            model.train()
            steps = []
            for step in range(3):
                optimizer.zero_grad(set_to_none=True)
                loss = F.cross_entropy(model(x), labels)
                loss.backward()
                assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
                gradients = {}
                for name, block in model.readout_blocks():
                    gradients[name] = {key: float(parameter.grad.norm())
                                       for key, parameter in block.named_parameters()}
                    assert block.project.weight.grad.norm() > 0
                    hidden = list(block.scores.parameters()) + list(block.mixer.parameters())
                    if step == 0:
                        assert all(torch.count_nonzero(p.grad) == 0 for p in hidden)
                    else:
                        assert all(p.grad.norm() > 0 for p in hidden)
                optimizer.step()
                steps.append(dict(step=step + 1, cross_entropy=float(loss.detach()),
                                  all_parameter_gradients_finite=True, readout_gradient_norms=gradients))
            model.eval()
            state = {name: value.clone() for name, value in model.state_dict().items()}
            rng = torch.get_rng_state().clone()
            diagnostics = model.diagnostics(x)
            assert torch.equal(rng, torch.get_rng_state())
            assert all(torch.equal(value, model.state_dict()[name]) for name, value in state.items())
            readout_records = diagnostics['learned_readout']['records']
            assert [row['block'] for row in readout_records] == ['time.readout', 'behavior.readout']
            assert all(row['relative_output_change_mean'] > 0 for row in readout_records)
            assert all(row['attention_sum_max_abs_error'] <= 4 * torch.finfo(x.dtype).eps
                       for row in readout_records), (
                variant, seed, [row['attention_sum_max_abs_error'] for row in readout_records])
            with torch.no_grad():
                actual = model(x)
                singleton = torch.cat([model(packet[None]) for packet in x])
                phase = model(rotate_pair(x, torch.tensor([.43, -.6, 1.2, -2.])))
                torch.testing.assert_close(actual, singleton, atol=2e-4, rtol=2e-4)
                torch.testing.assert_close(actual, phase, atol=2e-3, rtol=2e-3)
            records.append(dict(
                variant=variant, model_seed=seed, status='VERIFIED',
                parameters=sum(p.numel() for p in model.parameters()),
                new_trainable_parameters=sum(p.numel() for p in model.readout_parameters()),
                contract=model.contract(), initial_exact=initial_exact,
                old_state_exact=old_state_exact, cpu_rng_preserved=rng_preserved,
                paired_score_project_initialization=paired_exact,
                steps=steps, diagnostics=diagnostics,
                packet_independence_max_abs_error=float((actual - singleton).abs().max()),
                common_phase_max_abs_error=float((actual - phase).abs().max()),
                inference_state_unchanged=True, checkpoint_source=None,
            ))
    result = dict(status='VERIFIED', data_access=False, target_access=False,
                  public_synthetic_input=True, checkpoint_loading=False,
                  optimizer='AdamW(lr=0.0002, weight_decay=0.0001)',
                  loss='single cross_entropy', steps_per_row=3, records=records)
    output.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation prevents accidentally replacing earlier evidence.
    with output.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    reread = json.loads(output.read_text(encoding='utf-8'))
    assert reread['status'] == 'VERIFIED' and len(reread['records']) == 8
    print(output)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    run(parser.parse_args().output)
