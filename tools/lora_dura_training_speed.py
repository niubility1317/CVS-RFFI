"""Measure complete DURA updates on synthetic IQ without accessing experiment data."""

import argparse
from copy import deepcopy
import json
from time import perf_counter

import numpy as np
import torch

from experiments.pamf_lora_main.dura_capacity import DuraCapacity, objective
from experiments.pamf_lora_main.dura_capacity_source import RECIPES
from experiments.pamf_lora_main.dura_repair import PacketRestorer, real_iq


def capture_cores(model, batch):
    # Each source packet is expanded to three observations with six chirps each.
    shapes = [(model.frequency, 10, False), (model.coarse, 18, True), (model.risk, 18, True),
              *((prior, 26, True) for prior in model.priors)]
    inputs = [(torch.zeros((18 * batch, channels, 128), device='cuda', requires_grad=grad),)
              for _, channels, grad in shapes]
    with torch.autocast('cuda', dtype=torch.bfloat16, cache_enabled=False):
        # Modules keep their parameter tensors. Only their forwards are rebound during training.
        torch.cuda.make_graphed_callables(tuple(core for core, _, _ in shapes), tuple(inputs))


class Update:
    def __init__(self, model, tracker, iq, recipe, *, fused):
        self.model = model
        self.tracker = tracker
        self.iq = iq
        self.recipe = recipe
        self.rng = torch.Generator(device='cuda').manual_seed(6458841)
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4, fused=fused)
        self.schedule = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, recipe.steps, eta_min=2e-5)
        self.step = 0

    def __call__(self):
        self.step += 1
        indices = torch.randint(len(self.iq), (self.recipe.batch,), device='cuda', generator=self.rng)
        loss, terms = objective(self.model, self.tracker, self.iq[indices], self.rng, with_identity=True,
                                recipe=self.recipe, step=self.step)
        if not torch.isfinite(terms).all():
            raise FloatingPointError('Nonfinite synthetic training objective')
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1., error_if_nonfinite=True)
        self.optimizer.step()
        self.schedule.step()
        return terms.detach()


def compare_models(reference, other):
    with torch.no_grad():
        max_weight_error = 0.
        max_gradient_error = 0.
        for a, b in zip(reference.parameters(), other.parameters(), strict=True):
            max_weight_error = max(max_weight_error, (a - b).abs().max().item())
            max_gradient_error = max(max_gradient_error, (a.grad - b.grad).abs().max().item())
    return dict(max_weight_error=max_weight_error, max_gradient_error=max_gradient_error)


def measure(arm, rounds, steps, *, graphs):
    device = 'cuda'
    torch.manual_seed(6458840)
    tracker = PacketRestorer('consensus').to(device).eval()
    recipe = RECIPES[arm]
    phase = torch.randn(64, 6, 1, device=device) * .35
    synthetic = torch.polar(torch.ones_like(phase), phase) * tracker.reference.chirp
    iq = real_iq(synthetic + .03 * torch.randn(synthetic.shape, dtype=torch.complex64, device=device))
    reference = DuraCapacity(arm).to(device).train()
    methods = {'eager': Update(reference, tracker, iq, recipe, fused=None),
               'fused': Update(deepcopy(reference), tracker, iq, recipe, fused=True)}
    graph_setup_ms = None
    if graphs:
        graphed = deepcopy(reference)
        torch.cuda.synchronize()
        started = perf_counter()
        capture_cores(graphed, recipe.batch)
        torch.cuda.synchronize()
        graph_setup_ms = 1000. * (perf_counter() - started)
        methods['graph_cores'] = Update(graphed, tracker, iq, recipe, fused=None)
    samples = {name: [] for name in methods}
    last = {}
    for _ in range(8):
        for name, train in methods.items():
            last[name] = train()
    order = tuple(methods)
    for iteration in range(rounds):
        for name in order if iteration % 2 == 0 else reversed(order):
            torch.cuda.synchronize()
            begun = perf_counter()
            for _ in range(steps):
                last[name] = methods[name]()
            torch.cuda.synchronize()
            samples[name].append(1000. * (perf_counter() - begun) / steps)
    agreement = {}
    for name, train in methods.items():
        if name != 'eager':
            agreement[name] = compare_models(reference, train.model)
            agreement[name]['max_term_error'] = (last[name]-last['eager']).abs().max().item()
    return dict(arm=arm, batch=recipe.batch, device=torch.cuda.get_device_name(), torch=torch.__version__,
                samples_ms=samples, median_ms={name: float(np.median(x)) for name, x in samples.items()},
                graph_setup_ms=graph_setup_ms, updates_per_method=8 + rounds*steps, agreement=agreement, data_reads=0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--arm', choices=tuple(RECIPES), default='expanded')
    parser.add_argument('--rounds', type=int, default=3)
    parser.add_argument('--steps', type=int, default=16)
    parser.add_argument('--graphs', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(1)
    print(json.dumps(measure(**vars(args)), allow_nan=False), flush=True)
