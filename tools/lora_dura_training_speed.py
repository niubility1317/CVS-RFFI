"""Measure complete DURA updates on synthetic or registered source U IQ."""

import argparse
from copy import deepcopy
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from experiments.cric_dura_comparison.bootstrap import numerical_policy
from experiments.cric_dura_comparison.io import read
from experiments.pamf_lora_main.channel_blind_source import SEED, held_waveform, setup
from experiments.pamf_lora_main.data import SOURCE
from experiments.pamf_lora_main.dura_capacity import objective
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
    def __init__(self, model, tracker, iq, recipe, *, fused, receiver_batches=False):
        self.model = model
        self.tracker = tracker
        self.iq = iq
        self.recipe = recipe
        self.receiver_batches = receiver_batches
        self.rng = torch.Generator(device='cuda').manual_seed(SEED if receiver_batches else 6458841)
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4, fused=fused)
        self.schedule = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, recipe.steps, eta_min=2e-5)
        self.step = 0

    def __call__(self):
        self.step += 1
        device = self.iq.device
        if self.receiver_batches:
            count = len(self.iq) // len(SOURCE)
            receiver = torch.randint(len(SOURCE), (1,), device=device, generator=self.rng)
            indices = receiver * count + torch.randint(count, (self.recipe.batch,), device=device,
                                                        generator=self.rng)
        else:
            indices = torch.randint(len(self.iq), (self.recipe.batch,), device=device, generator=self.rng)
        loss, terms = objective(self.model, self.tracker, self.iq[indices], self.rng, with_identity=True,
                                recipe=self.recipe, step=self.step)
        if not torch.isfinite(terms).all():
            raise FloatingPointError('Nonfinite training objective')
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1., error_if_nonfinite=True)
        self.optimizer.step()
        self.schedule.step()
        return terms.detach()


def compare_updates(reference, other):
    with torch.no_grad():
        max_weight_error = 0.
        max_gradient_error = 0.
        max_optimizer_error = 0.
        for a, b in zip(reference.model.parameters(), other.model.parameters(), strict=True):
            max_weight_error = max(max_weight_error, (a - b).abs().max().item())
            max_gradient_error = max(max_gradient_error, (a.grad - b.grad).abs().max().item())
            state_a = reference.optimizer.state[a]
            state_b = other.optimizer.state[b]
            max_optimizer_error = max(max_optimizer_error, *(
                (value - state_b[key]).abs().max().item() for key, value in state_a.items()))
    return dict(max_weight_error=max_weight_error, max_gradient_error=max_gradient_error,
                max_optimizer_error=max_optimizer_error, rng_equal=torch.equal(reference.rng.get_state(),
                                                                                other.rng.get_state()))


def measure(arm, rounds, steps, *, graphs, source_runtime=None):
    device = 'cuda'
    real_source = source_runtime is not None
    torch.manual_seed(SEED if real_source else 6458840)
    recipe = RECIPES[arm]
    held = None
    if real_source:
        numerical_policy()
        (iq, _, held), _ = setup(read(source_runtime), device, fit_per_rx=recipe.fit_per_rx)
    tracker = PacketRestorer('consensus').to(device).eval()
    if not real_source:
        phase = torch.randn(64, 6, 1, device=device) * .35
        synthetic = torch.polar(torch.ones_like(phase), phase) * tracker.reference.chirp
        iq = real_iq(synthetic + .03 * torch.randn(synthetic.shape, dtype=torch.complex64, device=device))
    reference = recipe.model().to(device).train()
    methods = {'eager': Update(reference, tracker, iq, recipe, fused=None, receiver_batches=real_source)}
    if not real_source:
        methods['fused'] = Update(deepcopy(reference), tracker, iq, recipe, fused=True)
    graph_setup_ms = None
    if graphs:
        graphed = deepcopy(reference)
        torch.cuda.synchronize()
        started = perf_counter()
        capture_cores(graphed, recipe.batch)
        torch.cuda.synchronize()
        graph_setup_ms = 1000. * (perf_counter() - started)
        methods['graph_cores'] = Update(graphed, tracker, iq, recipe, fused=None, receiver_batches=real_source)
    samples = {name: [] for name in methods}
    traces = {name: [] for name in methods}
    for _ in range(8):
        for name, train in methods.items():
            traces[name].append(train())
    order = tuple(methods)
    for iteration in range(rounds):
        for name in order if iteration % 2 == 0 else reversed(order):
            torch.cuda.synchronize()
            begun = perf_counter()
            for _ in range(steps):
                traces[name].append(methods[name]())
            torch.cuda.synchronize()
            samples[name].append(1000. * (perf_counter() - begun) / steps)
    held_risk = {}
    if held is not None:
        with torch.no_grad():
            for name, train in methods.items():
                held_risk[name] = held_waveform(train.model, tracker, held, objective, recipe, train.step,
                                                receiver_batches=True)
    agreement = {}
    eager_terms = torch.stack(traces['eager'])
    for name, train in methods.items():
        if name != 'eager':
            agreement[name] = compare_updates(methods['eager'], train)
            agreement[name]['max_term_error'] = (torch.stack(traces[name]) - eager_terms).abs().max().item()
            if held_risk:
                agreement[name]['max_held_error'] = max(abs(a - b) for a, b in
                                                       zip(held_risk['eager'], held_risk[name], strict=True))
    return dict(arm=arm, batch=recipe.batch, device=torch.cuda.get_device_name(), torch=torch.__version__,
                samples_ms=samples, median_ms={name: float(np.median(x)) for name, x in samples.items()},
                graph_setup_ms=graph_setup_ms, updates_per_method=8 + rounds*steps, agreement=agreement,
                data_reads='U_s' if real_source else 0, fit_iq=len(iq) if real_source else 0,
                held_iq=len(held) if held is not None else 0, held_waveform=held_risk)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--arm', choices=tuple(RECIPES), default='expanded')
    parser.add_argument('--rounds', type=int, default=3)
    parser.add_argument('--steps', type=int, default=16)
    parser.add_argument('--graphs', action='store_true')
    parser.add_argument('--source-runtime', type=Path, help='Registered LoRa source run runtime JSON')
    args = parser.parse_args()
    torch.set_num_threads(1)
    print(json.dumps(measure(**vars(args)), allow_nan=False), flush=True)
