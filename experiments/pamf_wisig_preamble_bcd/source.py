"""Insert the same zero-initialized cycle path in all three new arms."""

from contextlib import contextmanager, nullcontext
from functools import partial
from pathlib import Path

import torch

from experiments.cric_dura_comparison.io import read, write
from experiments.pamf_repl_view_shuffle.source import assert_structure
from experiments.pamf_wisig_branch_style import source as matched

from .features import WIDTH, CycleProjection, CycleResponse, cycle_descriptor, fit_drift
from .plan import ARMS, validate_config


@contextmanager
def modified_install(module, noise_factor, seed, *, allow_prefix=False, holder=None):
    old_install = module.install

    def install(model, cfg):
        model = old_install(model, cfg)
        ref = model.id_backbone.encoder
        ref.response = CycleResponse(ref.response, noise_factor, seed)
        ref.response_projection[0] = CycleProjection(ref.response_projection[0], allow_prefix)
        if holder is not None:
            holder['cycle_encoder'] = ref
        return model

    module.install = install
    try:
        yield
    finally:
        module.install = old_install


def fit_source(c):
    from experiments.cric_dura_comparison.f_data import SourceDataset

    validate_config(c)
    dataset = SourceDataset(c['source_cache'], 'L_s', read(c['source_contract']))
    metadata = dataset.metadata
    if (len(dataset) != 6300 or set(metadata['rx_i']) != set(c['source_rxs'])
            or set(metadata['day_i']) != set(c['source_days']) or set(metadata['y']) != set(range(6))):
        raise ValueError('Expected legally labeled Main split L_s')
    with torch.no_grad():
        values = torch.cat([cycle_descriptor(x) for x in dataset.x.split(256)])
    return fit_drift(values, metadata['y'].tolist(), metadata['rx_i'].tolist(),
                     metadata['day_i'].tolist(), expected=(6, 5, 3))


@contextmanager
def installed(c, holder, *, fit):
    from experiments.cric_dura_comparison import t1_model

    arm = c['preamble_bcd_arm']
    factor = None if ARMS[arm] is None else fit[ARMS[arm]]
    seed = c['model_seed'] + 17017 + c['epoch_offset']
    with matched.installed(c, holder), modified_install(
        t1_model, factor, seed, allow_prefix=c['epoch_offset'] == 20, holder=holder
    ):
        yield


def train(c):
    validate_config(c)
    fit = None if ARMS[c['preamble_bcd_arm']] is None else fit_source(c)

    def save_fit(root):
        if fit is not None:
            write(root / 'source_drift.json', dict(source_role='L_s', source_packets=6300, **{
                k: v.tolist() if torch.is_tensor(v) else v for k, v in fit.items()
            }))

    return matched.train(c, validator=validate_config, installer=partial(installed, fit=fit),
                         source_init=save_fit, extra_state=checked_state)


def checked_state(c, holder):
    ref = holder['cycle_encoder']
    count = ref.response.perturbed_packets
    learned_norm = float(ref.response_projection[0].cycle_weight.detach().norm())
    if learned_norm <= 0 or (count > 0) != (ARMS[c['preamble_bcd_arm']] is not None):
        raise ValueError('Expected learned cycle path and the registered perturbation exposure')
    return dict(cycle_weight_norm=learned_norm, cycle_perturbed_packets=count)


def structure(model, c):
    return assert_structure(model, c, added_parameters=64 * WIDTH)


def smoke(runtime, train_configs):
    configs = [validate_config(read(path)) for path in train_configs]
    fit = fit_source(configs[0])

    def check(c, model):
        ref = model.id_backbone.encoder
        added = ref.response_projection[0].cycle_weight
        count = ref.response.perturbed_packets
        if (added.grad is None or float(added.grad.norm()) <= 0 or float(added.detach().norm()) != 0
                or (count > 0) != (ARMS[c['preamble_bcd_arm']] is not None)):
            raise ValueError('Cycle gradient, zero prefix initialization or noise application differs')
        return dict(cycle_gradient_norm=float(added.grad.norm()), cycle_perturbed_packets=count,
                    comparable_noise_rms=float(fit['diagonal'].norm()))

    return matched.smoke(runtime, train_configs, validator=validate_config,
                         installer=partial(installed, fit=fit), structure=structure, extra_checks=check)


def load_reader(folder, contract, device):
    from experiments.cric_dura_comparison import f_model

    config = validate_config(read(Path(folder) / 'complete.json')['config'])
    with modified_install(f_model, noise_factor=None, seed=config['model_seed']):
        return matched.load_reader(folder, contract, device, validator=validate_config)
