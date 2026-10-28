"""One source-only E50+222 PAMF increment, with optional uniform OE."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from copy import deepcopy
import json
import math
from pathlib import Path
import random
import time
from types import SimpleNamespace

import numpy as np
import torch
from torch.nn import functional as F

from . import runner


BASE_KEYS = runner.COMMON | {'source_cache', 'arm', 'steps', 'lr', 'L_seed', 'OE_seed'}
ARMS = ('supervised_only', 'uniform_oe')
CLASSES = ['14-10', '14-7', '20-15', '20-19', '6-15', '8-20']
RX = [1, 3, 4, 6, 8]
PARENT_RUN = '20261027-phase1-pamf-repl-shuffled-views-m16-r02'
PARENT_ROW = 'main_split-repl-shuffled_time-3view-s2026092701'
ADDED_STEPS = 222
PARENT_STEPS = 11100
OE_BATCH = 64
OE_WEIGHT = .1
SCHEMA = 'pamf_phase3_source_embedding_increment_v1'
SINC_PREFIX = 'id_backbone.encoder.core.id_backbone.sinc.'
HEAD_NAME = 'id_backbone.encoder.core.id_backbone.cls_head.weight'


def validate_config(config):
    if not isinstance(config, dict) or config.get('arm') not in ARMS:
        raise ValueError('A declared native supervised or uniform-OE arm is required')
    expected = BASE_KEYS | ({'diverse_data_root'} if config['arm'] == 'uniform_oe' else set())
    if set(config) != expected:
        raise ValueError('Source increment input whitelist differs: ' + str(sorted(set(config) ^ expected)))
    base = {key: config[key] for key in runner.COMMON}
    runner.validate_config('calibrate', dict(base, source_cache=config['source_cache'], retention=.99))
    fixed = dict(steps=ADDED_STEPS, lr=.0001, L_seed=2026092701, OE_seed=2026102907,
                 model_seed=2026092701, batch_size=128)
    if any(type(config[key]) is not type(value) or config[key] != value for key, value in fixed.items()):
        raise ValueError('Fixed seed, 222-update increment and constant learning rate required')
    if config['registered_classes'] != CLASSES:
        raise ValueError('Original ordered six-class roster required')
    if len(config['code_commit']) != 40 or any(c not in '0123456789abcdef' for c in config['code_commit']):
        raise ValueError('Actual Git execution commit required')
    if config['arm'] == 'uniform_oe' and (not isinstance(config['diverse_data_root'], str) or not config['diverse_data_root'].strip()):
        raise ValueError('Validated source21 OE root required')
    return config


def _contract(config):
    value = runner._read(config['source_contract'])
    if (value.get('classes') != CLASSES or value.get('source_rxs') != RX or value.get('source_days') != [1, 2, 3]
            or value.get('counts') != {'L_s': 6300, 'U_s': 56700, 'V': 27000}):
        raise ValueError('Original six-class source physical roles required; no resplitting')
    return value


def _training_config(config, parent):
    value = deepcopy(parent)
    value.update(run_id=config['run_id'], row_id=config['row_id'], output_root=config['output_root'],
        source_cache=config['source_cache'], source_contract=config['source_contract'], model_seed=config['model_seed'],
        ablation='F20', regime='matched_sat_supervised', U_use=False, own_EMA=False, native_U_gate=False,
        native_unlabeled_loop=False, satellite_concat=False, step_offset=PARENT_STEPS, epoch_offset=50,
        segment_epochs=1, parent_source=None, num_classes=6)
    value.pop('core_mechanisms', None)
    value['repl_row'] = dict(value['repl_row'], U=False)
    return value


def _verified_parent(config, contract):
    # Reuse the native E50 ancestry/data checks. Its frozen reader is discarded;
    # in particular, its no-grad Sinc method is never used by the trainable model.
    frozen = runner._load_model(config, contract, 'cpu')
    del frozen
    done = runner._read(Path(config['source_output']) / 'complete.json')
    original = done['config']
    if (original.get('run_id') != PARENT_RUN or original.get('row_id') != PARENT_ROW
            or original.get('split_seed') != 392005
            or Path(original.get('source_cache', '')).resolve() != Path(config['source_cache']).resolve()
            or original.get('repl_row', {}).get('L_A') is not True
            or original.get('repl_row', {}).get('L_B') is not True):
        raise ValueError('Only the verified original full-six-class E50 source parent is allowed')
    payload = torch.load(Path(config['source_output']) / 'final_ssdg.pth', map_location='cpu', weights_only=False)
    args = payload.get('args')
    if (not isinstance(args, dict) or not isinstance(args.get('weight_decay'), (float, int))
            or not math.isfinite(float(args['weight_decay'])) or args['weight_decay'] < 0):
        raise ValueError('Native parent optimizer/augmentation arguments required')
    identity = dict(source_run_id=original['run_id'], source_row_id=original['row_id'], model_seed=config['model_seed'],
        source_output=config['source_output'], source_contract=config['source_contract'], source_cache=config['source_cache'],
        base_release=config['base_release'], frozen_sources=config['frozen_sources'], parent_epoch=50,
        parent_updates=PARENT_STEPS, selection='original_fixed_final_E50', original_code_commit=original['code_commit'])
    return original, payload, identity


def _new_native(config, native_config, device):
    from experiments.cric_dura_comparison import f_train, t1_model
    from experiments.cvs_phase1_stack.runtime import installed as native_scope
    from experiments.pamf_repl_view_shuffle.source import installed
    with installed(native_config, {}, False), f_train.class_dimension(native_config), native_scope(f_train.original_config(config['model_seed'])) as native:
        args = f_train.arguments(native_config, device)
        args.input_len = 256
        args.from_scratch = args.a1_scratch_only = True
        args.baseline_ckpt = args.teacher_ckpt = ''
        model = t1_model.install(native.build_baseline_model(args, device), native_config).to(device)
    return model


def _assert_trainable(model):
    from experiments.pamf_repl_view_shuffle.source import assert_structure
    assert_structure(model, {'view_comparison': {'view_count': 3}})
    reader = model.id_backbone.encoder
    sinc = reader.core.id_backbone.sinc
    if (any(v.window_order.tolist() != runner.ORDER for v in reader.sequence_pool.views)
            or tuple(reader.core.id_backbone.cls_head.weight.shape) != (6, 160)
            or sinc.mode != 'CENTER_25K' or sinc.scale_hz != 25000.
            or getattr(getattr(getattr(sinc, '_filters'), '__func__', None), '__name__', '') != '_filters'
            or any(not p.requires_grad for p in model.parameters())):
        raise ValueError('Trainable current full PAMF/default head/Sinc structure differs')


def make_trainable_model(config, device='cuda:0'):
    """Verify the original E50 source parent, rebuild its graph, then strict-load."""
    validate_config(config)
    contract = _contract(config)
    original, payload, identity = _verified_parent(config, contract)
    native_config = _training_config(config, original)
    model = _new_native(config, native_config, device)
    model.load_state_dict(payload['model'], strict=True)
    model.train().requires_grad_(True)
    _assert_trainable(model)
    model._phase3_parent_identity = identity
    model._phase3_parent_args = deepcopy(payload['args'])
    model._phase3_source_handle = dict(classes=CLASSES, source_rxs=RX, source_days=[1, 2, 3],
        split_seed=392005, counts=contract['counts'], source_contract=config['source_contract'])
    return model


def _checkpoint_identity(config):
    keys = ('run_id', 'row_id', 'model_seed', 'source_output', 'source_contract',
        'source_cache', 'base_release', 'frozen_sources', 'registered_classes', 'arm', 'code_commit')
    return {key: config[key] for key in keys + (('diverse_data_root',) if config['arm'] == 'uniform_oe' else ())}


def load_trained_model(config, checkpoint, device='cuda:0'):
    """Load this increment without applying the historical E50 terminal-step rule."""
    model = make_trainable_model(config, device)
    value = torch.load(checkpoint, map_location='cpu', weights_only=False)
    if (value.get('schema') != SCHEMA or value.get('identity') != _checkpoint_identity(config)
            or value.get('original_parent_identity') != model._phase3_parent_identity
            or value.get('source_data_handle') != model._phase3_source_handle
            or value.get('updates') != ADDED_STEPS or value.get('cumulative_updates') != 11322
            or value.get('optimizer_reset') is not True or value.get('U_reads') != 0
            or value.get('target_reads') != 0 or value.get('validation_reads') != 0):
        raise ValueError('This frozen Phase3 increment identity or source-only budget differs')
    model.load_state_dict(value['model'], strict=True)
    _assert_trainable(model)
    return model.eval().requires_grad_(False)


def _reader_logits(model, x):
    reader = model.id_backbone.encoder
    logits = reader.classify_features(reader.features(x))
    if logits.shape != (len(x), 6) or not bool(torch.isfinite(logits).all()):
        raise ValueError('Finite original six-class full default-head logits required')
    return logits


def uniform_oe_loss(logits):
    if logits.ndim != 2 or logits.shape[1] != 6 or not len(logits) or not bool(torch.isfinite(logits).all()):
        raise ValueError('Uniform OE requires finite full-six-class logits')
    return -F.log_softmax(logits, dim=1).mean() - math.log(6)


def _cuda_devices(model):
    return sorted({p.device.index for p in model.parameters() if p.device.type == 'cuda'})


@contextmanager
def _temporary_eval(model):
    flags = [(module, module.training) for module in model.modules()]
    model.eval()
    try:
        yield
    finally:
        for module, flag in flags:
            module.training = flag


def oe_forward(model, x):
    # Eval disables BN updates and dropout, but leaves the entire identity graph
    # differentiable. Forking preserves the known-L stochastic stream.
    with torch.random.fork_rng(devices=_cuda_devices(model)), _temporary_eval(model), torch.enable_grad():
        return uniform_oe_loss(_reader_logits(model, x))


def gradient_smoke(model, x, y=None):
    """No-step two-row gradient check, restoring tensors, flags and all RNGs."""
    if x.shape != (2, 2, 256) or not bool(torch.isfinite(x).all()):
        raise ValueError('Two legal finite source observations required')
    state = {key: value.detach().clone() for key, value in model.state_dict().items()}
    # Native Sinc keeps nonpersistent versioned filter caches outside state_dict.
    caches = [(module, key, value) for module in model.modules() for key, value in vars(module).items()
              if 'cache' in key and key not in ('_buffers', '_parameters', '_modules')]
    numpy_rng, python_rng = np.random.get_state(), random.getstate()
    model.zero_grad(set_to_none=True)
    try:
        with torch.random.fork_rng(devices=_cuda_devices(model)), _temporary_eval(model), torch.enable_grad():
            logits = _reader_logits(model, x)
            loss = uniform_oe_loss(logits) if y is None else F.cross_entropy(logits, y, label_smoothing=.01)
            loss.backward()
            named = dict(model.named_parameters())
            sinc = [named[SINC_PREFIX + leaf].grad for leaf in ('low_coordinate', 'band_coordinate')]
            head = named[HEAD_NAME].grad
            if (any(grad is None or not bool(torch.isfinite(grad).all()) for grad in sinc)
                    or head is None or not bool(torch.isfinite(head).all())):
                raise ValueError('Actual Sinc coordinate/default-head gradient is disconnected or nonfinite')
            sinc_norm = sum(float(grad.norm()) for grad in sinc)
            head_norm = float(head.norm())
            if sinc_norm <= 0 or head_norm <= 0:
                raise ValueError('Actual Sinc parameter set and default head must receive nonzero gradients')
            return dict(status='PASS', source_samples=2, objective='uniform_oe' if y is None else 'known_CE',
                sinc_gradient_norm=sinc_norm, head_gradient_norm=head_norm, optimizer_updates=0,
                state_restored=True, RNG_restored=True, query_reads=0, U_reads=0)
    finally:
        model.load_state_dict(state, strict=True)
        for module, key, value in caches:
            setattr(module, key, value)
        model.zero_grad(set_to_none=True)
        np.random.set_state(numpy_rng)
        random.setstate(python_rng)


class DomainSampler:
    """IID uniform source RX/day, then IID uniform records within that domain."""
    def __init__(self, domains, seed):
        domains = np.asarray(domains)
        if domains.ndim != 1 or domains.dtype.kind not in 'iu' or set(np.unique(domains).tolist()) != set(range(15)):
            raise ValueError('All fifteen source domains must be nonempty')
        self.rows = [np.flatnonzero(domains == index) for index in range(15)]
        self.generator = torch.Generator().manual_seed(seed)

    def sample(self, count):
        domain = torch.randint(15, (count,), generator=self.generator).tolist()
        result = np.empty(count, dtype=np.int64)
        for index in range(15):
            positions = [i for i, value in enumerate(domain) if value == index]
            if positions:
                offsets = torch.randint(len(self.rows[index]), (len(positions),), generator=self.generator).tolist()
                result[positions] = self.rows[index][offsets]
        return result


def _load_oe(config):
    from .source_diverse_export import _diverse_handle, _public
    handle = _diverse_handle(config['diverse_data_root'])
    iq, metadata = _public(config['diverse_data_root'])
    return iq, metadata, handle


def _load_labeled(config, contract):
    # Do not call historical f_data.context: that function also opens V.
    from experiments.cric_dura_comparison.f_data import SourceDataset
    return SourceDataset(config['source_cache'], 'L_s', contract)


def _augmentor(parent_args):
    from cvsrffi.schedule import build_aug_base_cfg, make_augmentor, configure_augmentor_for_epoch
    prior = SimpleNamespace(**deepcopy(parent_args))
    prior.seed = 2026092701
    prior.epochs, prior.label_epochs, prior.pseudo_epochs = 20, 13, 7
    cfg = build_aug_base_cfg(prior)
    augmentor = make_augmentor(cfg)
    state = configure_augmentor_for_epoch(augmentor, cfg, 13, prior)
    return augmentor, not bool(prior.aug_enable_pa_normal), state


def supervised_loss(model, raw, y, domains, meta, augmentor, no_pa, step):
    ordinary = augmentor(raw, labels=y, no_pa=no_pa)
    if ordinary.shape != raw.shape or not bool(torch.isfinite(ordinary).all()):
        raise ValueError('Native ordinary source augmentation is nonfinite or malformed')
    try:
        with model._phase1_labeled_feature_context():
            output = model(ordinary, return_aux=True, domain_labels=domains)
        identity = F.cross_entropy(output['tx_logits'], y, label_smoothing=.01)
        correction, metrics = model._source_labeled_feature_hook(model=model, labeled_output=output,
            labels=y, extra=(None, meta), domains=domains, optimizer_step=step, identity_loss=identity,
            labeled_count=len(y), labeled_inputs=raw)
        return identity + correction, metrics
    finally:
        from experiments.cric_dura_comparison.f_model import clear
        if hasattr(model, '_cd_handles'):
            clear(model)


def _train_steps(model, batches, optimizer, augmentor, no_pa, device, steps, oe_batch, clip,
                 *, labeled_loss=supervised_loss, on_step=None):
    trace = []
    model.train()
    iterator = iter(batches)
    params = list(model.parameters())
    for step in range(1, steps + 1):
        started = time.perf_counter()
        raw, y, domains, meta = next(iterator)
        raw, y, domains = raw.to(device), y.to(device).long(), domains.to(device).long()
        if raw.shape != (128, 2, 256) or not bool(torch.isfinite(raw).all()) or y.min() < 0 or y.max() >= 6:
            raise ValueError('Only legal clean source-L128 may enter supervised updates')
        optimizer.zero_grad(set_to_none=True)
        loss_l, metrics = labeled_loss(model, raw, y, domains, meta, augmentor, no_pa, step)
        if float(metrics['CRIC_supcon_weight']) != .1 or int(metrics.get('T1_PL_seen', 0)):
            raise ValueError('E51 cross-RX weight or U-off execution differs')
        if int(metrics['T1_L_packets']) != 768:
            raise ValueError('Original six source supervised packet views are required')
        loss_oe = loss_l.new_zeros(())
        if oe_batch is not None:
            loss_oe = oe_forward(model, oe_batch(OE_BATCH).to(device))
        loss = loss_l + OE_WEIGHT * loss_oe
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError('Nonfinite source increment loss')
        loss.backward()
        if any(p.grad is not None and not bool(torch.isfinite(p.grad).all()) for p in params):
            raise FloatingPointError('Nonfinite source increment gradient')
        grad_norm = clip(optimizer, params)
        if not bool(torch.isfinite(torch.as_tensor(grad_norm))):
            raise FloatingPointError('Nonfinite physical global gradient norm')
        optimizer.step()
        row = dict(step=step, cumulative_updates=PARENT_STEPS + step, epoch=51, lr=float(optimizer.param_groups[0]['lr']),
            loss=float(loss.detach()), supervised_loss=float(loss_l.detach()), oe_loss=float(loss_oe.detach()),
            oe_weight=OE_WEIGHT if oe_batch is not None else 0., L_samples=128, ordinary_forward_samples=384,
            As_generated_samples=64, As_forward_samples=128, raw_forward_samples=256, L_forward_calls=6,
            L_packet_views=768, oe_samples=OE_BATCH if oe_batch is not None else 0,
            oe_forward_calls=int(oe_batch is not None), physical_grad_norm=float(grad_norm),
            seconds=time.perf_counter() - started,
            native_metrics={key: float(value.detach()) if torch.is_tensor(value) else value for key, value in metrics.items()})
        trace.append(row)
        if on_step is not None:
            on_step(row)
    return trace


def train(config, device='cuda:0'):
    validate_config(config)
    output = Path(config['output_root'])
    if output.exists():
        raise FileExistsError('A new immutable Phase3 source increment root is required')
    model = make_trainable_model(config, device)
    contract = _contract(config)
    labeled = _load_labeled(config, contract)
    augmentor, no_pa, augmentation_state = _augmentor(model._phase3_parent_args)
    torch.manual_seed(config['L_seed'])
    if torch.device(device).type == 'cuda':
        with torch.cuda.device(torch.device(device)):
            torch.cuda.manual_seed(config['L_seed'])
    np.random.seed(config['L_seed'])
    random.seed(config['L_seed'])
    output.mkdir(parents=True, exist_ok=False)
    # This bare full identity forward does not advance the L128 hook or loader.
    smoke = gradient_smoke(model, labeled.x[:2].to(device),
        torch.tensor(labeled.metadata['y'][:2].tolist(), dtype=torch.long, device=device))
    runner._write(output / 'smoke.json', dict(smoke, actual_checkpoint=True, original_parent_identity=model._phase3_parent_identity))
    print('SOURCE_EMBEDDING_GRADIENT_SMOKE_PASS', config['row_id'], 'known_L', flush=True)
    oe_iq = None
    oe_handle = None
    oe_next = None
    try:
        if config['arm'] == 'uniform_oe':
            oe_iq, oe_meta, oe_handle = _load_oe(config)
            if set(oe_meta['ids'].tolist()) & set(labeled.metadata['ids'].tolist()):
                raise ValueError('Original L and auxiliary source21 observations overlap')
            sampler = DomainSampler(oe_meta['domain'], config['OE_seed'])
            from experiments.cric_dura_comparison.io import tensor
            def oe_next(count):
                values = np.ascontiguousarray(oe_iq[sampler.sample(count)], dtype=np.float32)
                if not np.isfinite(values).all():
                    raise ValueError('Nonfinite raw normalized auxiliary source IQ')
                return tensor(values, device)
            sampler_state = sampler.generator.get_state()
            try:
                oe_smoke = gradient_smoke(model, oe_next(2))
            finally:
                sampler.generator.set_state(sampler_state)
            runner._write(output / 'oe_smoke.json', dict(oe_smoke, actual_checkpoint=True, source_OE_handle=oe_handle))
            print('SOURCE_EMBEDDING_GRADIENT_SMOKE_PASS', config['row_id'], 'source_OE', flush=True)
        from experiments.cric_dura_comparison.f_data import FixedEpochLoader
        from experiments.cric_dura_comparison.f_optimizer import build, clip_physical
        batches = FixedEpochLoader(labeled, 128, config['L_seed'], steps=ADDED_STEPS, drop_last=True)
        optimizer = build(model, list(model.parameters()), 'CENTER_UNITS', lr=config['lr'],
            weight_decay=float(model._phase3_parent_args['weight_decay']), betas=(.9, .999), eps=1e-8, foreach=False)
        if optimizer.state:
            raise ValueError('The increment optimizer must start without inherited moments')
        runner._write(output / 'resolved_config.json', config)
        start = time.perf_counter()
        with (output / 'steps.jsonl').open('x', encoding='utf-8') as stream:
            def on_step(row):
                stream.write(json.dumps(row, allow_nan=False) + '\n')
                stream.flush()
                print('SOURCE_EMBEDDING_PROGRESS', config['row_id'], row['step'], ADDED_STEPS,
                      'loss', row['loss'], 'oe', row['oe_loss'], flush=True)
            trace = _train_steps(model, batches, optimizer, augmentor, no_pa, device, ADDED_STEPS,
                oe_next, lambda opt, params: clip_physical(opt, params, 5.), on_step=on_step)
        if (len(trace) != ADDED_STEPS or optimizer._sinc_unit_steps != ADDED_STEPS
                or optimizer._sinc_physical_clip_calls != ADDED_STEPS or model._cd_calls != ADDED_STEPS
                or model._feature_l_steps != ADDED_STEPS or model._cd_U_seen or model._cd_U_accepted
                or labeled.role != 'L_s' or batches.actual_examples != 28416):
            raise ValueError('The complete fixed 222-step source increment is missing')
        model.zero_grad(set_to_none=True)
        model.eval().requires_grad_(False)
        summary = dict(status='SOURCE_OPEN_EMBEDDING_FROZEN', schema=SCHEMA, config=config,
            original_parent_identity=model._phase3_parent_identity, source_data_handle=model._phase3_source_handle,
            source_OE_handle=oe_handle, arm=config['arm'], updates=ADDED_STEPS, parent_updates=PARENT_STEPS,
            cumulative_updates=11322, parent_epochs=50, increment_epoch=51, optimizer_reset=True,
            optimizer='AdamW_CENTER_UNITS', lr=config['lr'], lr_schedule='constant',
            weight_decay=float(model._phase3_parent_args['weight_decay']), ordinary_augmentation_epoch=13,
            augmentation_state=augmentation_state, cross_RX_weight=.1, source_only=True,
            U_reads=0, validation_reads=0, novel_validation_reads=0, target_reads=0, OE_truth_reads=0,
            model_classes=6, parameters=232993, default_head_joint=True, identity_head_replaced=False,
            L_sampled_exposures=28416, As_generated_exposures=14208, L_forward_packet_views=170496,
            OE_sampled_exposures=OE_BATCH * ADDED_STEPS if oe_next is not None else 0,
            L_smoke_samples=2, OE_smoke_samples=2 if oe_next is not None else 0,
            checkpoint='final.pt', seconds=time.perf_counter() - start,
            claim='independent Phase3 source increment; does not replace historical E50 or establish open-set performance')
        checkpoint = dict(schema=SCHEMA, model={key: value.detach().cpu() for key, value in model.state_dict().items()},
            config=config, identity=_checkpoint_identity(config), original_parent_identity=model._phase3_parent_identity,
            source_data_handle=model._phase3_source_handle, source_OE_handle=oe_handle, arm=config['arm'],
            updates=ADDED_STEPS, cumulative_updates=11322, optimizer_reset=True,
            U_reads=0, validation_reads=0, target_reads=0, training_summary=summary)
        with (output / 'final.pt').open('xb') as stream:
            torch.save(checkpoint, stream)
        runner._write(output / 'training.json', summary)
        runner._write(output / 'complete.json', summary)
        return summary
    finally:
        if isinstance(oe_iq, np.memmap):
            oe_iq._mmap.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args(argv)
    result = train(runner._read(args.config), args.device)
    print(json.dumps(dict(status=result['status'], arm=result['arm'], updates=result['updates'])))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
