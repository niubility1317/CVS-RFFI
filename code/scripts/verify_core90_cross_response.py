"""Bounded synthetic integration verification, never a formal WiSig experiment.

The actual SSDG train entrypoint, CORE90 model, optimizer, EMA, data loader and
cross-response runtime execute. Only test budgets and gate thresholds change;
expensive unrelated tail/leakage and final heldout reports are explicitly omitted.
"""
from __future__ import annotations

import argparse
import contextlib
import gc
import json
import os
import pickle
import shutil
import sys
import time
from pathlib import Path
from unittest.mock import patch

import numpy as np

CODE = Path(__file__).resolve().parents[1]
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))
from scripts.core90_cross_response_matrix import DEFAULT_CONFIG, VARIANTS, V2_VARIANTS, build_matrix, load_config

TEST_BOUNDARIES = {
    'dataset': 'entirely synthetic, never actual source/target records',
    'model': 'real CORE90 lite_d dual backbone',
    'epochs': 'bounded fixture override after authoritative method configuration',
    'gate': 'lenient synthetic activation threshold; not a scientific activation claim',
    'tail_and_leakage': 'unrelated expensive reports omitted by test patch',
    'heldout_evaluation': 'omitted; no target selection or target score claim',
    'initialization': 'scratch only; no external checkpoint loaded',
}


def make_fixture(path: Path, *, records_per_cell: int = 128, seed: int = 413, n_tx: int = 4, n_rx: int = 5) -> dict:
    """Create genuine distinct physical records in the compact WiSig schema."""
    rng = np.random.default_rng(seed)
    n_day, length = 2, 256
    t = np.arange(length, dtype=np.float32) / length
    data = []
    for tx in range(n_tx):
        receivers = []
        for rx in range(n_rx):
            days = []
            for day in range(n_day):
                records = []
                for record in range(records_per_cell):
                    phase = .12 * rx + .04 * day + rng.normal(0, .01)
                    # Separate spectral fingerprints, with TX/RX interaction and
                    # unique noise. Frequency target varies across TX and RX.
                    frequency = 8 + 20 * tx + .7 * rx + .13 * tx * rx
                    iq = np.exp(2j * np.pi * frequency * t + 1j * phase)
                    iq += (.2 + .025 * rx) * np.exp(2j * np.pi * (frequency + 9 + tx) * t)
                    iq *= 1 + .06 * np.sin(2 * np.pi * (tx + 1) * t)
                    iq += .015 * (rng.normal(size=length) + 1j * rng.normal(size=length))
                    records.append(np.stack([iq.real, iq.imag], axis=-1).astype(np.float32))
                days.append([np.stack(records)])
            receivers.append(days)
        data.append(receivers)
    payload = dict(data=data, tx_list=list(range(n_tx)), rx_list=list(range(n_rx)),
                   capture_date_list=[0, 1], equalized_list=[1])
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return dict(path=str(path), tx=n_tx, rx=n_rx, days=n_day, records_per_cell=records_per_cell,
                iq_length=length, unique_physical_records=n_tx*n_rx*n_day*records_per_cell,
                source_receivers=list(range(n_rx-1)), target_receivers=[n_rx-1], source_days=[0], target_days=[1],
                synthetic=True, seed=seed)


def make_test_config(path: Path) -> None:
    config = load_config(DEFAULT_CONFIG)
    config['cross_response'].update(gate_error_ratio=1e9, gate_stable_checks=1,
                                   gate_min_blocks=1, source_eval_max_blocks=2,
                                   source_fit_max_records=128, update_interval=1,
                                   scheduler_candidate_limit=16)
    with path.open('x', encoding='utf-8', newline='\n') as handle:
        json.dump(config, handle, indent=2)


def _omitted(*args, **kwargs):
    return {'status': 'OMITTED_SYNTHETIC_TEST', 'reason': 'unrelated bounded-fixture report'}


def assert_activation(report: dict) -> None:
    c, n, maxima = report['config'], report['counts'], report['maxima']
    assert report['status'] == 'ACTIVE_VERIFIED', report.get('missing')
    assert n['successful_steps'] > 0
    if not c['enabled']:
        assert report['auxiliary_parameters'] == 0
        assert n['effective_blocks'] == 0
        assert n['head_gradient_steps'] == n['joint_gradient_steps'] == n['domain_gradient_steps'] == 0
        return
    assert n['effective_blocks'] > 0
    assert maxima.get('response_grad_shared', 0) == 0
    assert maxima.get('response_grad_identity_front', 0) == 0
    if c['response_enabled']:
        assert report['auxiliary_parameters'] > 0 and n['head_gradient_steps'] > 0
        if c['head_only']:
            assert n['domain_gradient_steps'] == n['joint_gradient_steps'] == 0
        else:
            assert n['domain_gradient_steps'] > 0
            if c['permanent_detach']:
                assert n['joint_gradient_steps'] == 0
            else:
                assert report['gate']['opened'] and n['joint_gradient_steps'] > 0
                assert maxima['response_grad_identity_tail'] > 0
    else:
        assert report['auxiliary_parameters'] == 0 and n['response_batches'] == 0
    if c['decision_enabled']:
        assert n['decision_batches'] > 0 and n['decision_gradient_steps'] > 0
        assert maxima['decision_valid_comparisons'] > 0
    if c['identity_interaction_enabled']:
        assert n['cross_batches'] > 0 and n['cross_gradient_steps'] > 0
        assert maxima['cross_identity_grad_norm'] > 0


def run_synthetic_variant(variant: str, root: Path, *, device: str = 'cuda:0', epochs: int = 2,
                          resume_from: Path | None = None, output_suffix: str = '', deterministic: bool = False,
                          amp: bool = False, runtime_disabled_reference: bool = False,
                          records_per_cell: int = 128, transmitters: int = 4, source_receivers: int = 4) -> dict:
    if deterministic:
        os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    import torch
    from SSDG import train_ssdg as train
    from cvsrffi.cross_response import config as cr_config
    from cvsrffi.cross_response import integration as cr_integration

    if variant not in V2_VARIANTS:
        raise ValueError('unregistered variant')
    if runtime_disabled_reference and (variant != 'U0' or resume_from is not None):
        raise ValueError('only scratch U0 can form the package-absent reference')
    root.mkdir(parents=True, exist_ok=True)
    fixture = root / 'synthetic_wisig.pkl'
    config_path = root / 'synthetic_cross_response_config.json'
    if not fixture.exists():
        manifest = make_fixture(fixture,records_per_cell=records_per_cell,n_tx=transmitters,n_rx=source_receivers+1)
        (root/'fixture_manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    elif json.loads((root/'fixture_manifest.json').read_text(encoding='utf-8'))['records_per_cell'] != records_per_cell:
        raise ValueError('existing fixture has a different physical-record budget')
    manifest = json.loads((root/'fixture_manifest.json').read_text(encoding='utf-8'))
    if manifest['tx'] != transmitters or len(manifest['source_receivers']) != source_receivers:
        raise ValueError('existing fixture has different device counts')
    if not config_path.exists():
        make_test_config(config_path)
    roles = dict(wisig_train_rxs=','.join(map(str,manifest['source_receivers'])),
                 wisig_test_rxs=','.join(map(str,manifest['target_receivers'])),
                 wisig_train_days='0', wisig_test_days='1')
    row = build_matrix(config_path=config_path, wisig_pkl=str(fixture), output_root=root,
                       roles=roles, seeds=[392002], variants=[variant])[0]
    args = train.build_arg_parser().parse_args(row['argv'] +
        ['--device', device, '--num_workers', '0', '--amp', str(bool(amp)).lower(), '--eval_batch_size', '128'])
    if output_suffix:
        if not output_suffix.replace('_', '').isalnum():
            raise ValueError('output_suffix must contain only alphanumeric characters and underscores')
        args.output_dir += '_' + output_suffix
    if resume_from is not None:
        args.cross_response_resume = str(resume_from)
    actual_apply = cr_config.apply_configuration
    actual_build = train.build_baseline_model
    actual_save = train.save_payload
    actual_seed = train.set_seed
    actual_nonfinite = train._first_nonfinite_gradient
    captured = {}

    def deterministic_seed(seed):
        actual_seed(seed)
        if deterministic:
            # The normal project seed helper enables cuDNN benchmarking. The
            # exact-replay fixture explicitly overrides only that test setting.
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True

    @contextlib.contextmanager
    def deterministic_context():
        previous = torch.are_deterministic_algorithms_enabled()
        benchmark, cudnn_det = torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic
        try:
            if deterministic:
                torch.use_deterministic_algorithms(True)
            yield
        finally:
            torch.use_deterministic_algorithms(previous)
            torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic = benchmark, cudnn_det

    def capture_model(*build_args, **build_kwargs):
        model = actual_build(*build_args, **build_kwargs)
        if captured:
            raise AssertionError('synthetic scratch fixture must not construct an external teacher')
        captured['model'] = model
        captured['initial'] = {name: p.detach().cpu().clone() for name, p in model.named_parameters()}
        return model
    overrides = dict(epochs=epochs, label_epochs=1, pseudo_epochs=max(0, epochs-1),
                     eval_max_batches=1, zid_leakage_probe_max_batches=1,
                     source_val_heavy_eval_start_epoch=epochs+1,
                     source_val_heavy_eval_final_window=0,
                     stage1_epochs=1, stage2_epochs=1, stage3_ramp_epochs=1,
                     zid_compact_start_epoch=1, zid_compact_warmup_epochs=0,
                     ow_feat_start_epoch=1, ow_feat_warmup_epochs=0,
                     proxy_unknown_start_epoch=1, proxy_unknown_warmup_epochs=0,
                     soft_unknown_mixup_start_epoch=1, soft_unknown_mixup_warmup_epochs=0,
                     source_episode_start_epoch=1, source_episode_warmup_epochs=0,
                     sat_cons_start_epoch=1)

    def bounded_apply(parsed):
        resolved = actual_apply(parsed)
        for key, value in overrides.items():
            setattr(parsed, key, value)
        return resolved

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    epoch1_path = output / 'epoch1_training_state.pth'
    nonfinite_events = []

    def observe_nonfinite(model):
        result = actual_nonfinite(model)
        if result is not None:
            nonfinite_events.append(dict(result))
            print('[SYNTHETIC-NONFINITE] '+json.dumps(result),flush=True)
            (output/'synthetic_nonfinite_gradients.json').write_text(json.dumps(nonfinite_events,indent=2),encoding='utf-8')
        return result

    def capture_epoch_one(path, payload):
        result = actual_save(path, payload)
        if (Path(path).name == 'cross_response_training_state.pth'
                and payload.get('checkpoint_role') == 'cross_response_epoch_resume'
                and payload.get('epoch') == 1):
            if epoch1_path.exists():
                raise FileExistsError(epoch1_path)
            shutil.copyfile(path, epoch1_path)
        return result
    log_path = output / 'synthetic_entrypoint.log'
    started = time.monotonic()
    runtime_patch = (patch.object(cr_integration, 'CrossResponseRuntime', lambda *a, **kw: None)
                     if runtime_disabled_reference else contextlib.nullcontext())
    with log_path.open('x', encoding='utf-8', newline='\n') as log:
        with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log), deterministic_context(), runtime_patch, \
             patch.object(cr_config, 'apply_configuration', bounded_apply), \
             patch.object(train, 'set_seed', deterministic_seed), \
             patch.object(train, 'build_baseline_model', capture_model), \
             patch.object(train, 'save_payload', capture_epoch_one), \
             patch.object(train, '_first_nonfinite_gradient', observe_nonfinite), \
             patch.object(train, '_evaluate_source_val_tail_geometry', _omitted), \
             patch.object(train, '_evaluate_zid_leakage_probes', _omitted), \
             patch.object(train, '_run_final_heldout_evaluation', _omitted):
            code = train.train(args)
    activation_path = output / 'cross_response_activation.json'
    terminal_path = output / 'phase1_terminal_status.json'
    terminal = json.loads(terminal_path.read_text(encoding='utf-8')) if terminal_path.exists() else {}
    frozen_path = output/'frozen_phase1_heldout_eval.json'
    frozen = json.loads(frozen_path.read_text(encoding='utf-8')) if frozen_path.exists() else {}
    # The only expected diagnostic exit is our deliberately omitted heldout
    # scoring. Independently read both terminal and frozen-evaluation artifacts.
    diagnostic_terminal = (code == 7 and terminal.get('status') == 'HELDOUT_EVAL_INCOMPLETE'
                           and terminal.get('exit_code') == 7 and terminal.get('promotion_ready') is False
                           and terminal.get('heldout_eval', {}).get('status') == 'OMITTED_SYNTHETIC_TEST'
                           and frozen.get('status') == 'OMITTED_SYNTHETIC_TEST')
    # The intentionally package-absent comparison follows the ordinary trainer's
    # legacy terminal classification. This exception never applies to any actual
    # cross-response runtime row.
    if runtime_disabled_reference:
        diagnostic_terminal = diagnostic_terminal or (
            code == 8 and terminal.get('status') == 'NON_PROMOTABLE_P0_DISABLED'
            and terminal.get('exit_code') == 8 and terminal.get('promotion_ready') is False
            and frozen.get('status') == 'OMITTED_SYNTHETIC_TEST')
    if code != 0 and not diagnostic_terminal:
        raise AssertionError(f'real train returned {code}; inspect {log_path}')
    if not activation_path.exists() and not runtime_disabled_reference:
        raise AssertionError(f'real runtime activation artifact missing: {activation_path}')
    activation = None
    if not runtime_disabled_reference:
        activation = json.loads(activation_path.read_text(encoding='utf-8'))
        assert_activation(activation)
    else:
        assert not activation_path.exists(), 'package-absent reference must not emit auxiliary evidence'
    checkpoint_path = output / 'final_ssdg.pth'
    if not checkpoint_path.exists():
        raise AssertionError(f'real final checkpoint missing: {checkpoint_path}')
    changes = {'identity_squared_delta': 0., 'domain_squared_delta': 0., 'all_parameters_finite': True}
    for name, parameter in captured['model'].named_parameters():
        final = parameter.detach().cpu()
        if not torch.isfinite(final).all():
            changes['all_parameters_finite'] = False
        delta = float((final-captured['initial'][name]).float().square().sum())
        if name.startswith('id_backbone.'):
            changes['identity_squared_delta'] += delta
        elif name.startswith('dom_backbone.'):
            changes['domain_squared_delta'] += delta
    assert changes['all_parameters_finite'] and changes['identity_squared_delta'] > 0
    result = dict(variant=variant, status='TRAIN_EXECUTED', activation=activation,
                  train_exit_code=code, terminal_status=terminal.get('status'),
                  elapsed_seconds=time.monotonic()-started, log_path=str(log_path),
                  checkpoint_path=str(checkpoint_path), test_overrides=overrides,
                  training_state_path=str(output/'cross_response_training_state.pth'),
                  epoch1_training_state_path=str(epoch1_path) if epoch1_path.exists() else None,
                  resume_from=str(resume_from) if resume_from is not None else None,
                  deterministic_cuda_fixture=deterministic,
                  amp_enabled=amp,
                  runtime_disabled_reference=runtime_disabled_reference,
                  records_per_cell=records_per_cell, nonfinite_gradient_events=nonfinite_events,
                  parameter_changes=changes, boundaries=TEST_BOUNDARIES, formal_experiment=False)
    if resume_from is not None:
        result['boundaries'] = dict(TEST_BOUNDARIES,
            initialization='validated epoch resume of this synthetic scratch lineage; no historical external weights')
    (output/'synthetic_verification.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


def assert_state_equal(left, right, path='state'):
    """Exact same-host state replay, excluding only declared timing telemetry."""
    import torch
    if torch.is_tensor(left):
        torch.testing.assert_close(left, right, rtol=0, atol=0, equal_nan=True, msg=path)
    elif isinstance(left, np.ndarray):
        np.testing.assert_array_equal(left, right, err_msg=path)
    elif isinstance(left, dict):
        assert left.keys() == right.keys(), path
        for key in left:
            assert_state_equal(left[key], right[key], f'{path}.{key}')
    elif isinstance(left, (list, tuple)):
        assert type(left) is type(right) and len(left) == len(right), path
        for index, (a,b) in enumerate(zip(left,right)):
            assert_state_equal(a,b,f'{path}[{index}]')
    elif isinstance(left, float) and np.isnan(left):
        assert isinstance(right, float) and np.isnan(right), path
    else:
        assert left == right, path


def run_resume_verification(root: Path, *, variant='U5', device='cuda:0', records_per_cell=128) -> dict:
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    import copy
    import torch
    from types import SimpleNamespace
    from cvsrffi.cross_response.integration import CrossResponseRuntime
    uninterrupted = run_synthetic_variant(variant, root, device=device, epochs=2,
                                         output_suffix='uninterrupted', deterministic=True,records_per_cell=records_per_cell)
    epoch1 = Path(uninterrupted['epoch1_training_state_path'])
    resumed = run_synthetic_variant(variant, root, device=device, epochs=2,
                                   resume_from=epoch1, output_suffix='resumed', deterministic=True,records_per_cell=records_per_cell)
    full = torch.load(uninterrupted['training_state_path'], map_location='cpu', weights_only=False)
    replay = torch.load(resumed['training_state_path'], map_location='cpu', weights_only=False)
    checked = ['model','ema_model','optimizer','scaler','prototype_memory','pseudo_temporal_bank','rng_state']
    for key in checked:
        assert_state_equal(full[key], replay[key], key)
    for key in ('auxiliary','statistics','normalizer','sampler','loader_generator','gate',
                'counts','rotation_counts','source_evaluations','config','source_contract'):
        assert_state_equal(full['cross_response'][key], replay['cross_response'][key], 'cross_response.'+key)
    if full['cross_response']['config'].get('implementation_version') == 2:
        for key in ('auxiliary_transaction','mechanism_gate','extended_identity_module'):
            assert_state_equal(full['cross_response'][key],replay['cross_response'][key],'cross_response.'+key)
    for result in (uninterrupted,resumed):
        final = torch.load(result['checkpoint_path'], map_location='cpu', weights_only=False)
        assert_state_equal(full['model'], final['model'], 'final.model')
        assert_state_equal(full['ema_model'], final['ema_model'], 'final.ema_model')
    source = torch.load(epoch1, map_location='cpu', weights_only=False)
    fake_runtime = SimpleNamespace(config=source['cross_response']['config'], variant=variant,
                                   source_contract=source['cross_response']['source_contract'])
    restored_args = SimpleNamespace(**source['args'])
    negatives = []
    for name in ('variant','source_contract','checkpoint_selection','checkpoint_role'):
        bad = copy.deepcopy(source)
        if name in ('variant','source_contract'):
            bad['cross_response'][name] = 'WRONG'
        else:
            bad[name] = 'joint_safe' if name == 'checkpoint_selection' else 'training_final_only'
        try:
            CrossResponseRuntime.validate_checkpoint(fake_runtime, bad, restored_args)
        except ValueError:
            negatives.append(name)
        else:
            raise AssertionError(f'invalid resume accepted: {name}')
    report = dict(status='EXACT_REPLAY_VERIFIED', variant=variant,
                  uninterrupted=uninterrupted, resumed=resumed,
                  tensor_tolerance=dict(rtol=0,atol=0), negative_cases_rejected=negatives,
                  excluded=['timing/max-memory telemetry', 'output metadata', 'wall-clock continuation telemetry'],
                  compared=checked+['cross_response state and final model/EMA'])
    (root/'synthetic_resume_verification.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    return report


def run_u0_off_verification(root: Path, *, device='cuda:0') -> dict:
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    import torch
    configured = run_synthetic_variant('U0',root,device=device,epochs=2,
                                      output_suffix='configured',deterministic=True)
    absent = run_synthetic_variant('U0',root,device=device,epochs=2,
                                  output_suffix='runtime_absent',deterministic=True,
                                  runtime_disabled_reference=True)
    a = torch.load(configured['checkpoint_path'],map_location='cpu',weights_only=False)
    b = torch.load(absent['checkpoint_path'],map_location='cpu',weights_only=False)
    keys = ['model','ema_model','optimizer','scaler','prototype_memory','rng_state']
    for key in keys:
        assert_state_equal(a[key],b[key],key)
    baseline_stats = {k:v for k,v in a['stats']['train'].items()
                      if ('loss' in k or 'grad' in k) and not k.startswith('cross_response/')}
    assert baseline_stats and any('grad' in key for key in baseline_stats)
    for key,value in baseline_stats.items():
        assert_state_equal(value,b['stats']['train'][key],'baseline_stats.'+key)
    report = dict(status='U0_PACKAGE_OFF_EXACT_PARITY',configured=configured,package_absent=absent,
                  compared=keys,baseline_loss_gradient_metrics=len(baseline_stats),
                  tensor_tolerance=dict(rtol=0,atol=0),
                  scope='real two-epoch loop; historical bindings/config remain identical; only runtime constructor bypassed')
    (root/'synthetic_u0_off_verification.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--variants', choices=VARIANTS, nargs='+', default=list(VARIANTS))
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--epochs', type=int, default=2)
    parser.add_argument('--verify-resume', action='store_true')
    parser.add_argument('--amp', action='store_true', help='Exercise actual CUDA GradScaler/autocast routing')
    parser.add_argument('--verify-u0-off', action='store_true')
    parser.add_argument('--records-per-cell', type=int, default=128,
                        help='Synthetic physical records per cell;512 provides4clean batches per epoch')
    args = parser.parse_args()
    if args.epochs < 2 or args.epochs > 4:
        parser.error('synthetic verification supports only 2 to 4 epochs')
    if not 32 <= args.records_per_cell <= 1024:
        parser.error('synthetic records-per-cell must be in32..1024')
    results = []
    if args.verify_u0_off:
        report = run_u0_off_verification(args.output_root,device=args.device)
        print(report['status'],flush=True)
        return
    if args.verify_resume:
        if len(args.variants) != 1:
            parser.error('--verify-resume requires exactly one variant')
        report = run_resume_verification(args.output_root, variant=args.variants[0], device=args.device)
        print(report['status'], flush=True)
        return
    for variant in args.variants:
        print(f'[SYNTHETIC] start {variant}', flush=True)
        result = run_synthetic_variant(variant, args.output_root, device=args.device, epochs=args.epochs,
                                       amp=args.amp,records_per_cell=args.records_per_cell)
        results.append(result)
        print(f'[SYNTHETIC] completed {variant} in {result["elapsed_seconds"]:.1f}s', flush=True)
    report = args.output_root/'synthetic_verification_summary.json'
    report.write_text(json.dumps(results, indent=2), encoding='utf-8')
    print(report, flush=True)


if __name__ == '__main__':
    main()
