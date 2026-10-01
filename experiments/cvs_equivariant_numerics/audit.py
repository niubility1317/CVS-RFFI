"""Frozen-source, public-synthetic numerical attribution. No formal IQ or fitting."""
import argparse
from contextlib import contextmanager
import json
import os
from pathlib import Path
import sys
import time

MODES = ('default_fp32', 'cudnn_tf32_off_fp32', 'cudnn_tf32_off_fp64')
SEEDS = (2026092701, 2026092702, 2026092703, 2026092704)
PHASES = (.37, -1.2, 2.9)
BASE_COMMIT = '4e3958bcb085a469dd7fb9829e29de544ea95b09'
PROJECT = '/home/szu2070436088/2510044040/CV-SincNet'
BASE = PROJECT + '/releases/cvs_equivariant_clean_eval_20261002_r01'
SOURCE = PROJECT + '/runs/20261002-phase1-cvs-equivariant-identity-manysig-m4-r01'
RUN = '20261002-diagnostic-cvs-equivariant-numerics-public-m12-r01'


def write(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, allow_nan=False, indent=2)


def flags(torch):
    return dict(cudnn_allow_tf32=torch.backends.cudnn.allow_tf32,
                cuda_matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,
                cudnn_benchmark=torch.backends.cudnn.benchmark,
                cudnn_deterministic=torch.backends.cudnn.deterministic,
                matmul_precision=torch.get_float32_matmul_precision())


@contextmanager
def precision_mode(torch, mode):
    if mode not in MODES:
        raise ValueError('Unregistered precision mode')
    original = torch.backends.cudnn.allow_tf32
    try:
        if mode != 'default_fp32':
            torch.backends.cudnn.allow_tf32 = False
        yield torch.float64 if mode.endswith('fp64') else torch.float32
    finally:
        torch.backends.cudnn.allow_tf32 = original


def capture(model, x):
    """Observe existing modules; hooks return None and cannot alter the forward."""
    import torch
    from torch.nn import functional as F
    traces = {}
    counters = {}
    handles = []

    def hook(name):
        def observe(module, inputs, output):
            count = counters.get(name, 0)
            counters[name] = count + 1
            key = name + '#' + str(count)
            traces[key] = output.detach().clone()
        return observe

    names = ['id_backbone.sinc', 'readout', 'id_backbone.t_proj',
             'id_backbone.pa_proj', 'id_backbone.freq_gate',
             'id_backbone.f1', 'id_backbone.f2', 'id_backbone.f3',
             'id_backbone.f_proj', 'id_backbone.freq_stats_proj',
             'id_backbone.pa_stats_proj', 'id_backbone.fuse']
    names += [f'{branch}.{i}{suffix}' for branch in ('time', 'behavior')
              for i in range(3) for suffix in ('.conv', '')]
    modules = dict(model.named_modules())
    try:
        for name in names:
            if name in modules:
                handles.append(modules[name].register_forward_hook(hook(name)))
        with torch.no_grad():
            # forward_iq_pair is called directly, so a Module hook cannot observe it.
            traces['id_backbone.sinc#0'] = model.id_backbone._sinc_on_iq(x).detach().clone()
            features = model.features(x)
            traces['features'] = features.detach().clone()
            traces['unit_features'] = F.normalize(features, dim=1, eps=1e-4)
            traces['logits'] = model.classify_features(features).detach().clone()
        return traces
    finally:
        for handle in handles:
            handle.remove()


def stage_error(name, baseline, observed, phase):
    from experiments.cvs_equivariant_identity.model import rotate_pair
    if baseline.shape != observed.shape:
        raise ValueError('Stage shape mismatch')
    if name.startswith(('time.', 'behavior.')):
        expected = rotate_pair(baseline, baseline.new_tensor(phase))
        transformation = 'charge1_complex_phase_equivariance'
    elif name.startswith('id_backbone.sinc'):
        # Native Sinc flattens [B,2,24,L] into [B,48,L].
        expected = rotate_pair(baseline.reshape(len(baseline), 2, 24, -1),
                               baseline.new_tensor(phase)).reshape_as(baseline)
        transformation = 'charge1_shared_sinc_equivariance'
    else:
        expected = baseline
        transformation = 'constant_phase_invariance'
    error = (observed - expected).abs()
    return dict(stage=name, transformation=transformation,
                dtype=str(observed.dtype), shape=list(observed.shape),
                max_abs_error=float(error.max()),
                rms_error=float(error.square().mean().sqrt()),
                expected_abs_max=float(expected.abs().max()))


def audit_model(model, x):
    import torch
    from experiments.cvs_equivariant_identity.model import rotate_pair, behavior_basis
    from experiments.cvs_equivariant_identity.physics import controlled_diagnostics
    # Reproduce the original frozen audit first with the exact same code/input order.
    original = controlled_diagnostics(model)
    baseline = capture(model, x)
    phases = []
    for phase in PHASES:
        rotated = rotate_pair(x, x.new_tensor(phase))
        observed = capture(model, rotated)
        stages = [stage_error(name, value, observed[name], phase)
                  for name, value in baseline.items()]
        stages.insert(0, stage_error('behavior.basis', behavior_basis(x),
                                    behavior_basis(rotated), phase))
        phases.append(dict(theta_radians=phase, stages=stages,
            synthetic_argmax_change_count=int((baseline['logits'].argmax(1) != observed['logits'].argmax(1)).sum()),
            whole_unit_embedding_max_distance=float((baseline['unit_features'] - observed['unit_features']).norm(dim=1).max())))
    if any(not torch.isfinite(t).all() for t in baseline.values()):
        raise ValueError('Nonfinite frozen observations')
    return dict(original_audit=original, stage_audit=phases,
                synthetic_identity_accuracy=None,
                argmax_scope='Consistency only; synthetic TX settings have no actual source identity labels')


def main(output, device):
    # Pin the immutable model implementation before importing it.
    if (Path(BASE) / 'release_commit.txt').read_text().strip() != BASE_COMMIT:
        raise ValueError('Immutable model release commit mismatch')
    sys.path[:0] = [BASE, BASE + '/code']
    import numpy as np
    import torch
    from experiments.cvs_clean_eval.contracts import checkpoint_contract, build_model, read
    from experiments.cvs_equivariant_identity.dispatch import read_source_record
    from experiments.cvs_reference_identity.physics import TX_ROWS, RX_ROWS, received
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    expected = read(PROJECT + '/runs/phase1_daot_rc4_pure_game_m3_20260917_r2/source_contract.json')
    initial_flags = flags(torch)
    write(output / 'resolved_config.json', dict(run_id=RUN, pid=os.getpid(), python=sys.executable,
        cwd=os.getcwd(), argv=sys.argv, model_code_commit=BASE_COMMIT, source_training_commit='555aaadea1ab5c40dab7f9d8a80a51b1cd7b5c76',
        torch_version=torch.__version__, device=device, hardware=torch.cuda.get_device_name(device),
        initial_flags=initial_flags, nvidia_tf32_override=os.environ.get('NVIDIA_TF32_OVERRIDE'),
        seeds=list(SEEDS), modes=list(MODES), phases=list(PHASES), synthetic_combinations=30,
        optimizer_steps=0, model_updated=False, formal_iq_access=False, target_access=False,
        target_truth_access=False, target_score_used=False,
        diagnostic_commit=(Path(__file__).parent / 'release_commit.txt').read_text().strip()))
    results = []
    for seed in SEEDS:
        source = Path(SOURCE) / ('equivariant_memory-s' + str(seed)) / 'source'
        row = dict(variant='equivariant_memory', model_seed=seed, source_output=str(source))
        read_source_record(row, expected, 'cvs_equivariant_identity')
        done, initial, contract, resolved = [read(source / n) for n in
            ('completion.json', 'initialization.json', 'source_contract.json', 'resolved_config.json')]
        payload = torch.load(source / 'last.pt', map_location='cpu', weights_only=False)
        checkpoint_contract(row, done, initial, contract, expected, resolved, payload)
        original_phase = read(source / 'source_physical_diagnostics.json')['phase_audit']
        for mode in MODES:
            rid = f's{seed}-{mode}'
            folder = output / rid
            folder.mkdir(exist_ok=False)
            tic = time.perf_counter()
            result = dict(row_id=rid, seed=seed, mode=mode, checkpoint=str(source / 'last.pt'),
                provenance='VERIFIED: own scratch E200 full physical contract/payload matched',
                original_source_phase_audit=original_phase, model_updated=False, target_access=False)
            with precision_mode(torch, mode) as dtype:
                result['flags'] = flags(torch)
                model = build_model('equivariant_memory')
                model.load_state_dict(payload['model'], strict=True)
                model.to(device=device, dtype=dtype).eval().requires_grad_(False)
                result['precision_scope'] = 'Requested model/input dtype; native Sinc filter synthesis/convolution and classifier normalization explicitly retain FP32. FP64 mode is mixed internally, not end-to-end FP64.'
                if seed == SEEDS[0] and mode == MODES[0]:
                    with torch.no_grad():
                        smoke = model(torch.zeros(2, 2, 256, device=device, dtype=dtype))
                    if smoke.shape != (2, 6) or not torch.isfinite(smoke).all():
                        raise ValueError('Real frozen checkpoint no-query smoke failed')
                before = {name: value.detach().clone() for name, value in model.state_dict().items()}
                x = torch.tensor(np.stack([received(t, r) for t in TX_ROWS for r in RX_ROWS]).tolist(),
                                 dtype=dtype, device=device)
                try:
                    result.update(audit_model(model, x), status='DIAGNOSTIC_COMPLETE')
                except RuntimeError as error:
                    if mode != 'cudnn_tf32_off_fp64':
                        raise
                    # Native code explicitly casts some operations to FP32; no model patch is permitted here.
                    result.update(status='UNSUPPORTED_NATIVE_FP64', error=str(error),
                        stage_audit=None, original_audit=None,
                        limitation='Unmodified native model is not guaranteed end-to-end FP64; no diagnostic wrapper or dtype repair applied')
                if any(not torch.equal(value, before[name]) for name, value in model.state_dict().items()):
                    raise ValueError('Frozen model state changed')
                result['state_unchanged'] = True
                result['seconds'] = time.perf_counter() - tic
                write(folder / 'result.json', result)
                results.append(result)
                print('NUMERICAL_ROW ' + json.dumps(dict(row_id=rid, status=result['status'], flags=result['flags'], seconds=result['seconds'])), flush=True)
                del model, x, before
                torch.cuda.empty_cache()
    if flags(torch) != initial_flags:
        raise ValueError('Process backend flags not restored')
    write(output / 'completion.json', dict(status='DIAGNOSTIC_MATRIX_COMPLETE', attempted_rows=12,
        complete_rows=sum(r['status'] == 'DIAGNOSTIC_COMPLETE' for r in results),
        unsupported_rows=sum(r['status'] == 'UNSUPPORTED_NATIVE_FP64' for r in results),
        backend_flags_restored=True, target_access=False, model_updated=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    main(args.output, args.device)
