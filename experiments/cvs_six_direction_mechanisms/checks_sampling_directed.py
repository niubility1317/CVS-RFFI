"""Focused mathematical and source-boundary checks for directions 2 and 5."""
from pathlib import Path
import json
import tempfile

import torch
from torch import nn

from . import sampling


def checks():
    try:
        from . import directed
    except ImportError as exc:
        raise AssertionError('Direction 5 implementation is required by this check') from exc

    # The S2 representation must be lossless, and integer shifts must use the
    # phase-exchange/carry action rather than an arbitrary phase concatenation.
    sampling_result = sampling.structure_checks('cpu')
    assert sampling_result['passed'], sampling_result

    # For a shift of two input samples at stride two, the exact aligned crop has
    # zero error. A one-sample shift must remain an explicit floor/ceil bracket.
    x = torch.arange(64, dtype=torch.float32).reshape(1, 1, 64)
    reference = x[..., ::2]
    shifted2 = sampling.shift(x, 2)[..., ::2]
    exact = sampling.paired_sums(reference, shifted2, 2, 2, 0., 1, 64)
    assert exact['exact_alignment'] and exact['alternatives'][0]['relative_L2'].max() == 0
    bracket = sampling.paired_sums(reference, sampling.shift(x, 1)[..., ::2], 1, 2, 0., 1, 64)
    assert not bracket['exact_alignment'] and len(bracket['alternatives']) == 2
    assert 'bracket' in bracket['alignment']

    class NativeT3(nn.Module):
        def __init__(self):
            super().__init__()
            self.dw = nn.Conv1d(96, 96, 3, padding=1, groups=96, bias=False)
            self.pw = nn.Conv1d(96, 96, 1, bias=False)
            self.norm = nn.GroupNorm(16, 96)
            self.act = nn.ReLU()
            self.drop = nn.Dropout1d(0.)

        def forward(self, x):
            return self.drop(self.act(self.norm(self.pw(self.dw(x)))))

    replacements = sampling.replacements(NativeT3(), 'cpu')
    local = torch.randn(3, 96, 64)
    shapes = {name: list(module.eval()(local).shape) for name, module in replacements.items()}
    assert shapes['ordinary_global_GN'] == [3, 96, 32]
    assert shapes['ordinary_point_GN'] == [3, 96, 32]
    assert shapes['lowpass_point_GN'] == [3, 96, 32]
    assert shapes['carry_point_GN'] == [3, 96, 64]

    # Build the real NoCFO architecture from its registered source config with
    # random weights only. This checks actual hook paths and whole-model swaps,
    # without loading a checkpoint or touching source/target data.
    from . import design as six_direction_design
    from experiments.cvs_scratch30_baseline import design as scratch30_design
    from experiments.cvs_nocfo_foundation.model import build as build_nocfo
    from experiments.cvs_source_pseudotarget_health.frozen import preserve_rng

    row = next(row for row in scratch30_design.rows()
               if row['arm'] == 'CE30' and row['model_seed'] == six_direction_design.SEEDS[0])
    source_config = scratch30_design.config(row)['source_loader_config']
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(2)
    try:
        with preserve_rng('cpu'):
            real_model = build_nocfo(source_config, 'cpu').eval().requires_grad_(False)
        synthetic_iq = torch.randn(2, 2, 256)
        with sampling.capture_levels(real_model) as (actual_maps, actual_native):
            actual_output = real_model.forward_identity_only(synthetic_iq)
        assert actual_output['id_feat_joint'].shape == (2, 160)
        assert actual_maps['time_down'].shape == (2, 48, 128)
        assert actual_maps['t2.pool'].shape == (2, 96, 32)
        shallow_descriptor = torch.cat((actual_maps['time_down'].mean(-1),
                                        actual_maps['time_down'].std(-1, unbiased=False)), dim=1)
        static_descriptor = torch.cat((actual_maps['t2.pool'].mean(-1),
                                       actual_maps['t2.pool'].std(-1, unbiased=False)), dim=1)
        assert shallow_descriptor.shape == (2, 96) and static_descriptor.shape == (2, 192)
        assert not real_model._identity_input_policy['explicit_CFO_identity_input']
        def source_cache(prefix, count):
            return dict(x=torch.randn(count, 2, 256), y=torch.arange(count) % 6,
                        rx=torch.tensor([[1, 3, 4, 6, 8][i % 5] for i in range(count)]),
                        day=torch.tensor([[1, 2, 3][i % 3] for i in range(count)]),
                        ids=[f'{prefix}{i}' for i in range(count)])
        sampling_context = dict(model=real_model, L=source_cache('L', 9), V=source_cache('V', 3),
                                folds=torch.arange(9) % 3)
        temp_root = Path(__file__).resolve().parents[2] / '.codex_tmp'
        temp_root.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(prefix='cvs_sampling_native_', dir=temp_root) as tmp:
            output = Path(tmp) / 'sampling'
            sampling_summary = sampling.diagnose(sampling_context, output, 'cpu')
            sampling_audit = torch.load(output / 'audit.pt', map_location='cpu', weights_only=False)
            assert sampling_summary['applicability']['criteria']['structural']
            assert sampling_summary['identity_state']['parameters_buffers_exactly_unchanged']
            assert set(sampling_summary['costs']) == {'ordinary_global_GN', 'ordinary_point_GN',
                                                       'lowpass_point_GN', 'carry_point_GN'}
            assert all(value['t3_MAC_per_packet'] > 0 and value['output_bytes_per_packet'] > 0
                       and value['whole_model_parameters'] > 0 and value['whole_model_forward_seconds'] > 0
                       for value in sampling_summary['costs'].values())
            assert sampling_audit['roles']['L_s']['values']['base_joint160'].shape == (9, 160)
    finally:
        torch.set_num_threads(previous_threads)

    directed_result = directed.structure_checks('cpu')
    assert directed_result['passed'], directed_result

    # Exercise the complete source-only linear-probe path on a small synthetic
    # L/V fixture. No U role or target input is supplied to this context.
    generator = torch.Generator().manual_seed(20261013)
    n_l, n_v = 108, 36
    y_l = (torch.arange(n_l) // 3) % 6
    y_v = (torch.arange(n_v) // 3) % 6
    prototype = torch.randn(6, 160, generator=generator)
    joint_l = prototype[y_l] + .35 * torch.randn(n_l, 160, generator=generator)
    joint_v = prototype[y_v] + .35 * torch.randn(n_v, 160, generator=generator)
    sequence_l = torch.randn(n_l, 96, 32, generator=generator)
    sequence_v = torch.randn(n_v, 96, 32, generator=generator)
    L = dict(ids=[f'L{i}' for i in range(n_l)], y=y_l,
             rx=torch.tensor([[1, 3, 4, 6, 8][(i // 3) % 5] for i in range(n_l)]),
             day=torch.tensor([[1, 2, 3][(i // 3) % 3] for i in range(n_l)]))
    V = dict(ids=[f'V{i}' for i in range(n_v)], y=y_v,
             rx=torch.tensor([[1, 3, 4, 6, 8][(i // 3) % 5] for i in range(n_v)]),
             day=torch.tensor([[1, 2, 3][(i // 3) % 3] for i in range(n_v)]))
    context = dict(
        L=L, V=V, folds=torch.arange(n_l) % 3, seed=2026092701, device='cpu',
        features={
            'L_s': {'joint160': joint_l, 'middle_sequence': sequence_l},
            'V': {'joint160': joint_v, 'middle_sequence': sequence_v},
        },
    )
    temp_root = Path(__file__).resolve().parents[2] / '.codex_tmp'
    temp_root.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='cvs_sampling_directed_', dir=temp_root) as tmp:
        output = Path(tmp) / 'directed'
        result = directed.diagnose(context, output, 'cpu')
        audit = torch.load(output / 'audit.pt', map_location='cpu', weights_only=False)
        assert output.is_dir() and (output / 'diagnostic.json').is_file()
        assert len(result['probes']) == 12
        assert len(audit['probes']) == 12
        for record in audit['probes'].values():
            train_ids = set(record['training_ids'])
            eval_ids = set(record['L_fold2_ids'])
            assert not train_ids & eval_ids
            assert record['fitted_steps'] == 100
        try:
            directed.diagnose(context, output, 'cpu')
        except FileExistsError:
            pass
        else:
            raise AssertionError('Diagnostic output directory must be exclusive')

    return dict(status='PASS', checks=[
        'S2 inverse, phase exchange/carry, and arbitrary phase-concatenation counterexample',
        'exact aligned shift versus non-exact phase-bracket reporting',
        'all four local t2 replacement output shapes',
        'actual random NoCFO build, native 96/192/160 feature capture, source diagnose, and four full-model replacement timings',
        'directed reversal, lag-zero, shuffle, and gradient recovery properties',
        '12 source-only probes, held-fold exclusion, full audit, and exclusive output root',
    ], sampling=sampling_result, directed=directed_result,
       synthetic_source_fixture=True, target_access=False, U_truth_read=False)


if __name__ == '__main__':
    print(json.dumps(checks(), ensure_ascii=False))
