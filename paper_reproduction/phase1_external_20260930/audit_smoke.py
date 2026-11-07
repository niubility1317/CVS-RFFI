"""Synthetic-only checks; no WiSig IQ or author checkpoints are read."""
import argparse
import ast
import contextlib
import importlib
import io
import json
import sys
import traceback
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset, BatchSampler


@contextlib.contextmanager
def source_path(path):
    # Author repositories reuse top-level names ('main', 'utils', 'backbones').
    prefixes = ('main', 'utils', 'backbones', 'Network1', 'teacherNet', 'complex')
    saved = {n: m for n, m in sys.modules.items()
             if any(n == p or n.startswith(p + '.') for p in prefixes)}
    for n in saved:
        del sys.modules[n]
    sys.path.insert(0, str(path))
    try:
        yield
    finally:
        sys.path.pop(0)
        for n in list(sys.modules):
            if any(n == p or n.startswith(p + '.') for p in prefixes):
                del sys.modules[n]
        sys.modules.update(saved)


def model_step(method, folder):
    with source_path(folder):
        if method == 'asknet':
            model = importlib.import_module('backbones.PatchNet').PatchNet()
        elif method == 'wavemlp':
            model = importlib.import_module('backbones.MyModel').MyModel(
                wavelet_levels=4, wavelet_kernel_size=16)
        else:
            model = importlib.import_module('Network1').DNN()
        x = torch.randn(4, 2, 256)
        y = torch.tensor([0, 1, 2, 3])
        model.train()
        result = model(x)
        logits = result if method == 'asknet' else result[0 if method == 'wavemlp' else 1]
        assert logits.shape == (4, 6)
        loss = torch.nn.functional.cross_entropy(logits, y)
        loss.backward()
        gradients = [p.grad for p in model.parameters() if p.grad is not None]
        assert gradients and all(torch.isfinite(g).all() for g in gradients)
        assert torch.isfinite(loss)
        before = model.cls_head.weight.detach().clone() if method != 'difl' else model.fc4.weight.detach().clone()
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        opt.step()
        after = model.cls_head.weight if method != 'difl' else model.fc4.weight
        assert not torch.equal(before, after)
        model.eval()
        state = {n: p.detach().clone() for n, p in model.state_dict().items()}
        with torch.no_grad():
            model(x)
        assert all(torch.equal(p, state[n]) for n, p in model.state_dict().items())
        if method == 'difl':
            teacher = importlib.import_module('teacherNet').DNN()
            tf, _ = teacher(torch.angle(torch.fft.fft(torch.complex(x[:, 0], x[:, 1]))).unsqueeze(1))
            assert result[0].shape[1] == 256 and tf.shape[1] == 256
            assert result[0][:, 256:].shape[1] == 0
        return {'finite_loss_and_gradients': True, 'optimizer_updated': True,
                'eval_state_unchanged': True, 'logits_shape': list(logits.shape)}


def native_loop(method, folder, output):
    with source_path(folder):
        m = importlib.import_module('main')
        model = (m.PatchNet() if method == 'asknet' else
                 m.MyModel(wavelet_levels=4, wavelet_kernel_size=16))
        ds = TensorDataset(torch.randn(8, 2, 256), torch.arange(8) % 6)
        loader = DataLoader(ds, batch_size=4)
        kwargs = dict(model=model, train_loader=loader, val_loader=loader, epochs=1,
                      save_path=str(output / (method + '.pth')), device='cpu')
        with contextlib.redirect_stdout(io.StringIO()):
            m.train_and_evaluate(**kwargs)
        state = torch.load(kwargs['save_path'], weights_only=True, map_location='cpu')
        model.load_state_dict(state, strict=True)
        # Check all native LOGO folds and source-only validation preparation.
        for count in (12, 32):
            test_sets = []
            for fold in range(4):
                train, test = m.split_receivers(count, 4, fold)
                assert not set(train) & set(test)
                assert set(train) | set(test) == set(range(count))
                test_sets.extend(test)
            assert sorted(test_sets) == list(range(count))
        def fixture(dataset, receiver, day, tx_count, representation):
            signals = np.ones((20, 2, 256), dtype=np.float32)
            signals[:, 0, 0] = np.arange(20) + 1000 * receiver + 100 * day
            return signals, np.arange(20) % tx_count
        m.load_single_dataset = fixture
        train, val = m.prepare_dataset('ManySig', [0, 1], [1, 2], 6, True, 2023)
        assert not set(train[0][:, 0, 0]) & set(val[0][:, 0, 0])
        assert len(train[0]) + len(val[0]) == 80
        return {'one_epoch_native_loop': True, 'state_dict_roundtrip': True,
                'all_receiver_folds_disjoint': True, 'source_split_disjoint': True}


def extract_difl(path):
    tree = ast.parse(path.read_text(encoding='utf-8'))
    nodes = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))
             and n.name in ('coral', 'MultiFileBatchSampler')]
    ns = {'torch': torch, 'BatchSampler': BatchSampler}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), ns)
    with torch.no_grad():
        value = ns['coral'](torch.randn(32, 0), torch.randn(32, 0))
    assert torch.isnan(value), 'Expected upstream empty-branch NaN no longer reproduced'
    sampler = ns['MultiFileBatchSampler']([list(range(64)), list(range(64, 128))], 32)
    emitted = list(sampler)
    assert len(sampler) == 4 and len(emitted) == 2
    main = next(n for n in tree.body if isinstance(n, ast.If) and
                '__name__' in ast.unparse(n.test))
    calls = [n.func.id for n in ast.walk(main) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)]
    assert 'train_and_evaluate' not in calls
    assert any(isinstance(n, ast.Name) and n.id == 'colors' for n in ast.walk(main))
    assert not any(isinstance(n, ast.Name) and n.id == 'colors' and isinstance(n.ctx, ast.Store)
                   for n in ast.walk(tree))
    return {'empty_coral_is_nan': True, 'sampler_reported_batches': len(sampler),
            'sampler_actual_batches': len(emitted), 'student_training_call_missing': True,
            'plot_colors_undefined': True}


def upstream_failures(root):
    found = {}
    wave = root / 'WaveMLP-RFF'
    assert not (wave / 'train.py').exists() and not (wave / 'eval.py').exists()
    with source_path(wave):
        m = importlib.import_module('main')
        ds = TensorDataset(torch.randn(4, 2, 256), torch.arange(4))
        try:
            m.train_and_evaluate(m.MyModel(), DataLoader(ds, batch_size=4),
                                 DataLoader(ds, batch_size=4), 1, 'unused.pth', device='cpu')
        except TypeError as e:
            assert 'verbose' in str(e)
            found['wavemlp_scheduler_error'] = str(e)
        else:
            raise AssertionError('Expected PyTorch verbose incompatibility not reproduced')
        w = importlib.import_module('backbones.WaveletOperator').LearnableWavelet1D()
        low, high = w.analysis(torch.randn(2, 1, 256, dtype=torch.cfloat))
        try:
            w.synthesis(low, high, out_len=257)
        except NameError as e:
            assert "'F'" in str(e)
            found['wavemlp_padding_error'] = str(e)
        else:
            raise AssertionError('Expected missing functional import not reproduced')
    for method, folder in [('asknet', root / 'ASKNet-RFF/WiSig'), ('wavemlp', wave)]:
        with source_path(folder):
            loader = importlib.import_module('utils.load_data')
            with np.errstate(invalid='ignore'):
                result = loader.preprocessing(np.zeros((1, 2, 256), dtype=np.float32))
            assert not np.isfinite(result).all()
            found[method + '_zero_power_nan'] = True
    found['difl'] = extract_difl(root / 'Receiver-agnostic-RFFI-CL/train_DIFEX1.py')
    return found


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--upstream-root', type=Path, required=True)
    p.add_argument('--working-root', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    args = p.parse_args()
    # Unique output directories preserve all prior evidence.
    args.output_dir.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    torch.manual_seed(2023)
    result = {'environment': {'python': sys.executable, 'torch': torch.__version__,
              'device': 'cpu'}, 'data': 'synthetic random tensors only', 'checks': {}}
    try:
        result['upstream_failures_reproduced'] = upstream_failures(args.upstream_root)
        for method in ('asknet', 'wavemlp', 'difl'):
            folder = args.working_root / method
            if method == 'asknet':
                folder = folder / 'WiSig'
            checks = model_step(method, folder)
            if method != 'difl':
                checks.update(native_loop(method, folder, args.output_dir))
                with source_path(folder):
                    loader = importlib.import_module('utils.load_data')
                    try:
                        loader.preprocessing(np.zeros((1, 2, 256), dtype=np.float32))
                    except ValueError:
                        checks['zero_power_rejected'] = True
                    else:
                        raise AssertionError('Zero-power input silently accepted')
                    if method == 'wavemlp':
                        w = importlib.import_module('backbones.WaveletOperator').LearnableWavelet1D()
                        low, high = w.analysis(torch.randn(2, 1, 256, dtype=torch.cfloat))
                        assert w.synthesis(low, high, 257).shape[-1] == 257
                        checks['padding_branch_fixed'] = True
            result['checks'][method] = checks
        result['status'] = 'PASS_WITH_DIFL_FORMAL_USE_BLOCKED'
    except Exception:
        result['status'] = 'FAIL'
        result['error'] = traceback.format_exc()
        raise
    finally:
        (args.output_dir / 'audit_results.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
        print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
