"""Regression checks for code-preserving repairs, using synthetic inputs only."""
import argparse
import ast
import contextlib
import importlib
import importlib.util
import io
import json
import os
import pickle
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from audit_smoke import source_path, model_step, native_loop


class Writer:
    def add_scalar(self, *args, **kwargs):
        pass


def run_cli(folder, entry, arguments, output):
    env = dict(os.environ, OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', PYTHONUTF8='1', CUDA_VISIBLE_DEVICES='')
    result = subprocess.run([sys.executable, '-X', 'utf8', str(folder / entry), *arguments],
        cwd=folder, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        encoding='utf-8', timeout=180)
    output.write_text(result.stdout, encoding='utf-8')
    if result.returncode:
        raise RuntimeError(f'CLI failed: {entry}; evidence {output}\n{result.stdout[-4000:]}')


def difl_checks(folder, upstream, output):
    with source_path(folder):
        mod = importlib.import_module('train_DIFEX1')
        pre = importlib.import_module('pretrain')
        net = importlib.import_module('Network1')
        teacher_net = importlib.import_module('teacherNet')
        spec = importlib.util.spec_from_file_location('original_network', upstream / 'Network1.py')
        original = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(original)
        old = original.DNN().eval()
        new = net.DNN().eval()
        x = torch.randn(8, 2, 256)
        with torch.no_grad():
            _, old_logits = old(x)
        new.load_state_dict(old.state_dict(), strict=True)
        with torch.no_grad():
            features, new_logits = new(x)
        assert features.shape == (8, 512)
        assert torch.equal(old_logits, new_logits), 'Classification function changed'
        assert {n: tuple(v.shape) for n, v in old.state_dict().items()} == {
            n: tuple(v.shape) for n, v in new.state_dict().items()}
        # Covariance formulation and the original loss weights remain unchanged.
        f1, f2 = torch.randn(4, 256), torch.randn(4, 256)
        cx, cy = f1 - f1.mean(0), f2 - f2.mean(0)
        reference = (f1.mean(0)-f2.mean(0)).square().mean() + (
            cx.T@cx/3 - cy.T@cy/3).square().mean()
        assert torch.allclose(mod.coral(f1, f2), reference)
        sampler = mod.MultiFileBatchSampler([list(range(12)), list(range(12, 28))], 4)
        assert len(sampler) == len(list(sampler)) == 3
        # Interleave receiver groups to ensure alignment follows domain IDs.
        domain_ids = torch.tensor([0, 1, 0, 1, 0, 1, 0, 1])
        loader = DataLoader(TensorDataset(x, torch.arange(8)%6, domain_ids), batch_size=8)
        teacher = teacher_net.DNN().eval()
        teacher(torch.randn(8, 1, 256))
        teacher_state = {n: v.clone() for n, v in teacher.state_dict().items()}
        metrics = mod.Metrics(output / 'direct_student_logs', {'fixture': 'random tensors'})
        log = mod.setup_logger(str(output / 'direct_logs'), 'direct')
        model = net.DNN()
        model(x)  # initialize lazy layers
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        with contextlib.redirect_stdout(io.StringIO()):
            mod.train(model, loader, opt, 1, Writer(), 'cpu', log,
                teacher=teacher, lam_value=0.05, beta_value=0.01, metrics=metrics)
        record = json.loads(metrics.step_path.read_text(encoding='utf-8').splitlines()[0])
        expected = (record['ce'] + 0.05 * record['kd'] + 0.01 * record['coral']) / 1.06
        assert abs(record['loss'] - expected) < 1e-6
        assert record['coral'] > 0 and record['kd'] > 0 and record['receiver_pairs'] == 1
        assert torch.equal(teacher.state_dict()['fc1.weight'], teacher_state['fc1.weight'])
        assert all(torch.equal(v, teacher_state[n]) for n, v in teacher.state_dict().items())
        assert all(p.grad is None for p in teacher.parameters())
        before = {n: v.clone() for n, v in model.state_dict().items()}
        singleton = DataLoader(TensorDataset(x[:3], torch.arange(3)%6, domain_ids[:3]), batch_size=2)
        mod.evaluate(model, torch.nn.CrossEntropyLoss(), singleton, 1, Writer(), 'cpu', log)
        assert all(torch.equal(v, before[n]) for n, v in model.state_dict().items())
        test_loader = DataLoader(TensorDataset(x[:3], torch.arange(3)%6), batch_size=1)
        pred, _, _, _ = mod.test(model, test_loader, log)
        assert len(pred) == 3
        support = importlib.import_module('runtime_support')
        support.save_state(model, output / 'synthetic_student.pth')
        restored = support.load_state(net.DNN(), output / 'synthetic_student.pth', 'cpu')
        restored.eval(); model.eval()
        with torch.no_grad():
            assert torch.equal(restored(x)[1], model(x)[1])
        # Legacy compatibility is tested only with our own synthetic saved object.
        torch.save(model, output / 'synthetic_legacy.pth')
        support.load_state(net.DNN(), output / 'synthetic_legacy.pth', 'cpu', allow_legacy=True)
        loader_module = importlib.import_module('data_loader')
        raw = np.random.default_rng(300).normal(size=(48, 256, 2)).astype('float32')
        labels = np.tile([6, 1, 5, 2, 4, 3], 8)
        raw[:, 0, 0] = np.arange(48)
        np.save(output / 'labels.npy', labels)
        sources = []
        for receiver in range(2):
            file = output / f'Rx{receiver}.npy'
            np.save(file, raw + receiver * 100)
            sources.append(str(file))
        selected, y = loader_module.read_test_data(sources[0], output / 'labels.npy', dev_range=[0, 2])
        assert np.array_equal(selected[:, 0, 0], np.arange(48)[np.isin(labels-1, [0, 2])])
        assert np.array_equal(y, (labels-1)[np.isin(labels-1, [0, 2])])
        split = loader_module.read_train_data(sources[0], 0, label_path=output / 'labels.npy')
        assert not set(split[0][:, 0, 0]) & set(split[1][:, 0, 0])
    print('DIFL direct regression checks passed', flush=True)
    args = ['--source-files', *sources, '--label-path', str(output / 'labels.npy'),
            '--epochs', '2', '--batch-size', '4', '--device', 'cpu']
    run_cli(folder, 'pretrain.py', [*args, '--output-dir', str(output / 'teacher_run'),
            '--log-dir', str(output / 'teacher_logs')], output / 'teacher_cli.txt')
    print('DIFL teacher CLI passed', flush=True)
    run_cli(folder, 'train_DIFEX1.py', [*args, '--output-dir', str(output / 'student_run'),
            '--log-dir', str(output / 'student_logs'), '--teacher-checkpoint',
            str(output / 'teacher_run/pretrain.pth'), '--code-state', 'only_train'], output / 'student_cli.txt')
    assert (output / 'student_run/model.pth').exists()
    for kind in ('teacher', 'student'):
        logs = output / (kind + '_logs')
        assert len((logs / 'epochs.jsonl').read_text(encoding='utf-8').splitlines()) == 2
        assert (logs / 'epochs.csv').exists() and (logs / 'steps.jsonl').exists()
    run_cli(folder, 'train_DIFEX1.py', [*args, '--output-dir', str(output / 'student_run'),
            '--log-dir', str(output / 'student_test_logs'), '--teacher-checkpoint',
            str(output / 'teacher_run/pretrain.pth'), '--code-state', 'only_test', '--test-data', sources[0],
            '--test-labels', str(output / 'labels.npy'), '--plot'], output / 'student_test_cli.txt')
    assert (output / 'student_run/predictions.npz').exists() and (output / 'student_run/after.pdf').exists()
    return dict(classification_bitwise_equal=True, parameter_shapes_unchanged=True,
        original_loss_formula_preserved=True, teacher_frozen=True, sampler_length_correct=True,
        IQ_label_pairing_preserved=True, singleton_batches_passed=True, safe_and_legacy_checkpoints_passed=True,
        two_epoch_teacher_and_student_CLI=True, test_and_plot_CLI=True, detailed_logs_jsonl_csv=True)


def native_cli(method, folder, output):
    with source_path(folder):
        loader = importlib.import_module('utils.load_data')
        receivers = (loader.RX_INDEXES_MANYSIG if method == 'asknet' else loader.rx_indexes_of_manysig)
    data_root = output / (method + '_data')
    for receiver in receivers:
        for day in (1, 2, 3, 4):
            file = data_root / 'ManySig/non_equalized' / f'date{day}' / f'rx_{receiver}_data.pkl'
            file.parent.mkdir(parents=True, exist_ok=True)
            data = [np.random.default_rng(tx+day).normal(size=(3, 256, 2)).astype('float32') for tx in range(6)]
            with file.open('wb') as f:
                pickle.dump({'data': data}, f)
    args = ['--dataset_name', 'ManySig', '--code_state', 'train_test', '--epochs', '1',
            '--batch_size', '128', '--test_round', '0', '--gpu', '', '--data-root', str(data_root),
            '--output-root', str(output / (method + '_cli_run')), '--log-root', str(output / (method + '_cli_logs'))]
    run_cli(folder, 'main.py', args, output / (method + '_cli.txt'))
    logs = list((output / (method + '_cli_logs')).glob('*/resolved_config.json'))
    assert len(logs) == 1
    config = json.loads(logs[0].read_text(encoding='utf-8'))
    assert config['dataset_name'] == 'ManySig' and config['test_round'] == 0
    assert not set(config['source_receivers']) & set(config['test_receivers'])
    assert len(list((output / (method + '_cli_run/weights')).glob('*.pth'))) == 1
    return {'native_train_test_CLI': True, 'portable_input_output_logs': True, 'resolved_config_complete': True}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--working-root', type=Path, required=True)
    p.add_argument('--upstream-root', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    args = p.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    torch.manual_seed(300)
    result = {'data': 'synthetic only', 'python': sys.executable, 'torch': torch.__version__, 'checks': {}}
    for method in ('asknet', 'wavemlp'):
        folder = args.working_root / method / ('WiSig' if method == 'asknet' else '')
        check = model_step(method, folder)
        check.update(native_loop(method, folder, args.output_dir))
        logs = args.output_dir / (method + '.pth.logs')
        for name in ('resolved_config.json', 'training.log', 'steps.jsonl', 'epochs.jsonl', 'epochs.csv'):
            assert (logs / name).exists(), name
        check['detailed_logs_jsonl_csv'] = True
        check.update(native_cli(method, folder, args.output_dir))
        result['checks'][method] = check
        print(f'{method} regressions passed', flush=True)
    result['checks']['difl'] = difl_checks(args.working_root / 'difl',
        args.upstream_root / 'Receiver-agnostic-RFFI-CL', args.output_dir)
    for path in args.working_root.rglob('*.py'):
        ast.parse(path.read_text(encoding='utf-8-sig'))
    result['status'] = 'PASS'
    (args.output_dir / 'repair_results.json').write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
