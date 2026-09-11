from pathlib import Path
import sys
from types import SimpleNamespace
import torch
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'code'), str(ROOT/'code/scripts')]
from SSDG import train_ssdg as train
from run_a1_e600_repair import repair_matrix
import run_a1_fast_v2 as runner
from cvsrffi.a1_budget_schedule import reference_epoch


def test_matched_scratch_full_precision_contract():
    matrix = repair_matrix()
    options = []
    for row in matrix['rows']:
        args = train.build_arg_parser().parse_args(runner.v2_command(matrix,
            project_root=Path('/p'), run_root=Path('/r'), row=row)[3:])
        assert args.epochs == 600 and not args.amp and args.a1_tail_lr == 'continuous'
        assert args.from_scratch and args.a1_scratch_only and not args.baseline_ckpt and not args.teacher_ckpt
        train._validate_a1_scratch_only(args)
        assert args.a1_source_screen_only and not args.use_a1_r3
        assert args.a1_ecrs_cross_rx_scope == 'clean'
        assert args.a1_periodic_target_start == 300 and args.a1_periodic_target_interval == 20
        assert reference_epoch(args, 600) == 200
        optimizer = torch.optim.AdamW([{'params': [torch.nn.Parameter(torch.ones(1))], 'fasttrust_role': 'backbone'}], lr=args.lr)
        for epoch in (1, 238, 300, 540, 600):
            train._apply_fasttrust_lr(optimizer, base_lr=args.lr, epoch=reference_epoch(args, epoch), tail_mode=args.a1_tail_lr)
            assert optimizer.param_groups[0]['lr'] > 0
        options.append({k:v for k,v in row['options'].items() if k not in ('--base_candidate', '--a1_ecrs_cross_rx_weight')})
    assert options[0] == options[1]
    assert [r['options']['--a1_ecrs_cross_rx_weight'] for r in matrix['rows']] == ['0', '0.05']


def test_pool_reserves_contextless_children_and_waits(monkeypatch):
    matrix = {'max_gpu_processes': 2, 'gpu_pool': [0, 1]}
    monkeypatch.setattr(runner, 'gpu_compute_pids', lambda gpu: {'11', '12'} if gpu == 0 else {'21'})
    row = {'gpu': 0}
    assert runner.available_gpu(matrix, row, {}) == 1
    process = SimpleNamespace(pid=22, poll=lambda: None)
    running = {'new': ({'gpu': 1}, process, None)}
    assert runner.available_gpu(matrix, row, running) is None
    process.poll = lambda: 0
    assert runner.available_gpu(matrix, row, running) == 1
    assert runner.available_gpu({'max_gpu_processes': 2}, row, {}) is None
