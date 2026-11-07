"""Isolate CUDA graph stream-capture behavior using synthetic, non-user data."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

from experiments.phase1_dura_orthogonal.benchmark_flow_n607_graph import wait_for_slot


def worker(case):
    import torch
    from experiments.phase1_dura_orthogonal.benchmark_flow_sync import train
    from experiments.phase1_dura_orthogonal.conditional_iq_flow import ConditionalIQFlow, flow_matching_loss
    from experiments.phase1_dura_orthogonal.flow_cuda_graph import FlowGraphStep

    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    generator = torch.Generator().manual_seed(42)
    iq = tuple(torch.randn((192, 2, 256), generator=generator).cuda() for _ in range(3))
    fields = (*iq, torch.full((192,), .1, device='cuda'), torch.randn((192, 2, 256), generator=generator).cuda())
    warm = train(fields, 'conditional', True, 1)
    del warm
    torch.manual_seed(123)
    model = ConditionalIQFlow(width=32, steps=4).cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=.0003, weight_decay=.001)
    inputs = [*(v[:32] for v in fields), torch.rand(32, generator=generator).cuda()]
    if case == 'baseline':
        stepper = FlowGraphStep(model, optimizer, inputs)
        loss = stepper.step(inputs)
    else:
        static = tuple(x.detach().clone() for x in inputs)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                optimizer.zero_grad(set_to_none=True)
                loss = flow_matching_loss(model, *static, validate_time=False)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=False)
            del loss
        torch.cuda.current_stream().wait_stream(stream)
        optimizer.zero_grad(set_to_none=True)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            loss = flow_matching_loss(model, *static, validate_time=False)
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=False)
        for target, value in zip(static, inputs):
            target.copy_(value)
        graph.replay()
        if not bool(torch.isfinite(loss) & torch.isfinite(norm)):
            raise FloatingPointError('Finite synthetic loss and gradients required')
        optimizer.step()
    torch.cuda.synchronize()
    if not torch.isfinite(loss):
        raise FloatingPointError('Finite synthetic loss required')
    print('CAPTURE_CASE_PASS', case, 'loss', float(loss.detach()), 'torch', torch.__version__,
          'cudnn', torch.backends.cudnn.version(), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--worker', choices=('baseline', 'same_stream'))
    parser.add_argument('--gpu', type=int)
    parser.add_argument('--out', type=Path)
    parser.add_argument('--commit')
    args = parser.parse_args()
    if args.worker:
        worker(args.worker)
        return
    if args.gpu is None or args.out is None or not args.commit:
        raise ValueError('GPU, unique output and code commit required')
    uuid = wait_for_slot(args.out, args.gpu)
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES=uuid, CUBLAS_WORKSPACE_CONFIG=':4096:8',
               OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2')
    results = {}
    for case in ('baseline', 'same_stream'):
        cmd = [sys.executable, '-X', 'utf8', '-u', '-m',
               'experiments.phase1_dura_orthogonal.flow_graph_capture_probe', '--worker', case]
        completed = subprocess.run(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=120)
        results[case] = dict(exit_code=completed.returncode, output=completed.stdout)
        print('CAPTURE_PROBE_CASE', case, 'exit', completed.returncode, flush=True)
    args.out.mkdir(parents=True)
    result = dict(status='CAPTURE_PROBE_COMPLETE', commit=args.commit, physical_gpu=args.gpu, physical_gpu_uuid=uuid,
                  synthetic_only=True, user_data_reads=0, weights_saved=False, results=results)
    (args.out/'result.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n', encoding='utf8')
    print('CAPTURE_PROBE_COMPLETE', flush=True)


if __name__ == '__main__':
    main()
