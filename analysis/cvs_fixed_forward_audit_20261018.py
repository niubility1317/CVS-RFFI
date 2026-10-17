"""Synthetic-only inspection of the actual fixed F release; never loads IQ/CP/truth.

Example: python analysis/cvs_fixed_forward_audit_20261018.py \
  --release /path/to/cvs_FcrossRX_impulse_factorial_E30_20261018_r01 \
  --output /new/path/audit.json

This is structural evidence, not a source-performance or training experiment.
"""
import argparse
import collections
import importlib.util
import inspect
import json
import os
from pathlib import Path
import time


def summarize_graph(model):
    groups = collections.defaultdict(lambda: dict(
        parameters=0, parameters_in_tensors_with_graph_gradient=0,
        parameters_in_tensors_with_any_nonzero_gradient=0,
        nonzero_gradient_elements=0))
    for name, p in model.named_parameters():
        out = groups[name.split('.')[0]]
        out['parameters'] += p.numel()
        if p.grad is not None:
            out['parameters_in_tensors_with_graph_gradient'] += p.numel()
            nonzero = int(p.grad.count_nonzero())
            out['nonzero_gradient_elements'] += nonzero
            if nonzero:
                out['parameters_in_tensors_with_any_nonzero_gradient'] += p.numel()
    return dict(groups)


def run(release):
    os.environ.update(CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='2',
                      MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2')
    spec = importlib.util.spec_from_file_location(
        'fixed_forward_audit_runtime',
        release / 'experiments/cvs_noise_shape_factorial43/runner.py')
    runtime = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runtime)
    runtime.bootstrap()
    import torch
    from experiments.cvs_noise_shape_factorial43 import design as d, model
    from experiments.cvs_clock_factorial30 import channel as tail
    torch.set_num_threads(2)
    torch.manual_seed(202610184401)
    row = next(r for r in d.rows() if r['arm'] == 'B01')
    net = model.build(d.config(row, 20), 'cpu').train()
    before = {k: v.clone() for k, v in net.state_dict().items()}
    norms = []
    for name, layer in net.named_modules():
        if 'norm' in type(layer).__name__.lower():
            norms.append(dict(name=name, type=type(layer).__name__,
                              track_running_stats=getattr(layer, 'track_running_stats', None),
                              buffers=list(dict(layer.named_buffers(recurse=False)))))
    running_buffers = [k for k, _ in net.named_buffers()
                       if k.endswith(('running_mean', 'running_var', 'num_batches_tracked'))]
    called, handles = [], []
    for name, layer in net.named_modules():
        if name:
            def capture(module, inputs, output, name=name):
                called.append(name)
            handles.append(layer.register_forward_hook(capture))
    generator = torch.Generator().manual_seed(202610184402)
    synthetic = torch.randn((2, 2, 256), generator=generator)
    try:
        logits = net.forward_identity_only(synthetic)['tx_logits']
        if logits.shape != (2, 6) or not bool(torch.isfinite(logits).all()):
            raise ValueError('Actual default-head synthetic output is invalid')
        torch.nn.functional.cross_entropy(logits, torch.tensor([0, 1])).backward()
    finally:
        for handle in handles:
            handle.remove()
    groups = summarize_graph(net)
    head = net.id_backbone.encoder.core.id_backbone.cls_head
    head_name = next(k for k, layer in net.named_modules() if layer is head)
    state_unchanged = all(torch.equal(v, net.state_dict()[k]) for k, v in before.items())
    if not state_unchanged or running_buffers:
        raise ValueError('Unexpected persistent-state update or running-stat norm')
    identity = groups['id_backbone']
    if identity['parameters_in_tensors_with_graph_gradient'] != identity['parameters']:
        raise ValueError('An identity parameter is disconnected from the actual head graph')
    if head.weight.grad is None or not bool(torch.isfinite(head.weight.grad).all()):
        raise ValueError('Actual existing classifier W is disconnected or nonfinite')
    if any(not bool(torch.isfinite(p.grad).all()) for p in net.parameters()
           if p.grad is not None):
        raise ValueError('Nonfinite synthetic graph derivative')

    # Synthetic DSP only: physical RX/day/EQ session is unchanged while opaque
    # sample coordinates and epoch/step change. No TX label enters this API.
    channel = tail.Practical('train')
    t = torch.arange(256, dtype=torch.float32)
    packet = torch.stack((torch.cos(t * .2), torch.sin(t * .2)))
    x = packet.unsqueeze(0).repeat(2, 1, 1)
    meta = dict(eq_i=torch.ones(2, dtype=torch.long),
                rx_i=torch.ones(2, dtype=torch.long),
                day_i=torch.ones(2, dtype=torch.long),
                sig_i=torch.tensor([0, 1]), base_index=torch.tensor([0, 1]))
    a, rec_a = channel.apply(x, meta, epoch=21, step=4441)
    hardware_a = dict(channel.sessions)
    b, rec_b = channel.apply(x, meta, epoch=22, step=4663)
    if hardware_a != channel.sessions or len(channel.sessions) != 1:
        raise ValueError('RX hardware depends on sample coordinates or epoch')
    if torch.equal(a, b) or rec_a[0]['namespace'] == rec_b[0]['namespace']:
        raise ValueError('Training channel realization failed to refresh across epochs')
    cfg = channel.audit()
    for c in cfg['configurations'].values():
        if c['processing_route'] != 'residual' or c['mode'] != 'post_sync' or c['equalization_enabled']:
            raise ValueError('Actual channel is not the declared residual/no-EQ route')
    if any(r['channel_equalization_applied'] for r in rec_a + rec_b):
        raise ValueError('Synthetic diagnostic unexpectedly enabled EQ')
    files = dict(model=inspect.getfile(type(net)),
                 encoder=inspect.getfile(type(net.id_backbone.encoder)),
                 default_head=inspect.getfile(type(head)),
                 training_channel=tail.__file__,
                 leo_practical_channel=channel.ch.__file__,
                 leo_practical_residual=channel.residual.__file__)
    return dict(status='VERIFIED_SYNTHETIC_ONLY', seed=202610184401,
                actual_module_files=files, norms=norms,
                running_stat_buffers=running_buffers, called_modules=called,
                graph_groups=groups, default_head_type=type(head).__name__,
                default_head_module_forward_hook_calls=called.count(head_name),
                default_head_W_graph_gradient=True,
                persistent_state_unchanged=state_unchanged,
                channel=dict(configurations=cfg['configurations'],
                             hardware_session_count=len(channel.sessions),
                             session_ids=list(channel.sessions),
                             TX_labels_in_apply_api=False,
                             hardware_stable_across_sample_and_epoch=True,
                             realization_refreshed_across_epoch=True,
                             namespaces=[rec_a[0]['namespace'], rec_b[0]['namespace']],
                             channel_equalization_applied=False,
                             epoch_step_role='training',
                             formal_test_or_validation_streams_accessed=False),
                cost=dict(synthetic_identity_NN=2, synthetic_channel_packets=4,
                          real_IQ=0, checkpoint_reads=0, optimizer_updates=0,
                          teacher_NN=0, U_truth=0, target_access=False),
                interpretation=[
                    'Graph connectivity does not establish useful learning, training health or accuracy.',
                    'A module forward hook can be silent when its methods or functional operations are called directly.',
                    'This rejects a BatchNorm-running-stat explanation, not every stochastic ordering effect.',
                    'The channel test checks its declared training API on synthetic packets; it does not verify target performance.',
                    'Absent identity gradients on the unused domain modules do not authorize changing the fixed architecture.'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--release', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    start = time.monotonic()
    result = run(args.release.resolve())
    result['elapsed_seconds'] = time.monotonic() - start
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8', newline='\n') as f:
        json.dump(result, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')
    print(json.dumps(dict(status=result['status'], cost=result['cost'],
                         output=str(args.output), elapsed_seconds=result['elapsed_seconds'])))


if __name__ == '__main__':
    main()
