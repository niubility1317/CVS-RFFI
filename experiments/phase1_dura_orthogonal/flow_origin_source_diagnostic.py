"""Small matched source-only flow training and held-out path diagnosis."""
import argparse
import json
import os
from pathlib import Path
import time

os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
import torch

from experiments.phase1_dura_orthogonal.conditional_iq_flow import ConditionalIQFlow, flow_matching_loss, sample_flow_times
from experiments.phase1_dura_orthogonal.source_context_cache import _contract, _validate_data


def source_split(total, fit, heldout):
    if fit < 1 or heldout < 1 or fit + heldout > total:
        raise ValueError('Disjoint source fit and held-out records required')
    return (torch.cat([torch.arange(k*total, k*total+fit) for k in range(3)]),
            torch.cat([torch.arange(k*total+fit, k*total+fit+heldout) for k in range(3)]))


def velocity_error(velocity, anchor, clean):
    energy = clean.square().mean((1, 2)).clamp_min(1e-8)
    return (velocity-(clean-anchor)).square().mean((1, 2))/energy


def train(fields, origin_fraction):
    torch.manual_seed(2026110612)
    model = ConditionalIQFlow(width=32, steps=4).to('cuda:0')
    optimizer = torch.optim.AdamW(model.parameters(), lr=.0003, weight_decay=.001)
    order_generator = torch.Generator().manual_seed(2026110613)
    time_generator = torch.Generator().manual_seed(2026110713)
    epoch_losses, origin_count = [], 0
    started = time.monotonic()
    for epoch in range(5):
        order = torch.randperm(len(fields[0]), generator=order_generator).to('cuda:0')
        total = torch.zeros((), device='cuda:0')
        for start in range(0, len(order), 32):
            index = order[start:start+32]
            x, anchor, direction, variance, clean = (v[index] for v in fields)
            times = sample_flow_times(len(index), time_generator, origin_fraction=origin_fraction).to('cuda:0')
            origin_count += int((times == 0).sum().cpu())
            optimizer.zero_grad(set_to_none=True)
            loss = flow_matching_loss(model, x, anchor, direction, variance, clean, times)
            if not torch.isfinite(loss):
                raise FloatingPointError('Finite source flow loss required')
            loss.backward()
            if model.out.weight.grad is None or not torch.isfinite(model.out.weight.grad).all():
                raise FloatingPointError('Finite flow gradients required')
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            optimizer.step()
            total += loss.detach()*len(index)
        epoch_losses.append(float(total.cpu())/len(order))
        print('FLOW_ORIGIN_DIAGNOSTIC_EPOCH', origin_fraction, epoch+1, epoch_losses[-1], flush=True)
    torch.cuda.synchronize()
    return model.eval(), epoch_losses, origin_count, time.monotonic()-started


@torch.no_grad()
def evaluate(model, fields):
    x, anchor, direction, variance, clean = fields
    energy = clean.square().mean((1, 2)).clamp_min(1e-8)
    result = {'anchor_nmse': float(((anchor-clean).square().mean((1, 2))/energy).mean().cpu()),
              'output_nmse': float(((model(x, anchor, direction, variance)-clean).square().mean((1, 2))/energy).mean().cpu()),
              'velocity_nmse': {}}
    for t in (0., .01, .25, .5, .75):
        times = torch.full((len(x),), t, device=x.device, dtype=x.dtype)
        state = (1-t)*anchor+t*clean
        velocity = model.velocity(state, x, anchor, direction, variance, times)
        result['velocity_nmse'][str(t)] = float(velocity_error(velocity, anchor, clean).mean().cpu())
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--commit', required=True)
    args = parser.parse_args()
    if args.out.exists() or not torch.cuda.is_available():
        raise ValueError('Unique output and CUDA device required')
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    payload = torch.load(args.cache, map_location='cpu', weights_only=True)
    contract = payload['contract']
    fit_ids = contract['fit_ids']
    if len(fit_ids) != 32768 or contract != _contract(fit_ids, 32, 'cuda:0', True):
        raise ValueError('Fixed legal source-only DURA cache required')
    _validate_data(payload['data'], payload['counts'], 3*len(fit_ids))
    train_index, eval_index = source_split(32768, 384, 128)
    training = tuple(v[train_index].to('cuda:0') for v in payload['data'])
    heldout = tuple(v[eval_index].to('cuda:0') for v in payload['data'])
    smoke = ConditionalIQFlow(width=32, steps=4).to('cuda:0')
    with torch.no_grad():
        if not torch.equal(smoke(*(v[:8] for v in training[:4])), training[1][:8]):
            raise ValueError('Real source context zero-head smoke required')
    del smoke, payload
    args.out.mkdir(parents=True)
    print('FLOW_ORIGIN_SOURCE_SMOKE_PASS', flush=True)
    result = {}
    for name, fraction in (('uniform', 0.), ('origin_half', .5)):
        model, losses, origin_count, seconds = train(training, fraction)
        result[name] = dict(losses=losses, origin_count=origin_count,
                            optimization_seconds=seconds, heldout=evaluate(model, heldout))
    out = dict(status='FLOW_ORIGIN_SOURCE_DIAGNOSTIC_COMPLETE', commit=args.commit,
               fit_ids=384, heldout_ids=128, views_per_record=3, epochs=5, batch_size=32,
               methods=result, source_only=True, F_weights_read=0, F_outputs_read=0,
               TX_labels_read=0, V_reads=0, target_reads=0, weights_saved=False,
               caveat='Small source-fit subset; held-out only for diagnostic flow training, DURA may have seen related source data. Source waveform and velocity metrics are not identity accuracy.')
    (args.out/'result.json').write_text(json.dumps(out, indent=2, allow_nan=False)+'\n', encoding='utf8')
    print('FLOW_ORIGIN_SOURCE_DIAGNOSTIC_COMPLETE', flush=True)


if __name__ == '__main__':
    main()
