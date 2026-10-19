import json
import math
import os
import sys
import time
import numpy as np
import torch
from . import design as d
from .models import UncertaintyRestorer, variance_nll
from .runtime import frozen_context, save_checkpoint, load_checkpoint, validate_calibration
from experiments.pamf_reception_restore.runner import pair_loss
from experiments.pamf_restoration.data import write_json
from experiments.pamf_restoration.runner import array_tensor, tensor_array, read_json


def learning_rate(update):
    if not 0 <= update < 4940:
        raise ValueError('Update exceeds preregistered budget')
    if update < 247:
        return 2e-4*(update+1)/247
    return 2e-4*(.1+.9*(1+math.cos(math.pi*(update-247)/4692))/2)


def initialize_model(kind, frozen, floor):
    # Reset after loading parents: their constructors consume different RNG draws.
    torch.manual_seed(d.MODEL_SEED)
    torch.cuda.manual_seed_all(d.MODEL_SEED)
    return UncertaintyRestorer(kind, frozen, floor)


def train(args):
    if args.kind not in d.NAMES:
        raise ValueError('Registered trained candidate required')
    torch.manual_seed(d.MODEL_SEED); torch.cuda.manual_seed_all(d.MODEL_SEED)
    frozen, parent = frozen_context(args, args.kind, 'cuda:0')
    calibration = read_json(args.output/'calibration.json'); validate_calibration(calibration)
    used_calibration = calibration if args.kind == 'uncertain_map_prior' else None
    floor = calibration['excess_variance'] if used_calibration is not None else None
    model = initialize_model(args.kind, frozen, floor).to('cuda:0')
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=2e-4, weight_decay=1e-4)
    row = args.output/args.kind; row.mkdir()
    write_json(row/'gpu_ready.json', dict(pid=os.getpid(), cwd=os.getcwd(), argv=sys.argv,
        physical_GPU=os.environ.get('CUDA_VISIBLE_DEVICES')))
    write_json(row/'resolved_config.json', dict(d.configuration(), kind=args.kind,
        excess_variance=floor, frozen_parent=parent))
    cache = args.parent_action/'cache'
    clean = np.load(cache/'train/clean.npy', mmap_mode='r', allow_pickle=False)
    roles = np.load(cache/'train/roles.npy', allow_pickle=False)
    if clean.shape != (63000, 2, 256) or len(roles) != 63000 or (roles == 'L_s').sum() != 6300 or (roles == 'U_s').sum() != 56700:
        raise ValueError('All original L/U IQ and no V required')
    permutation = np.random.default_rng(d.DATA_SEED)
    updates = seen_l = seen_u = 0; started = time.perf_counter(); torch.cuda.reset_peak_memory_stats()
    with (row/'training.jsonl').open('x', encoding='utf-8') as log:
        for epoch in range(1, d.EPOCHS+1):
            folder = cache/'epochs'/f'{epoch:02d}'
            received = np.load(folder/'received.npy', mmap_mode='r', allow_pickle=False)
            if received.shape != clean.shape:
                raise ValueError('Original complete epoch observations required')
            with np.load(folder/'groups.npz', allow_pickle=False) as archive:
                groups = {key: archive[key] for key in archive.files}
            order = permutation.permutation(len(clean)); model.train()
            totals = dict(loss=0., iq_nmse=0., log_power=0., variance_nll=0., variance_mean=0.)
            strata = {}; clipped = 0
            for begin in range(0, len(order), d.BATCH_SIZE):
                selected = order[begin:begin+d.BATCH_SIZE]
                x, r = array_tensor(clean[selected], 'cuda:0'), array_tensor(received[selected], 'cuda:0')
                output, variance, action, _ = model.training_outputs(r)
                paired, metrics = pair_loss(output, x)
                nll = variance_nll(action, r, x, variance) if args.kind == 'uncertain_map_prior' else torch.zeros_like(paired)
                loss = paired.mean()+.01*nll.mean()
                if not torch.isfinite(loss):
                    raise FloatingPointError('Nonfinite reconstruction/variance objective')
                lr = learning_rate(updates)
                for group in optimizer.param_groups:
                    group['lr'] = lr
                optimizer.zero_grad(set_to_none=True); loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(parameters, 1., error_if_nonfinite=True)
                clipped += int(norm > 1); optimizer.step(); updates += 1
                seen_l += int((roles[selected] == 'L_s').sum()); seen_u += int((roles[selected] == 'U_s').sum())
                totals['loss'] += float(loss)*len(selected)
                totals['iq_nmse'] += float(metrics['iq_nmse'].sum())
                totals['log_power'] += float(metrics['log_power'].sum())
                totals['variance_nll'] += float(nll.sum())
                if variance is not None:
                    totals['variance_mean'] += float(variance.mean(-1).sum())
                values = tensor_array(metrics['iq_nmse'])
                masks = {'L_s': roles[selected] == 'L_s', 'U_s': roles[selected] == 'U_s'}
                for key, labels in groups.items():
                    masks.update({key+':'+str(label): labels[selected] == label for label in np.unique(labels)})
                for key, mask in masks.items():
                    item = strata.setdefault(key, dict(count=0, nmse_sum=0.))
                    item['count'] += int(mask.sum()); item['nmse_sum'] += float(values[mask].sum())
                if updates == 1:
                    save_checkpoint(row/'initial_source_smoke.pt', model, optimizer, epoch, updates, parent, used_calibration)
                    fresh, _ = load_checkpoint(row/'initial_source_smoke.pt', 'cuda:0'); model.eval()
                    with torch.no_grad():
                        if not torch.allclose(model(r[:2]), fresh(r[:2]), atol=1e-6, rtol=1e-5):
                            raise ValueError('Actual-source trained checkpoint roundtrip differs')
                    del fresh; model.train()
            record = dict(epoch=epoch, updates=updates, losses={key: value/len(clean) for key, value in totals.items()},
                L_exposures=seen_l, U_exposures=seen_u, learning_rate_last=lr, gradient_clip_fraction=clipped/247,
                strata={key: dict(count=value['count'], mean_iq_nmse=value['nmse_sum']/max(value['count'], 1)) for key, value in strata.items()},
                seconds=time.perf_counter()-started)
            log.write(json.dumps(record, allow_nan=False)+'\n'); log.flush()
            print(json.dumps(dict(event='UNCERTAINTY_EPOCH_COMPLETE', kind=args.kind, **record)), flush=True)
    if (updates, seen_l, seen_u) != (4940, 126000, 1134000):
        raise ValueError('Full registered incremental training exposure required')
    save_checkpoint(row/'final.pt', model, optimizer, 20, updates, parent, used_calibration)
    write_json(row/'training_complete.json', dict(status='SOURCE_TRAINED', kind=args.kind, epochs=20, updates=updates,
        L_exposures=seen_l, U_exposures=seen_u, TX_labels_read=0, F_outputs_read=0, target_reads=0,
        frozen_context_updates=0, parameters=sum(p.numel() for p in model.parameters()),
        trainable_parameters=sum(p.numel() for p in parameters), seconds=time.perf_counter()-started,
        peak_allocated_bytes=torch.cuda.max_memory_allocated()))
