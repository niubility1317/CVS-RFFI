"""Reuse only the source-matched E20 prefix, then execute native E30/E50."""

import shutil
import time
from pathlib import Path

from experiments.cric_dura_comparison.io import read, write

from .objective import installed
from .plan import ARMS, check_prefix, validate_config


def train(c):
    import torch
    from experiments.cric_dura_comparison.f_train import train as train_segment

    validate_config(c)
    prefix = check_prefix(c)
    root = Path(c['output_root'])
    root.mkdir(parents=True, exist_ok=False)
    write(root / 'resolved_config.json', c)
    segments = [prefix]
    states = []
    started = time.monotonic()
    for epochs, offset in ((10, 20), (20, 30)):
        destination = root / f'segment{epochs + offset}'
        segment = dict(c, output_root=str(destination), segment_epochs=epochs,
                       epoch_offset=offset, step_offset=offset * 222, parent_source=segments[-1]['folder'])
        holder = {}
        with installed(segment, holder):
            train_segment(segment)
        objective = holder['objective']
        state = objective.state_dict()
        if (state['calls'] != epochs * 222 or state['EMA_updates'] or state['U_truth_read']
                or state['student_views'] != 6 * state['seen']):
            raise ValueError('The complete six-view REPL budget is required')
        shift = getattr(objective, 'branch_sampled_shift', {})
        state['branch_style_sampled_batch_rms'] = {
            key: [float(value) for value in torch.stack(values).cpu()] for key, values in shift.items()
        }
        write(destination / 'repl_state.json', state)
        states.append(state)
        done = read(destination / 'complete.json')
        if (done['status'] != 'SOURCE_FROZEN' or done['updates'] != epochs * 222
                or done['target_access'] or done['U_truth_read'] or done['head_fit_updates']):
            raise ValueError('Incomplete source-only REPL segment')
        segments.append(dict(folder=str(destination), **done))
        write(root / 'progress.json', dict(epoch=epochs + offset, updates=(epochs + offset) * 222),
              exclusive=False)
    shutil.copyfile(Path(segments[-1]['folder']) / 'final_ssdg.pth', root / 'final_ssdg.pth')
    write(root / 'source_contract.json', read(c['source_contract']))
    write(root / 'complete.json', dict(
        status='SOURCE_FROZEN', config=c, epoch=50, updates=11100, segments=segments,
        initialization='same_contract_scratch20_own10_own20', ancestors=[s['folder'] for s in segments],
        source_contract=c['source_contract'], target_access=False, U_truth_read=False, head_fit_updates=0,
        actual_REPL={k: sum(s[k] for s in states) for k in ('calls', 'seen', 'accepted', 'student_views')},
        incremental_training_seconds=time.monotonic() - started,
        cumulative_training_seconds=sum(s['seconds'] for s in segments)))


def load_reader(folder, contract, device):
    from experiments.pamf_repl_view_shuffle.source import seed_scope, single_base
    from experiments.pamf_repl_completion.source import load_reader as read_original

    done = read(Path(folder) / 'complete.json')
    c = validate_config(done['config'])
    if [s['folder'] for s in done['segments']] != [c['shared_prefix'], str(Path(folder) / 'segment30'),
                                                   str(Path(folder) / 'segment50')]:
        raise ValueError('Registered E20 and same-row REPL ancestry required')
    with seed_scope(), single_base(c, reader=True):
        return read_original(folder, contract, device)


def smoke(runtime, train_configs):
    import torch
    from torch.nn import functional as F
    from experiments.cric_dura_comparison import f_train, t1_model
    from experiments.cvs_phase1_stack.runtime import installed as native_scope
    from experiments.pamf_repl_view_shuffle.source import assert_structure
    from .objective import sites_for

    device = 'cuda:0'
    configs = [validate_config(read(path)) for path in train_configs]
    prefix = check_prefix(configs[0])
    payload = torch.load(Path(prefix['folder']) / 'final_ssdg.pth', map_location='cpu', weights_only=False)
    results = {}
    for c in configs:
        cfg = dict(c, hard_pl_smoke=True, epoch_offset=20, step_offset=4440,
                   segment_epochs=10, parent_source=prefix['folder'])
        holder = {}
        torch.manual_seed(c['model_seed'])
        torch.cuda.manual_seed_all(c['model_seed'])
        with installed(cfg, holder):
            loaders = f_train.context(cfg)
            with f_train.class_dimension(cfg), native_scope(f_train.original_config(c['model_seed'])) as native:
                args = f_train.arguments(cfg, device)
                args.input_len = 256
                args.baseline_ckpt = args.teacher_ckpt = ''
                args.from_scratch = args.a1_scratch_only = True
                net = t1_model.install(native.build_baseline_model(args, device), cfg).to(device)
            shape = assert_structure(net, cfg)
            net.load_state_dict(payload['model'], strict=True)
            net.eval()
            with torch.no_grad():
                logits = net(loaders['train_loader'].dataset.x[:2].to(device))
                if logits.shape != (2, 6) or not bool(torch.isfinite(logits).all()):
                    raise ValueError('Historical E20 no-query inference failed')
            net.train()
            net._cd_U_loader = loaders['unlabeled_loader']
            net._cd_U_iter = iter(net._cd_U_loader)
            batch = next(iter(loaders['train_loader']))
            x, y, domains = [v.to(device) for v in batch[:3]]
            meta = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch[3].items()}
            with net._phase1_labeled_feature_context():
                output = net(x, return_aux=True, domain_labels=domains)
            ce = F.cross_entropy(output['tx_logits'], y, label_smoothing=.01)
            correction, metrics = net._source_labeled_feature_hook(
                model=net, labeled_output=output, labels=y, extra=(None, meta), domains=domains,
                optimizer_step=1, identity_loss=ce, labeled_count=128, labeled_inputs=x)
            loss = ce + correction
            loss.backward()
            objective = holder['objective']
            state = objective.state_dict()
            head = net.id_backbone.encoder.core.id_backbone.cls_head.weight
            if (not bool(torch.isfinite(loss)) or head.grad is None or float(head.grad.norm()) <= 0
                    or any(p.grad is not None and not bool(torch.isfinite(p.grad).all()) for p in net.parameters())
                    or state['calls'] != 1 or state['student_views'] != 1536
                    or state['U_truth_read'] or state['EMA_updates'] or metrics['T1_L_packets'] != 768
                    or not state['accepted'] or not state['U_style']['eligible']
                    or state['U_style']['plan_calls'] != 1 or state['U_joint_contrast']['calls'] != 1
                    or state['U_receiver_extrapolation']['calls'] != 1
                    or state['U_gradient_guard']['calls'] != 1):
                raise ValueError('Actual source-only REPL inference/backward failed')
            arm = c['branch_style_arm']
            time_enabled, branches = ARMS[arm]
            sites = set(sites_for(branches))
            expected = sites | (set(c['hard_pl_mixstyle']['levels']) if time_enabled else set())
            applied = {k: n for k, n in state['U_style']['applied'].items() if n > 0}
            shifts = {k: float(v[0]) for k, v in getattr(objective, 'branch_sampled_shift', {}).items()}
            if (set(applied) != expected or set(shifts) != sites or any(not 0 < v < float('inf') for v in shifts.values())
                    or any(m._forward_hooks for m in (net.id_backbone.encoder.get_submodule(path)
                           for path, _ in _direct_modules(sites)))):
                raise ValueError('Actual direct branch dose, placement or hook cleanup failed: ' + arm)
            results[arm] = dict(shape=shape, accepted=state['accepted'], applied=applied,
                                sampled_rms=shifts, finite_backward=True)
            print('SOURCE_SMOKE_PASS', arm, flush=True)
        del net, objective, holder, loaders, loss, output, ce, correction, head
        torch.cuda.empty_cache()
    write(Path(runtime['run_root']) / 'smoke.json', dict(
        status='PASS', checks=results, actual_checkpoint=prefix['folder'], optimizer_updates=0,
        target_reads=0, U_truth_reads=0))


def _direct_modules(sites):
    from .objective import SITES
    return [SITES[name] for name in sorted(sites)]
