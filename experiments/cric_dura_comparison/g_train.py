"""Same-contract TX-label-free/F-free eight-stage DURA training."""
import json
import math
from pathlib import Path
from shutil import copy2
import time
import numpy as np
import torch
from .io import read, write, tensor
from .g_cache import CacheView
from .g_lineage import STAGES, construct, load, objective, trainable
from .teacher_targets import EpochTargets, SharedEpochTargets, binding, exposure_count, frozen_prefix_predictor


def learning_rate(step):
    return 2e-4 * (step + 1) / 247 if step < 247 else 2e-4 * (.1 + .9 * (1 + math.cos(math.pi * (step - 247) / 4692)) / 2)


def batches(n, rng):
    order = rng.permutation(n)
    position = 0
    for _ in range(247):
        if position >= n:
            order = rng.permutation(n)
            position = 0
        result = order[position:position + 256]
        position += len(result)
        yield result


def inherit_source_stage(frozen,root,contract,roles,seed,stage):
    """Carry forward a frozen source prefix stage without updating its weights."""
    source=Path(frozen)
    proof=read(source/'complete.json')
    _,checkpoint=load(source/'final.pt',contract,roles)
    ancestors=[str(source.parent/'coarse'/'final.pt')] if stage=='action' else []
    if (proof['status']!='SOURCE_FROZEN' or proof['stage']!=stage or proof['updates']!=4940
            or checkpoint['stage']!=stage or checkpoint['model_seed']!=seed
            or checkpoint['ancestors']!=ancestors
            or any(proof[k] for k in ('TX_labels_read','F_reads','target_reads'))):
        raise ValueError('Require the registered source-only prefix stage for the same seed')
    dest=root/stage
    dest.mkdir(exist_ok=False)
    for name in ('final.pt','training.jsonl','complete.json'):
        copy2(source/name,dest/name)
    return dest/'final.pt',proof['seconds']


def train(c, device='cuda:0', *, view=None, prepare_model=None, frozen_coarse=None,
          frozen_action=None, stage_loss=None, reconstruction=None, log_action_terms=False,
          original_recipe=False):
    if any(c.get(k) for k in ('truth', 'target_truth', 'target_inputs', 'tests', 'reader', 'F_checkpoint')):
        raise ValueError('G source training rejects F/target inputs')
    contract = read(c['cache']['source_contract'])
    roles = c['cache']['roles']
    view = CacheView(c['cache']) if view is None else view
    if not view.ready():
        raise ValueError('Required source IQ observations missing')
    root = Path(c['output_root'])
    root.mkdir(parents=True, exist_ok=False)
    write(root / 'config.json', c)
    previous = None
    timings = {}
    stages=STAGES
    if frozen_action is not None and (frozen_coarse is None or Path(frozen_coarse).parent!=Path(frozen_action).parent):
        raise ValueError('An inherited action stage needs its original coarse parent')
    if frozen_coarse is not None:
        previous,timings['coarse']=inherit_source_stage(frozen_coarse,root,contract,roles,c['model_seed'],'coarse')
        stages=STAGES[1:]
    if frozen_action is not None:
        previous,timings['action']=inherit_source_stage(frozen_action,root,contract,roles,c['model_seed'],'action')
        stages=STAGES[2:]
    if stage_loss is not None:
        stage_loss=stage_loss.to(device)
    if reconstruction is not None:
        reconstruction=reconstruction.to(device)
    acceleration = c.get('training_acceleration', {})
    mode = acceleration.get('teacher_cache', 'online')
    if mode not in ('online', 'epoch', 'shared'):
        raise ValueError('Unknown teacher acceleration mode')
    reduce_syncs = acceleration.get('reduce_syncs', False)
    for stage in stages:
        folder = root / stage
        folder.mkdir()
        torch.manual_seed(c['model_seed'])
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(c['model_seed'])
        if previous is None:
            parent = None
        else:
            parent, _ = load(previous, contract, roles)
        if stage == 'r17' and original_recipe:
            # The original R17 resets the RNG after reconstructing the frozen action checkpoint.
            torch.manual_seed(c['model_seed'])
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(c['model_seed'])
        model = construct(stage, parent).to(device)
        if prepare_model is not None:
            prepare_model(model, stage)
        teacher = parent.to(device).eval().requires_grad_(False) if stage in ('r38', 'r44', 'r46', 'r65', 'r69') else None
        parameters = trainable(model, stage)
        optimizer = torch.optim.AdamW(parameters, lr=2e-4, weight_decay=1e-4)
        rng = np.random.default_rng(2026101901 if stage in ('coarse', 'action') else 2026102001)
        # The two additional G replicas vary training order as well as init;
        # all variants of each seed share this declared order and cached IQ.
        if c['model_seed'] != 2026092701:
            rng = np.random.default_rng((2026101901 if stage in ('coarse', 'action') else 2026102001) + c['model_seed'] - 2026092701)
        generator = torch.Generator(device=device).manual_seed(2026101919)
        updates = seen_l = seen_u = 0
        started = time.perf_counter()
        with (folder / 'training.jsonl').open('x', encoding='utf-8') as stream:
            for epoch in range(1, 21):
                model.train()
                total = torch.zeros((), dtype=torch.float64, device=device) if reduce_syncs else 0.
                trace = dict(action_error=0., log_variance_error=0.) if stage == 'action' and log_action_terms else None
                observed = 0
                target_cache = prefix_cache = None
                if teacher is not None and mode == 'shared':
                    target_cache = SharedEpochTargets(acceleration['shared_roots'][stage],
                        binding(c['cache'], previous, STAGES[STAGES.index(stage)-1]), view.ids, epoch,
                        positions=(-2,-1) if stage=='r69' else (0,1,2))
                elif mode == 'epoch' and exposure_count(len(view)) > len(view):
                    options = dict(storage=acceleration.get('storage','auto'),
                        device_limit_mib=acceleration.get('device_limit_mib',64), max_mib=acceleration.get('max_mib',512))
                    identity = binding(c['cache'], previous, STAGES[STAGES.index(stage)-1]) if previous else None
                    if teacher is not None:
                        prediction = (lambda x: teacher.stage_outputs(x)[0][1:]) if stage=='r69' else None
                        target_cache = EpochTargets(teacher,len(view),epoch,identity,predict=prediction,**options)
                    elif stage in ('action','r17') and acceleration.get('frozen_prefix',True):
                        prefix, prediction = frozen_prefix_predictor(model,stage)
                        prefix_cache = EpochTargets(prefix,len(view),epoch,identity,predict=prediction,**options)
                for selected in batches(len(view), rng):
                    arrays = view.batch(selected, epoch, coarse=stage == 'coarse', action=stage == 'action')
                    optimizer.zero_grad(set_to_none=True)
                    for group in optimizer.param_groups:
                        group['lr'] = learning_rate(updates)
                    micro = 64 if stage in ('r38','r44','r46') else 256
                    finite = torch.ones((),dtype=torch.bool,device=device) if reduce_syncs else None
                    for offset in range(0, len(selected), micro):
                        take = slice(offset, offset + micro)
                        clean = tensor(arrays['clean'][take], device)
                        received = tensor(arrays['coarse' if stage == 'coarse' else 'received'][take], device)
                        action = tuple(tensor(arrays[k][take], device) for k in ('kernel', 'coefficients', 'variance')) if stage == 'action' else None
                        desired = target_cache.get(selected[take],received,epoch) if target_cache is not None else None
                        context = prefix_cache.get(selected[take],received,epoch) if prefix_cache is not None else None
                        loss = objective(model, teacher, stage, received, clean, epoch, action, generator,
                            desired=desired,frozen_context=context,skip_zero_spectral=acceleration.get('skip_zero_spectral',False),
                            stage_loss=stage_loss,reconstruction=reconstruction,action_trace=trace,
                            original_recipe=original_recipe)
                        if reduce_syncs:
                            finite.logical_and_(torch.isfinite(loss).all())
                        elif not bool(torch.isfinite(loss).all()):
                            raise FloatingPointError('Nonfinite G source objective')
                        (loss.sum() / len(selected)).backward()
                        total += loss.detach().sum().double() if reduce_syncs else float(loss.detach().sum())
                    if reduce_syncs and not bool(finite):
                        raise FloatingPointError('Nonfinite G source objective')
                    gradient_norm = torch.nn.utils.clip_grad_norm_(parameters, 1., error_if_nonfinite=True)
                    optimizer.step()
                    updates += 1
                    if updates == 1 and trace is not None:
                        first = dict(updates=updates, loss=float(total) / len(selected), gradient_norm=float(gradient_norm),
                                     **{k: v / len(selected) for k, v in trace.items()})
                        write(folder / 'first_update.json', first)
                        print('R07_ESTIMATED_FIRST_UPDATE', first, flush=True)
                    seen_l += int(view.is_L[selected].sum())
                    seen_u += int((~view.is_L[selected]).sum())
                    observed += len(selected)
                if stage == 'r65' and epoch == 10:
                    from experiments.pamf_r46_delayed_prune.models import transition
                    model = transition(model.cpu(), 10).to(device)
                    parameters = list(model.parameters())
                    optimizer = torch.optim.AdamW(parameters, lr=learning_rate(updates), weight_decay=1e-4)
                row = dict(epoch=epoch, updates=updates, loss=float(total) / observed, L_exposures=seen_l,
                    U_exposures=seen_u, seconds=time.perf_counter() - started,
                    parameters=sum(p.numel() for p in model.parameters()), TX_labels_read=0, F_reads=0, target_reads=0)
                if acceleration:
                    row['teacher_cache'] = target_cache.stats() if target_cache else None
                    row['prefix_cache'] = prefix_cache.stats() if prefix_cache else None
                if trace is not None:
                    row.update({k: v / observed for k, v in trace.items()})
                del target_cache, prefix_cache
                stream.write(json.dumps(row, allow_nan=False) + '\n')
                stream.flush()
                write(root / 'progress.json', dict(stage=stage, **row), exclusive=False)
                print(json.dumps(dict(event='DURA_LINEAGE_EPOCH', stage=stage, **row)), flush=True)
        if updates != 4940 or (roles == 'L' and seen_u):
            raise ValueError('G role or complete update budget differs')
        if teacher is not None and any(p.grad is not None for p in teacher.parameters()):
            raise ValueError('G teacher acquired gradients')
        previous = folder / 'final.pt'
        torch.save(dict(schema='contract_matched_dura_stage_v1', stage=stage,
            source_contract=contract, allowed_roles=roles, model_seed=c['model_seed'],
            model={k: v.detach().cpu() for k, v in model.state_dict().items()},
            initialization_report=getattr(model, 'initialization_report', None),
            ancestors=[str(root / s / 'final.pt') for s in STAGES[:STAGES.index(stage)]],
            epoch=20, updates=4940, L_exposures=seen_l, U_exposures=seen_u,
            TX_labels_read=0, F_reads=0, target_reads=0, code_commit=c['code_commit']), previous)
        fresh, _ = load(previous, contract, roles, device)
        with torch.no_grad():
            model.eval()
            if not torch.allclose(model(received[:2]), fresh(received[:2]), atol=1e-6, rtol=1e-5):
                raise ValueError('G stage checkpoint round trip differs')
        timings[stage] = time.perf_counter() - started
        write(folder / 'complete.json', dict(status='SOURCE_FROZEN', stage=stage, updates=updates,
            seconds=timings[stage], L_exposures=seen_l, U_exposures=seen_u, TX_labels_read=0, F_reads=0, target_reads=0))
        del model, optimizer, fresh, parent, teacher
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    write(root / 'complete.json', dict(status='SOURCE_FROZEN', stages=list(STAGES), updates=39520,
        checkpoint=str(previous), source_contract=c['cache']['source_contract'], allowed_roles=roles,
        seconds_by_stage=timings, TX_labels_read=0, F_reads=0, target_reads=0))
