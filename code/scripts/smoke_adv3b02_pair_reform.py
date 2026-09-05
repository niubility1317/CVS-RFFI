"""Real-checkpoint, synthetic-source pair runtime smoke; no dataset or query I/O."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import torch
import torch.nn.functional as F

CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from post_stage_common import build_baseline_model
from cvsrffi import pair_reform_runtime as runtime
from cvsrffi.deployment_orbit import stable_orbit_key_tensor
from cvsrffi.orbit_teacher import DenseTemporalOrbitMemory
from SSDG.train_ssdg import build_arg_parser, _update_ema_model


def load_model(checkpoint: Path, device: torch.device):
    """Reconstruct saved baseline architecture without adding auxiliary heads."""
    payload = torch.load(checkpoint, map_location='cpu', weights_only=False)
    if not isinstance(payload, dict) or not isinstance(payload.get('model'), dict):
        raise ValueError('checkpoint must contain a model state dictionary')
    raw_args = payload.get('baseline_args') or payload.get('args') or {}
    model_args = dict(raw_args) if isinstance(raw_args, dict) else vars(raw_args).copy()
    state = payload['model']
    if 'num_classes' not in model_args:
        key = 'id_backbone.cls_head.head.weight'
        if key not in state:
            raise ValueError('checkpoint is missing baseline num_classes and classifier weight')
        model_args['num_classes'] = int(state[key].shape[0])
    if 'num_domains' not in model_args:
        raise ValueError('checkpoint baseline_args must preserve num_domains')
    model_args.setdefault('input_len', 256)
    model = build_baseline_model(SimpleNamespace(**model_args), device)
    # Unlike the historical DAOT smoke, do not force use_daot_nuisance_head.
    incompatible = model.load_state_dict(state, strict=False)
    details = {'missing': list(incompatible.missing_keys), 'unexpected': list(incompatible.unexpected_keys)}
    if details['missing'] or details['unexpected']:
        raise RuntimeError(f'checkpoint compatibility failure: {details}')
    if model.dom_backbone is None or not callable(getattr(model, 'forward_identity_only', None)):
        raise ValueError('pair smoke requires the dual backbone and identity-only teacher')
    return payload, model, model_args, details


def run_smoke(checkpoint: Path, device: torch.device, *, amp=False, training_manifest=None) -> dict:
    torch.manual_seed(392005)
    payload, model, model_args, compatibility = load_model(checkpoint, device)
    initial = copy.deepcopy(model.state_dict())
    teacher = copy.deepcopy(model).eval()
    teacher.requires_grad_(False)
    batch = 2
    clean = torch.randn(batch, 2, int(model_args['input_len']), device=device)
    channel = clean + .03 * torch.randn_like(clean)
    domains = torch.arange(batch, device=device) % int(model_args['num_domains'])
    ids = [('synthetic_source_smoke', index) for index in range(batch)]
    keys = stable_orbit_key_tensor(ids, device=device)
    domain_forwards = []
    identity_forwards = []
    handle = teacher.dom_backbone.register_forward_hook(lambda *args: domain_forwards.append(1))
    identity_handle = teacher.id_backbone.register_forward_hook(lambda *args: identity_forwards.append(1))
    rows = {}
    manifest_rows = json.loads(Path(training_manifest).read_text(encoding='utf-8'))['rows'] if training_manifest else []
    names = dict(point='A_POINT',safe='B_SAFE',asymmetric='ASYMMETRIC',
                 tangent_only='TANGENT',route_only='ROUTE',memory='POINT_MEMORY')
    try:
        with torch.no_grad():
            teacher_clean = teacher.forward_identity_only(clean, domain_labels=domains)
        weights = runtime._fixed_head_weights(teacher, teacher_clean)
        # Synthetic labels activate the correct-anchor branch; no real TX truth
        # or accuracy is inferred from teacher-selected synthetic labels.
        labels = teacher_clean['tx_logits'].argmax(-1)
        for name in ('point', 'safe', 'asymmetric', 'tangent_only', 'route_only', 'memory'):
            model.load_state_dict(initial, strict=True)
            teacher.load_state_dict(initial, strict=True)
            model.train()
            selected = next((r for r in manifest_rows if r['candidate']==names[name]),None)
            # Use the real training namespace; artificial fields concealed the E11 failure.
            args = build_arg_parser().parse_args(selected['argv'][3:] if selected else [
                '--output_dir','synthetic_smoke_unused','--pair_reform',name if name in {'safe','asymmetric'} else 'point'])
            if selected is None:
                args.pair_tangent_weight=.035 if name=='tangent_only' else 0.
                args.pair_route_weight=.05 if name=='route_only' else 0.
            args.pair_direction_ratio=1.  # Always exercise directions on a two-sample smoke.
            optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)
            scaler = torch.cuda.amp.GradScaler(enabled=amp,init_scale=128.)
            tracked = model.id_backbone.cls_head.head.weight
            before = tracked.detach().clone()
            memory = DenseTemporalOrbitMemory(train_physical_ids=keys,
                feature_dim=teacher_clean['z_id'].shape[-1]) if name=='memory' else None
            step_rows = []
            for step in range(2):
                optimizer.zero_grad(set_to_none=True)
                transaction = runtime.StepStateTransaction()
                epoch=max(args.pair_start_epoch,args.pair_pseudo_start_epoch)+step
                with torch.autocast(device_type=device.type,dtype=torch.float16,enabled=amp):
                    out = model(torch.cat([clean, channel]), y_tx=None, return_aux=True,
                        domain_labels=torch.cat([domains, domains]))
                    paired = [{k: v[start:start+batch] for k,v in out.items()
                        if torch.is_tensor(v) and v.ndim and len(v)==2*batch} for start in (0,batch)]
                    result = runtime.compute_pair_batch(model=model, teacher=teacher, clean=clean,
                        channel=channel, student_clean=paired[0], student_channel=paired[1],
                        domains=domains, labels=labels if step==0 else None,
                        metadata={'snr_db':torch.full((batch,),25.,device=device)},
                        args=args, epoch=epoch, physical_ids=ids, memory=memory, transaction=transaction)
                    # Synthetic CE permits a valid optimizer step even with zero safe hinge.
                    loss = result['loss'] + F.cross_entropy(out['tx_logits'], labels.repeat(2))
                if not torch.isfinite(loss):
                    raise RuntimeError(f'{name}: nonfinite forward loss')
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                gradients = [p.grad for p in model.parameters() if p.grad is not None]
                if not gradients or not all(torch.isfinite(g).all() for g in gradients):
                    transaction.finish(applied=False)
                    raise RuntimeError(f'{name}: missing or nonfinite gradients')
                if any(p.grad is not None for p in teacher.parameters()):
                    raise RuntimeError(f'{name}: teacher accumulated gradients')
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5., error_if_nonfinite=True)
                old_scale=scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                if scaler.get_scale()<old_scale:
                    transaction.finish(applied=False)
                    raise RuntimeError(f'{name}: optimizer step skipped')
                if not all(torch.isfinite(p).all() for p in model.parameters()):
                    transaction.finish(applied=False)
                    raise RuntimeError(f'{name}: optimizer produced nonfinite parameters')
                transaction.finish(applied=True)
                _update_ema_model(teacher,model,.5)
                with torch.no_grad(), torch.autocast(device_type=device.type,dtype=torch.float16,enabled=amp):
                    refreshed=teacher.forward_identity_only(clean,domain_labels=domains)
                    runtime._fixed_head_weights(teacher,refreshed)
                step_rows.append({'loss':float(loss.detach()), 'pair_loss':float(result['loss'].detach()),
                    'ema_updated':True,'post_ema_head_consistency':'VERIFIED',
                    'role':'L' if step==0 else 'U','epoch':epoch,'amp':amp,
                    'components':sorted(result['weighted_components']),
                    'student_extra_samples':int(result['diagnostics']['student_extra_views'])})
            changed = not torch.equal(before, tracked.detach())
            if not changed:
                raise RuntimeError(f'{name}: optimizer did not change the classifier weights')
            if name == 'tangent_only' and 'tangent' not in step_rows[0]['components']:
                raise RuntimeError('tangent-only path did not execute')
            if name == 'route_only' and 'route' not in step_rows[0]['components']:
                raise RuntimeError('route-only path did not execute')
            rows[name] = {'status':'VERIFIED','optimizer_steps':len(step_rows),
                          'parameters_changed':changed, 'steps':step_rows}
            if memory is not None:
                _, found, _ = memory.lookup(keys, step=epoch)
                rows[name]['memory_hit_rate'] = float(found.float().mean())
                if not found.all():
                    raise RuntimeError('memory did not commit after the successful step')
        if domain_forwards or not identity_forwards:
            raise RuntimeError('teacher did not remain identity-only')
        return {'schema':'cvs.phase1.adv3b02_pair_reform_checkpoint_smoke.v1','status':'VERIFIED',
            'checkpoint':str(checkpoint),'checkpoint_epoch':int(payload.get('epoch',-1)),
            'checkpoint_compatibility':compatibility,'input_role':'source_shaped_synthetic_smoke_only',
            'query_inputs':0,'target_inputs':0,'teacher_domain_forwards':len(domain_forwards),
            'teacher_identity_forwards':len(identity_forwards),'fixed_head_consistency':'VERIFIED',
            'classifier_weight_shape':list(weights.shape),'rows':rows,'amp':amp,
            'training_manifest':str(training_manifest) if training_manifest else None,
            'claim_boundary':'Runtime correctness only; synthetic labels, no measured scientific accuracy or throughput.'}
    finally:
        handle.remove()
        identity_handle.remove()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--output-json', '--output_json', dest='output_json', required=True)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--amp', action='store_true')
    parser.add_argument('--training-manifest')
    args = parser.parse_args(argv)
    if args.threads < 1:
        parser.error('--threads must be positive')
    if args.amp and torch.device(args.device).type!='cuda':
        parser.error('--amp requires CUDA')
    torch.set_num_threads(args.threads)
    checkpoint = Path(args.checkpoint).resolve()
    try:
        output = run_smoke(checkpoint, torch.device(args.device),amp=args.amp,training_manifest=args.training_manifest)
        code = 0
    except Exception as exc:
        output = {'schema':'cvs.phase1.adv3b02_pair_reform_checkpoint_smoke.v1',
                  'status':'FAILED','checkpoint':str(checkpoint),'error':f'{type(exc).__name__}: {exc}'}
        code = 1
    path = Path(args.output_json).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(output, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    if json.loads(path.read_text(encoding='utf-8')) != output:
        raise RuntimeError('smoke JSON readback mismatch')
    print(json.dumps(output, ensure_ascii=False, sort_keys=True))
    return code


if __name__ == '__main__':
    raise SystemExit(main())
