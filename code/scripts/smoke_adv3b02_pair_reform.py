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


def run_smoke(checkpoint: Path, device: torch.device) -> dict:
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
    try:
        with torch.no_grad():
            teacher_clean = teacher.forward_identity_only(clean, domain_labels=domains)
        weights = runtime._fixed_head_weights(teacher, teacher_clean)
        # Synthetic labels activate the correct-anchor branch; no real TX truth
        # or accuracy is inferred from teacher-selected synthetic labels.
        labels = teacher_clean['tx_logits'].argmax(-1)
        for name in ('point', 'safe', 'asymmetric', 'tangent_only', 'route_only', 'memory'):
            model.load_state_dict(initial, strict=True)
            model.train()
            args = SimpleNamespace(pair_reform=name if name in {'safe','asymmetric'} else 'point',
                pair_weight=.5, pair_pseudo_weight=.2, pair_start_epoch=1, pair_pseudo_start_epoch=1,
                pair_alpha=.5, pair_u_tolerance=.1, pair_teacher_mix=.25, pair_unknown_quality='neutral',
                pair_tangent_weight=.035 if name=='tangent_only' else 0.,
                pair_route_weight=.05 if name=='route_only' else 0.,
                pair_direction_ratio=1., pair_direction_delta=.05,
                pair_direction_scale=.05, pair_direction_budget=0.,
                seed=392005, sat_fs_hz=float(model_args.get('sample_rate_hz', 25e6)))
            optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)
            tracked = model.id_backbone.cls_head.head.weight
            before = tracked.detach().clone()
            memory = DenseTemporalOrbitMemory(train_physical_ids=keys,
                feature_dim=teacher_clean['z_id'].shape[-1]) if name=='memory' else None
            step_rows = []
            for step in range(2 if memory is not None else 1):
                optimizer.zero_grad(set_to_none=True)
                out = model(torch.cat([clean, channel]), y_tx=None, return_aux=True,
                    domain_labels=torch.cat([domains, domains]))
                paired = [{k: v[start:start+batch] for k,v in out.items()
                    if torch.is_tensor(v) and v.ndim and len(v)==2*batch} for start in (0,batch)]
                transaction = runtime.StepStateTransaction()
                result = runtime.compute_pair_batch(model=model, teacher=teacher, clean=clean,
                    channel=channel, student_clean=paired[0], student_channel=paired[1],
                    domains=domains, labels=labels, metadata={'snr_db':torch.full((batch,),25.,device=device)},
                    args=args, epoch=step+1, physical_ids=ids, memory=memory, transaction=transaction)
                # Core clean/LEO CE makes a zero safe auxiliary a valid optimizer
                # step, rather than treating zero hinge as a failure.
                loss = result['loss'] + F.cross_entropy(out['tx_logits'], labels.repeat(2))
                if not torch.isfinite(loss):
                    raise RuntimeError(f'{name}: nonfinite forward loss')
                loss.backward()
                gradients = [p.grad for p in model.parameters() if p.grad is not None]
                if not gradients or not all(torch.isfinite(g).all() for g in gradients):
                    transaction.finish(applied=False)
                    raise RuntimeError(f'{name}: missing or nonfinite gradients')
                if any(p.grad is not None for p in teacher.parameters()):
                    raise RuntimeError(f'{name}: teacher accumulated gradients')
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5., error_if_nonfinite=True)
                optimizer.step()
                if not all(torch.isfinite(p).all() for p in model.parameters()):
                    transaction.finish(applied=False)
                    raise RuntimeError(f'{name}: optimizer produced nonfinite parameters')
                transaction.finish(applied=True)
                step_rows.append({'loss':float(loss.detach()), 'pair_loss':float(result['loss'].detach()),
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
                _, found, _ = memory.lookup(keys, step=2)
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
            'classifier_weight_shape':list(weights.shape),'rows':rows,
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
    args = parser.parse_args(argv)
    if args.threads < 1:
        parser.error('--threads must be positive')
    torch.set_num_threads(args.threads)
    checkpoint = Path(args.checkpoint).resolve()
    try:
        output = run_smoke(checkpoint, torch.device(args.device))
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
