"""Restore every immutable source-only snapshot on CPU before releasing workers."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'code'))
import torch
from cvsrffi.game_tracking.runtime import build_model
from cvsrffi.game_tracking.legacy.losses import PrototypeMemoryBank
from cvsrffi.xuc_fusion.response_solver import ResponseSolver
from cvsrffi.xuc_fusion.dr_objective import DROT
from cvsrffi.xuc_fusion.resume import validate_resume_payload, restore_checkpoint


def equal(expected, actual):
    if torch.is_tensor(expected):
        return torch.equal(expected.cpu(), actual.cpu())
    if isinstance(expected, dict):
        return expected.keys() == actual.keys() and all(equal(v, actual[k]) for k, v in expected.items())
    if isinstance(expected, (list, tuple)):
        return len(expected) == len(actual) and all(equal(a, b) for a, b in zip(expected, actual))
    return expected == actual


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--destination', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    manifest = json.loads((args.destination / 'manifest.json').read_text())
    run = Path(manifest['run'])
    contract = json.loads((run / 'source_contract.json').read_text())
    result = dict(status='RUNNING', scope='CPU strict reconstruction; no target or optimizer updates', rows={})
    reference = json.loads((ROOT / 'configs/a1_native_recipe_reference.json').read_text())
    for rid, saved in manifest['rows'].items():
        payload = torch.load(saved['checkpoint'], map_location='cpu', weights_only=False)
        config = json.loads((Path(manifest['original_state']['release']) / 'configs/separate_controls' / (rid + '.json')).read_text())['row']
        source_info = json.loads((run / rid / 'source_contract.json').read_text())
        if source_info['role_ids'] != contract['role_ids']:
            raise ValueError('CHECKPOINT_DATA_CONTRACT_MISMATCH')
        restored_args = SimpleNamespace(**deepcopy(payload['args']))
        restored_args.device = 'cpu'
        validate_resume_payload(payload, config, source_info, restored_args, allow_legacy=True)
        model = build_model(restored_args, len(source_info['domains']) if 'domains' in source_info else 15, torch.device('cpu'))
        ema = deepcopy(model).eval()
        grouped = {role: [] for role in ('backbone', 'other')}
        for name, parameter in model.named_parameters():
            grouped['backbone' if 'backbone' in name.lower() else 'other'].append(parameter)
        optimizer = torch.optim.AdamW([dict(params=v, fasttrust_role=k) for k, v in grouped.items()],
                                     lr=restored_args.lr, weight_decay=restored_args.weight_decay)
        solver = ResponseSolver(model, optimizer, restored_args.response, nonfinite='raise', max_grad_norm=5.,
            predictor_lr_ratio=restored_args.joint['predictor_lr_ratio'], telemetry_interval=restored_args.joint['diagnostic_interval'])
        proto = PrototypeMemoryBank(restored_args.num_classes, 15, momentum=restored_args.proto_momentum,
            margin=restored_args.proto_margin, domain_align_weight=restored_args.proto_domain_align_weight,
            push_weight=restored_args.proto_push_weight, min_count=restored_args.proto_min_count)
        stream = SimpleNamespace()
        dr = DROT(model, ema, SimpleNamespace(info=source_info), restored_args, reference)
        restore_checkpoint(payload, model, ema, optimizer, proto, solver, stream, dr)
        checks = dict(model=equal(payload['model'], model.state_dict()), ema=equal(payload['ema'], ema.state_dict()),
            optimizer=equal(payload['optimizer'], optimizer.state_dict()), prototype=equal(payload['prototype'], vars(proto)),
            solver=equal(payload['solver'], solver.state_dict()), dr=equal(payload['daot_rc4']['scale'], dr.scale.state_dict()),
            tickets=all(equal(payload['tickets'][k], getattr(stream, k)) for k in ('next_epoch','pending','consumed','window','window_consumed')))
        if not all(checks.values()):
            raise RuntimeError('Restored state differs: ' + rid + ' ' + repr(checks))
        result['rows'][rid] = dict(status='PASS', checkpoint=saved['checkpoint'], epoch=payload['epoch'],
                                  step=payload['step'], checks=checks, initialization='same-run continuation')
        print(json.dumps(dict(row=rid, status='PASS', epoch=payload['epoch'])), flush=True)
        del model, ema, optimizer, solver, proto, dr, payload
    result['status'] = 'PASS'
    target = args.destination / 'restore_validation.json'
    if target.exists():
        raise FileExistsError(target)
    target.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
