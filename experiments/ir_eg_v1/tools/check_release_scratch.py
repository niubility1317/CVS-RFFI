"""One real scratch-checkpoint smoke on a bounded existing source-L slice.

This command does not train, load historical weights, inspect target samples,
or rebuild the existing source role contract. A successful exit is sufficient
for the release launcher to proceed immediately; no permission artifact exists.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT / 'code'
sys.path.insert(0, str(CODE))

import torch
from dataset_wisig import WiSigCompactDataset, load_wisig_compact_pkl
from cvsrffi.game_tracking.data import opaque_id
from cvsrffi.game_tracking.runtime import build_model
from cvsrffi.tensors import set_seed
from cvsrffi.xuc_fusion.response_config import validate_prepared_row
from cvsrffi.xuc_fusion.runtime import resolve_args
from cvsrffi.xuc_fusion.ir_types import audit_runtime_imports


def read_json(path):
    return Path(path).read_text(encoding='utf-8')


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def source_l_slice(args, contract, batch_size):
    """Materialize only declared L IDs; inspect index metadata, never U truth."""
    if contract.get('schema') != 'core90_game_source_roles_v1':
        raise ValueError('unsupported existing source contract schema')
    if contract.get('checkpoint_init') != 'scratch_only' or contract.get('target_access_before_freeze') is not False:
        raise ValueError('source contract is not scratch/source-only')
    rxs = [int(v) for v in args.wisig_train_rxs.split(',')]
    days = [int(v) for v in args.wisig_train_days.split(',')]
    if contract['source_rxs'] != rxs or contract['source_days'] != days:
        raise ValueError('source receiver/day settings differ from the existing contract')
    ids = contract['role_ids']['L_s'][:batch_size]
    if len(ids) != batch_size or len(set(ids)) != batch_size:
        raise ValueError('existing source-L contract lacks the requested distinct records')
    raw = load_wisig_compact_pkl(args.wisig_pkl)
    if len(raw['tx_list']) != contract['num_classes']:
        raise ValueError('native classifier shape differs from existing contract')
    base = WiSigCompactDataset(raw, out_len=args.wisig_out_len, crop_mode='center',
        normalize=True, equalized=int(args.wisig_equalized), day_keep=days,
        rx_keep=rxs, domain='rx_day', seed=args.game_split_seed)
    selected = {}
    wanted = set(ids)
    # Index metadata only. No split regeneration or whole-role revalidation.
    for index, item in enumerate(base.index):
        identity = opaque_id(item)
        if identity in wanted:
            selected[identity] = index
            if len(selected) == batch_size:
                break
    if set(selected) != wanted:
        raise ValueError('requested source-L records absent from supplied dataset')
    domains = {(int(rx), int(day)): int(domain) for rx, day, domain in contract['domain_map']}
    samples = [base[selected[identity]] for identity in ids]
    x = torch.stack([sample[0] for sample in samples])
    y = torch.as_tensor([sample[1] for sample in samples], dtype=torch.long)
    d = torch.as_tensor([domains[(int(sample[3]['rx_i']), int(sample[3]['day_i']))]
                         for sample in samples], dtype=torch.long)
    return x, y, d, ids, len(domains)


def finite_outputs(model, x, y, d):
    model.eval()
    with torch.no_grad():
        result = model(x, y_tx=y, domain_labels=d, grl_lambda=0., return_aux=True)
    checked = {}
    for key in ('tx_logits', 'adv_dom_logits', 'z_id', 'z_dom'):
        value = result.get(key)
        if not torch.is_tensor(value) or value.shape[0] != len(x) or not torch.isfinite(value).all():
            raise RuntimeError('native source forward failed at ' + key)
        checked[key] = value.detach().cpu()
    return checked


def run(cli):
    output = Path(cli.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    try:
        document = json.loads(read_json(cli.config))
        row = validate_prepared_row(document['row'])
        contract = json.loads(read_json(cli.source_contract))
        recipe = json.loads(read_json(ROOT / 'configs/core90_recipe_reference.json'))
        args = resolve_args(recipe, row, dataset=Path(cli.dataset).resolve(),
                            output=output, device=cli.device)
        args.num_classes = int(contract['num_classes'])
        if not args.from_scratch or args.baseline_ckpt or args.game_resume or getattr(args, 'teacher_ckpt', ''):
            raise ValueError('scratch smoke cannot inherit any model source')
        import model_dual_cvsincnet  # required native dependency must be pinned
        from cvsrffi.xuc_fusion import dr_objective, joint_normalization, ir_solver
        imports = audit_runtime_imports(CODE)
        device = torch.device(cli.device)
        if device.type == 'cuda' and not torch.cuda.is_available():
            raise RuntimeError('requested CUDA device unavailable')
        torch.set_num_threads(2)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        set_seed(int(row['joint']['model_seed']))
        x, y, d, ids, num_domains = source_l_slice(args, contract, cli.batch_size)
        model = build_model(args, num_domains, device).eval()
        checkpoint = output / 'scratch_checkpoint.pth'
        torch.save(dict(schema='ir_release_scratch_smoke_v1', model=model.state_dict(),
            args=vars(args), num_domains=num_domains, initialization='scratch_only',
            checkpoint_sources=[], source_contract=str(Path(cli.source_contract).resolve()),
            source_L_ids=ids, target_access=False, optimizer_steps=0), checkpoint)
        original = finite_outputs(model, x.to(device), y.to(device), d.to(device))
        # CPU deserialize is also the production optimizer-safe restore route.
        saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
        if any(value.device.type != 'cpu' for value in saved['model'].values()):
            raise RuntimeError('checkpoint CPU deserialization failed')
        restored = build_model(SimpleNamespace(**saved['args']), saved['num_domains'], device).eval()
        restored.load_state_dict(saved['model'], strict=True)
        for name, value in model.state_dict().items():
            if not torch.equal(value.detach().cpu(), restored.state_dict()[name].detach().cpu()):
                raise RuntimeError('checkpoint state roundtrip mismatch: ' + name)
        replay = finite_outputs(restored, x.to(device), y.to(device), d.to(device))
        for key in original:
            torch.testing.assert_close(replay[key], original[key], atol=1e-6, rtol=1e-5)
        result = dict(status='PASS', schema='ir_release_scratch_smoke_v1',
            row_id=row['id'], model_seed=row['joint']['model_seed'], device=str(device),
            checkpoint=str(checkpoint), checkpoint_sources=[], initialization='scratch_only',
            source_contract=str(Path(cli.source_contract).resolve()),
            role='L_s', physical_ids=ids, samples=len(ids), target_access=False,
            full_data_revalidation=False, optimizer_steps=0,
            cpu_deserialization=True, strict_state_roundtrip=True,
            finite_output_shapes={key:list(value.shape) for key,value in replay.items()},
            imports=imports)
        write_json(output / 'smoke.json', result)
        print(json.dumps(dict(status='PASS', output=str(output), samples=len(ids),
                              target_access=False, next_action='continue_launcher'), ensure_ascii=False))
        return 0
    except Exception as exc:
        write_json(output / 'failure.json', dict(status='FAILED', error_type=type(exc).__name__,
            error=str(exc), target_access=False, partial_artifacts_preserved=True))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--source-contract', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--batch-size', type=int, default=4)
    cli = parser.parse_args()
    if not 1 <= cli.batch_size <= 16:
        parser.error('--batch-size must be between 1 and 16')
    return run(cli)


if __name__ == '__main__':
    raise SystemExit(main())
