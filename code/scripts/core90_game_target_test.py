"""One frozen, jointly predicted target test of the authorized 15 E200 rows.

Closed-set diagnostic: no fitting, selection, support adaptation or unknown claim.
All models finish all four scenes before a separate artifact scorer sees truth.
"""
from __future__ import annotations
import argparse
from contextlib import ExitStack
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace
import torch
from torch.utils.data import Dataset, DataLoader

CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE))
from core90_game_evaluate import EVAL_SCENES, score_source_predictions, _write_json
from cvsrffi.game_tracking.data import build_source, opaque_id
from cvsrffi.game_tracking.runtime import build_model, configure_determinism
from cvsrffi.eval import apply_sat_channel_for_scenario
from dataset_wisig import WiSigCompactDataset

ROWS = ('B0','B1','B2','B3','B3_lr_high','B3_head_slow','B3_head_fast',
        'B7','S1','S3','S4','C1','C2','H1','J1')
CONTRACT_KEYS = ('source_rxs','source_days','domain_map','role_ids','num_classes',
                 'counts','split_rule','split_seed')

def check_checkpoint(checkpoint, reference=None, *, expected_seed=392005):
    if (checkpoint.get('epoch') != 200 or checkpoint.get('initialization') != 'scratch_only'
        or checkpoint.get('checkpoint_selection') != 'final_only'
        or checkpoint.get('target_contact') is not False):
        raise ValueError('Checkpoint is not a clean frozen scratch E200 final')
    args = checkpoint['args']
    info = checkpoint['source_info']
    if args['seed'] != expected_seed or args.get('game_synthetic') or info.get('checkpoint_init') != 'scratch_only' or info.get('target_access_before_freeze') is not False:
        raise ValueError('Checkpoint provenance is unverified')
    if reference is not None:
        for key in CONTRACT_KEYS:
            if info.get(key) != reference.get(key):
                raise ValueError('Checkpoint data contract mismatch: ' + key)

class TargetInputs(Dataset):
    def __init__(self, base): self.base = base
    def __len__(self): return len(self.base)
    def __getitem__(self, i):
        x, _, _, _ = self.base[i]
        item = self.base.index[i]
        return x, dict(sample_id=opaque_id(item), rx_i=item.rx_i, day_i=item.day_i)

def check_target(base, source_info, rxs, days):
    if not len(base) or set(rxs) & set(source_info['source_rxs']):
        raise ValueError('Empty target or source/target receiver overlap')
    if {i.rx_i for i in base.index} != set(rxs) or {i.day_i for i in base.index} != set(days):
        raise ValueError('Target physical IDs differ from contract')
    ids = [opaque_id(i) for i in base.index]
    source_ids = set().union(*(set(v) for v in source_info['role_ids'].values()))
    if len(set(ids)) != len(ids) or source_ids.intersection(ids):
        raise ValueError('Duplicate or overlapping target physical sample')

def assign_devices(rows, devices):
    if not devices or len(set(devices)) != len(devices):
        raise ValueError('Expected distinct evaluation devices')
    return {row: devices[i % len(devices)] for i, row in enumerate(rows)}

def predict_group(items, x, device, num_classes):
    # Inference mode is thread-local; each GPU worker must enter it itself.
    with torch.inference_mode():
        local_x = x.to(device)
        result = {}
        for row, model in items:
            logits = model(local_x, return_aux=True)['tx_logits']
            if logits.shape != (len(local_x), num_classes) or not torch.isfinite(logits).all():
                raise ValueError('Invalid all-class logits: '+row)
            conf, pred = logits.float().softmax(-1).max(-1)
            result[row] = (conf.cpu().tolist(), pred.cpu().tolist())
        return result

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    inputs=parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument('--run-root')
    inputs.add_argument('--input-manifest',help='Frozen explicit completed E200 checkpoints and training seeds')
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--devices', nargs='+', help='Balanced model placement; shared augmentation stays on --device')
    parser.add_argument('--rows', nargs='+', choices=ROWS+('B8','B3_head_scale','B3_adv_low',
        'B3_adv_high','B4','B4_fixedk','B5','S2','S2_random','C3'), default=list(ROWS))
    opts = parser.parse_args(argv)
    explicit=json.loads(Path(opts.input_manifest).read_text()) if opts.input_manifest else None
    if explicit is not None:
        if explicit.get('scope')!='frozen_completed_v2_target_recheck' or explicit.get('selection_permitted') is not False:
            raise ValueError('Invalid frozen target input scope')
        rows=tuple(r['run_id'] for r in explicit['models'])
        seeds={r['run_id']:r['seed'] for r in explicit['models']}
        if not rows or any(type(s) is not int or s<0 for s in seeds.values()):raise ValueError('Invalid frozen training seeds')
        if any(not row.startswith('V2_') or not row.endswith('_seed'+str(seeds[row])) for row in rows):
            raise ValueError('Frozen V2 run ID and training seed disagree')
    else:
        rows = tuple(opts.rows);seeds={r:392005 for r in rows}
    if len(set(rows)) != len(rows): raise ValueError('Duplicate requested row')
    devices = opts.devices or [opts.device]
    row_devices = assign_devices(rows, devices)
    root, out = Path(opts.run_root) if opts.run_root else None, Path(opts.output_dir)
    out.mkdir(parents=True, exist_ok=False)
    # Freeze and verify every candidate before opening target data.
    checkpoints = {}
    for row in rows:
        if explicit is not None:
            entry=next(r for r in explicit['models'] if r['run_id']==row)
            path=Path(entry['checkpoint'])
            if path.name!='final_ssdg.pth' or path.parent.name!=row or not path.is_file():
                raise ValueError('Expected existing final checkpoint for '+row)
            completion=json.loads((path.parent/'completion.json').read_text())
            if completion.get('status')!='SOURCE_ARTIFACTS_COMPLETE' or completion.get('epoch')!=200 or completion.get('target_evaluated') is not False:
                raise ValueError('Incomplete source row: '+row)
            candidates=[path]
        else:candidates = list(root.glob('runs/' + row + '_seed392005/final_ssdg.pth'))
        if len(candidates) != 1:
            raise ValueError(f'Expected exactly one final checkpoint for {row}: {candidates}')
        checkpoints[row] = candidates[0]
    models, buffers, reference, args = {}, {}, None, None
    for row, path in checkpoints.items():
        saved = torch.load(path, map_location='cpu', weights_only=False)
        check_checkpoint(saved, reference, expected_seed=seeds[row])
        if reference is None:
            reference = saved['source_info']
            args = SimpleNamespace(**saved['args'])
            configure_determinism(args)
        for key in ('wisig_test_rxs','wisig_test_days','wisig_out_len','wisig_equalized','game_split_seed'):
            if saved['args'][key] != getattr(args,key):
                raise ValueError('Test configuration differs: ' + key)
        row_args = SimpleNamespace(**saved['args'])
        row_args.device = row_devices[row]
        model = build_model(row_args, len(reference['domain_map']), torch.device(row_devices[row]))
        model.load_state_dict(saved['model'], strict=True)
        model.eval().requires_grad_(False)
        models[row] = model
        buffers[row] = {k: v.detach().clone() for k,v in model.named_buffers()}
        del saved
    source = build_source(args)
    for key in CONTRACT_KEYS:
        if source.info[key] != reference[key]:
            raise ValueError('Actual source contract mismatch: ' + key)
    with torch.inference_mode():
        smoke_x = next(iter(source.loader('val', 16)))[0].to(opts.device)
        for row, model in models.items():
            logits = model(smoke_x.to(row_devices[row]), return_aux=True)['tx_logits']
            if logits.shape != (16,args.num_classes) or not torch.isfinite(logits).all():
                raise ValueError('Source-only checkpoint smoke failed: '+row)
    groups = {device: [(row, model) for row, model in models.items() if row_devices[row] == device] for device in devices}
    if smoke_x.is_cuda: torch.cuda.synchronize(smoke_x.device)
    reference_predictions = {}
    for device, items in groups.items():
        reference_predictions.update(predict_group(items, smoke_x, device, args.num_classes))
    with ThreadPoolExecutor(max_workers=len(devices)) as smoke_pool:
        smoke_futures = [smoke_pool.submit(predict_group, items, smoke_x, device, args.num_classes) for device, items in groups.items()]
        parallel_predictions = {}
        for future in smoke_futures: parallel_predictions.update(future.result())
    if parallel_predictions != reference_predictions:
        raise ValueError('Parallel source checkpoint smoke differs from serial inference')
    print('SOURCE_CHECKPOINT_SMOKE_PASS',flush=True)
    rxs = [int(v) for v in args.wisig_test_rxs.split(',')]
    days = [int(v) for v in args.wisig_test_days.split(',')]
    if explicit is not None and (rxs!=explicit['target_rxs'] or days!=explicit['target_days'] or args.game_split_seed!=explicit['augmentation_seed']):
        raise ValueError('Frozen target test configuration mismatch')
    base = WiSigCompactDataset(source.train.base.ds, out_len=args.wisig_out_len,
                              crop_mode='center', normalize=True, equalized=args.wisig_equalized,
                              rx_keep=rxs, day_keep=days, domain='rx_day', seed=args.game_split_seed)
    check_target(base, reference, rxs, days)
    if explicit is not None and len(base)!=explicit['samples_per_scene']:raise ValueError('Frozen target count mismatch')
    manifest = dict(schema='core90_target_frozen_multiseed_v2' if explicit else 'core90_target_frozen_15_v1', rows=list(rows), seed=392005 if not explicit else None,
                    training_seeds=seeds,target_exposure_status='previously_exposed_benchmark_recheck',
                    checkpoints={k:str(v) for k,v in checkpoints.items()}, target_rxs=rxs,
                    target_days=days, samples_per_scene=len(base), scenes=list(EVAL_SCENES),
                    batch_size=256, augmentation_seed=args.game_split_seed,
                    num_classes=args.num_classes, source_only=False,
                    scope='four_view_closed_set_target_diagnostic', fitting=False,
                    selection=False, all_rows_predicted_before_truth=True)
    manifest['model_devices'] = row_devices
    _write_json(out/'frozen_manifest.json', manifest)
    counts = {}
    started = time.perf_counter()
    with ExitStack() as stack, torch.inference_mode():
        pool = stack.enter_context(ThreadPoolExecutor(max_workers=len(devices)))
        streams = {row: stack.enter_context((out/(row+'_predictions.jsonl')).open('x',encoding='utf-8')) for row in rows}
        for si, scene in enumerate(EVAL_SCENES):
            gen = torch.Generator(device=opts.device).manual_seed(args.game_split_seed+100003*si)
            loader = DataLoader(TargetInputs(base), batch_size=256, shuffle=False, num_workers=0)
            counts[scene] = 0
            for batch_i, (x, meta) in enumerate(loader):
                x = x.to(opts.device)
                if scene != 'clean':
                    x = apply_sat_channel_for_scenario(x, scene, args, gen=gen)[0]
                # Finish the one shared view before worker streams consume it.
                if x.is_cuda: torch.cuda.synchronize(x.device)
                futures = [pool.submit(predict_group, items, x, device, args.num_classes) for device, items in groups.items()]
                predictions = {}
                for future in futures: predictions.update(future.result())
                for row in rows:
                    conf, pred = predictions[row]
                    streams[row].writelines(json.dumps(dict(sample_id=meta['sample_id'][j],scene=scene,
                        rx_i=int(meta['rx_i'][j]),day_i=int(meta['day_i'][j]),prediction=pred[j],confidence=conf[j]))+'\n' for j in range(len(x)))
                counts[scene] += len(x)
                if batch_i % 100 == 0:
                    print(json.dumps(dict(stage='predict',scene=scene,samples=counts[scene],total=len(base))),flush=True)
    if any(n != len(base) for n in counts.values()): raise ValueError('Incomplete predictions')
    for row, model in models.items():
        if any(not torch.equal(v,dict(model.named_buffers())[k]) for k,v in buffers[row].items()):
            raise ValueError('Evaluation mutated buffers: '+row)
    _write_json(out/'predictions_complete.json',dict(complete=True,rows=list(rows),counts=counts,
                prediction_seconds=time.perf_counter()-started,truth_accessed_for_scoring=False))
    # A separate scorer consumes only closed artifacts and builder-side truth.
    truth_path = out/'target_truth.jsonl'
    with truth_path.open('x',encoding='utf-8') as stream:
        for item in base.index:
            stream.write(json.dumps(dict(sample_id=opaque_id(item),truth=item.tx_i,rx_i=item.rx_i,day_i=item.day_i))+'\n')
    for row in rows:
        scores = score_source_predictions(out/(row+'_predictions.jsonl'),truth_path,
                    expected_samples=len(base),num_classes=args.num_classes)
        scores.update(schema='core90_game_target_scores_v1',source_only=False,target_evaluated=True,
                      scope=manifest['scope'],row=row,seed=seeds[row],selection_permitted=False)
        _write_json(out/(row+'_target_scores.json'),scores)
        print(json.dumps(dict(stage='scored',row=row)),flush=True)
    _write_json(out/'complete.json',dict(complete=True,rows=list(rows),manifest='frozen_manifest.json',
                total_seconds=time.perf_counter()-started))

if __name__ == '__main__': main()
