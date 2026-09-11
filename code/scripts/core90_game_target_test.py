"""One frozen, jointly predicted target test of the authorized 15 E200 rows.

Closed-set diagnostic: no fitting, selection, support adaptation or unknown claim.
All models finish all four scenes before a separate artifact scorer sees truth.
"""
from __future__ import annotations
import argparse
from contextlib import ExitStack
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
from cvsrffi.game_tracking.runtime import build_model
from cvsrffi.eval import apply_sat_channel_for_scenario
from dataset_wisig import WiSigCompactDataset

ROWS = ('B0','B1','B2','B3','B3_lr_high','B3_head_slow','B3_head_fast',
        'B7','S1','S3','S4','C1','C2','H1','J1')
CONTRACT_KEYS = ('source_rxs','source_days','domain_map','role_ids','num_classes',
                 'counts','split_rule','split_seed')

def check_checkpoint(checkpoint, reference=None):
    if (checkpoint.get('epoch') != 200 or checkpoint.get('initialization') != 'scratch_only'
        or checkpoint.get('checkpoint_selection') != 'final_only'
        or checkpoint.get('target_contact') is not False):
        raise ValueError('Checkpoint is not a clean frozen scratch E200 final')
    args = checkpoint['args']
    info = checkpoint['source_info']
    if args['seed'] != 392005 or args.get('game_synthetic') or info.get('checkpoint_init') != 'scratch_only' or info.get('target_access_before_freeze') is not False:
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

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-root', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--rows', nargs='+', choices=ROWS+('B8',), default=list(ROWS))
    opts = parser.parse_args(argv)
    rows = tuple(opts.rows)
    if len(set(rows)) != len(rows): raise ValueError('Duplicate requested row')
    root, out = Path(opts.run_root), Path(opts.output_dir)
    out.mkdir(parents=True, exist_ok=False)
    # Freeze and verify every candidate before opening target data.
    checkpoints = {}
    for row in rows:
        candidates = list(root.glob('runs/' + row + '_seed392005/final_ssdg.pth'))
        if len(candidates) != 1:
            raise ValueError(f'Expected exactly one final checkpoint for {row}: {candidates}')
        checkpoints[row] = candidates[0]
    models, buffers, reference, args = {}, {}, None, None
    for row, path in checkpoints.items():
        saved = torch.load(path, map_location='cpu', weights_only=False)
        check_checkpoint(saved, reference)
        if reference is None:
            reference = saved['source_info']
            args = SimpleNamespace(**saved['args'])
        for key in ('wisig_test_rxs','wisig_test_days','wisig_out_len','wisig_equalized','game_split_seed'):
            if saved['args'][key] != getattr(args,key):
                raise ValueError('Test configuration differs: ' + key)
        row_args = SimpleNamespace(**saved['args'])
        row_args.device = opts.device
        model = build_model(row_args, len(reference['domain_map']), torch.device(opts.device))
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
            logits = model(smoke_x, return_aux=True)['tx_logits']
            if logits.shape != (16,args.num_classes) or not torch.isfinite(logits).all():
                raise ValueError('Source-only checkpoint smoke failed: '+row)
    print('SOURCE_CHECKPOINT_SMOKE_PASS',flush=True)
    rxs = [int(v) for v in args.wisig_test_rxs.split(',')]
    days = [int(v) for v in args.wisig_test_days.split(',')]
    base = WiSigCompactDataset(source.train.base.ds, out_len=args.wisig_out_len,
                              crop_mode='center', normalize=True, equalized=args.wisig_equalized,
                              rx_keep=rxs, day_keep=days, domain='rx_day', seed=args.game_split_seed)
    check_target(base, reference, rxs, days)
    manifest = dict(schema='core90_target_frozen_15_v1', rows=list(rows), seed=392005,
                    checkpoints={k:str(v) for k,v in checkpoints.items()}, target_rxs=rxs,
                    target_days=days, samples_per_scene=len(base), scenes=list(EVAL_SCENES),
                    batch_size=256, augmentation_seed=args.game_split_seed,
                    num_classes=args.num_classes, source_only=False,
                    scope='four_view_closed_set_target_diagnostic', fitting=False,
                    selection=False, all_rows_predicted_before_truth=True)
    _write_json(out/'frozen_manifest.json', manifest)
    counts = {}
    started = time.perf_counter()
    with ExitStack() as stack, torch.inference_mode():
        streams = {row: stack.enter_context((out/(row+'_predictions.jsonl')).open('x',encoding='utf-8')) for row in rows}
        for si, scene in enumerate(EVAL_SCENES):
            gen = torch.Generator(device=opts.device).manual_seed(args.game_split_seed+100003*si)
            loader = DataLoader(TargetInputs(base), batch_size=256, shuffle=False, num_workers=0)
            counts[scene] = 0
            for batch_i, (x, meta) in enumerate(loader):
                x = x.to(opts.device)
                if scene != 'clean':
                    x = apply_sat_channel_for_scenario(x, scene, args, gen=gen)[0]
                for row, model in models.items():
                    logits = model(x, return_aux=True)['tx_logits']
                    if logits.shape != (len(x),args.num_classes) or not torch.isfinite(logits).all():
                        raise ValueError('Invalid all-class logits: '+row)
                    conf, pred = logits.float().softmax(-1).max(-1)
                    conf, pred = conf.cpu().tolist(), pred.cpu().tolist()
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
                      scope=manifest['scope'],row=row,seed=392005,selection_permitted=False)
        _write_json(out/(row+'_target_scores.json'),scores)
        print(json.dumps(dict(stage='scored',row=row)),flush=True)
    _write_json(out/'complete.json',dict(complete=True,rows=list(rows),manifest='frozen_manifest.json',
                total_seconds=time.perf_counter()-started))

if __name__ == '__main__': main()
