"""Complete four-scene prediction fixation followed by independent truth scoring.

Prediction receives only IQ and opaque record IDs for bookkeeping. WiSig labels
are accessed only by the separate truth provider, after every scene is on disk
and independently checked for complete, unique record coverage.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Callable, Mapping, Sequence

import numpy as np
import torch

from cvsrffi.eval import apply_sat_channel_for_scenario
from cvsrffi.tensors import make_torch_generator
from training_test_eval import select_main_test_keys

SCENARIOS = ('clean', 'leo_clear_weak', 'leo_low_elev_weak', 'leo_rain_weak')


@dataclass(frozen=True)
class ReceivedRecord:
    record_id: str
    read_iq: Callable[[], torch.Tensor]
    receiver: int | None = None
    day: int | None = None


@contextmanager
def _fixed_model(model):
    states = [(module, module.training) for module in model.modules()]
    try:
        model.eval()
        with torch.no_grad():
            yield
    finally:
        for module, was_training in states:
            module.training = was_training


def _received_wisig_iq(dataset, index: int):
    """Match dataset input preprocessing without invoking its label-returning API."""
    from dataset_wisig import _pad_or_crop_2t, _rms_normalize_iq, _safe_to_torch_float_tensor
    base = getattr(dataset, 'base', dataset)
    for item in (dataset, base):
        if getattr(item, 'transform', None) is not None:
            raise ValueError('frozen prediction requires unaugmented received IQ')
    row = dataset.index[index]
    raw = np.asarray(base.data[row.tx_i][row.rx_i][row.day_i][row.eq_i][row.sig_i], dtype=np.float32).T
    if raw.ndim != 2 or raw.shape[0] != 2 or not np.isfinite(raw).all():
        raise ValueError('invalid received WiSig IQ record')
    if raw.shape[1] != base.out_len:
        raw = _pad_or_crop_2t(raw, base.out_len, mode=base.crop_mode)
    if base.normalize:
        raw = _rms_normalize_iq(raw, center=base.center)
    return _safe_to_torch_float_tensor(raw)


def wisig_received_catalog(named_loaders):
    """One prediction per physical record, allowing named-subset aliases.

    Opaque IDs are deterministic ordinals of the fixed physical-address catalog;
    addressing fields never enter the model and no TX label is stored in the
    prediction manifest. Dataset indexing here identifies received arrays only.
    """
    addresses = {}
    named_addresses = {}
    corpus = None
    for name, loader in named_loaders.items():
        dataset = loader.dataset
        base = getattr(dataset, 'base', dataset)
        if not hasattr(dataset, 'index') or not hasattr(base, 'data'):
            raise TypeError('final scoring requires physical WiSig dataset indexing')
        if corpus is None:
            corpus = base.data
        elif corpus is not base.data:
            raise ValueError('named loaders refer to different physical corpora')
        for item in (dataset, base):
            if getattr(item, 'transform', None) is not None:
                raise ValueError('final heldout datasets must have no random/label-dependent transform')
        keys = []
        seen = set()
        for index, row in enumerate(dataset.index):
            key = (int(row.tx_i), int(row.rx_i), int(row.day_i), int(row.eq_i), int(row.sig_i))
            if key in seen:
                raise ValueError(f'duplicate physical record in named loader {name}')
            seen.add(key)
            keys.append(key)
            addresses.setdefault(key, (dataset, index))
        named_addresses[str(name)] = keys
    if not addresses:
        raise ValueError('no heldout physical records for complete prediction')
    ids = {key: f'wisig_record_{i:010d}' for i, key in enumerate(sorted(addresses))}
    records = []
    truth_addresses = {}
    for key in sorted(addresses):
        dataset, index = addresses[key]
        record_id = ids[key]
        records.append(ReceivedRecord(record_id,
            lambda ds=dataset, idx=index: _received_wisig_iq(ds, idx), key[1], key[2]))
        truth_addresses[record_id] = (dataset, index)
    named_ids = {name: [ids[key] for key in keys] for name, keys in named_addresses.items()}

    def truth_provider(expected_ids):
        # Separate late role: only now interpret the archived WiSig TX field as
        # a label. No dataset.__getitem__ call materializes truth during forward.
        return {record_id: int(truth_addresses[record_id][0].index[truth_addresses[record_id][1]].tx_i)
                for record_id in expected_ids}

    return records, named_ids, truth_provider


def write_fixed_predictions(model, records: Sequence[ReceivedRecord], named_ids: Mapping[str, Sequence[str]],
                            args, device, output_dir: str | Path, *, channel_apply=apply_sat_channel_for_scenario):
    """Write all classes for all records in all four scenes, without a truth API."""
    if int(getattr(args, 'eval_max_batches', 0)) > 0 or int(getattr(args, 'sat_eval_max_batches', 0)) > 0:
        raise ValueError('final truth-last prediction requires complete datasets, not batch caps')
    record_ids = [record.record_id for record in records]
    if not record_ids or len(record_ids) != len(set(record_ids)):
        raise ValueError('prediction record IDs must be nonempty and unique')
    expected = set(record_ids)
    if any(len(ids) != len(set(ids)) or not set(ids) <= expected for ids in named_ids.values()):
        raise ValueError('named subset contains duplicate or unknown record IDs')
    if set().union(*(set(ids) for ids in named_ids.values())) != expected:
        raise ValueError('named subsets must cover every predicted record')
    classes = int(getattr(model, 'num_classes', getattr(args, 'num_classes', 0)))
    batch_size = int(getattr(args, 'eval_batch_size', 256))
    if classes < 2 or batch_size < 1:
        raise ValueError('registered class count and batch size must be positive')
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=False)
    seed_base = int(getattr(args, 'sat_seed', 2027))
    scene_seeds = {}
    with _fixed_model(model):
        for si, scenario in enumerate(SCENARIOS):
            seed = seed_base + max(0, si-1) * 1009
            scene_seeds[scenario] = seed if scenario != 'clean' else None
            generator = make_torch_generator(device, seed)
            path = directory / f'predictions.{scenario}.jsonl'
            with path.open('x', encoding='utf-8', newline='\n') as handle:
                for start in range(0, len(records), batch_size):
                    batch = records[start:start+batch_size]
                    iq = torch.stack([record.read_iq() for record in batch]).to(device)
                    if not torch.isfinite(iq).all():
                        raise ValueError('non-finite received IQ cannot be silently scored')
                    if scenario != 'clean':
                        iq, _ = channel_apply(iq, scenario, args, gen=generator, return_meta=False)
                    # No truth, domain, role, quota or target class count enters
                    # this actual deployment-semantic classification call.
                    output = model(iq, y_tx=None, domain_labels=None, return_aux=False)
                    logits = output['tx_logits'] if isinstance(output, dict) else output
                    if logits.shape != (len(batch), classes) or not torch.isfinite(logits).all():
                        raise ValueError('model must produce finite logits for every registered class')
                    for record, values in zip(batch, logits.detach().float().cpu().tolist()):
                        handle.write(json.dumps(dict(record_id=record.record_id, scenario=scenario,
                                                     logits=values), allow_nan=False) + '\n')
    manifest = dict(schema='cross_response_fixed_predictions_v1', state='PREDICTIONS_FIXED',
                    scenarios=list(SCENARIOS), registered_class_count=classes,
                    record_ids=record_ids, named_ids={k:list(v) for k,v in named_ids.items()},
                    record_metadata={r.record_id:dict(receiver=r.receiver, day=r.day) for r in records},
                    scene_seeds=scene_seeds, model_inputs='received_iq_only',
                    truth_accessed=False, record_count=len(records), rows_written=len(records)*len(SCENARIOS))
    with (directory/'prediction_manifest.json').open('x', encoding='utf-8', newline='\n') as handle:
        json.dump(manifest, handle, indent=2, allow_nan=False)
    return directory/'prediction_manifest.json'


def _load_complete_predictions(directory: Path):
    manifest = json.loads((directory/'prediction_manifest.json').read_text(encoding='utf-8'))
    if (manifest.get('state') != 'PREDICTIONS_FIXED' or manifest.get('scenarios') != list(SCENARIOS)
            or manifest.get('truth_accessed') is not False):
        raise ValueError('all four scenarios must be fixed before truth scoring')
    ids = manifest['record_ids']
    if not ids or len(ids) != len(set(ids)) or manifest['record_count'] != len(ids):
        raise ValueError('invalid expected record coverage')
    id_set = set(ids)
    classes = manifest['registered_class_count']
    if type(classes) is not int or classes < 2 or manifest.get('rows_written') != len(ids)*len(SCENARIOS):
        raise ValueError('invalid complete all-class scene manifest')
    subsets = manifest['named_ids']
    if any(len(v) != len(set(v)) or not set(v) <= id_set for v in subsets.values()):
        raise ValueError('invalid named subset coverage')
    if set().union(*(set(v) for v in subsets.values())) != id_set:
        raise ValueError('named subsets do not cover all received records')
    if set(manifest['record_metadata']) != id_set:
        raise ValueError('missing physical bookkeeping metadata')
    predictions = {}
    for scenario in SCENARIOS:
        rows = {}
        with (directory/f'predictions.{scenario}.jsonl').open(encoding='utf-8') as handle:
            for line in handle:
                row = json.loads(line)
                if set(row) != {'record_id','scenario','logits'}:
                    raise ValueError('prediction artifact must not carry truth or unknown fields')
                rid, logits = row['record_id'], row['logits']
                if row.get('scenario') != scenario or rid in rows or rid not in id_set:
                    raise ValueError('duplicate, unknown or wrong-scenario prediction')
                if len(logits) != classes or not all(isinstance(v,(int,float)) and math.isfinite(v) for v in logits):
                    raise ValueError('incomplete/non-finite all-class logits')
                rows[rid] = int(np.argmax(logits))
        if set(rows) != id_set:
            raise ValueError(f'missing predictions for scenario {scenario}')
        predictions[scenario] = rows
    return manifest, predictions


def _accuracy(predictions, truth, ids):
    correct = sum(predictions[rid] == truth[rid] for rid in ids)
    return dict(tx_acc=100.*correct/len(ids) if ids else float('nan'),
                tx_correct=correct, tx_total=len(ids), dom_acc=float('nan'), probe_dom_acc=float('nan'))


def _json_metrics(value):
    if isinstance(value,dict):
        return {key:_json_metrics(item) for key,item in value.items()}
    if isinstance(value,list):
        return [_json_metrics(item) for item in value]
    if isinstance(value,float) and not math.isfinite(value):
        return None
    return value


def score_fixed_predictions(directory: str | Path, truth_provider: Callable[[Sequence[str]], Mapping[str,int]]):
    """Independent scorer: fully validate prediction coverage before any truth read."""
    directory = Path(directory)
    manifest, predictions = _load_complete_predictions(directory)
    ids = manifest['record_ids']
    truth = dict(truth_provider(tuple(ids)))  # FIRST truth access in this path.
    if set(truth) != set(ids) or any(type(y) is not int or not 0 <= y < manifest['registered_class_count'] for y in truth.values()):
        raise ValueError('truth must exactly match prediction IDs and registered classes')
    with (directory/'scoring_truth.jsonl').open('x', encoding='utf-8', newline='\n') as handle:
        for rid in ids:
            handle.write(json.dumps(dict(record_id=rid, label=truth[rid])) + '\n')
    named_scenes = {scenario:{name:_accuracy(rows,truth,nids) for name,nids in manifest['named_ids'].items()}
                    for scenario,rows in predictions.items()}
    named = named_scenes['clean']
    main_keys = select_main_test_keys(named,'wisig')
    # Main named groups can alias when day lists overlap. Overall accuracy must
    # count each physical record once, while retaining the named subgroup views.
    main_ids = list(dict.fromkeys(rid for key in main_keys for rid in manifest['named_ids'][key]))
    sats = {}
    for scenario in SCENARIOS[1:]:
        scene = named_scenes[scenario]
        seen = {k:v for k,v in scene.items() if k.startswith('test_rx_')}
        strict_rx = {k:v for k,v in scene.items() if k.startswith('test_unseen_day_rx_')}
        receiver = {**seen,**strict_rx}
        def floor(rows):
            values = [v['tx_acc'] for v in rows.values() if v['tx_total'] > 0]
            return min(values) if values else float('nan')
        sats[scenario] = dict(aggregate=_accuracy(predictions[scenario],truth,main_ids), named=scene,
            strict_udu=scene.get('test_unseen_day_unseen_rx',{}).get('tx_acc',float('nan')),
            selected_names=list(scene), receiver_named=receiver, receiver_seen_day_named=seen,
            receiver_strict_named=strict_rx, receiver_floor=floor(receiver),
            receiver_seen_day_floor=floor(seen), receiver_strict_floor=floor(strict_rx),
            evaluation_seed=dict(base=manifest['scene_seeds'][scenario], policy='one_view_per_physical_record'))
    by_receiver = {}
    by_class = {}
    for scenario, rows in predictions.items():
        rx_values = sorted({v['receiver'] for v in manifest['record_metadata'].values() if v['receiver'] is not None})
        by_receiver[scenario] = {str(rx):_accuracy(rows,truth,[rid for rid in ids if manifest['record_metadata'][rid]['receiver']==rx])
                                 for rx in rx_values}
        by_class[scenario] = {str(cls):_accuracy(rows,truth,[rid for rid in ids if truth[rid]==cls])
                              for cls in range(manifest['registered_class_count'])}
    report = dict(status='COMPLETE', test=_accuracy(predictions['clean'],truth,main_ids), named_test=named,
                  sat_test_named=sats, per_receiver=by_receiver, per_class=by_class,
                  truth_last=True, scenarios=list(SCENARIOS), unique_records=len(ids),
                  prediction_rows=manifest['rows_written'], prediction_artifacts=str(directory),
                  selection_source='final_only', claim='FROZEN_PREDICTIONS_INDEPENDENT_TRUTH_LAST_SCORING')
    with (directory/'independent_scores.json').open('x', encoding='utf-8', newline='\n') as handle:
        # Missing domains/classes remain N/A, represented as JSON null on disk.
        # In-memory NaNs preserve the existing evaluation caller's convention.
        json.dump(_json_metrics(report),handle,indent=2,allow_nan=False)
    return report


def evaluate_final_truth_last(model, args, data_ctx, device, output_dir: str | Path):
    if getattr(args,'checkpoint_selection',None) != 'final_only':
        raise ValueError('cross-response heldout scoring requires final-only checkpoint selection')
    records, named_ids, truth = wisig_received_catalog(data_ctx['named_test_loaders'])
    directory = Path(output_dir)/'final_predictions'
    write_fixed_predictions(model,records,named_ids,args,device,directory)
    return score_fixed_predictions(directory,truth)
