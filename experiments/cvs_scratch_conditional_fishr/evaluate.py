"""Clean-only frozen checkpoint prediction and independent truth-last scoring."""
import argparse
import json
import os
from pathlib import Path
import sys
import time

from . import design as d

COUNT = 168000
VIEWS = ('clean',)
HELDOUT_RX = ('1-1', '14-7', '2-1', '20-1', '7-14', '7-7', '8-8')
TARGET_DAYS = ('2021_03_01', '2021_03_08', '2021_03_15', '2021_03_23')


def snapshot(row=None):
    rows = [r for r in d.rows() if row is None or r['row_id'] == row]
    if not rows:
        raise ValueError('Unregistered evaluation row')
    configs = [d.config(r) for r in rows]
    for c in configs:
        d.validate(c)
        frozen = d.read(d.BASE / c['row_id'] / 'source_frozen.json')
        if (frozen['status'] != 'SOURCE_FROZEN' or frozen['config'] != c
                or frozen['target_access'] or frozen['epoch'] != 20 or frozen['optimizer_steps'] != 4440
                or frozen['total_epochs']!=20 or frozen['parent_checkpoint'] is not None
                or frozen.get('selection')!=c['checkpoint_selection']
                or frozen['checkpoint'] != str(Path(c['output_root']) / 'final_ssdg.pth')):
            raise ValueError('Source freeze differs from preregistered row')
    return configs


def source_provenance(c):
    d.validate(c)
    source = Path(c['output_root'])
    done = d.read(source / 'completion.json')
    if (done['status'] != 'SOURCE_TRAINED' or done['config'] != c
            or done['epoch'] != 20 or done['optimizer_steps'] != 4440
            or done['logged_steps'] != 4440 or done['target_access']
            or done['checkpoint_selection'] != c['checkpoint_selection']):
        raise ValueError('Scratch checkpoint source provenance differs')
    init=d.read(source/'initialization.json')
    from .provenance import expected_sources,validate as validate_teachers
    validate_teachers(c,init.get('teacher_provenance',{}))
    _scratch_initialization(c,init['parent_provenance'])
    if (init['scratch_only'] is not True or init['scratch_scope']!='student_identity_only'
            or init['checkpoint_sources']!=expected_sources(c) or init['ancestors']!=expected_sources(c)
            or init['target_contact'] or init['source_roles']!='EXACT_MATCH'
            or init['reference_E20_loaded'] or init['parent_checkpoint'] is not None
            or init['parent_training_epochs']!=0 or init['incremental_epochs']!=20 or init['total_epochs']!=20
            or init!=done['initialization'] or done['parent_checkpoint'] is not None or done['total_epochs']!=20
            or done['target_evaluated']):raise ValueError('Scratch student/teacher ancestry differs')
    from experiments.cvs_phase1_stack.recover import verify_contract
    verify_contract(d.read(source/'source_contract.json'),d.read(d.SOURCE))
    return source


def _scratch_initialization(c,proof):
    expected=dict(status='SCRATCH_INITIALIZATION',seed=c['model_seed'],checkpoint=None,parent_checkpoint=None,
                  initialization_checkpoint_sources=[],ancestors=[],scratch_only=True,scratch_scope='student_identity_only',
                  source_roles='EXACT_MATCH',training_target_access=False,reference_E20_loaded=False,
                  parent_training_epochs=0,auxiliary_inheritance=False)
    if c['model_initialization']!='scratch' or c['parent_checkpoint'] is not None or c['initialization_checkpoint_sources']:
        raise ValueError('Student initialization config is not strictly scratch')
    if not isinstance(proof,dict) or any(proof.get(k)!=v for k,v in expected.items()):
        raise ValueError('Student initialization inherited weights or lacks scratch provenance')


def checkpoint_provenance(c, checkpoint):
    expected = dict(epoch=20, total_epochs=20, parent_training_epochs=0,incremental_epochs=20,
                    model_seed=c['model_seed'],optimizer_steps=4440, run_id=c['run_id'], candidate_id=c['row_id'],target_access=False,
                    scratch_only=True,scratch_scope='student_identity_only',reference_E20_loaded=False,
                    config=c, parent_checkpoint=None,
                    checkpoint_selection=c['checkpoint_selection'])
    if any(checkpoint.get(k) != v for k, v in expected.items()):
        raise ValueError('Scratch checkpoint identity, no-parent contract, or budget differs')
    _scratch_initialization(c,checkpoint.get('parent_provenance'))
    if c['enabled']['averaging']:
        avg=checkpoint.get('averaging') or {}
        if avg.get('epochs')!=c['averaging']['snapshot_epochs'] or avg.get('count')!=5:raise ValueError('Incomplete fixed average')
    if 'model' not in checkpoint or 'baseline_args' not in checkpoint:
        raise ValueError('Incomplete identity checkpoint')
    from .provenance import expected_sources,validate as validate_teachers
    validate_teachers(c,checkpoint.get('teacher_provenance',{}))
    if checkpoint.get('checkpoint_sources')!=expected_sources(c):raise ValueError('Checkpoint teacher sources differ')
    calibration=d.read(Path(c['output_root'])/'calibration.json')
    state=checkpoint.get('fixed_sensitivity_state')
    expected_state=dict(schema='fixed_conditional_sensitivity_v1',objective=c['multilevel']['sensitivity_objective'],normalization='fixed_layer_source_RMS',
        components=c['multilevel']['conditional_components'],fixed_component_coefficients=calibration['fixed_component_coefficients'],after_identity_updates=444,fixed_scales=calibration['fixed_scales'],fixed_coefficients=calibration['fixed_coefficients'])
    if state!=expected_state:raise ValueError('Checkpoint sensitivity scales differ from own source calibration')


def physical_ids():
    import numpy as np
    # Reuse the capsule's already validated observations without generating views.
    manifest = d.read(Path(d.CAPSULE) / 'manifest.json')
    expected = dict(status='VALIDATED_ONCE', count=COUNT, classes=d.CLASSES,
                    source_target_disjoint=True, unique_ids=True,
                    channel='residual/post_sync/noeq')
    if any(manifest.get(k) != v for k, v in expected.items()):
        raise ValueError('Validated clean observation contract differs')
    with np.load(Path(d.CAPSULE) / 'index.npz', allow_pickle=False) as ix:
        ids = ix['ids'].copy()
    if ids.shape != (COUNT,) or len(set(ids.tolist())) != COUNT:
        raise ValueError('Clean physical ID coverage differs')
    return ids


def iq_tensor(values, device):
    """Copy clean float32 IQ through the buffer API, independent of NumPy ABI."""
    import torch
    if values.dtype.name != 'float32' or values.ndim != 3 or values.shape[1:] != (2, 256):
        raise ValueError('Clean IQ tensor contract differs')
    return torch.frombuffer(bytearray(values.tobytes(order='C')), dtype=torch.float32).reshape(values.shape).to(device)


def predict(row):
    import numpy as np
    import torch
    from experiments.cvs_equivariant_identity.precision import numerical_context, actual_flags
    from .model import build
    c = snapshot(row)[0]
    source = source_provenance(c)
    checkpoint_path = source / 'final_ssdg.pth'
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    checkpoint_provenance(c, checkpoint)
    del checkpoint
    output = d.BASE / row / 'prediction'
    if output.exists():
        raise FileExistsError('Existing prediction must be reconciled, never overwritten')
    torch.set_num_threads(2)
    device = torch.device('cuda:0')
    started = time.perf_counter()
    with numerical_context(d.FULL_FP32_POLICY):
        # Only this row's fixed final20 student is loaded. Original E20 weights
        # are never used to initialize or supplement this prediction model.
        model = build(c, device, checkpoint=checkpoint_path)
        model.eval()
        with torch.no_grad():
            logits = model(torch.zeros(2, 2, 256, device=device))
        if logits.shape != (2, 6) or not torch.isfinite(logits).all():
            raise ValueError('Identity checkpoint no-query forward failed')
        ids = physical_ids()
        array = np.load(Path(d.CAPSULE) / 'clean.npy', mmap_mode='r', allow_pickle=False)
        if array.shape != (COUNT, 2, 256):
            raise ValueError('Complete clean observation shape differs')
        output.mkdir(parents=True, exist_ok=False)
        d.write(output / 'provenance.json', dict(status='VERIFIED', config=c,
            checkpoint=str(checkpoint_path), parent_checkpoint=c['parent_checkpoint'],
            own_scratch20=True,student_initialization='random',parent_training_epochs=0,
            incremental_epochs=20,total_epochs=20,optimizer_steps=4440,source_roles='EXACT_MATCH',
            teacher_models_used_for_prediction=False,P0_used_for_student_initialization=False,
            truth_read=False, query_fit=False))
        d.write(output / 'resolved_config.json', dict(row_id=row, run_id=d.RUN,
            pid=os.getpid(), cwd=os.getcwd(), python=sys.executable,
            evaluation_commit=(d.ROOT / 'release_commit.txt').read_text().strip(),
            views=['clean'], batch_size=256, backend_flags=actual_flags(),
            parameters=sum(p.numel() for p in model.parameters()), trainable_parameters=0,
            hardware=torch.cuda.get_device_name(device), torch_version=torch.__version__))
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        infer_started = time.perf_counter()
        predictions = []; log_probabilities=[]
        with torch.no_grad():
            for begin in range(0, COUNT, 256):
                values = np.array(array[begin:begin + 256], dtype=np.float32, copy=True)
                logits = model(iq_tensor(values, device))
                if logits.shape != (len(values), 6) or not torch.isfinite(logits).all():
                    raise ValueError('Nonfinite or malformed clean classifier output')
                predictions.extend(logits.argmax(1).cpu().tolist())
                log_probabilities.extend(logits.log_softmax(1).cpu().tolist())
        torch.cuda.synchronize()
        seconds = time.perf_counter() - infer_started
        np.savez(output / 'predictions.npz', ids=ids, clean=np.asarray(predictions, dtype=np.int64), clean_log_probabilities=np.asarray(log_probabilities,dtype=np.float32))
        import resource
        d.write(output / 'complete.json', dict(status='PREDICTIONS_COMPLETE', count=COUNT,
            views=['clean'], truth_read=False, query_fit=False, inference_seconds=dict(clean=seconds),
            wall_seconds=time.perf_counter() - started,
            peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),
            peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved(),
            peak_process_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024))
        print(json.dumps(dict(row_id=row, status='PREDICTIONS_COMPLETE', count=COUNT,
                             views=['clean'], truth_read=False)), flush=True)


def prediction_preflight(configs):
    import numpy as np
    ids = physical_ids()
    for c in configs:
        output = d.BASE / c['row_id'] / 'prediction'
        done = d.read(output / 'complete.json')
        if (done['status'] != 'PREDICTIONS_COMPLETE' or done['count'] != COUNT
                or done['views'] != ['clean'] or done['truth_read'] or done['query_fit']):
            raise ValueError('Truth remains closed: incomplete clean predictions')
        provenance = d.read(output / 'provenance.json')
        if (provenance['config'] != c or provenance['parent_checkpoint'] != c['parent_checkpoint']
                or provenance['status'] != 'VERIFIED' or not provenance['own_scratch20'] or provenance['total_epochs']!=20
                or provenance['student_initialization']!='random' or provenance['parent_training_epochs']!=0
                or provenance['teacher_models_used_for_prediction'] or provenance['P0_used_for_student_initialization']
                or provenance['optimizer_steps'] != 4440 or provenance['truth_read'] or provenance['query_fit']
                or provenance['checkpoint'] != str(Path(c['output_root'])/'final_ssdg.pth')):
            raise ValueError('Prediction provenance differs')
        with np.load(output / 'predictions.npz', allow_pickle=False) as p:
            if set(p.files) != {'ids', 'clean','clean_log_probabilities'} or not np.array_equal(ids, p['ids']):
                raise ValueError('Prediction physical identity differs')
            validate_prediction(p['clean'])
            validate_log_probabilities(p['clean_log_probabilities'],p['clean'])
    return ids


def validate_prediction(a):
    if a.shape != (COUNT,) or a.dtype.kind not in 'iu' or a.min() < 0 or a.max() >= 6:
        raise ValueError('Invalid full clean prediction array')


def validate_log_probabilities(values,prediction):
    import numpy as np
    if values.shape!=(COUNT,6) or not np.isfinite(values).all() or not np.allclose(np.exp(values).sum(1),1.,atol=1e-5) or not np.array_equal(values.argmax(1),prediction):raise ValueError('Full fixed probability prediction differs')


def parent_predictions(configs, ids):
    """P0 is a score reference only; validate metadata without loading its weights."""
    import numpy as np
    values = {}
    for c in configs:
        path = c.get('reference_prediction')
        if not path or c['model_seed'] in values:
            continue
        from experiments.cvs_e20_foundation.verify import config as reference_config,validate_metadata
        native=reference_config(c['model_seed']);native_source=Path(native['output_root'])
        if (native_source/'final_ssdg.pth').as_posix()!=c['reference_checkpoint']:
            raise ValueError('Fixed E20 score-reference checkpoint metadata differs')
        validate_metadata(native,d.read(native_source/'initialization.json'),d.read(native_source/'completion.json'),
                          d.read(native_source/'resolved_config.json'),d.read(native_source/'resolved_native_args.json'))
        from experiments.cvs_phase1_stack.recover import verify_contract
        verify_contract(d.read(native_source/'source_contract.json'),d.read(d.SOURCE))
        path = Path(path)
        done = d.read(path.parent/'complete.json')
        provenance = d.read(path.parent/'provenance.json')
        if (provenance['status'] != 'VERIFIED' or provenance['checkpoint'] != c['reference_checkpoint']
                or provenance['config']['model_seed'] != c['model_seed'] or provenance['config']['arm'] != 'native'
                or provenance['truth_read'] or provenance['query_fit']
                or done['status'] != 'PREDICTIONS_COMPLETE' or done['count'] != COUNT
                or 'clean' not in done['views'] or done['truth_read'] or done['query_fit']):
            raise ValueError('Registered fixed P0 score-reference prediction provenance differs')
        with np.load(path, allow_pickle=False) as prediction:
            if not np.array_equal(prediction['ids'], ids):
                raise ValueError('P0 score-reference clean physical identities differ')
            validate_prediction(prediction['clean'])
            values[c['model_seed']] = prediction['clean'].copy()
    return values


def control_predictions(configs,ids):return {}


def truth_arrays(truth, ids):
    import numpy as np
    if set(truth) != set(ids.tolist()):
        raise ValueError('Truth physical identities differ')
    y = np.asarray([truth[s]['label'] for s in ids])
    rx = np.asarray([str(truth[s]['receiver']) for s in ids])
    days = np.asarray([str(truth[s]['day']) for s in ids])
    if (y.dtype.kind not in 'iu' or set(y) != set(range(6))
            or set(rx) != set(HELDOUT_RX) or set(days) != set(TARGET_DAYS)):
        raise ValueError('Truth class/receiver/day contract differs')
    for label in range(6):
        for receiver in HELDOUT_RX:
            for day in TARGET_DAYS:
                if int(((y == label) & (rx == receiver) & (days == day)).sum()) != 1000:
                    raise ValueError('Truth registered per-condition count differs')
    return y, rx, days


def metric_rows(config, prediction, ids, y, rx, days):
    import numpy as np
    from comparison_suite.score import metrics
    records, day_records = [], []
    identity = {k: config[k] for k in ('row_id', 'arm', 'model_seed')}
    identity['stage'] = config.get('stage', 'phase1_scratch_boundary_style')
    for dimension, strata in [('overall', ['ALL']), ('receiver', sorted(set(rx))),
                              ('transmitter', d.CLASSES), ('day', list(TARGET_DAYS))]:
        for stratum in strata:
            mask = (np.ones(len(ids), bool) if dimension == 'overall' else rx == stratum
                    if dimension == 'receiver' else days == stratum if dimension == 'day'
                    else y == d.CLASSES.index(stratum))
            result = metrics(y[mask], prediction[mask], 6)
            cm = np.bincount(6 * y[mask] + prediction[mask], minlength=36).reshape(6, 6)
            denominator = cm.sum(0) + cm.sum(1)
            f1 = np.divide(2 * np.diag(cm), denominator, out=np.zeros(6), where=denominator > 0).mean()
            if (result['confusion'] != cm.tolist() or result['query_count'] != int(cm.sum())
                    or abs(result['accuracy'] - float(np.trace(cm) / cm.sum())) > 1e-12
                    or abs(result['macro_f1'] - f1) > 1e-12):
                raise ValueError('Independent metric recount differs')
            (day_records if dimension == 'day' else records).append(
                dict(**identity, view='clean', dimension=dimension, stratum=stratum, **result))
    return records, day_records


def summarize(results):
    import numpy as np
    output = []
    for arm in ('P0', *d.ARMS):
        group = [r for r in results if r['arm'] == arm and r['dimension'] == 'overall']
        if not group:
            continue
        values = [r['accuracy'] for r in group]
        worst = [min(r['accuracy'] for r in results if r['row_id'] == g['row_id']
                     and r['dimension'] == 'receiver') for g in group]
        output.append(dict(arm=arm, view='clean', seed_count=len(group),
            seeds=[r['model_seed'] for r in group],
            seed_accuracy={str(r['model_seed']): r['accuracy'] for r in group},
            accuracy_mean=float(np.mean(values)), accuracy_sd=float(np.std(values, ddof=1)) if len(values) > 1 else None,
            accuracy_min=min(values), macro_f1_mean=float(np.mean([r['macro_f1'] for r in group])),
            worst_rx_mean=float(np.mean(worst)), complete_three_seeds={r['model_seed'] for r in group} == set(d.SEEDS)))
    return output


def paired_results(results):
    import numpy as np
    overall = {(r['arm'], r['model_seed']): r for r in results if r['dimension'] == 'overall'}
    result = []
    for treatment, control in d.PAIRS:
        seeds = [s for s in d.SEEDS if (treatment, s) in overall and (control, s) in overall]
        if not seeds:
            continue
        differences = [overall[treatment, s]['accuracy'] - overall[control, s]['accuracy'] for s in seeds]
        result.append(dict(treatment=treatment, control=control, metric='accuracy', view='clean',
            seeds=seeds, seed_count=len(seeds), differences=differences,
            mean_difference=float(np.mean(differences)),
            sd_difference=float(np.std(differences, ddof=1)) if len(seeds) > 1 else None,
            positive_seeds=sum(v > 0 for v in differences), complete_three_seeds=len(seeds) == 3))
    return result


def source_resources(c):
    source=Path(c['output_root']);done=d.read(source/'completion.json');resolved=d.read(source/'resolved_config.json')
    return dict(parent_checkpoint=None,parent_epochs=0,incremental_epochs=20,total_epochs=20,
                incremental_identity_updates=4440,hardware=resolved['hardware'],measured_training_resources=done['resources'],
                shared_IQ_preparation_reused=True,training_flops=None,
                scope='random student20; pretrained source-U teacher history and initial calibration separately measured; shared GPU wall cost',
                P0_role='fixed score reference only; never loaded into the student')


def score(row=None):
    import numpy as np
    from comparison_suite.score import csvwrite
    configs = snapshot(row)
    # This process alone opens truth, only after all requested-row predictions are fixed.
    ids = prediction_preflight(configs)
    root = d.BASE / 'row_scoring' / row if row else d.BASE
    if (root / 'scoring_complete.json').is_file():
        from .dispatch import require_scored
        require_scored(d.read(root / 'scoring_complete.json'), len(configs))
        for filename in ('scores.json', 'scores.csv', 'summary.json', 'paired_results.json',
                         'day_scores.json', 'analysis.md', 'goal_assessment.json', 'resources.json'):
            if not (root / filename).is_file():
                raise ValueError('Existing scoring completion lacks artifact: ' + filename)
        return  # Completed score is reused; no repeated truth read or overwrite.
    if (root / 'scores.json').exists():
        raise FileExistsError('Partial scoring must be reconciled, never overwritten')
    parent_values = parent_predictions(configs, ids)
    control_values = control_predictions(configs, ids)
    truth = d.read(d.TRUTH)
    y, rx, days = truth_arrays(truth, ids)
    results, day_results, resources, parent_rows = [], [], [], []
    seen_parent = set()
    for c in configs:
        out = d.BASE / c['row_id'] / 'prediction'
        with np.load(out / 'predictions.npz', allow_pickle=False) as p:
            records, day_records = metric_rows(c, p['clean'], ids, y, rx, days)
            parent=parent_values.get(c['model_seed'])
            for record in records+day_records:
                dim,key=record['dimension'],record['stratum']
                mask=np.ones(len(y),bool) if dim=='overall' else rx==key if dim=='receiver' else days==key if dim=='day' else y==d.CLASSES.index(key)
                record['NLL']=float(-p['clean_log_probabilities'][np.arange(len(y))[mask],y[mask]].mean())
                if parent is not None:
                    parent_correct=parent[mask]==y[mask];now_correct=p['clean'][mask]==y[mask]
                    record['parent_fix_count']=int((~parent_correct&now_correct).sum())
                    record['parent_break_count']=int((parent_correct&~now_correct).sum())
                    record['net_correct_gain']=record['parent_fix_count']-record['parent_break_count']
        results.extend(records)
        day_results.extend(day_records)
        resources.append(dict(row_id=c['row_id'], prediction=d.read(out / 'complete.json'), source=source_resources(c)))
        if c['model_seed'] in parent_values and c['model_seed'] not in seen_parent:
            parent = dict(c, arm='P0', row_id='P0-s' + str(c['model_seed']))
            parent_rows.extend(metric_rows(parent, parent_values[c['model_seed']], ids, y, rx, days)[0])
            seen_parent.add(c['model_seed'])
    for (arm,seed),prediction in control_values.items():
        c=next(c for c in configs if c['model_seed']==seed);control=dict(c,arm=arm,row_id=arm+'-s'+str(seed))
        parent_rows.extend(metric_rows(control,prediction,ids,y,rx,days)[0])
    all_results = results + parent_rows
    summary, pairs = summarize(all_results), paired_results(all_results)
    d.write(root / 'scores.json', dict(status='SCORED', results=results, target_feedback_forbidden=True))
    csvwrite(root / 'scores.csv', [{k: v for k, v in r.items() if k != 'confusion'} for r in results])
    d.write(root / 'day_scores.json', dict(results=day_results))
    csvwrite(root / 'day_scores.csv', [{k: v for k, v in r.items() if k != 'confusion'} for r in day_results])
    d.write(root / 'parent_scores.json', dict(results=parent_rows, source='Reused fixed clean predictions only'))
    d.write(root / 'summary.json', dict(results=summary, aggregation='ACTUAL_COMPLETED_SEEDS_ONLY'))
    csvwrite(root / 'summary.csv', summary)
    d.write(root / 'paired_results.json', dict(comparisons=pairs, inferential_significance_claim=False))
    d.write(root / 'resources.json', dict(rows=resources, shared_preparation_counted_once_per_seed=True))
    main = next((r for r in summary if r['arm'] == d.PRIMARY), None)
    increment = next((r for r in pairs if r['treatment'] == d.PRIMARY and r['control'] == 'CE_INTERACTION'), None)
    d.write(root / 'goal_assessment.json', dict(primary_arm=d.PRIMARY, threshold=.84,
        performance_status='PARTIAL' if main is None or not main['complete_three_seeds'] else 'MET' if main['accuracy_mean'] >= .84 else 'NOT_MET',
        mechanism_increment_status='PARTIAL' if increment is None or not increment['complete_three_seeds'] else 'POSITIVE' if increment['mean_difference'] > 0 else 'NOT_POSITIVE',
        complete_arms_reaching_goal=[r['arm'] for r in summary if r['arm'] in d.ARMS and r['complete_three_seeds'] and r['accuracy_mean']>=.84],
        each_seed_must_be_reported=True, scores_must_not_control_training=True))
    lines = ['# Scratch20 conditional sensitivity and controlled L style: clean test', '',
        'Each row uses 168000 fixed physical queries across 7 receivers and 4 days. Predictions precede independent truth access.',
        'P0 predictions are reused when registered. Results do not feed model, hyperparameter, or rerun selection.', '',
        '| Arm | Seeds | Mean accuracy (%) | SD (pp) | Minimum (%) | Macro-F1 (%) |',
        '|---|---:|---:|---:|---:|---:|']
    for r in summary:
        sd = 'N/A' if r['accuracy_sd'] is None else f"{100*r['accuracy_sd']:.3f}"
        lines.append(f"| {r['arm']} | {r['seed_count']} | {100*r['accuracy_mean']:.3f} | {sd} | {100*r['accuracy_min']:.3f} | {100*r['macro_f1_mean']:.3f} |")
    lines.extend(['', '## Every seed', '', '| Arm | Model seed | Accuracy (%) |', '|---|---:|---:|'])
    for r in all_results:
        if r['dimension'] == 'overall':
            lines.append(f"| {r['arm']} | {r['model_seed']} | {100*r['accuracy']:.3f} |")
    lines.extend(['', '## Paired differences', '', '| Comparison | Seeds | Mean (pp) | Positive seeds |', '|---|---:|---:|---:|'])
    for r in pairs:
        lines.append(f"| {r['treatment']} − {r['control']} | {r['seed_count']} | {100*r['mean_difference']:.3f} | {r['positive_seeds']} |")
    lines.extend(['', 'Primary CE_MOMENT_INTERACTION success requires all three seeds and clean mean >=84%.',
        'All 3 preregistered arms are reported. A complete arm meeting 84% is recognized separately; target scores never select or change training.',
        'Every student starts from random identity weights, trains20 epochs and freezes final20. P0 reuses exposed original E20 predictions only for scoring; those weights are never loaded into the student.',
        'Non-CE methods retain two pretrained original E20 source-U teachers: historical model cost is student20 plus teachers40 =60 epochs, reference CE20 is reused from the prior fixed scratch matrix. Random student initialization does not mean the full method is free of pretraining. Scratch20 and E20+20 have different cumulative student budgets, so any later cross-batch comparison cannot isolate initialization.',
        'No EMA, satellite generator, auxiliary learned heads or head warmup. After444 updates,24 source-L pair forwards set RMS and24 more calibrate each component separately. Components are equally averaged then share one detached joint gradient cap0.2. Shallow/middle conditional CE or margin sensitivities preserve mixed second derivatives. Clean-inclusive hard style evaluates four style candidates plus clean, then replays the selected style or clean with the clean dropout mask; clean replay has no style hooks. Two L-only epochs then four-epoch ramp. U TX truth stays hidden.',
        'Three seed differences are descriptive evidence, not an automatic statistical significance claim.',
        'Pretrained teacher history, shared preparation, scratch student training, calibration, style search and test costs are referenced separately in resources.json.'])
    root.mkdir(parents=True, exist_ok=True)
    (root / 'analysis.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    d.write(root / 'scoring_complete.json', dict(status='SCORED_COMPLETE', models=len(configs), views=1,
        query_count_per_view=COUNT, prediction_count=len(configs) * COUNT,
        metric_records=len(results), day_metric_records=len(day_results), truth_last=True,
        independent_recount='VERIFIED'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['predict', 'score'], required=True)
    parser.add_argument('--row')
    args = parser.parse_args()
    if args.mode == 'predict':
        if not args.row:
            parser.error('predict requires --row')
        predict(args.row)
    else:
        score(args.row)
