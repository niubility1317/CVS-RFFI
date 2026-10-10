"""Finish source diagnostics from immutable E200 artifacts; never train or open query."""
import argparse
from copy import deepcopy
import gzip
import json
import math
import os
from pathlib import Path
import shutil
import sys
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / 'code')]
OLD_COMMIT = 'd4723b1aa227fce6368fd25680bb74b1190557d7'
OLD_RUN = '20261010-phase1-feature-disentangle-manysig-m48-r01'
MODE = 'post_training_diagnostic_recovery'
COPY_FILES = ('final_ssdg.pth', 'auxiliary_final.pth', 'source_contract.json',
              'initialization.json', 'resolved_config.json', 'resolved_native_args.json',
              'epoch_metrics.jsonl', 'epoch_metrics.csv', 'step_metrics.jsonl.gz')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def validate_pair(c_new, c_old):
    from . import design as d
    d.validate(c_new)
    expected = deepcopy(c_old)
    expected.update(run_id=c_new['run_id'], output_root=c_new['output_root'],
                    checkpoint_origin=dict(config=c_old, commit=OLD_COMMIT, mode=MODE))
    if expected != c_new or c_new['run_id'] == c_old['run_id'] or c_old['run_id'] != OLD_RUN:
        raise ValueError('Recovery may only change run/output and declare exact checkpoint origin')
    if (c_old['target_access'] or c_old['checkpoint_sources'] or
            c_old['model_initialization'] != 'scratch' or c_old['risk_recipe']['resume']):
        raise ValueError('CHECKPOINT_TARGET_CONTAMINATED or non-scratch origin')
    old, new = Path(c_old['output_root']), Path(c_new['output_root'])
    if old.resolve() == new.resolve() or old.resolve() in new.resolve().parents:
        raise ValueError('Recovery output must be separate from immutable source')
    if new.exists():
        raise FileExistsError(new)


def inspect_logs(source):
    """Stream all update evidence; no sampled/inferred budget accounting."""
    steps = 0
    with gzip.open(source / 'step_metrics.jsonl.gz', 'rt', encoding='utf-8') as handle:
        for line in handle:
            value = json.loads(line); steps += 1
            if value['step'] != steps or value['epoch'] != (steps - 1) // 222 + 1:
                raise ValueError('Noncontiguous training update evidence')
            m = value['metrics']
            if (not math.isfinite(m['train/loss']) or m.get('train/skipped_nonfinite_loss', 0)
                    or m.get('train/skipped_nonfinite_grad', 0)):
                raise ValueError('Failed optimizer step in training evidence')
    epochs = [json.loads(line) for line in (source / 'epoch_metrics.jsonl').read_text(encoding='utf-8').splitlines()]
    if steps != 44400 or [row['epoch'] for row in epochs] != list(range(1, 201)):
        raise ValueError('Incomplete E200/44400 budget')
    durations = [row['epoch_time_s'] for row in epochs]
    if any(not isinstance(v, (int, float)) or not math.isfinite(v) or v < 0 for v in durations):
        raise ValueError('Missing measured epoch duration')
    return dict(logged_steps=steps, logged_epochs=len(epochs),
                training_epoch_seconds=sum(durations),
                training_time_scope='sum of 200 measured epoch_time_s; excludes startup and post-training diagnostics')


def mechanism_audit(c, execution):
    """Keep the original source.py execution requirements exactly."""
    missing = []
    for kind in c['arm_plan']['paths']:
        if execution['fit_steps'].get(kind, 0) <= 0: missing.append(kind + '_fit')
        if (kind != 'receiver' or c['feature_plan']['r_identity']) and execution['attempted_identity_steps'].get(kind, 0) <= 0:
            missing.append(kind + '_identity_attempt')
    if c['arm_plan'].get('source_u_fit') and not execution['exposure_totals'].get('source_U_action_only_packets'):
        missing.append('source_U_action_only')
    feature = execution['feature_counts']
    if c['feature_plan']['relation'] != 'none' and feature.get('relation_calls') != 5550:
        missing.append('relation_B48_expected_5550')
    if c['feature_plan']['fishr'] != 'none':
        if feature.get('fishr_statistics_updates') != 5550: missing.append('fishr_statistics_expected_5550')
        if feature.get('fishr_positive_weight_calls') != 4995: missing.append('fishr_positive_expected_4995')
        if execution['fishr_calibration']['lambda_max'] is None: missing.append('fishr_fixed_E20_calibration')
    return dict(status='FAILED_METHOD_NOT_EXECUTED' if missing else 'VERIFIED', missing=missing, **execution)


def inspect_artifacts(c_old):
    import torch
    from . import design as d
    from .source import clean, require_budget
    from experiments.cvs_phase1_stack.recover import verify_contract
    from experiments.cvs_phase1_stack.runtime import installed
    source = Path(c_old['output_root'])
    if (source / 'completion.json').exists():
        raise ValueError('Diagnostic recovery is only for unfinished source diagnostics')
    for name in COPY_FILES:
        if not (source / name).is_file(): raise FileNotFoundError(source / name)
    verify_contract(read(source / 'source_contract.json'), read(d.SOURCE))
    init = read(source / 'initialization.json')
    if (not init['scratch_only'] or init['checkpoint_sources'] or init['ancestors'] or
            init['target_contact'] or init['source_roles'] != 'EXACT_MATCH' or init['seed'] != c_old['model_seed']):
        raise ValueError('CHECKPOINT_PROVENANCE_UNVERIFIED: source initialization')
    resolved = read(source / 'resolved_config.json')
    if any(resolved.get(k) != v for k, v in c_old.items()) or resolved.get('commit') != OLD_COMMIT:
        raise ValueError('Original training configuration or commit differs')
    with installed(c_old):
        from scripts.train_daot_rc4_baseline import resolved_config
        ck = torch.load(source / 'final_ssdg.pth', map_location='cpu', weights_only=False)
        actual = clean(resolved_config(SimpleNamespace(**ck['args'])))
        if actual != read(source / 'resolved_native_args.json'):
            raise ValueError('Actual checkpoint native args differ from original record')
    if (ck['epoch'] != 200 or ck['candidate_id'] != c_old['row_id'] or ck['run_id'] != c_old['run_id']
            or ck['checkpoint_selection'] != 'final_only' or actual['baseline_ckpt'] or actual['teacher_ckpt']
            or not actual['from_scratch'] or actual['output_dir'] != c_old['output_root']
            or actual['candidate_id'] != c_old['row_id'] or actual['run_id'] != c_old['run_id']
            or actual['seed'] != c_old['model_seed'] or actual['wisig_pkl'] != c_old['dataset']
            or actual['use_unlabeled'] or actual['use_ema_teacher'] or actual['use_concat_sat_channel_aug']
            or actual['label_smoothing'] != 0 or actual['label_epochs'] != 200 or actual['pseudo_epochs'] != 0
            or not actual['a1_source_screen_only'] or actual['a1_periodic_target_inputs']
            or actual['a1_periodic_target_truth'] or actual['a1_periodic_target_start']
            or actual['a1_periodic_target_interval'] or actual['a1_final_weak_reference']):
        raise ValueError('Checkpoint actual identity, source or pure-CE initialization differs')
    auxiliary = torch.load(source / 'auxiliary_final.pth', map_location='cpu', weights_only=False)
    if auxiliary['epoch'] != 200 or auxiliary['config'] != c_old or auxiliary['target_access']:
        raise ValueError('Auxiliary final-state provenance mismatch')
    evidence = inspect_logs(source)
    require_budget(evidence['logged_steps'], ck['a1_ema_successful_updates'])
    mechanism = mechanism_audit(c_old, auxiliary['mechanism_execution'])
    if mechanism['missing']:
        raise ValueError('FAILED_METHOD_NOT_EXECUTED: ' + ','.join(mechanism['missing']))
    return ck, mechanism, evidence


def smoke(c_old):
    """Read original complete artifacts and run fixed final model on zero IQ only."""
    import torch
    from experiments.cvs_phase1_stack.runtime import installed
    ck, mechanism, evidence = inspect_artifacts(c_old)
    with installed(c_old) as native:
        model = native.build_baseline_model(SimpleNamespace(**ck['baseline_args']), torch.device('cpu'))
        model.load_state_dict(ck['model'], strict=True); model.eval()
        with torch.no_grad(): logits = model(torch.zeros(2, 2, 256))
        if logits.shape != (2, 6) or not torch.isfinite(logits).all():
            raise ValueError('Actual checkpoint no-query smoke failed')
    return dict(status='VERIFIED', row_id=c_old['row_id'], no_query=True, target_access=False,
                actual_checkpoint_forward=True, mechanism_status=mechanism['status'], **evidence)


def recover(c_new, c_old, device='cuda:0'):
    """Caller must first establish original worker exit and exact known probe failure."""
    import torch
    from . import design as d
    from .source import source_validation
    from .validation import evaluate_source_stress
    from experiments.cvs_phase1_stack.runtime import installed
    from experiments.cvs_equivariant_identity.precision import numerical_context
    validate_pair(c_new, c_old)
    ck, mechanism, evidence = inspect_artifacts(c_old)
    old, new = Path(c_old['output_root']), Path(c_new['output_root'])
    new.mkdir(parents=True, exist_ok=False)
    for name in COPY_FILES: shutil.copy2(old / name, new / name)
    for name in ('source_cost.json', 'execution_epoch.jsonl'):
        if (old / name).is_file(): shutil.copy2(old / name, new / name)
    started = time.perf_counter(); dev = torch.device(device); torch.set_num_threads(2)
    with numerical_context(d.FULL_FP32_POLICY), installed(c_old) as native:
        from cvsrffi.xuc_fusion.native import role_ids_from_native
        # Keep immutable checkpoint args intact. The separate in-memory data
        # construction args route any incidental outputs to the new directory.
        data_args = SimpleNamespace(**deepcopy(ck['args']))
        data_args.output_dir = str(new); data_args.device = str(dev)
        context = native._build_ssdg_wisig_data(data_args, dev)
        if context['named_test_loaders'] or role_ids_from_native(context) != read(d.SOURCE)['role_ids']:
            raise ValueError('Recovery source physical roles/target construction mismatch')
        model = native.build_baseline_model(SimpleNamespace(**ck['baseline_args']), dev)
        model.load_state_dict(ck['model'], strict=True); model.eval()
        metrics = source_validation(model, context['val_loader'], dev)
        stress = evaluate_source_stress(model, context['val_loader'], dev, c_new, new)
    diagnostic_seconds = time.perf_counter() - started
    provenance = dict(status='VERIFIED', mode=MODE, original_config=c_old, original_commit=OLD_COMMIT,
        original_checkpoint=str(old / 'final_ssdg.pth'), original_artifacts_unchanged=True,
        checkpoint_metadata_rewritten=False, training_repeated=False, optimizer_updates=0,
        query_constructed=False, target_access=False, source_roles='EXACT_MATCH',
        diagnostic_elapsed_seconds=diagnostic_seconds, recovery_commit=(ROOT / 'release_commit.txt').read_text().strip(),
        recovery_pid=os.getpid(), **evidence)
    d.write(new / 'recovery_provenance.json', provenance)
    d.write(new / 'mechanism_execution.json', mechanism)
    d.write(new / 'completion.json', dict(status='SOURCE_TRAINED', epoch=200,
        checkpoint=str(new / 'final_ssdg.pth'), elapsed_seconds=evidence['training_epoch_seconds'],
        elapsed_seconds_scope=evidence['training_time_scope'], diagnostic_elapsed_seconds=diagnostic_seconds,
        final_source_metrics=metrics, target_access=False, target_evaluated=False, checkpoint_sources=[],
        checkpoint_origin=c_new['checkpoint_origin'], config=c_new, logged_steps=evidence['logged_steps'],
        optimizer_steps=ck['a1_ema_successful_updates'], mechanism_execution=mechanism, source_stress=stress,
        recovery_provenance=str(new / 'recovery_provenance.json')))
    return provenance


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--old-config', required=True)
    parser.add_argument('--config')
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    if not args.check_only and not args.config: parser.error('--config is required for recovery')
    result = smoke(read(args.old_config)) if args.check_only else recover(read(args.config), read(args.old_config))
    print(json.dumps(result, ensure_ascii=False, allow_nan=False), flush=True)
