"""Read complete immutable source artifacts; no weights, query or truth access."""
import argparse
import csv
import inspect
import io
import json
import math
from pathlib import Path
import statistics


def audit_logs(epochs, steps, csv_text, stdout):
    """Reconcile every step, epoch, compact CSV and detailed text record."""
    if len(epochs) != 200 or len(steps) != 10000:
        raise ValueError('Incomplete E200 x50 logs')
    if [r['epoch'] for r in epochs] != list(range(1, 201)):
        raise ValueError('Missing or duplicate epoch')
    if [r['step'] for r in steps] != list(range(1, 10001)):
        raise ValueError('Missing or duplicate optimizer step')
    csv_rows = list(csv.DictReader(io.StringIO(csv_text)))
    text_epochs = [json.loads(line[6:]) for line in stdout.splitlines() if line.startswith('EPOCH ')]
    if text_epochs != epochs or len(csv_rows) != 200:
        raise ValueError('Detailed stdout/compact CSV differs from epoch JSONL')
    runtime = stdout.split('RESOLVED_CONFIG ', 1)
    if len(runtime) != 2:
        raise ValueError('Missing actual resolved configuration in stdout')
    errors = [m for m in ('Traceback (most recent call last)', 'CUDA out of memory', 'FloatingPointError', 'Killed') if m in runtime[1]]
    if errors:
        raise ValueError('Runtime error after resolved configuration: ' + repr(errors))
    fixed = dict(ce_weight=1., augmentation_active=False, domain_backbone_active=False,
                 pseudo_labels_active=False, extra_losses_active=False, equivariant_active=True,
                 gradient_used_parameters=202553)
    full_fp32=any('cudnn_allow_tf32' in r for r in [*epochs,*steps])
    for epoch, compact in zip(epochs, csv_rows):
        number = epoch['epoch']
        batch_steps = steps[(number - 1) * 50:number * 50]
        for rec in [epoch, *batch_steps]:
            if full_fp32 and rec.get('cudnn_allow_tf32') is not False:
                raise ValueError('Full FP32 step/epoch policy missing or changed')
            if any(rec.get(k) != v for k, v in fixed.items()):
                raise ValueError('Plain CE / physical execution flags changed')
            if rec['total_loss'] != rec['clean_ce'] or any(not math.isfinite(rec[k]) for k in ('clean_ce', 'total_loss', 'learning_rate', 'gradient_norm')):
                raise ValueError('Nonfinite metrics or non-CE loss')
            if rec['learning_rate'] <= 0 or rec['gradient_norm'] < 0:
                raise ValueError('Invalid optimization metrics')
        if epoch['optimizer_steps'] != 50 or epoch['optimizer_steps_total'] != number * 50 or epoch['source_sample_exposure'] != 6300 or epoch['satellite_sample_exposure'] != 0:
            raise ValueError('Epoch budget changed')
        if [s['source_samples'] for s in batch_steps] != [128] * 49 + [28]:
            raise ValueError('Actual batch exposure differs')
        for s in batch_steps:
            if s['epoch'] != number or s['satellite_samples'] != 0 or s['learning_rate'] != epoch['learning_rate']:
                raise ValueError('Step role / epoch / learning rate differs')
        for key in ('clean_ce', 'gradient_norm'):
            if not math.isclose(statistics.mean(s[key] for s in batch_steps), epoch[key], rel_tol=1e-12, abs_tol=1e-12):
                raise ValueError('Epoch average differs from complete steps')
        if epoch['source_val_count'] != 27000 or set(epoch['source_val_rx_accuracy']) != {'1', '3', '4', '6', '8'}:
            raise ValueError('Source validation roles/count differ')
        rates = [epoch['source_val_accuracy'], *epoch['source_val_rx_accuracy'].values()]
        if any(not math.isfinite(v) or not 0 <= v <= 1 for v in rates) or epoch['source_val_worst_rx'] != min(epoch['source_val_rx_accuracy'].values()):
            raise ValueError('Invalid source validation metrics')
        diag = epoch['equivariant_diagnostics']
        if any(not math.isfinite(v) for v in diag.values() if isinstance(v, (int, float))):
            raise ValueError('Nonfinite response/gradient diagnostic')
        expected_compact = {k: v for k, v in epoch.items() if not isinstance(v, dict)}
        if set(compact) != set(expected_compact):
            raise ValueError('Compact CSV fields differ')
        for k, v in expected_compact.items():
            actual = compact[k]
            if isinstance(v, (bool, str)):
                if actual != str(v):
                    raise ValueError('Compact CSV value differs: ' + k)
            elif isinstance(v, (int, float)) and float(actual) != v:
                raise ValueError('Compact CSV numeric value differs: ' + k)
            elif v is None and actual != '':
                raise ValueError('Compact CSV null differs: ' + k)
    return dict(epochs=200, steps=10000, csv_epochs=200, full_stdout_lines=len(stdout.splitlines()),
                all_step_flags_match=True, all_step_metrics_finite=True, step_epoch_csv_stdout_reconciled=True,
                errors=[], numpy_bridge_warning='A module that was compiled using NumPy 1.x' in stdout,
                startup_traceback_warning='Traceback (most recent call last)' in runtime[0])


REMOTE = r'''
import json,time,csv,io,math,statistics
from pathlib import Path
cfg=CONFIG
p=Path(cfg['project'])/'runs'/cfg['run']
def read(q):return json.loads(q.read_text(encoding='utf-8'))
state=read(p/'pipeline_state.json')
if state['status'] not in ('SOURCE_RESEARCH_COMPLETE_AWAITING_FROZEN_CLEAN_TEST','SOURCE_RESEARCH_COMPLETE_BASELINE_RETAINED'):
    print(json.dumps(dict(status=state['status'],ready=False)))
else:
    PROJECT=cfg['project'];VARIANTS=('equivariant_memory',)
    def equivariant_contract(variant):return cfg['equivariant'][variant]
    exec(VALIDATE_SOURCE,globals())
    exec(SOURCE_READER,globals())
    exec(LOG_AUDITOR,globals())
    original=read(Path(PROJECT)/'runs/phase1_daot_rc4_pure_game_m3_20260917_r2/source_contract.json')
    controls=[read_source_record(r,original,'cvs_residual_identity') for r in cfg['controls']]
    result=dict(ready=True,read_at=time.time(),run_id=cfg['run'],pipeline=state,
        source_selection=read(p/'source_selection.json'),source_controls=controls,rows=[])
    for rid,row in state['rows'].items():
        q=Path(row['source_output'])
        actual=read_source_record(dict(source_output=str(q),model_seed=row['model_seed'],variant=row['variant']),original,'cvs_equivariant_identity')
        epochs=[json.loads(s) for s in (q/'epoch_metrics.jsonl').read_text(encoding='utf-8').splitlines()]
        steps=[json.loads(s) for s in (q/'step_metrics.jsonl').read_text(encoding='utf-8').splitlines()]
        audit=audit_logs(epochs,steps,(q/'epoch_metrics.csv').read_text(encoding='utf-8'),Path(row['log']).read_text(encoding='utf-8',errors='strict'))
        audit.update(roles_match=True,full_source_record_reverified=True)
        result['rows'].append(dict(row_id=rid,source_record=actual,log_audit=audit,epochs=epochs,
            completion=read(q/'completion.json'),initialization=read(q/'initialization.json'),
            resolved=read(q/'resolved_config.json'),profile=read(q/'resource_profile.json'),
            source_diagnostics=read(q/'source_final_diagnostics.json'),physical_diagnostics=read(q/'source_physical_diagnostics.json')))
    print(json.dumps(result,allow_nan=False))
'''


def validate_completed(data):
    from experiments.cvs_equivariant_identity.dispatch import select_source_candidate, SEEDS
    if len(data['rows']) != 4 or {r['resolved']['model_seed'] for r in data['rows']} != SEEDS:
        raise ValueError('Incomplete four-seed source completion')
    records = list(data['source_controls'])
    for row in data['rows']:
        if row['completion']['final_source_metrics'] != {k: row['epochs'][-1][k] for k in row['completion']['final_source_metrics']}:
            raise ValueError('Final source metrics differ from E200')
        profile = row['profile']
        if (profile['total_parameters'] != 202553 or profile['gradient_used_parameters'] != 202553
                or profile['trainable_parameters'] != 202553
                or profile['conv_linear_macs_per_sample'] != 7199008
                or profile.get('all_identity_paths_phase_constrained') is not True
                or profile.get('equivariant_contract') != row['resolved']['equivariant_actual']):
            raise ValueError('Actual measured resource accounting differs')
        diag = row['source_diagnostics']
        keys = [(g['tx'], g['receiver'], g['day']) for g in diag['groups']]
        if (diag['role'] != 'V' or diag['count'] != 27000 or len(keys) != 90 or len(set(keys)) != 90
                or sum(g['count'] for g in diag['groups']) != 27000
                or diag['target_access'] or diag['used_for_training'] or diag['used_for_selection']):
            raise ValueError('Source-only final geometry contract differs')
        phys = row['physical_diagnostics']
        if (phys['status'] != 'FROZEN_SYNTHETIC_DIAGNOSTIC_COMPLETE' or len(phys['records']) != 30
                or len(phys['isolated_tx_changes']) != 4 or not phys['synthetic_only']
                or any(phys[k] for k in ('target_access', 'training_augmentation', 'formal_data_access', 'model_updated', 'hardware_parameter_recovery'))
                or max(phys['confounds'].values()) > 1e-12):
            raise ValueError('Frozen synthetic-only physics contract differs')
        phases = phys.get('phase_audit', [])
        if (len(phases) != 3 or [r['theta_radians'] for r in phases] != [.37, -1.2, 2.9]
                or phys.get('whole_identity_global_phase_invariance_claimed') is not True
                or any(not math.isfinite(r[k]) or r[k] < 0 for r in phases
                       for k in ('whole_logit_max_abs_error', 'whole_unit_embedding_max_distance'))):
            raise ValueError('Missing or invalid measured whole-phase evidence')
        # Numerical tolerance outcome is reported, never a source selection/stop gate.
        records.append(row['source_record'])
    selection = select_source_candidate(records)
    if selection != data['source_selection']:
        raise ValueError('Stored source selection differs from independent ranking')
    return selection


def collect(root, output, run_id=None):
    from experiments.cvs_equivariant_identity.prepare import PROJECT, RUN
    RUN=run_id or RUN
    from experiments.cvs_equivariant_identity.publish import ssh
    from experiments.cvs_equivariant_identity.dispatch import read_source_record, control_rows
    from experiments.cvs_equivariant_identity.source import validate_config
    from experiments.cvs_equivariant_identity.model import equivariant_contract
    cfg = dict(project=PROJECT, run=RUN, controls=control_rows(), equivariant={'equivariant_memory': equivariant_contract('equivariant_memory')})
    script = (REMOTE.replace('CONFIG', repr(cfg)).replace('VALIDATE_SOURCE', repr(inspect.getsource(validate_config)))
              .replace('SOURCE_READER', repr(inspect.getsource(read_source_record))).replace('LOG_AUDITOR', repr(inspect.getsource(audit_logs))))
    data = json.loads(ssh(script))
    if not data['ready']:
        print(json.dumps(data)); return False
    selection = validate_completed(data)
    validation = dict(status='VERIFIED',new_rows=4,control_rows=4,new_epochs=800,new_steps=40000,
                      full_stdout_scanned=True,step_epoch_csv_stdout_reconciled=True,
                      full_source_records_reverified=True,source_rule_recomputed=True,target_access=False,
                      source_selection=selection)
    for base in (output, root/'automation_reports/CV-SincNet'/RUN/'evidence'):
        base.mkdir(parents=True, exist_ok=True)
        for name, value in [('source_research_complete.json', data), ('source_selection.json', selection), ('source_completion_validation.json', validation)]:
            (base/name).write_text(json.dumps(value,ensure_ascii=False,allow_nan=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(validation,ensure_ascii=False))
    return True


if __name__ == '__main__':
    p=argparse.ArgumentParser(); p.add_argument('--root',type=Path,required=True); p.add_argument('--output',type=Path,required=True)
    a=p.parse_args(); collect(a.root,a.output)
