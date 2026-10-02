"""Read complete immutable source artifacts; no weights, query or truth access."""
import argparse
import csv
import inspect
import io
import json
import math
from pathlib import Path
import statistics


def audit_input(lift,count,expected_envelope_lag=None):
    if lift.get('active') is not True or len(lift.get('records',[]))!=1:raise ValueError('Actual orthogonal conv input missing')
    rec=lift['records'][0]
    lag=rec.get('actual_envelope_lag')
    if rec.get('block')!='behavior.0.conv' or rec.get('packets')!=count or rec.get('complex_terms')!=12:raise ValueError('Orthogonal conv input scope differs')
    if lag not in (0,4) or (expected_envelope_lag is not None and lag!=expected_envelope_lag):raise ValueError('Actual orthogonal envelope lag differs')
    if rec.get('envelope_lag')!=lag or rec.get('per_packet_only') is not True or rec.get('persistent_moments') is not False or rec.get('energy_whitening') is not False:raise ValueError('Packet IQ-only projection scope differs')
    for k in ('input_formula_max_abs_error','input_abs_max'):
        if not math.isfinite(rec.get(k,float('nan'))) or rec[k]<0:raise ValueError('Invalid actual input measurement')
    rr=rec.get('records',[])
    if len(rr)!=4 or [r.get('delay') for r in rr]!=[0,1,2,3]:raise ValueError('Four delay/order measurements missing')
    for r in rr:
        n=r.get('eligible_packets',-1)
        if r.get('packets')!=count or not 0<=n<=count or not 0<=r.get('near_degenerate_packets',-1)<=count:raise ValueError('Orthogonal basis packet counts differ')
        for k in ('raw_order1_max_abs_error','reconstruction_max_abs_error','mean_envelope','mean_envelope_variance'):
            if not math.isfinite(r.get(k,float('nan'))) or r[k]<0:raise ValueError('Invalid reconstruction/moment measurement')
        for k in ('original_normalized_gram_offdiag_mean','orthogonal_normalized_gram_offdiag_max'):
            v=r.get(k)
            if (v is None)!=(n==0) or (v is not None and (not math.isfinite(v) or v<0)):raise ValueError('Invalid Gram eligibility/measurement')
        energy=r.get('order_energy_mean',[])
        if len(energy)!=3 or any(not math.isfinite(v) or v<0 for v in energy):raise ValueError('Missing three actual order energies')
    return dict(packets=count,envelope_lag=lag,delay_order_records=4)


def audit_logs(epochs, steps, csv_text, stdout, expected_parameters, expected_alignment_strength, compact_jsonl=None, expected_envelope_lag=None):
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
    if compact_jsonl is not None:
        actual=[json.loads(line) for line in compact_jsonl.splitlines()]
        expected=[{k:v for k,v in epoch.items() if not isinstance(v,dict)} for epoch in epochs]
        if actual!=expected:raise ValueError('Compact epoch JSONL differs from measured full epochs')
    runtime = stdout.split('RESOLVED_CONFIG ', 1)
    if len(runtime) != 2:
        raise ValueError('Missing actual resolved configuration in stdout')
    errors = [m for m in ('Traceback (most recent call last)', 'CUDA out of memory', 'FloatingPointError', 'Killed') if m in runtime[1]]
    if errors:
        raise ValueError('Runtime error after resolved configuration: ' + repr(errors))
    fixed = dict(ce_weight=1., augmentation_active=False, domain_backbone_active=False,
                 pseudo_labels_active=False, extra_losses_active=False, orthopoly_active=True,
                 gradient_used_parameters=expected_parameters)
    # This new family always registers full FP32. Even omission from every
    # record is a missing execution measurement, not a legacy policy exemption.
    full_fp32=True
    learned_alignment=False
    previous_alpha=expected_alignment_strength
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
            if rec.get('alignment_gradient_used_parameters')!=int(learned_alignment):raise ValueError('Alignment gradient-use count differs')
            ag=rec.get('alignment_gradient_norm')
            if learned_alignment and (ag is None or not math.isfinite(ag) or ag<0):raise ValueError('Actual alignment gradient missing/nonfinite')
            if not learned_alignment and ag is not None:raise ValueError('Fixed alignment gradient must be N/A')
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
        if learned_alignment and not math.isclose(statistics.mean(s['alignment_gradient_norm'] for s in batch_steps),epoch['alignment_gradient_norm'],rel_tol=1e-12,abs_tol=1e-12):raise ValueError('Alignment gradient epoch mean differs')
        alpha=epoch.get('alignment_strength')
        if alpha is None or not math.isfinite(alpha) or not 0<=alpha<=1:raise ValueError('Missing/invalid actual alignment strength')
        if not learned_alignment and alpha!=expected_alignment_strength:raise ValueError('Raw received waveform alignment changed')
        if alpha!=batch_steps[-1]['alignment_strength_after']:raise ValueError('Laststep/epoch alignment mismatch')
        for step in batch_steps:
            if any(not math.isfinite(step[k]) or not 0<=step[k]<=1 for k in ('alignment_strength_before','alignment_strength_after')):raise ValueError('Invalid perstep alpha')
            if step['alignment_strength_before']!=previous_alpha:raise ValueError('Broken alignment step continuity')
            if not learned_alignment and step['alignment_strength_after']!=expected_alignment_strength:raise ValueError('Raw received waveform alignment changed in step')
            previous_alpha=step['alignment_strength_after']
        diag = epoch['orthopoly_diagnostics']
        if diag.get('alignment_strength')!=alpha:raise ValueError('Alpha telemetry mismatch')
        audit_input(diag.get('orthopoly_input',{}),28,expected_envelope_lag)
        normalization=diag['normalization'];norm_records=normalization['records']
        if normalization['blocks']!=6 or len(norm_records)!=6 or {r['block'] for r in norm_records}!={p+'.'+str(i) for p in ('time','behavior') for i in range(3)}:raise ValueError('Actual six-block normalization telemetry missing')
        for record in norm_records:
            n=record['eligible_packets']
            if not 0<=n<=28:raise ValueError('Normalization diagnostic packet scope changed')
            for k in ('relative_energy_fraction_max_error','relative_energy_fraction_variance_mean'):
                v=record[k]
                if (v is None)!=(n==0) or (v is not None and (not math.isfinite(v) or v<0)):raise ValueError('Invalid shared energy normalization measurement')
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
import json,time,csv,io,math,statistics,sys
from pathlib import Path
cfg=CONFIG
p=Path(cfg['project'])/'runs'/cfg['run']
def read(q):return json.loads(q.read_text(encoding='utf-8'))
state=read(p/'pipeline_state.json')
if state['status'] not in ('SOURCE_RESEARCH_COMPLETE_AWAITING_FROZEN_CLEAN_TEST','SOURCE_RESEARCH_COMPLETE_BASELINE_RETAINED'):
    print(json.dumps(dict(status=state['status'],ready=False)))
else:
    PROJECT=cfg['project'];VARIANTS=tuple(cfg['orthopoly'])
    def orthopoly_contract(variant):return cfg['orthopoly'][variant]
    exec(VALIDATE_SOURCE,globals())
    exec(SOURCE_READER,globals())
    exec(INPUT_AUDITOR,globals())
    exec(LOG_AUDITOR,globals())
    original=read(Path(PROJECT)/'runs/phase1_daot_rc4_pure_game_m3_20260917_r2/source_contract.json')
    controls=[read_source_record(r,original,'cvs_adaptive_volterra_identity') for r in cfg['controls']]
    result=dict(ready=True,read_at=time.time(),run_id=cfg['run'],pipeline=state,
        source_selection=read(p/'source_selection.json'),source_controls=controls,rows=[])
    for rid,row in state['rows'].items():
        q=Path(row['source_output'])
        actual=read_source_record(dict(source_output=str(q),model_seed=row['model_seed'],variant=row['variant']),original,'cvs_orthopoly_identity')
        epochs=[json.loads(s) for s in (q/'epoch_metrics.jsonl').read_text(encoding='utf-8').splitlines()]
        steps=[json.loads(s) for s in (q/'step_metrics.jsonl').read_text(encoding='utf-8').splitlines()]
        audit=audit_logs(epochs,steps,(q/'epoch_metrics.csv').read_text(encoding='utf-8'),Path(row['log']).read_text(encoding='utf-8',errors='strict'),cfg['parameter_counts'][row['variant']],cfg['alignment_strengths'][row['variant']],(q/'epoch_compact.jsonl').read_text(encoding='utf-8'),cfg['orthopoly'][row['variant']]['envelope_lag'])
        audit.update(roles_match=True,full_source_record_reverified=True)
        result['rows'].append(dict(row_id=rid,source_record=actual,log_audit=audit,epochs=epochs,
            completion=read(q/'completion.json'),initialization=read(q/'initialization.json'),
            resolved=read(q/'resolved_config.json'),profile=read(q/'resource_profile.json'),
            source_diagnostics=read(q/'source_final_diagnostics.json'),physical_diagnostics=read(q/'source_physical_diagnostics.json')))
    print(json.dumps(result,allow_nan=False))
'''


def validate_completed(data):
    from experiments.cvs_orthopoly_identity.dispatch import select_source_candidate, SEEDS
    from experiments.cvs_orthopoly_identity.model import VARIANTS
    if len(data['rows']) != 8 or {(r['resolved']['variant'],r['resolved']['model_seed']) for r in data['rows']} != {(v,s) for v in VARIANTS for s in SEEDS}:
        raise ValueError('Incomplete two-variant/four-seed source completion')
    records = list(data['source_controls'])
    for row in data['rows']:
        if row['completion']['final_source_metrics'] != {k: row['epochs'][-1][k] for k in row['completion']['final_source_metrics']}:
            raise ValueError('Final source metrics differ from E200')
        profile = row['profile']
        parameter_count=202553
        if (profile['total_parameters'] != parameter_count or profile['gradient_used_parameters'] != parameter_count
                or profile['trainable_parameters'] != parameter_count
                or not math.isfinite(profile['conv_linear_macs_per_sample']) or profile['conv_linear_macs_per_sample']<=0
                or profile.get('all_identity_paths_shared_energy_normalization') is not True
                or profile.get('orthopoly_contract') != row['resolved']['orthopoly_actual']):
            raise ValueError('Actual measured resource accounting differs')
        diag = row['source_diagnostics']
        keys = [(g['tx'], g['receiver'], g['day']) for g in diag['groups']]
        expected_cells={(t,r,d) for t in range(6) for r in (1,3,4,6,8) for d in (1,2,3)}
        if (diag['role'] != 'V' or diag['count'] != 27000 or len(keys) != 90 or len(set(keys)) != 90
                or set(keys)!=expected_cells or any(g['count']!=300 for g in diag['groups'])
                or sum(g['count'] for g in diag['groups']) != 27000
                or diag['target_access'] or diag['used_for_training'] or diag['used_for_selection']):
            raise ValueError('Source-only final geometry contract differs')
        alpha=row['epochs'][-1]['alignment_strength']
        if diag['groups'][0]['alignment_strength']!=alpha:raise ValueError('E200/finalV alignment differs')
        for group in diag['groups']:
            if group['alignment_strength']!=alpha or not 0<=group['residual_valid_fraction']<=1 or not 0<=group['residual_formula_eligible_count']<=group['count']:raise ValueError('Actual fractional source execution differs')
            if (group['residual_formula_max_error_hz'] is None)!=(group['residual_formula_eligible_count']==0):raise ValueError('Residual frequency eligibility/missing metric mismatch')
            if group['residual_formula_max_error_hz'] is not None and (not math.isfinite(group['residual_formula_max_error_hz']) or group['residual_formula_max_error_hz']<0):raise ValueError('Invalid residual frequency measurement')
            if any(not math.isfinite(group[k]) for k in ('relative_cfo_hz_mean','relative_cfo_hz_min','relative_cfo_hz_max','coherence_mean','fallback_fraction','residual_estimated_cfo_hz_mean','nominal_residual_cfo_hz_mean')):
                raise ValueError('Missing or nonfinite complete source synchronization measurements')
            if not (-625000-1<=group['relative_cfo_hz_min']<=group['relative_cfo_hz_mean']<=group['relative_cfo_hz_max']<=625000+1) or not 0<=group['fallback_fraction']<=1:
                raise ValueError('Invalid source synchronization units or fallback count')
        phys = row['physical_diagnostics']
        expected_lag=row['resolved']['orthopoly_actual']['envelope_lag']
        lifts=[(epoch['orthopoly_diagnostics'].get('orthopoly_input',{}),28) for epoch in row['epochs']]+[(phys.get('orthopoly_input',{}),30)]
        for lift,count in lifts:audit_input(lift,count,expected_lag)
        norm=phys.get('normalization',{})
        if norm.get('blocks')!=6 or len(norm.get('records',[]))!=6 or {r['block'] for r in norm.get('records',[])}!={p+'.'+str(i) for p in ('time','behavior') for i in range(3)}:raise ValueError('Frozen six-block energy normalization evidence missing')
        for r in norm['records']:
            n=r['eligible_packets']
            if not 0<=n<=30:raise ValueError('Frozen normalization count differs')
            for k in ('relative_energy_fraction_max_error','relative_energy_fraction_variance_mean'):
                v=r[k]
                if (v is None)!=(n==0) or (v is not None and (not math.isfinite(v) or v<0)):raise ValueError('Invalid frozen normalization measurement')
        if (phys['status'] != 'FROZEN_SYNTHETIC_DIAGNOSTIC_COMPLETE' or len(phys['records']) != 30
                or len(phys['isolated_tx_changes']) != 4 or not phys['synthetic_only']
                or any(phys[k] for k in ('target_access', 'training_augmentation', 'formal_data_access', 'model_updated', 'hardware_parameter_recovery'))
                or max(phys['confounds'].values()) > 1e-12):
            raise ValueError('Frozen synthetic-only physics contract differs')
        if phys.get('whole_affine_phase_invariant_claim') is not False or phys.get('alignment_strength')!=alpha or not math.isfinite(phys.get('coordinate_reconstruction_max_error',float('nan'))):raise ValueError('Fractional physical boundary/alpha/reconstruction evidence missing')
        phases = phys.get('phase_audit', [])
        if (len(phases) != 3 or [r['theta_radians'] for r in phases] != [.37, -1.2, 2.9]
                or [r['received_cfo_hz'] for r in phases] != [0.,80000.,-80000.]
                or any(not math.isfinite(r[k]) or r[k] < 0 for r in phases
                       for k in ('whole_logit_max_abs_error', 'whole_unit_embedding_max_distance'))):
            raise ValueError('Missing or invalid measured whole-phase evidence')
        for phase in phases:
            n=phase['covariance_eligible_count'];error=phase['partial_waveform_covariance_max_error']
            if not 0<=n<=30 or ((error is None)!=(n==0)) or (error is not None and (not math.isfinite(error) or error<0)):
                raise ValueError('Missing or invalid partial waveform covariance evidence')
        # Numerical tolerance outcome is reported, never a source selection/stop gate.
        records.append(row['source_record'])
    selection = select_source_candidate(records)
    if selection != data['source_selection']:
        raise ValueError('Stored source selection differs from independent ranking')
    return selection


def collect(root, output, run_id=None):
    from experiments.cvs_orthopoly_identity.prepare import PROJECT, RUN, RELEASE
    RUN=run_id or RUN
    from experiments.cvs_orthopoly_identity.publish import ssh
    from experiments.cvs_orthopoly_identity.dispatch import read_source_record, control_rows
    from experiments.cvs_orthopoly_identity.source import validate_config
    from experiments.cvs_orthopoly_identity.model import orthopoly_contract,VARIANTS
    cfg = dict(project=PROJECT, run=RUN, controls=control_rows(), orthopoly={v:orthopoly_contract(v) for v in VARIANTS},parameter_counts={v:202553 for v in VARIANTS},alignment_strengths={v:orthopoly_contract(v)['alignment_strength'] for v in VARIANTS})
    script = (REMOTE.replace('CONFIG', repr(cfg)).replace('VALIDATE_SOURCE', repr(inspect.getsource(validate_config)))
              .replace('INPUT_AUDITOR',repr(inspect.getsource(audit_input))).replace('SOURCE_READER', repr(inspect.getsource(read_source_record))).replace('LOG_AUDITOR', repr(inspect.getsource(audit_logs))))
    # The control reader imports the committed energy source/model validators.
    # Execute in that immutable release with the verified Torch environment,
    # rather than SSH's system Python and unrelated default working directory.
    release=PROJECT+'/releases/'+RELEASE
    command=['/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python','-c',script]
    wrapper='import os,subprocess\nsubprocess.run('+repr(command)+',cwd='+repr(release)+',env=dict(os.environ,PYTHONPATH='+repr(release+':'+release+'/code')+'),check=True)\n'
    data = json.loads(ssh(wrapper))
    if not data['ready']:
        print(json.dumps(data)); return False
    selection = validate_completed(data)
    validation = dict(status='VERIFIED',new_rows=8,control_rows=4,new_epochs=1600,new_steps=80000,
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
