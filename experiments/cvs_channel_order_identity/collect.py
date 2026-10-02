"""Read complete immutable source artifacts; no weights, query or truth access."""
import argparse
import csv
import inspect
import io
import json
import math
from pathlib import Path
import statistics


def validate_channel_diagnostics(d, contract, packets):
    if d.get('active') is not True or d.get('variant')!=contract['mode'] or d.get('packets')!=packets:
        raise ValueError('Channel diagnostic scope differs')
    compensation=contract['channel_compensation_active'];order=contract['channel_order_active']
    if d.get('compensation_active')!=compensation or d.get('order_difference_active')!=order:
        raise ValueError('Channel diagnostic activation flags differ')
    groups=[
        (True,('f_gradient_norm','residual_relative_output_mean')),
        (compensation,('g_magnitude_mean','g_magnitude_min','g_magnitude_max','g_operator_delta_bound_max',
                      'g_coefficient_l1_max','g_basis_l1_max','g_context_gradient_norm','g_basis_gradient_norm','g_context_exit_gradient_norm')),
        (order,('d_relative_output_mean','d_relative_output_max','d_gradient_norm','d_readout_gradient_norm')),
    ]
    for active,keys in groups:
        for key in keys:
            if key not in d:raise ValueError('Missing channel measurement: '+key)
            value=d[key]
            if active:
                if value is None or not math.isfinite(value) or value<0:raise ValueError('Invalid active channel measurement: '+key)
            elif value is not None:raise ValueError('Inactive channel measurement must be N/A: '+key)
    if compensation and not d['g_magnitude_min']<=d['g_magnitude_mean']<=d['g_magnitude_max']:
        raise ValueError('G magnitude extrema mismatch')
    if order and d['d_relative_output_mean']>d['d_relative_output_max']:
        raise ValueError('D magnitude extrema mismatch')


def audit_logs(epochs, steps, csv_text, stdout, expected_parameters, expected_alignment_strength, compact_jsonl=None, expected_phase_lag=None, expected_channel_contract=None):
    """Reconcile every step, epoch, compact CSV and detailed text record."""
    if expected_channel_contract is None:raise ValueError('Missing actual registered channel contract')
    if expected_parameters!=expected_channel_contract['total_trainable_parameters']:raise ValueError('Total parameter contract mismatch')
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
                 pseudo_labels_active=False, extra_losses_active=False, channel_active=True,
                 gradient_used_parameters=expected_parameters,channel_compensation_active=expected_channel_contract['channel_compensation_active'],channel_order_active=expected_channel_contract['channel_order_active'])
    # This new family always registers full FP32. Even omission from every
    # record is a missing execution measurement, not a legacy policy exemption.
    full_fp32=True
    learned_alignment=False
    previous_alpha=expected_alignment_strength;previous_mix=[0.,0.]
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
            if step.get('mixture_gradient_used_parameters')!=2:raise ValueError('Missing two gate CE gradients')
            for j,order in enumerate((3,5)):
                before=step['mix_raw'+str(order)+'_before'];after=step['mix_raw'+str(order)+'_after']
                cb=step['mix_coefficient'+str(order)+'_before'];ca=step['mix_coefficient'+str(order)+'_after']
                gradient=step['mix_gradient'+str(order)]
                if any(not math.isfinite(v) for v in (before,after,cb,ca,gradient)) or before!=previous_mix[j]:raise ValueError('Broken/nonfinite adaptive gate step continuity')
                if abs(cb)>1 or abs(ca)>1 or not math.isclose(cb,math.tanh(before),abs_tol=1e-7) or not math.isclose(ca,math.tanh(after),abs_tol=1e-7):raise ValueError('Actual tanh residual coefficient mismatch')
                previous_mix[j]=after
            if not math.isfinite(step.get('mix_gradient_norm',float('nan'))) or step['mix_gradient_norm']<0:raise ValueError('Missing gate gradient norm')
            if any(not math.isfinite(step[k]) or not 0<=step[k]<=1 for k in ('alignment_strength_before','alignment_strength_after')):raise ValueError('Invalid perstep alpha')
            if step['alignment_strength_before']!=previous_alpha:raise ValueError('Broken alignment step continuity')
            if not learned_alignment and step['alignment_strength_after']!=expected_alignment_strength:raise ValueError('Raw received waveform alignment changed in step')
            previous_alpha=step['alignment_strength_after']
        if epoch.get('mixture_gradient_used_parameters')!=2:raise ValueError('Missing epoch gate gradient measurement')
        for order in (3,5):
            if epoch['mix_raw'+str(order)]!=batch_steps[-1]['mix_raw'+str(order)+'_after'] or epoch['mix_coefficient'+str(order)]!=batch_steps[-1]['mix_coefficient'+str(order)+'_after']:raise ValueError('Epoch/step residual gate state differs')
        for key in ('mix_gradient3','mix_gradient5','mix_gradient_norm'):
            if not math.isclose(statistics.mean(s[key] for s in batch_steps),epoch[key],rel_tol=1e-12,abs_tol=1e-12):raise ValueError('Epoch gate gradient average differs')
        if any(not math.isfinite(step.get('channel_gradient_norm',float('nan'))) or step['channel_gradient_norm']<0 or step['channel_gradient_used_parameters']!=expected_channel_contract['new_trainable_parameters'] for step in batch_steps):raise ValueError('Missing measured neural CE gradients')
        if not math.isclose(statistics.mean(step['channel_gradient_norm'] for step in batch_steps),epoch['channel_gradient_norm'],rel_tol=1e-12,abs_tol=1e-12):raise ValueError('Neural gradient mean mismatch')
        diag = epoch['channel_diagnostics']
        validate_channel_diagnostics(diag['channel_order'],expected_channel_contract,28)
        if epoch.get('channel_gradient_used_parameters')!=expected_channel_contract['new_trainable_parameters']:raise ValueError('Epoch channel gradient count differs')
        if diag.get('alignment_strength')!=alpha:raise ValueError('Alpha telemetry mismatch')
        lift=diag.get('adaptive_input',{})
        if lift.get('active') is not True or len(lift.get('records',[]))!=1:raise ValueError('Actual volterra input telemetry missing')
        if lift.get('raw_parameters')!=[epoch['mix_raw3'],epoch['mix_raw5']] or lift.get('coefficients')!=[epoch['mix_coefficient3'],epoch['mix_coefficient5']]:raise ValueError('Actual input gates differ from optimizer state')
        record=lift['records'][0]
        if record.get('block')!='behavior.0.conv' or record.get('packets')!=28 or record.get('complex_terms')!=12:
            raise ValueError('Adaptive Volterra input diagnostic scope changed')
        lag=record.get('actual_phase_lag')
        if record.get('actual_envelope_lag')!=4:raise ValueError('Actual fixed envelope lag changed')
        if lag not in (1,4) or (expected_phase_lag is not None and lag!=expected_phase_lag):raise ValueError('Actual volterra lag changed')
        for key in ('input_formula_max_abs_error','input_abs_max','degree3_relative_input_change_mean','degree5_relative_input_change_mean'):
            if not math.isfinite(record.get(key,float('nan'))) or record[key]<0:raise ValueError('Invalid volterra input measurement')
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
    PROJECT=cfg['project'];VARIANTS=tuple(cfg['channel'])
    def channel_contract(variant):return cfg['channel'][variant]
    exec(VALIDATE_SOURCE,globals())
    exec(SOURCE_READER,globals())
    exec(CHANNEL_AUDITOR,globals())
    exec(LOG_AUDITOR,globals())
    original=read(Path(PROJECT)/'runs/phase1_daot_rc4_pure_game_m3_20260917_r2/source_contract.json')
    controls=[read_source_record(r,original,'cvs_neural_residual_identity') for r in cfg['controls']]
    result=dict(ready=True,read_at=time.time(),run_id=cfg['run'],pipeline=state,
        source_selection=read(p/'source_selection.json'),source_controls=controls,rows=[])
    for rid,row in state['rows'].items():
        q=Path(row['source_output'])
        actual=read_source_record(dict(source_output=str(q),model_seed=row['model_seed'],variant=row['variant']),original,'cvs_channel_order_identity')
        epochs=[json.loads(s) for s in (q/'epoch_metrics.jsonl').read_text(encoding='utf-8').splitlines()]
        steps=[json.loads(s) for s in (q/'step_metrics.jsonl').read_text(encoding='utf-8').splitlines()]
        audit=audit_logs(epochs,steps,(q/'epoch_metrics.csv').read_text(encoding='utf-8'),Path(row['log']).read_text(encoding='utf-8',errors='strict'),cfg['parameter_counts'][row['variant']],cfg['alignment_strengths'][row['variant']],(q/'epoch_compact.jsonl').read_text(encoding='utf-8'),cfg['channel'][row['variant']]['phase_lag'],cfg['channel'][row['variant']])
        audit.update(roles_match=True,full_source_record_reverified=True)
        result['rows'].append(dict(row_id=rid,source_record=actual,log_audit=audit,epochs=epochs,
            completion=read(q/'completion.json'),initialization=read(q/'initialization.json'),
            resolved=read(q/'resolved_config.json'),profile=read(q/'resource_profile.json'),
            source_diagnostics=read(q/'source_final_diagnostics.json'),physical_diagnostics=read(q/'source_physical_diagnostics.json')))
    print(json.dumps(result,allow_nan=False))
'''


def validate_completed(data):
    from experiments.cvs_channel_order_identity.dispatch import select_source_candidate, SEEDS
    from experiments.cvs_channel_order_identity.model import VARIANTS
    if len(data['rows']) != 16 or {(r['resolved']['variant'],r['resolved']['model_seed']) for r in data['rows']} != {(v,s) for v in VARIANTS for s in SEEDS}:
        raise ValueError('Incomplete four-variant/four-seed source completion')
    records = list(data['source_controls'])
    for row in data['rows']:
        if row['completion']['final_source_metrics'] != {k: row['epochs'][-1][k] for k in row['completion']['final_source_metrics']}:
            raise ValueError('Final source metrics differ from E200')
        profile = row['profile']
        parameter_count=row['resolved']['channel']['total_trainable_parameters']
        if (profile['total_parameters'] != parameter_count or profile['gradient_used_parameters'] != parameter_count
                or profile['trainable_parameters'] != parameter_count
                or not math.isfinite(profile['conv_linear_macs_per_sample']) or profile['conv_linear_macs_per_sample']<=0
                or profile.get('all_identity_paths_shared_energy_normalization') is not False
                or profile.get('base_identity_paths_shared_energy_normalization') is not True
                or profile.get('channel_contract') != row['resolved']['channel_actual']):
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
        validate_channel_diagnostics(phys['channel_order'],row['resolved']['channel'],30)
        expected_lag=row['resolved']['channel_actual']['phase_lag']
        lifts=[(epoch['channel_diagnostics'].get('adaptive_input',{}),28) for epoch in row['epochs']]+[(phys.get('adaptive_input',{}),30)]
        for lift,count in lifts:
            if lift.get('active') is not True or len(lift.get('records',[]))!=1:raise ValueError('Actual volterra input measurements missing')
            if len(lift.get('raw_parameters',[]))!=2 or len(lift.get('coefficients',[]))!=2 or any(not math.isfinite(v) for v in lift['raw_parameters']+lift['coefficients']):raise ValueError('Missing measured source/public adaptive gates')
            if any(abs(a)>1 or not math.isclose(a,math.tanh(r),abs_tol=1e-7) for a,r in zip(lift['coefficients'],lift['raw_parameters'])):raise ValueError('Source/public tanh gate formula differs')
            record=lift['records'][0]
            if record.get('actual_envelope_lag')!=4:raise ValueError('Actual fixed envelope lag changed')
            if record.get('packets')!=count or record.get('complex_terms')!=12 or record.get('actual_phase_lag')!=expected_lag:raise ValueError('Actual source/public volterra input differs')
            for key in ('input_formula_max_abs_error','input_abs_max','degree3_relative_input_change_mean','degree5_relative_input_change_mean'):
                if not math.isfinite(record.get(key,float('nan'))) or record[key]<0:raise ValueError('Invalid source/public volterra input measurement')
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
    from experiments.cvs_channel_order_identity.prepare import PROJECT, RUN, RELEASE
    RUN=run_id or RUN
    from experiments.cvs_channel_order_identity.publish import ssh
    from experiments.cvs_channel_order_identity.dispatch import read_source_record, control_rows
    from experiments.cvs_channel_order_identity.source import validate_config
    from experiments.cvs_channel_order_identity.model import channel_contract,VARIANTS
    cfg = dict(project=PROJECT, run=RUN, controls=control_rows(), channel={v:channel_contract(v) for v in VARIANTS},parameter_counts={v:channel_contract(v)['total_trainable_parameters'] for v in VARIANTS},alignment_strengths={v:channel_contract(v)['alignment_strength'] for v in VARIANTS})
    script = (REMOTE.replace('cfg=CONFIG', 'cfg='+repr(cfg),1).replace('VALIDATE_SOURCE', repr(inspect.getsource(validate_config)))
              .replace('SOURCE_READER', repr(inspect.getsource(read_source_record))).replace('CHANNEL_AUDITOR', repr(inspect.getsource(validate_channel_diagnostics))).replace('LOG_AUDITOR', repr(inspect.getsource(audit_logs))))
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
    validation = dict(status='VERIFIED',new_rows=16,control_rows=4,new_epochs=3200,new_steps=160000,
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
