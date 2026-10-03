"""Read complete immutable source artifacts; no weights, query or truth access."""
import argparse
import csv
import inspect
import io
import json
import math
from pathlib import Path
import statistics


def validate_frontfilter_diagnostics(diagnostic, expected_packets, expected_contract=None):
    """Validate measured bounded-filter telemetry; missing gradients remain N/A.

    Bounds concern the linear operator with coefficients held fixed. They do
    not establish invertibility of a dynamic packet-conditioned mapping.
    """
    def number(value, label, lower=0., upper=None):
        if (isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value)
                or value < lower or (upper is not None and value > upper)):
            raise ValueError('Invalid frontfilter measurement: ' + label)
        return value

    variant=diagnostic.get('variant')
    if variant not in ('frontfilter_static', 'frontfilter_dynamic'):
        raise ValueError('Unregistered frontfilter diagnostic variant')
    dynamic=variant=='frontfilter_dynamic'
    if (diagnostic.get('active') is not True or diagnostic.get('dynamic') is not dynamic or
            diagnostic.get('fixed_coefficient_bound_only') is not True or
            diagnostic.get('dynamic_map_invertible_claim') is not False or
            not isinstance(diagnostic.get('records'), list) or len(diagnostic['records'])!=1):
        raise ValueError('Missing actual bounded-frontfilter diagnostic contract')
    if expected_contract is not None:
        from experiments.cvs_frontfilter_identity.model import filter_contract
        if expected_contract!=filter_contract(variant):
            raise ValueError('Frontfilter measured variant differs from registered contract')
    record=diagnostic['records'][0];tolerance=1e-5;rho=.25
    if record.get('block')!='frontfilter' or record.get('packets')!=expected_packets:
        raise ValueError('Frontfilter diagnostic packet scope differs')
    for key in ('relative_input_change_mean','relative_input_change_max','coefficient_l1_mean','coefficient_l1_max',
                'kernel_l1_mean','kernel_l1_max','basis_l1_max','bound_active_fraction','basis_bound_active_fraction',
                'coefficient_packet_variance','fixed_coefficient_delta_operator_bound_max'):
        number(record.get(key),key)
    if any(record[key]>1+tolerance for key in ('coefficient_l1_max','kernel_l1_max','basis_l1_max')):
        raise ValueError('Frontfilter complex-modulus L1 bound differs')
    if any(record[key]>1 for key in ('bound_active_fraction','basis_bound_active_fraction')):
        raise ValueError('Invalid frontfilter bound-active fraction')
    if not 0<=record['relative_input_change_mean']<=record['relative_input_change_max']+tolerance:
        raise ValueError('Frontfilter input-change extrema differ')
    eligible=record.get('input_norm_ratio_eligible_packets')
    if isinstance(eligible,bool) or not isinstance(eligible,int) or not 0<=eligible<=expected_packets:
        raise ValueError('Invalid frontfilter input-norm eligibility')
    for key in ('input_norm_ratio_min','input_norm_ratio_max'):
        if key not in record or (record[key] is None)!=(eligible==0):
            raise ValueError('Frontfilter input-norm eligibility/N/A differs')
        if eligible:number(record[key],key,1-rho-tolerance,1+rho+tolerance)
    if eligible and record['input_norm_ratio_min']>record['input_norm_ratio_max']:
        raise ValueError('Frontfilter norm-ratio extrema differ')
    for key in ('basis_gradient_norm','context_gradient_norm','exit_gradient_norm'):
        if key not in record:raise ValueError('Missing frontfilter gradient measurement/N/A: '+key)
        if record[key] is not None:number(record[key],key)
    if not dynamic and record['context_gradient_norm'] is not None:
        raise ValueError('Static frontfilter context gradient must be N/A')
    coefficients=record.get('coefficients')
    if (not isinstance(coefficients,list) or len(coefficients)!=expected_packets or
            any(not isinstance(p,list) or len(p)!=2 or any(not isinstance(c,list) or len(c)!=4 for c in p) for p in coefficients)):
        raise ValueError('Frontfilter coefficients must preserve [packets,2,4]')
    for packet in coefficients:
        for component in packet:
            for value in component:number(value,'coefficient',-1-tolerance,1+tolerance)
    measured_l1=[sum(math.hypot(p[0][j],p[1][j]) for j in range(4)) for p in coefficients]
    for stem in ('coefficient','kernel'):
        values=record.get(stem+'_l1_by_packet')
        if not isinstance(values,list) or len(values)!=expected_packets:
            raise ValueError('Missing frontfilter per-packet '+stem+' L1')
        for value in values:number(value,stem+' L1',0,1+tolerance)
        if (not math.isclose(statistics.mean(values),record[stem+'_l1_mean'],abs_tol=1e-6,rel_tol=1e-6) or
                not math.isclose(max(values),record[stem+'_l1_max'],abs_tol=1e-6,rel_tol=1e-6)):
            raise ValueError('Frontfilter per-packet L1 summary mismatch')
    if any(not math.isclose(a,b,abs_tol=1e-6,rel_tol=1e-6) for a,b in zip(measured_l1,record['coefficient_l1_by_packet'])):
        raise ValueError('Frontfilter coefficients/L1 measurement mismatch')
    if any(k>c+tolerance for k,c in zip(record['kernel_l1_by_packet'],measured_l1)):
        raise ValueError('Frontfilter effective kernel exceeds coefficient bound')
    variance=statistics.mean(statistics.pvariance(p[i][j] for p in coefficients) for i in range(2) for j in range(4))
    if not math.isclose(variance,record['coefficient_packet_variance'],abs_tol=1e-7,rel_tol=1e-5):
        raise ValueError('Frontfilter packet coefficient variance mismatch')
    if not dynamic and any(packet!=coefficients[0] for packet in coefficients):
        raise ValueError('Static frontfilter coefficients vary across packets')
    bound=record['fixed_coefficient_delta_operator_bound_max']
    if (not math.isclose(bound,rho*record['kernel_l1_max'],abs_tol=1e-6,rel_tol=1e-6) or
            record['relative_input_change_max']>bound+tolerance):
        raise ValueError('Fixed-coefficient frontfilter operator-bound measurement differs')
    return record


def validate_diagnostic_input_scopes(diagnostic):
    expected=dict(coordinate_cfo_and_coherence='original_input_x',
        normalization_adaptive_input_neural_residual='actual_G_x(x), exactly once per diagnostic forward',
        core_weights_and_gradients='parameters and last CE gradients, no input-dependent estimate')
    if diagnostic.get('diagnostic_input_scopes')!=expected:
        raise ValueError('Original-input CFO and filtered-backbone diagnostic scopes differ')
    return expected


def validate_neural_diagnostics(diagnostic, packets):
    records=diagnostic.get('records',[])
    if (diagnostic.get('active') is not True or len(records)!=2 or
            {r.get('block') for r in records}!={'time.2.neural.0','behavior.2.neural.0'}):
        raise ValueError('Actual retained shallow neural branches missing')
    for record in records:
        if record.get('packets')!=packets:raise ValueError('Neural branch diagnostic packet scope differs')
        for key in ('relative_output_change_mean','projection_norm'):
            value=record.get(key)
            if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value) or value<0:
                raise ValueError('Invalid retained neural branch measurement')


def validate_frontfilter_groups(source_diagnostics, expected_contract):
    """Validate all 27000 source-V filter measurements; no epoch selection."""
    from experiments.cvs_frontfilter_identity.model import filter_contract
    variant=expected_contract.get('mode')
    if expected_contract!=filter_contract(variant):
        raise ValueError('Unregistered full-source frontfilter contract')
    rows=source_diagnostics.get('frontfilter_groups',[])
    expected={(t,r,d) for t in range(6) for r in (1,3,4,6,8) for d in (1,2,3)}
    if (len(rows)!=90 or {(r.get('tx'),r.get('receiver'),r.get('day')) for r in rows}!=expected or
            any(r.get('count')!=300 for r in rows) or not isinstance(source_diagnostics.get('frontfilter_scope'),str) or
            not source_diagnostics['frontfilter_scope'].strip() or
            source_diagnostics.get('coordinate_scope')!='Original input x; inherited nominal CFO diagnostic does not measure frontend compensation'):
        raise ValueError('Full-source frontfilter grid, count or scope differs')
    tolerance=1e-5;rho=expected_contract['frontfilter_rho']
    for row in rows:
        for key in ('relative_input_change_mean','relative_input_change_max','coefficient_l1_mean','coefficient_l1_max',
                    'kernel_l1_mean','kernel_l1_max','coefficient_trace_variance'):
            value=row.get(key)
            if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value) or value<0:
                raise ValueError('Invalid full-source frontfilter measurement: '+key)
        for stem in ('coefficient_l1','kernel_l1'):
            if not 0<=row[stem+'_mean']<=row[stem+'_max']+tolerance or row[stem+'_max']>1+tolerance:
                raise ValueError('Full-source frontfilter L1 mean/max differs')
        if (row['kernel_l1_mean']>row['coefficient_l1_mean']+tolerance or
                row['kernel_l1_max']>row['coefficient_l1_max']+tolerance or
                not 0<=row['relative_input_change_mean']<=row['relative_input_change_max']+tolerance or
                row['relative_input_change_max']>rho*row['kernel_l1_max']+tolerance):
            raise ValueError('Full-source fixed-coefficient frontfilter bound differs')
        eligible=row.get('input_norm_ratio_eligible_count')
        if isinstance(eligible,bool) or not isinstance(eligible,int) or not 0<=eligible<=300:
            raise ValueError('Full-source frontfilter norm-ratio count differs')
        for key in ('input_norm_ratio_min','input_norm_ratio_max'):
            value=row.get(key)
            if key not in row or (value is None)!=(eligible==0):
                raise ValueError('Full-source frontfilter norm-ratio eligibility/N/A differs')
            if value is not None and (isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value)
                                      or not 1-rho-tolerance<=value<=1+rho+tolerance):
                raise ValueError('Full-source frontfilter norm-ratio bound differs')
        if eligible and row['input_norm_ratio_min']>row['input_norm_ratio_max']:
            raise ValueError('Full-source frontfilter norm-ratio extrema differ')
        mean=row.get('coefficient_mean')
        if (not isinstance(mean,list) or len(mean)!=8 or
                any(isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v) or abs(v)>1+tolerance for v in mean)):
            raise ValueError('Full-source frontfilter coefficient mean must contain eight finite values')
        if (sum(math.hypot(mean[i],mean[i+4]) for i in range(4))>row['coefficient_l1_mean']+tolerance or
                sum(v*v for v in mean)+row['coefficient_trace_variance']>1+tolerance):
            raise ValueError('Full-source frontfilter coefficient moments violate measured bounds')
        if not expected_contract['frontfilter_dynamic'] and row['coefficient_trace_variance']>1e-10:
            raise ValueError('Static frontfilter has nonzero within-cell packet coefficient variance')
    if not expected_contract['frontfilter_dynamic']:
        reference=rows[0]['coefficient_mean']
        if any(abs(value-reference[j])>1e-10 for row in rows for j,value in enumerate(row['coefficient_mean'])):
            raise ValueError('Static frontfilter coefficients differ between source cells')
    return rows


def audit_logs(epochs, steps, csv_text, stdout, expected_parameters, expected_alignment_strength, compact_jsonl=None, expected_phase_lag=None, expected_frontfilter_contract=None):
    """Reconcile every step, epoch, compact CSV and detailed text record."""
    if expected_frontfilter_contract is None:raise ValueError('Missing actual registered frontfilter contract')
    if expected_parameters!=expected_frontfilter_contract['total_trainable_parameters']:
        raise ValueError('Total frontfilter parameter contract mismatch')
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
                 pseudo_labels_active=False, extra_losses_active=False, frontfilter_active=True,
                 gradient_used_parameters=expected_parameters)
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
            if rec['total_loss'] != rec['clean_ce'] or rec['clean_ce']<0 or any(not math.isfinite(rec[k]) for k in ('clean_ce', 'total_loss', 'learning_rate', 'gradient_norm')):
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
        if not math.isfinite(epoch.get('source_val_ce',float('nan'))) or epoch['source_val_ce']<0:
            raise ValueError('Missing/nonfinite source validation CE')
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
        new_parameters=expected_frontfilter_contract['new_trainable_parameters']
        if epoch.get('frontfilter_gradient_used_parameters') != new_parameters:
            raise ValueError('Frontfilter gradient parameter budget differs')
        if any(not math.isfinite(step.get('frontfilter_gradient_norm',float('nan'))) or step['frontfilter_gradient_norm']<0 or step['frontfilter_gradient_used_parameters']!=new_parameters for step in batch_steps):raise ValueError('Missing measured frontfilter CE gradients')
        if not math.isclose(statistics.mean(step['frontfilter_gradient_norm'] for step in batch_steps),epoch['frontfilter_gradient_norm'],rel_tol=1e-12,abs_tol=1e-12):raise ValueError('Frontfilter gradient mean mismatch')
        diag = epoch['frontfilter_diagnostics']
        record=validate_frontfilter_diagnostics(diag.get('frontfilter', {}), 28, expected_frontfilter_contract)
        validate_diagnostic_input_scopes(diag)
        validate_neural_diagnostics(diag.get('neural_residual', {}),28)
        measured_gradient_keys=('basis_gradient_norm','exit_gradient_norm')+(
            ('context_gradient_norm',) if expected_frontfilter_contract['frontfilter_dynamic'] else ())
        if any(record[key] is None for key in measured_gradient_keys):
            raise ValueError('Training epoch missing last CE frontfilter gradient measurements')
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
    PROJECT=cfg['project'];VARIANTS=tuple(cfg['frontfilter'])
    def filter_contract(variant):return cfg['frontfilter'][variant]
    exec(VALIDATE_SOURCE,globals())
    exec(SOURCE_READER,globals())
    exec(FRONTFILTER_AUDITOR,globals())
    exec(SCOPE_AUDITOR,globals())
    exec(NEURAL_AUDITOR,globals())
    exec(LOG_AUDITOR,globals())
    original=read(Path(PROJECT)/'runs/phase1_daot_rc4_pure_game_m3_20260917_r2/source_contract.json')
    controls=[read_source_record(r,original,r['method']) for r in cfg['controls']]
    result=dict(ready=True,read_at=time.time(),run_id=cfg['run'],pipeline=state,
        source_selection=read(p/'source_selection.json'),source_controls=controls,rows=[])
    for rid,row in state['rows'].items():
        q=Path(row['source_output'])
        actual=read_source_record(dict(source_output=str(q),model_seed=row['model_seed'],variant=row['variant']),original,'cvs_frontfilter_identity')
        epochs=[json.loads(s) for s in (q/'epoch_metrics.jsonl').read_text(encoding='utf-8').splitlines()]
        steps=[json.loads(s) for s in (q/'step_metrics.jsonl').read_text(encoding='utf-8').splitlines()]
        audit=audit_logs(epochs,steps,(q/'epoch_metrics.csv').read_text(encoding='utf-8'),Path(row['log']).read_text(encoding='utf-8',errors='strict'),cfg['parameter_counts'][row['variant']],cfg['alignment_strengths'][row['variant']],(q/'epoch_compact.jsonl').read_text(encoding='utf-8'),cfg['frontfilter'][row['variant']]['phase_lag'],cfg['frontfilter'][row['variant']])
        audit.update(roles_match=True,full_source_record_reverified=True)
        result['rows'].append(dict(row_id=rid,source_record=actual,log_audit=audit,epochs=epochs,
            completion=read(q/'completion.json'),initialization=read(q/'initialization.json'),
            resolved=read(q/'resolved_config.json'),profile=read(q/'resource_profile.json'),
            source_diagnostics=read(q/'source_final_diagnostics.json'),physical_diagnostics=read(q/'source_physical_diagnostics.json')))
    print(json.dumps(result,allow_nan=False))
'''


def validate_completed(data):
    from experiments.cvs_frontfilter_identity.dispatch import select_source_candidate, SEEDS
    from experiments.cvs_frontfilter_identity.model import VARIANTS, filter_contract
    from experiments.cvs_reference_identity.physics import TX_ROWS, RX_ROWS
    if len(data['rows']) != 8 or {(r['resolved']['variant'],r['resolved']['model_seed']) for r in data['rows']} != {(v,s) for v in VARIANTS for s in SEEDS}:
        raise ValueError('Incomplete two-variant/four-seed source completion')
    records = list(data['source_controls'])
    for row in data['rows']:
        resolved=row['resolved'];contract=filter_contract(resolved['variant'])
        if (resolved.get('frontfilter')!=contract or resolved.get('frontfilter_actual')!=contract or
                resolved.get('frontfilter_active') is not True):
            raise ValueError('Actual frontfilter architecture differs from registered variant')
        seed=resolved['model_seed']
        initial=dict(status='SCRATCH',scratch_only=True,checkpoint=None,ancestors=[],checkpoint_sources=[],
            target_access=False,target_contact=False,model_seed=seed,physical_roles='EXACT_MATCH',selection='fixed_last_epoch')
        if row.get('initialization')!=initial:
            raise ValueError('Frontfilter source provenance is not own scratch')
        if any(row['completion'].get(k)!=v for k,v in dict(status='SOURCE_TRAINED',epoch=200,steps=10000,target_access=False,target_evaluated=False).items()):
            raise ValueError('Incomplete or target-contaminated source completion')
        audit=row.get('log_audit',{})
        if any(audit.get(k)!=v for k,v in dict(epochs=200,steps=10000,csv_epochs=200,all_step_flags_match=True,
                all_step_metrics_finite=True,step_epoch_csv_stdout_reconciled=True,errors=[],
                roles_match=True,full_source_record_reverified=True).items()):
            raise ValueError('Complete source step/log/provenance audit is missing')
        if [epoch['epoch'] for epoch in row['epochs']] != list(range(1, 201)):
            raise ValueError('Incomplete ordered source E1 through E200 telemetry')
        for epoch in row['epochs']:
            telemetry=epoch['frontfilter_diagnostics']
            validate_frontfilter_diagnostics(telemetry.get('frontfilter', {}), 28, contract)
            validate_diagnostic_input_scopes(telemetry)
            validate_neural_diagnostics(telemetry.get('neural_residual', {}),28)
        if row['completion']['final_source_metrics'] != {k: row['epochs'][-1][k] for k in row['completion']['final_source_metrics']}:
            raise ValueError('Final source metrics differ from E200')
        profile = row['profile']
        parameter_count=contract['total_trainable_parameters']
        if (profile['total_parameters'] != parameter_count or profile['gradient_used_parameters'] != parameter_count
                or profile['trainable_parameters'] != parameter_count
                or resolved.get('total_parameters')!=parameter_count or resolved.get('trainable_parameters')!=parameter_count
                or not math.isfinite(profile['conv_linear_macs_per_sample']) or profile['conv_linear_macs_per_sample']<=0
                or profile.get('all_identity_paths_shared_energy_normalization') is not False
                or profile.get('base_complex_paths_shared_energy_normalization') is not True
                or profile.get('filter_contract') != contract):
            raise ValueError('Actual measured resource accounting differs')
        diag = row['source_diagnostics']
        validate_frontfilter_groups(diag,contract)
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
        validate_frontfilter_diagnostics(phys.get('frontfilter', {}), 30, contract)
        validate_diagnostic_input_scopes(phys)
        validate_neural_diagnostics(phys.get('neural_residual', {}),30)
        expected_lag=row['resolved']['frontfilter_actual']['phase_lag']
        lifts=[(epoch['frontfilter_diagnostics'].get('adaptive_input',{}),28) for epoch in row['epochs']]+[(phys.get('adaptive_input',{}),30)]
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
                or phys.get('synthetic_only') is not True
                or any(phys.get(k) is not False for k in ('target_access','training_augmentation','formal_data_access','model_updated','hardware_parameter_recovery',
                    'whole_affine_phase_invariant_claim','arbitrary_channel_rx_invariant_claim','dynamic_map_invertible_claim','exact_wisig_equalizer'))):
            raise ValueError('Frozen synthetic-only physics contract differs')
        expected_pairs={(tx['name'],rx['name']) for tx in TX_ROWS for rx in RX_ROWS}
        if {(r.get('tx'),r.get('rx')) for r in phys['records']}!=expected_pairs:
            raise ValueError('Frozen synthetic TX/RX matrix differs')
        for record in phys['records']:
            distance=record.get('unit_embedding_distance_same_tx_identity_rx')
            logits=record.get('source_classifier_logits')
            if (not isinstance(distance,(int,float)) or isinstance(distance,bool) or not math.isfinite(distance) or distance<0 or
                    not isinstance(logits,list) or len(logits)!=6 or
                    any(isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v) for v in logits)):
                raise ValueError('Invalid frozen synthetic embedding/logit measurement')
        phases = phys.get('phase_audit', [])
        if (len(phases) != 3 or [r['theta_radians'] for r in phases] != [.37, -1.2, 2.9]
                or [r['received_cfo_hz'] for r in phases] != [0.,0.,0.]
                or any(not math.isfinite(r.get('whole_logit_max_abs_error',float('nan'))) or r['whole_logit_max_abs_error']<0 for r in phases)):
            raise ValueError('Missing or invalid measured whole-phase evidence')
        if (phys.get('phase_numeric_tolerance')!=1e-3 or
                phys.get('phase_tolerance_scope')!='Report only; not ranking or stopping' or
                phys.get('phase_tolerance_pass') is not all(r['whole_logit_max_abs_error']<1e-3 for r in phases)):
            raise ValueError('Frozen phase-tolerance reporting differs from measurements')
        # Numerical tolerance outcome is reported, never a source selection/stop gate.
        record=row['source_record'];final=row['completion']['final_source_metrics']
        if (record.get('variant')!=resolved['variant'] or record.get('seed')!=seed or
                record.get('accuracy')!=final.get('source_val_accuracy') or record.get('worst_rx')!=final.get('source_val_worst_rx') or
                record.get('parameters')!=parameter_count or record.get('macs')!=profile['conv_linear_macs_per_sample']):
            raise ValueError('Frozen source-ranking record differs from actual E200 evidence')
        records.append(row['source_record'])
    selection = select_source_candidate(records)
    if selection != data['source_selection']:
        raise ValueError('Stored source selection differs from independent ranking')
    return selection


def collect(root, output, run_id=None):
    from experiments.cvs_frontfilter_identity.prepare import PROJECT, RUN, RELEASE
    RUN=run_id or RUN
    from experiments.cvs_frontfilter_identity.publish import ssh
    from experiments.cvs_frontfilter_identity.dispatch import read_source_record, control_rows
    from experiments.cvs_frontfilter_identity.source import validate_config
    from experiments.cvs_frontfilter_identity.model import filter_contract,VARIANTS
    cfg = dict(project=PROJECT, run=RUN, controls=control_rows(), frontfilter={v:filter_contract(v) for v in VARIANTS},parameter_counts={v:220987+filter_contract(v)['new_trainable_parameters'] for v in VARIANTS},alignment_strengths={v:filter_contract(v)['alignment_strength'] for v in VARIANTS})
    script = (REMOTE.replace('cfg=CONFIG', 'cfg='+repr(cfg),1).replace('VALIDATE_SOURCE', repr(inspect.getsource(validate_config)))
              .replace('SOURCE_READER', repr(inspect.getsource(read_source_record)))
              .replace('FRONTFILTER_AUDITOR', repr(inspect.getsource(validate_frontfilter_diagnostics)))
              .replace('SCOPE_AUDITOR', repr(inspect.getsource(validate_diagnostic_input_scopes)))
              .replace('NEURAL_AUDITOR', repr(inspect.getsource(validate_neural_diagnostics)))
              .replace('LOG_AUDITOR', repr(inspect.getsource(audit_logs))))
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
    validation = dict(status='VERIFIED',new_rows=8,control_rows=8,new_epochs=1600,new_steps=80000,
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
