"""Read frozen source evidence; never load checkpoints, datasets or target scores."""
import argparse
import base64
import csv
import inspect
import io
import json
import math
from pathlib import Path
import statistics

import numpy as np

SOURCE_COMMIT = 'bb0ffae4263419a992a44de8975ec7eb83c4596f'
RELATION_SCALARS = (
    'branch_output_norm', 'base_frequency_norm', 'relative_output_change',
    'relation_frobenius_mean', 'relation_frobenius_max',
    'relation_trace_mean', 'relation_trace_max', 'floor_fraction',
    'projected_energy_mean', 'projected_energy_min',
    'determinant_floor_fraction', 'determinant_mean', 'determinant_min', 'alpha_mean', 'alpha_min',
)
COMPACT_RELATION_KEYS = (
    'relative_output_change_mean', 'relative_output_change_max', 'floor_fraction',
    'relation_norm_mean', 'relation_norm_max', 'relation_trace_mean', 'relation_trace_max',
    'mix_norm', 'projection_norm', 'mix_gradient_norm', 'projection_gradient_norm',
    'encoder_gradient_norm', 'determinant_floor_fraction', 'determinant_mean', 'determinant_minimum', 'alpha_mean', 'alpha_minimum',
)


def finite_number(value, label, lower=0., upper=None):
    if (isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value)
            or value < lower or (upper is not None and value > upper)):
        raise ValueError('Invalid measured '+label)
    return value


def validate_mirror_relation_diagnostics(diagnostic, expected_packets, expected_contract):
    from experiments.cvs_mirror_subspace_identity.model import relation_contract
    variant = diagnostic.get('variant')
    if expected_contract != relation_contract(variant):
        raise ValueError('Mirror relation diagnostic contract differs')
    if diagnostic.get('active') is not True or len(diagnostic.get('records', [])) != 1:
        raise ValueError('Missing active mirror relation branch')
    record = diagnostic['records'][0]
    fixed = dict(block='frequency.mirror_relation', packets=expected_packets, frames=7,
        frequency_pairs=31, output_dimension=160, use_subspace=variant=='mirror_subspace')
    if any(record.get(key) != value for key, value in fixed.items()):
        raise ValueError('Actual mirror relation scope differs')
    bound = expected_contract['relation_theoretical_frobenius_upper_bound']
    for key in COMPACT_RELATION_KEYS:
        if key not in record: raise ValueError('Missing mirror relation measurement: '+key)
        if key.endswith('gradient_norm') and record[key] is None: continue
        finite_number(record[key], key)
    finite_number(record['floor_fraction'], 'floor fraction', upper=1.)
    for stem in ('relative_output_change', 'relation_norm', 'relation_trace'):
        if record[stem+'_mean'] > record[stem+'_max']+1e-5:
            raise ValueError('Mirror relation mean exceeds maximum')
    for stem in ('relation_norm', 'relation_trace', 'floor_fraction', 'determinant_floor_fraction'):
        values = record.get(stem+'_by_frequency')
        if not isinstance(values, list) or len(values) != 31:
            raise ValueError('Incomplete spectral frequency telemetry')
        upper = bound+1e-5 if stem == 'relation_norm' else 1+1e-5
        for value in values: finite_number(value, stem, upper=upper)
        key = stem if stem.endswith('floor_fraction') else stem+'_mean'
        if not math.isclose(statistics.mean(values), record[key], rel_tol=2e-6, abs_tol=2e-6):
            raise ValueError('Frequency means differ from scalar telemetry')
        if not stem.endswith('floor_fraction') and max(values) > record[stem+'_max']+1e-5:
            raise ValueError('Frequency mean exceeds packet-frequency maximum')
    for suffix in ('mean', 'max'):
        finite_number(record['relation_norm_'+suffix], 'Q norm', upper=bound+1e-5)
        finite_number(record['relation_trace_'+suffix], 'Q trace', upper=1+1e-5)
        if not record['relation_trace_'+suffix]/math.sqrt(2)-2e-5 <= record['relation_norm_'+suffix] <= record['relation_trace_'+suffix]+2e-5:
            raise ValueError('Rank-two PSD Q norm and trace disagree')
    if not diagnostic.get('scope'): raise ValueError('Missing diagnostic scope')
    return record


def validate_public_physics(physics, contract):
    fixed=dict(schema='mirror_relation_public_physics_v1',synthetic_only=True,target_access=False,
               training_augmentation=False,optimizer_updates=0,public_packets=30)
    if any(physics.get(k)!=v for k,v in fixed.items()):raise ValueError('Public mirror physics scope differs')
    validate_mirror_relation_diagnostics(physics.get('mirror_relation',{}),30,contract)
    for name,total in (('ideal_mixing',186),('time_iq',558),('all_public_domain',930)):
        r=physics[name]
        count_keys=('total_pair_count','eligible_pair_count','excluded_pair_count','energy_excluded_pair_count',
            'determinant_excluded_pair_count','original_energy_floor_count','changed_energy_floor_count',
            'original_determinant_floor_count','changed_determinant_floor_count')
        if any(type(r.get(k)) is not int or not 0<=r[k]<=total for k in count_keys):raise ValueError('Invalid mirror domain count')
        if r['total_pair_count']!=total or r['eligible_pair_count']+r['excluded_pair_count']!=total:
            raise ValueError('Incomplete mirror qualification accounting')
        if not max(r['energy_excluded_pair_count'],r['determinant_excluded_pair_count'])<=r['excluded_pair_count']<=r['energy_excluded_pair_count']+r['determinant_excluded_pair_count']:
            raise ValueError('Invalid union of mirror exclusions')
        finite_number(r['full_max_error'],'full domain error');finite_number(r['full_relative_distance'],'full domain distance')
        for count,error in (('eligible_pair_count','qualified_max_error'),('excluded_pair_count','excluded_max_error')):
            if (r.get(error) is None)!=(r[count]==0):raise ValueError('Mirror domain N/A mismatch')
            if r[error] is not None:finite_number(r[error],error,upper=r['full_max_error']+1e-7)
    ideal=physics['ideal_mixing'];iq=physics['time_iq']
    finite_number(ideal['unitary_mixing_max_error'],'unitary error')
    finite_number(ideal['minimum_mixing_abs_determinant'],'mixing determinant',lower=1e-12)
    finite_number(iq['commutation_max_error'],'IQ commutation error')
    if iq['receiver_a']!=[1.04,.12] or iq['receiver_b']!=[.14,-.07]:raise ValueError('Public IQ action changed')
    phase=physics['whole_model_phase']
    finite_number(phase['ordinary_max_abs_error'],'whole model phase error')
    if phase.get('zero_finite') is not True:raise ValueError('Nonfinite public zero')
    firs=physics['finite_fir_sensitivity'];taps={f'delay{d}':[[0,1.,0.],[d,.23,.09]] for d in (2,8,16)}
    if len(firs)!=3 or {r['name'] for r in firs}!=set(taps):raise ValueError('Incomplete FIR matrix')
    for r in firs:
        if r['taps']!=taps[r['name']]:raise ValueError('Public FIR changed')
    sensitivity=physics['tx_rx_sensitivity']
    if set(sensitivity)!={'am_am','am_pm','rx_iq'}:raise ValueError('Incomplete nonlinear/IQ interventions')
    for r in [*firs,*sensitivity.values()]:
        finite_number(r['relation_relative_distance'],'public relation sensitivity')
        finite_number(r['mean_unit_embedding_distance'],'public embedding sensitivity',upper=2+1e-5)
    if len(physics.get('signal_groups',[]))!=5 or not physics.get('limits'):raise ValueError('Missing public scope')
    return dict(status='VERIFIED',eligible_pairs=ideal['eligible_pair_count'],
                excluded_pairs=ideal['excluded_pair_count'],outcome_used_for_selection=False)


def recount_source_scalars(arrays, expected_ids, diagnostics, contract):
    """Independent NumPy recount; no source.py aggregation implementation reuse."""
    expected_fields = {'ids','tx','receiver','day',*RELATION_SCALARS}
    if set(arrays) != expected_fields or any(a.shape != (27000,) for a in arrays.values()):
        raise ValueError('Expected complete per-packet scalar evidence without feature vectors')
    ids = arrays['ids']
    if (ids.dtype.kind not in 'US' or len(set(ids.tolist())) != 27000 or len(expected_ids) != 27000
            or len(set(expected_ids)) != 27000 or set(ids.tolist()) != set(expected_ids)):
        raise ValueError('Source V physical ID set differs')
    for key in ('tx','receiver','day'):
        if arrays[key].dtype.kind not in 'iu': raise ValueError('Invalid physical grouping dtype')
    for key in RELATION_SCALARS:
        a = arrays[key]
        if a.dtype != np.float64 or not np.isfinite(a).all() or (a < 0).any():
            raise ValueError('Nonfinite or invalid FP32-measurement scalar storage')
    if (arrays['floor_fraction'] > 1).any(): raise ValueError('Invalid floor fraction')
    bound = contract['relation_theoretical_frobenius_upper_bound']
    for stem in ('relation_frobenius','relation_trace'):
        if (arrays[stem+'_max'] > (bound if stem=='relation_frobenius' else 1)+1e-5).any() or (arrays[stem+'_mean'] > arrays[stem+'_max']+1e-5).any():
            raise ValueError('Per-packet Q norm/trace bounds differ')
    for suffix in ('mean','max'):
        if ((arrays['relation_frobenius_'+suffix]<arrays['relation_trace_'+suffix]/math.sqrt(2)-2e-5) | (arrays['relation_frobenius_'+suffix]>arrays['relation_trace_'+suffix]+2e-5)).any():
            raise ValueError('Packet rank-two PSD Q trace/norm mismatch')
    if not np.allclose(arrays['relative_output_change'],
            arrays['branch_output_norm']/np.maximum(arrays['base_frequency_norm'],1e-12), rtol=2e-6, atol=1e-7):
        raise ValueError('Relative branch norm cannot be reproduced')
    if (arrays['projected_energy_min'] > arrays['projected_energy_mean']+1e-5).any():
        raise ValueError('Projected-energy minimum exceeds mean')
    for key in ('floor_fraction','determinant_floor_fraction','alpha_mean','alpha_min'):
        if (arrays[key]>1+1e-6).any():raise ValueError('Invalid floor/alpha bound')
    if (arrays['determinant_min']>arrays['determinant_mean']+1e-6).any() or (arrays['alpha_min']>arrays['alpha_mean']+1e-6).any():
        raise ValueError('Minimum exceeds mean for mirror determinant/alpha')
    expected_cells = {(t,r,d) for t in range(6) for r in (1,3,4,6,8) for d in (1,2,3)}
    actual_cells = set(zip(arrays['tx'].tolist(),arrays['receiver'].tolist(),arrays['day'].tolist()))
    rows = diagnostics.get('mirror_relation_groups',[])
    indexed = {(r['tx'],r['receiver'],r['day']):r for r in rows}
    if actual_cells != expected_cells or len(rows)!=90 or set(indexed)!=expected_cells:
        raise ValueError('Source relation evidence does not cover exactly 90 registered cells')
    for cell in sorted(expected_cells):
        mask = (arrays['tx']==cell[0]) & (arrays['receiver']==cell[1]) & (arrays['day']==cell[2])
        row = indexed[cell]
        expected_keys = {'tx','receiver','day','count',*(key+'_'+suffix for key in RELATION_SCALARS for suffix in ('mean','min','max'))}
        if int(mask.sum())!=300 or row.get('count')!=300 or set(row)!=expected_keys:
            raise ValueError('Incomplete 300-packet cell or changed summary fields')
        for key in RELATION_SCALARS:
            values = arrays[key][mask]
            for suffix,value in (('mean',values.mean()),('min',values.min()),('max',values.max())):
                measured = finite_number(row[key+'_'+suffix], 'cell '+key)
                if not math.isclose(measured,float(value),rel_tol=1e-12,abs_tol=1e-12):
                    raise ValueError('Full V cell scalar summary differs from NPZ recount')
    meta = diagnostics.get('mirror_relation_scalars',{})
    if (meta.get('schema')!='mirror_relation_source_scalars_v1' or meta.get('count')!=27000
            or len(meta.get('fields',[]))!=len(expected_fields) or set(meta['fields'])!=expected_fields
            or any(meta.get(k) is not False for k in ('target_access','used_for_training','used_for_selection'))
            or not all(meta.get(k) for k in ('scope','dtype','base_frequency_norm_scope','projected_energy_scope','floor_fraction_scope'))):
        raise ValueError('Full source scalar schema/scope differs')
    order = np.argsort(ids)
    metadata = [(str(ids[i]),int(arrays['tx'][i]),int(arrays['receiver'][i]),int(arrays['day'][i])) for i in order]
    return dict(status='VERIFIED',packets=27000,cells=90,scalars_per_packet=len(RELATION_SCALARS),
        all_cell_mean_min_max_recomputed=True,physical_id_set_matched=True,
        scope='Stored grouping coordinates and cross-model ID pairing; opaque IDs are not independently decoded'), metadata


def validate_source_geometry(diag, final_metrics):
    rows=diag.get('groups',[])
    expected={(t,r,d) for t in range(6) for r in (1,3,4,6,8) for d in (1,2,3)}
    if (diag.get('role')!='V' or diag.get('count')!=27000 or len(rows)!=90
            or {(r['tx'],r['receiver'],r['day']) for r in rows}!=expected
            or any(r.get('count')!=300 for r in rows)
            or any(diag.get(k) is not False for k in ('target_access','used_for_training','used_for_selection'))):
        raise ValueError('Final source geometry does not cover the registered V')
    for row in rows:
        finite_number(row.get('accuracy'),'source cell accuracy',upper=1.)
        if not math.isclose(row['accuracy']*300,round(row['accuracy']*300),abs_tol=1e-9):
            raise ValueError('Source cell accuracy is not a 300-packet count')
        centroid=row.get('unit_embedding_mean',[])
        if len(centroid)!=160: raise ValueError('Source centroid dimension changed')
        for value in centroid: finite_number(value,'source centroid',lower=-1.-1e-5,upper=1.+1e-5)
        finite_number(row.get('unit_embedding_trace_variance'),'source within-cell variance',upper=1.+1e-5)
        if row.get('alignment_strength')!=0.: raise ValueError('Raw waveform alignment changed')
        for key in ('relative_cfo_hz_mean','relative_cfo_hz_min','relative_cfo_hz_max',
                    'residual_estimated_cfo_hz_mean','nominal_residual_cfo_hz_mean'):
            finite_number(row.get(key),key,lower=-625001.,upper=625001.)
        if not row['relative_cfo_hz_min']<=row['relative_cfo_hz_mean']<=row['relative_cfo_hz_max']:
            raise ValueError('Source CFO extrema differ')
        for key in ('fallback_fraction','coherence_mean','residual_valid_fraction'):
            finite_number(row.get(key),key,upper=1.+1e-6)
        eligible=row.get('residual_formula_eligible_count')
        if type(eligible) is not int or not 0<=eligible<=300: raise ValueError('Invalid CFO eligibility')
        error=row.get('residual_formula_max_error_hz')
        if (error is None)!=(eligible==0): raise ValueError('CFO missing-value scope differs')
        if error is not None: finite_number(error,'CFO formula error')
    if not math.isclose(statistics.mean(r['accuracy'] for r in rows),final_metrics['source_val_accuracy'],abs_tol=1e-12):
        raise ValueError('Final full-V cells and E200 V accuracy differ')
    for rx in (1,3,4,6,8):
        actual=statistics.mean(r['accuracy'] for r in rows if r['receiver']==rx)
        if not math.isclose(actual,final_metrics['source_val_rx_accuracy'][str(rx)],abs_tol=1e-12):
            raise ValueError('Final full-V cells and E200 RX accuracy differ')
    for key in ('mean_between_tx_centroid_squared_distance','mean_within_tx_rx_centroid_squared_distance'):
        finite_number(diag.get(key),key)

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


def audit_logs(epochs, steps, csv_text, stdout, expected_parameters, expected_alignment_strength, compact_jsonl=None, expected_phase_lag=None, expected_mirror_relation_contract=None):
    """Reconcile every step, epoch, compact CSV and detailed text record."""
    if expected_mirror_relation_contract is None:raise ValueError('Missing actual registered mirror_relation contract')
    if expected_parameters!=expected_mirror_relation_contract['total_trainable_parameters']:
        raise ValueError('Total mirror_relation parameter contract mismatch')
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
                 pseudo_labels_active=False, extra_losses_active=False, mirror_relation_active=True,
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
        expected_lr=1e-6+(.0002-1e-6)*(1+math.cos(math.pi*(number-1)/200))/2
        if not math.isclose(epoch['learning_rate'],expected_lr,rel_tol=1e-10,abs_tol=1e-14):
            raise ValueError('Original cosine schedule differs')
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
        new_parameters=expected_mirror_relation_contract['new_trainable_parameters']
        if epoch.get('mirror_relation_gradient_used_parameters') != new_parameters:
            raise ValueError('Mirror relation gradient parameter budget differs')
        if any(not math.isfinite(step.get('mirror_relation_gradient_norm',float('nan'))) or step['mirror_relation_gradient_norm']<0 or step['mirror_relation_gradient_used_parameters']!=new_parameters for step in batch_steps):raise ValueError('Missing measured mirror_relation CE gradients')
        if not math.isclose(statistics.mean(step['mirror_relation_gradient_norm'] for step in batch_steps),epoch['mirror_relation_gradient_norm'],rel_tol=1e-12,abs_tol=1e-12):raise ValueError('Mirror relation gradient mean mismatch')
        diag = epoch['mirror_relation_diagnostics']
        record=validate_mirror_relation_diagnostics(diag.get('mirror_relation', {}), 28, expected_mirror_relation_contract)
        validate_neural_diagnostics(diag.get('neural_residual', {}),28)
        for key in ('mix_gradient_norm','projection_gradient_norm','encoder_gradient_norm'):
            if record[key] is None:raise ValueError('Missing last CE mirror relation gradient measurement')
        for key in COMPACT_RELATION_KEYS:
            if epoch.get('mirror_relation_'+key)!=record[key]:
                raise ValueError('Compact epoch spectral measurement differs from full diagnostics')
        if epoch.get('energy_diagnostic_scope')!='last source batch of epoch (28 packets); not complete source V':
            raise ValueError('Last-batch diagnostics cannot be presented as complete source V')
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
import base64,json,time,csv,io,math,statistics,sys
from pathlib import Path
cfg=CONFIG
if (Path.cwd()/'release_commit.txt').read_text().strip()!=cfg['source_commit']:
    raise ValueError('Source release commit differs')
exec(AUDIT_CODE,globals())
from experiments.cvs_mirror_subspace_identity.dispatch import read_source_record
p=Path(cfg['project'])/'runs'/cfg['run']
def read(q):return json.loads(q.read_text(encoding='utf-8'))
state=read(p/'pipeline_state.json')
if state['status'] not in ('SOURCE_RESEARCH_COMPLETE_AWAITING_FROZEN_CLEAN_TEST','SOURCE_RESEARCH_COMPLETE_BASELINE_RETAINED'):
    print(json.dumps(dict(status=state['status'],ready=False)))
else:
    if state.get('commit')!=cfg['source_commit'] or state.get('run_id')!=cfg['run'] or state.get('target_access') is not False:
        raise ValueError('Pipeline source identity differs')
    if set(state['rows'])!={r['row_id'] for r in cfg['rows']}:
        raise ValueError('Pipeline row matrix differs')
    original=read(Path(cfg['project'])/'runs/phase1_daot_rc4_pure_game_m3_20260917_r2/source_contract.json')
    controls=[read_source_record(r,original,r['method']) for r in cfg['controls']]
    result=dict(ready=True,read_at=time.time(),run_id=cfg['run'],pipeline=state,
        source_selection=read(p/'source_selection.json'),source_selection_inputs=read(p/'source_selection_inputs.json'),
        source_controls=controls,source_v_ids=original['role_ids']['V'],rows=[])
    for registered in cfg['rows']:
        rid=registered['row_id'];row=state['rows'][rid]
        q=p/rid/'source';log=Path(cfg['project'])/'logs'/cfg['run']/(rid+'.log')
        if (row.get('status')!='SOURCE_TRAINED' or row.get('exit_code')!=0 or row.get('source_output')!=str(q)
                or row.get('log')!=str(log) or row.get('variant')!=registered['variant']
                or row.get('model_seed')!=registered['model_seed']):
            raise ValueError('Terminal source row paths/status differ')
        actual=read_source_record(dict(source_output=str(q),model_seed=row['model_seed'],variant=row['variant']),original,'cvs_mirror_subspace_identity')
        resolved=read(q/'resolved_config.json')
        if resolved.get('commit')!=cfg['source_commit']:raise ValueError('Actual source training commit differs')
        epochs=[json.loads(s) for s in (q/'epoch_metrics.jsonl').read_text(encoding='utf-8').splitlines()]
        steps=[json.loads(s) for s in (q/'step_metrics.jsonl').read_text(encoding='utf-8').splitlines()]
        stdout=log.read_text(encoding='utf-8',errors='strict')
        printed=[json.loads(line[len('RESOLVED_CONFIG '):]) for line in stdout.splitlines() if line.startswith('RESOLVED_CONFIG ')]
        if printed!=[resolved]:raise ValueError('Printed/runtime resolved configuration differs')
        contract=cfg['contracts'][row['variant']]
        audit=audit_logs(epochs,steps,(q/'epoch_metrics.csv').read_text(encoding='utf-8'),stdout,
            contract['total_trainable_parameters'],0.,(q/'epoch_compact.jsonl').read_text(encoding='utf-8'),
            contract['phase_lag'],contract)
        audit.update(roles_match=True,full_source_record_reverified=True,resolved_stdout_matched=True)
        physics=read(q/'source_physical_diagnostics.json')
        printed_physics=[json.loads(line[len('FROZEN_PHYSICS '):]) for line in stdout.splitlines() if line.startswith('FROZEN_PHYSICS ')]
        if printed_physics!=[physics]:raise ValueError('Printed/frozen public physics differs')
        result['rows'].append(dict(row_id=rid,source_record=actual,log_audit=audit,epochs=epochs,
            completion=read(q/'completion.json'),initialization=read(q/'initialization.json'),resolved=resolved,
            profile=read(q/'resource_profile.json'),source_diagnostics=read(q/'source_final_diagnostics.json'),
            physical_diagnostics=physics,
            scalar_npz_base64=base64.b64encode((q/'source_mirror_relation_scalars.npz').read_bytes()).decode('ascii')))
    print(json.dumps(result,allow_nan=False))
'''


def validate_completed(data):
    from experiments.cvs_mirror_subspace_identity.dispatch import select_source_candidate, control_rows, SEEDS, PROJECT
    from experiments.cvs_mirror_subspace_identity.model import VARIANTS, relation_contract
    from experiments.cvs_mirror_subspace_identity.prepare import RUN
    from experiments.cvs_mirror_subspace_identity.source import validate_config
    from experiments.cvs_equivariant_identity.precision import FULL_FP32_POLICY
    if data.get('ready') is not True or data.get('run_id')!=RUN:
        raise ValueError('Unexpected source completion identity')
    state=data.get('pipeline',{})
    if state.get('commit')!=SOURCE_COMMIT or state.get('target_access') is not False or state.get('run_id')!=RUN:
        raise ValueError('Source pipeline origin differs')
    rows=data.get('rows',[])
    expected={(v,s) for v in VARIANTS for s in SEEDS}
    if len(rows)!=8 or {(r['resolved']['variant'],r['resolved']['model_seed']) for r in rows}!=expected:
        raise ValueError('Incomplete two-variant/four-seed source matrix')
    controls=data.get('source_controls',[])
    expected_controls={(r['variant'],r['model_seed'],r['source_output']) for r in control_rows()}
    if len(controls)!=12 or {(r['variant'],r['seed'],r['source_output']) for r in controls}!=expected_controls:
        raise ValueError('Control metadata does not match the twelve registered frozen controls')
    records=list(controls);reference_metadata=None
    for row in rows:
        resolved=row['resolved'];variant=resolved['variant'];seed=resolved['model_seed']
        contract=relation_contract(variant);validate_config(resolved)
        rid=variant+'-s'+str(seed);folder=PROJECT+'/runs/'+RUN+'/'+rid+'/source'
        if (row.get('row_id')!=rid or resolved.get('output_root')!=folder or resolved.get('commit')!=SOURCE_COMMIT
                or resolved.get('mirror_relation_actual')!=contract or resolved.get('mirror_relation_active') is not True
                or resolved.get('precision')!='float32' or resolved.get('backend_flags')!=FULL_FP32_POLICY
                or resolved.get('source_counts')!={'L_s':6300,'U_s':56700,'V':27000}
                or resolved.get('steps_per_epoch')!=50 or resolved.get('U_s_use')!='unused'
                or resolved.get('gradient_clipping') is not None or resolved.get('optimizer')!='AdamW+CosineAnnealingLR'
                or resolved.get('loader_seed')!=seed or resolved.get('classifier_scale')!=30.):
            raise ValueError('Actual source training/provenance contract differs')
        initial=dict(status='SCRATCH',scratch_only=True,checkpoint=None,ancestors=[],checkpoint_sources=[],
            target_access=False,target_contact=False,model_seed=seed,physical_roles='EXACT_MATCH',selection='fixed_last_epoch')
        if row.get('initialization')!=initial:raise ValueError('Source initialization is not own scratch')
        done=row['completion']
        if (any(done.get(k)!=v for k,v in dict(status='SOURCE_TRAINED',epoch=200,steps=10000,
                target_access=False,target_evaluated=False,backend_flags=FULL_FP32_POLICY).items())
                or done.get('checkpoint')!=folder+'/last.pt'):
            raise ValueError('Source completion/checkpoint destination differs')
        audit=row.get('log_audit',{})
        if any(audit.get(k)!=v for k,v in dict(epochs=200,steps=10000,csv_epochs=200,all_step_flags_match=True,
                all_step_metrics_finite=True,step_epoch_csv_stdout_reconciled=True,errors=[],roles_match=True,
                full_source_record_reverified=True,resolved_stdout_matched=True).items()):
            raise ValueError('Complete 10000-step source audit is absent')
        epochs=row['epochs']
        if len(epochs)!=200 or [e['epoch'] for e in epochs]!=list(range(1,201)):
            raise ValueError('Incomplete full source epoch history')
        for epoch in epochs:
            validate_mirror_relation_diagnostics(epoch['mirror_relation_diagnostics']['mirror_relation'],28,contract)
        final=done['final_source_metrics']
        metric_keys={'source_val_count','source_val_accuracy','source_val_ce','source_val_rx_accuracy','source_val_worst_rx'}
        if set(final)!=metric_keys or final!={k:epochs[-1][k] for k in metric_keys}:
            raise ValueError('Completion metrics differ from E200')
        profile=row['profile'];parameters=contract['total_trainable_parameters']
        if (any(profile.get(k)!=parameters for k in ('total_parameters','trainable_parameters','gradient_used_parameters'))
                or any(resolved.get(k)!=parameters for k in ('total_parameters','trainable_parameters'))
                or profile.get('mirror_relation_contract')!=contract
                or profile.get('all_identity_paths_shared_energy_normalization') is not False
                or profile.get('base_complex_paths_shared_energy_normalization') is not True):
            raise ValueError('Measured resource/normalization accounting differs')
        finite_number(profile.get('conv_linear_macs_per_sample'),'measured MACs',lower=1.)
        for key in ('resident_state_bytes','inference_batch1_ms','training_batch128_ms'):
            finite_number(profile.get(key),'measured '+key)
        if not profile.get('hardware') or not profile.get('torch_version'):
            raise ValueError('Resource measurements lack their execution environment')
        validate_source_geometry(row['source_diagnostics'],final)
        if row['source_diagnostics']['mirror_relation_scalars'].get('path')!=folder+'/source_mirror_relation_scalars.npz':
            raise ValueError('Scalar source path escapes registered row')
        if 'scalar_npz_base64' in row:
            raw=base64.b64decode(row['scalar_npz_base64'],validate=True)
        else:
            raw=Path(row['scalar_evidence']['path']).read_bytes()
        with np.load(io.BytesIO(raw),allow_pickle=False) as npz:
            arrays={key:npz[key] for key in npz.files}
        scalar_audit,metadata=recount_source_scalars(arrays,data['source_v_ids'],row['source_diagnostics'],contract)
        if reference_metadata is None:reference_metadata=metadata
        elif metadata!=reference_metadata:raise ValueError('Same physical V IDs have different metadata across models')
        row['scalar_audit']=dict(scalar_audit,bytes=len(raw),cross_model_metadata_matched=True)
        row['physics_audit']=validate_public_physics(row['physical_diagnostics'],contract)
        record=row['source_record']
        if (record.get('variant')!=variant or record.get('seed')!=seed or record.get('source_output')!=folder
                or record.get('accuracy')!=final['source_val_accuracy'] or record.get('worst_rx')!=final['source_val_worst_rx']
                or record.get('parameters')!=parameters or record.get('macs')!=profile['conv_linear_macs_per_sample']):
            raise ValueError('Source ranking record differs from actual E200 evidence')
        records.append(record)
    key=lambda r:(r['variant'],r['seed'])
    if sorted(data.get('source_selection_inputs',[]),key=key)!=sorted(records,key=key):
        raise ValueError('Stored source selection inputs differ from independently read metadata')
    selection=select_source_candidate(records)
    if selection!=data['source_selection']:raise ValueError('Stored source selection differs from independent ranking')
    expected_status=('SOURCE_RESEARCH_COMPLETE_AWAITING_FROZEN_CLEAN_TEST' if selection['new_candidate_selected']
                     else 'SOURCE_RESEARCH_COMPLETE_BASELINE_RETAINED')
    if state.get('status')!=expected_status:raise ValueError('Terminal status differs from frozen source choice')
    return selection


def remote_script():
    from experiments.cvs_mirror_subspace_identity.prepare import PROJECT, RUN
    from experiments.cvs_mirror_subspace_identity.dispatch import control_rows
    from experiments.cvs_mirror_subspace_identity.model import VARIANTS, relation_contract
    spec=json.loads((Path(__file__).parent/'configs/launch_spec.json').read_text(encoding='utf-8'))
    cfg=dict(project=PROJECT,run=RUN,source_commit=SOURCE_COMMIT,controls=control_rows(),
             rows=spec['rows'],contracts={v:relation_contract(v) for v in VARIANTS})
    functions=(finite_number,validate_mirror_relation_diagnostics,validate_neural_diagnostics,audit_logs)
    code='COMPACT_RELATION_KEYS='+repr(COMPACT_RELATION_KEYS)+'\n'+ '\n'.join(inspect.getsource(f) for f in functions)
    return REMOTE.replace('cfg=CONFIG','cfg='+repr(cfg),1).replace('AUDIT_CODE',repr(code))


def write_once_or_identical(path, raw):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():
        if path.read_bytes()!=raw:raise FileExistsError('Refusing to overwrite different evidence: '+str(path))
    else:
        with path.open('xb') as stream:stream.write(raw)


def save_evidence(data, root, output):
    """Large evidence stays in local_artifacts; report receives compact audit only."""
    from experiments.cvs_mirror_subspace_identity.prepare import RUN
    root=Path(root).resolve();output=Path(output).resolve()
    artifacts=root/'local_artifacts'
    if not output.is_relative_to(artifacts) or output==artifacts:
        raise ValueError('Collector large evidence must use a dedicated local_artifacts subdirectory')
    selection=validate_completed(data)
    manifest=[]
    for row in data['rows']:
        raw=base64.b64decode(row.pop('scalar_npz_base64'),validate=True)
        path=output/'source_scalar_npz'/(row['row_id']+'.npz')
        write_once_or_identical(path,raw)
        row['scalar_evidence']=dict(path=str(path),bytes=len(raw),schema='mirror_relation_source_scalars_v1',git_tracked=False)
        manifest.append(dict(row_id=row['row_id'],**row['scalar_evidence']))
    validation=dict(status='VERIFIED',source_commit=SOURCE_COMMIT,new_rows=8,control_rows=12,
        new_epochs=1600,new_steps=80000,source_scalar_packets=216000,source_scalar_cells=720,
        full_stdout_scanned=True,step_epoch_csv_stdout_reconciled=True,full_source_records_reverified=True,
        source_rule_recomputed=True,source_scalar_cells_recomputed=True,cross_model_metadata_matched=True,
        target_access=False,source_selection=selection,
        scalar_metadata_limit='Exact V ID set and matching stored TX/RX/day across rows; no independent decoding of opaque IDs')
    encode=lambda value:(json.dumps(value,ensure_ascii=False,allow_nan=False,indent=2)+'\n').encode('utf-8')
    complete_path=output/'source_research_complete.json'
    write_once_or_identical(complete_path,encode(data))
    evidence_manifest=dict(source_commit=SOURCE_COMMIT,complete_json=dict(path=str(complete_path),bytes=complete_path.stat().st_size,git_tracked=False),scalar_npz=manifest)
    for directory in (output,root/'automation_reports/CV-SincNet'/RUN/'evidence'):
        for name,value in [('source_selection.json',selection),('source_completion_validation.json',validation),
                           ('source_evidence_manifest.json',evidence_manifest)]:
            write_once_or_identical(directory/name,encode(value))
    return validation


def collect(root, output, run_id=None):
    from experiments.cvs_mirror_subspace_identity.prepare import PROJECT, RUN, RELEASE
    from experiments.cvs_mirror_subspace_identity.publish import ssh
    if run_id not in (None,RUN):raise ValueError('Collector is bound to the registered source run')
    release=PROJECT+'/releases/'+RELEASE
    command=['/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python','-c',remote_script()]
    wrapper='import os,subprocess\nsubprocess.run('+repr(command)+',cwd='+repr(release)+',env=dict(os.environ,PYTHONPATH='+repr(release+':'+release+'/code')+'),check=True)\n'
    data=json.loads(ssh(wrapper))
    if not data['ready']:
        print(json.dumps(data));return False
    validation=save_evidence(data,root,output)
    print(json.dumps(validation,ensure_ascii=False));return True


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();collect(a.root,a.output)
