"""Analyze completed frozen-F-task compact R17 aggregates without IQ or predictions."""
import math
import numpy as np

from experiments.pamf_r17_task import design as d
from experiments.pamf_r17_task.evaluation import select_source
from experiments.pamf_r17_task.calibration import validate_calibration


def classification_record(record):
    matrix=np.asarray(record['confusion'])
    if matrix.shape!=(6,6) or matrix.dtype.kind not in 'iu' or (matrix<0).any():
        raise ValueError('Nonnegative integer six-class confusion required')
    count=int(matrix.sum());correct=int(matrix.trace())
    if count<=0 or record['count']!=count:raise ValueError('Confusion count differs')
    precision=np.divide(matrix.diagonal(),matrix.sum(0),out=np.zeros(6),where=matrix.sum(0)>0)
    recall=np.divide(matrix.diagonal(),matrix.sum(1),out=np.zeros(6),where=matrix.sum(1)>0)
    f1=np.divide(2*precision*recall,precision+recall,out=np.zeros(6),where=precision+recall>0)
    accuracy=100*correct/count;macro_f1=100*float(f1.mean())
    for key,expected in (('accuracy',accuracy),('macro_f1',macro_f1)):
        if not math.isclose(record[key],expected,rel_tol=0,abs_tol=1e-9):raise ValueError('Invalid '+key)
    old_value=record['baseline_accuracy']*count/100;old_correct=round(old_value)
    if not 0<=old_correct<=count or abs(old_value-old_correct)>1e-6:raise ValueError('Invalid baseline count')
    repaired,damaged,net=(record[k] for k in ('repaired','damaged','net_repaired'))
    if (any(not isinstance(v,int) for v in (repaired,damaged,net)) or repaired<0 or damaged<0
            or repaired>count-old_correct or damaged>old_correct or repaired-damaged!=net or correct-old_correct!=net):
        raise ValueError('Invalid paired repair arithmetic')
    result=dict(count=count,correct=correct,accuracy=accuracy,macro_f1=macro_f1,baseline_accuracy=record['baseline_accuracy'],
                repaired=repaired,damaged=damaged,net_repaired=net,confusion=matrix.tolist())
    if 'iq_nmse' in record:
        if not math.isfinite(record['iq_nmse']) or record['iq_nmse']<0:raise ValueError('Finite grouped waveform NMSE required')
        result['iq_nmse']=record['iq_nmse']
    return result


def summary_metrics(summary,role,training_F_access=False):
    count=1800 if role=='source' else 168000
    expected=dict(status='ANALYZED',role=role,count=count,F_seed=2026092701,original_default_head=True,
                  F_run='20261016-phase1-F-supcon-e50-clean-manysig-m9-r02',target_access=role=='target',
                  all_predictions_before_truth=True,training_F_access=training_F_access,training_TX_labels_read=0,head_fit_updates=0)
    if role not in ('source','target') or any(summary.get(k)!=v for k,v in expected.items()):
        raise ValueError('Completed same-role same-head summary required')
    if set(summary['views'])!=set(d.VIEWS):raise ValueError('All seven views required')
    views={};checked=0
    for view in d.VIEWS:
        records=[r for r in summary['records'] if r['view']==view]
        keys=[(r['dimension'],r['value'],r['readout']) for r in records]
        if len(set(keys))!=len(keys) or any(r['readout']!='restored' for r in records):raise ValueError('Unique restored groups required')
        overall=[r for r in records if r['dimension']=='overall' and r['value']=='all']
        if len(overall)!=1 or overall[0]['count']!=count:raise ValueError('Complete overall count required')
        metrics=classification_record(overall[0]);checked+=1;groups={}
        for dimension in ('TX','RX','day'):
            rows=[r for r in records if r['dimension']==dimension]
            if not rows or (dimension=='TX' and len(rows)!=6):raise ValueError('Complete physical grouping required')
            group_values={str(r['value']):classification_record(r) for r in rows};checked+=len(rows)
            if sum(v['count'] for v in group_values.values())!=count or not np.array_equal(
                    sum((np.asarray(r['confusion']) for r in rows),start=np.zeros((6,6),np.int64)),overall[0]['confusion']):
                raise ValueError('Group confusions must partition the overall physical samples')
            groups[dimension]=group_values
        wave=summary['views'][view]['restored']['waveform']
        for key in ('iq_nmse','prefix232_nmse','tail24_nmse','fft_log_power_L1'):
            if not math.isfinite(wave[key]) or wave[key]<0:raise ValueError('Finite paired waveform metrics required')
        geometry={key:summary['views'][view]['restored'][key] for key in
                  ('paired_clean_feature_cosine_distance','paired_clean_vs_other_TX') if key in summary['views'][view]['restored']}
        views[view]=dict(**metrics,waveform=wave,paired_clean_geometry=geometry,groups=groups,
                        worst={dimension:min(values,key=lambda key:(values[key]['accuracy'],key)) for dimension,values in groups.items()})
    return dict(role=role,count=count,views=views,independently_recomputed_records=checked)


def compare(candidate,baseline):
    if candidate['role']!=baseline['role'] or candidate['count']!=baseline['count']:raise ValueError('Same physical evaluation scope required')
    result={}
    for view in d.VIEWS:
        left,right=candidate['views'][view],baseline['views'][view]
        if any(set(left['groups'][k])!=set(right['groups'][k]) for k in ('TX','RX','day')):raise ValueError('Same physical groups required')
        if any(left['groups'][dim][key]['count']!=right['groups'][dim][key]['count'] for dim in ('TX','RX','day') for key in left['groups'][dim]):
            raise ValueError('Same physical group counts required')
        result[view]=dict(correct_delta=left['correct']-right['correct'],
                         accuracy_delta_pp=100*(left['correct']-right['correct'])/candidate['count'],
                         macro_f1_delta_pp=left['macro_f1']-right['macro_f1'],
                         accuracy_nonregression=left['correct']>=right['correct'],
                         waveform_delta={k:left['waveform'][k]-right['waveform'][k] for k in left['waveform']},
                         group_accuracy_delta_pp={dim:{key:left['groups'][dim][key]['accuracy']-right['groups'][dim][key]['accuracy']
                            for key in left['groups'][dim]} for dim in ('TX','RX','day')})
    return dict(accuracy_nonregression=all(v['accuracy_nonregression'] for v in result.values()),views=result)


def training_metrics(records,complete,arm):
    if arm not in d.ARMS or len(records)!=20 or [r['epoch'] for r in records]!=list(range(1,21)):
        raise ValueError('All twenty registered training records required')
    previous=0.
    names=('loss','iq_nmse','teacher_stage_nmse','task_KL','task_feature_distance')
    for epoch,row in enumerate(records,1):
        if (row['updates'],row['L_exposures'],row['U_exposures'],row['teacher_IQ_examples'])!=(247*epoch,6300*epoch,56700*epoch,63000*epoch):
            raise ValueError('Full source exposure required')
        if not math.isfinite(row['seconds']) or not row['seconds']>previous or not 0<=row['clip_fraction']<=1:
            raise ValueError('Finite monotone training time and clip fraction required')
        previous=row['seconds']
        if set(row['losses'])!=set(names) or any(not math.isfinite(v) or v<0 for v in row['losses'].values()):
            raise ValueError('All finite registered waveform and task losses required')
        expected=10*row['losses']['teacher_stage_nmse']+row['losses']['task_KL']+row['losses']['task_feature_distance']
        if not math.isclose(row['losses']['loss'],expected,rel_tol=1e-5,abs_tol=1e-7):
            raise ValueError('Actual objective differs from registered task loss')
    cal=512 if arm=='activation_task' else 0
    expected=dict(status='SOURCE_TRAINED',arm=arm,epochs=20,updates=4940,L_exposures=126000,U_exposures=1134000,
        teacher_IQ_examples=1260000,inherited_prior_epochs=20,cumulative_prior_epochs=40,cumulative_serial_training_passes=80,
        parameters=d.PARAMETERS[arm],trainable_parameters=d.PARAMETERS[arm]-2157547,
        calibration_teacher_IQ_examples=cal,total_teacher_IQ_examples=1260000+cal,
        teacher_parameter_updates=0,frozen_context_updates=0,TX_labels_read=0,F_outputs_read=2520000,F_weights_read=1,
        training_F_access=True,F_teacher_forward_examples=1260000,F_student_forward_examples=1260000,
        F_parameter_updates=0,head_fit_updates=0,V_reads=0,target_reads=0)
    if any(complete.get(k)!=v for k,v in expected.items()):
        raise ValueError('Final budget/permissions/Fknowledge or parameter count differs')
    if (any(not isinstance(complete.get(k),int) or complete[k]<0 for k in ('calibration_L_exposures','calibration_U_exposures'))
            or complete['calibration_L_exposures']+complete['calibration_U_exposures']!=cal
            or not math.isfinite(complete.get('calibration_seconds',float('nan'))) or complete['calibration_seconds']<=0):
        raise ValueError('Actual arm-specific calibration exposure and initialization cost required')
    losses={}
    for name in names:
        values=np.asarray([r['losses'][name] for r in records],np.float64)
        losses[name]=dict(first=float(values[0]),last=float(values[-1]),minimum=float(values.min()),maximum=float(values.max()),
                          mean=float(values.mean()),slope_per_epoch=float(np.polyfit(np.arange(1,21),values,1)[0]))
    intervals=np.diff([0.,*[r['seconds'] for r in records]])
    return dict(full_log_records=20,losses=losses,epoch_seconds_median=float(np.median(intervals)),
                first_epoch_seconds=float(intervals[0]),last_epoch_seconds=float(intervals[-1]),
                training_complete=complete,checkpoint_selection='fixedfinalE20,notmintrainingloss')


def profile_metrics(profile,arm):
    if (profile['status']!='PROFILE_COMPLETE' or not profile['includes_full_frozen_context_and_CG']
            or profile['source_V_IQ']!=16 or profile['TX_labels_read']!=profile['state_updates'] or profile['state_updates']!=0):
        raise ValueError('Same-GPU full inference source-only profile required')
    original,student=(profile['results'][key] for key in ('original_R17','compact'))
    for item,parameters in ((original,4665073),(student,d.PARAMETERS[arm])):
        if item['samples']!=16 or item['warmup']!=2 or item['parameters']!=parameters:
            raise ValueError('Matched profile size/parameter scope required')
        if any(not math.isfinite(item[k]) or item[k]<=0 for k in ('median_ms','p90_ms','peak_allocated_bytes')):
            raise ValueError('Finite actual latency/memory required')
    return dict(**profile,median_latency_ratio=student['median_ms']/original['median_ms'],
                peak_allocated_ratio=student['peak_allocated_bytes']/original['peak_allocated_bytes'],
                limitation='16sourceV batch1 sequential profiles on sameGPU;warmcache/order/co-owner effects;not exclusiveGPU or statistical latency guarantee')


def completed_metrics(collection):
    if (collection['run_id']!=d.RUN or collection['method_commit']!='a310b3b5ff30e23584e7bc950e8a0ff96ac685fb'
            or collection['failure'] is not None or collection['complete']['status']!='ANALYZED'
            or collection['actual_live_owned_PIDs'] or set(collection['arms'])!=set(d.ARMS)):
        raise ValueError('Original completed immutable run and exited actual owned handles required')
    source_baseline=summary_metrics(collection['baseline_source'],'source')
    expected=select_source({arm:row['source_summary'] for arm,row in collection['arms'].items()},collection['baseline_source'])
    if collection['source_decision']!=expected or collection['complete']['target_rows']!=expected['target_rows']:
        raise ValueError('Actual source decision differs;target may not influence selection')
    public=dict(run_id=d.RUN,method_commit=collection['method_commit'],status='ANALYZED',goal_complete=False,
                source_decision=expected,baseline_source=source_baseline,arms={},
                source_screening_allowance_pp=1.,target_acceptance_allowance_pp=0.,training_F_access=True,F_parameter_updates=0,
                target_performance_evaluated=bool(expected['selected']),physical_unit='One physical IQ per view;single original F50seed2701/defaulthead')
    for arm,row in collection['arms'].items():
        validate_calibration(row['calibration'],arm,20)
        calibration=row['calibration'];complete=row['training_complete']
        if (complete['calibration_L_exposures'],complete['calibration_U_exposures'],complete['calibration_seconds'])!=(calibration['L_s'],calibration['U_s'],calibration['seconds']):
            raise ValueError('Calibration report and training exposure/cost counters differ')
        if len(calibration['factor_reports'])!=calibration['eligible_layers'] or any(
                not math.isfinite(v['regularized_output_relative_squared_error']) or v['regularized_output_relative_squared_error']<0
                or v['factor_kernel_parameters']>=v['original_kernel_parameters'] for v in calibration['factor_reports']):
            raise ValueError('All actual calibrated layer costs and errors required')
        source=summary_metrics(row['source_summary'],'source',training_F_access=True)
        public['arms'][arm]=dict(parameters=d.PARAMETERS[arm],parameter_reduction_percent=100*(1-d.PARAMETERS[arm]/4665073),
            calibration=calibration,checkpoint_bytes=row['checkpoint_bytes'],source=source,source_vs_R17=compare(source,source_baseline),
            training=training_metrics(row['training_records'],row['training_complete'],arm),
            profile=profile_metrics(row['matched_profile'],arm))
    if expected['selected']:
        arm=expected['selected'];row=collection['arms'][arm]
        baseline=summary_metrics(collection['baseline_target'],'target');target=summary_metrics(row['target_summary'],'target',training_F_access=True)
        comparison=compare(target,baseline)
        if row['assessment']['target_accuracy_nonregression']!=comparison['accuracy_nonregression']:
            raise ValueError('Runtime target assessment disagrees with exact correct-count comparison')
        public.update(baseline_target=baseline,target_accuracy_nonregression=comparison['accuracy_nonregression'])
        public['arms'][arm].update(target=target,target_vs_R17=comparison)
    else:
        if collection['baseline_target'] is not None or any(row.get('target_summary') is not None for row in collection['arms'].values()):
            raise ValueError('Unqualified candidates must not generate new target results')
        public.update(target_accuracy_nonregression=None)
    return public
