"""Summarize completed independent R24 scores; never open IQ, truth or F."""
import math
from experiments.pamf_factor_formal_identity import design as d

METHOD='8a2899f5998bd50d1595391f8c7fad9667b0d1fa'


def completed_metrics(value):
    if value.get('status')!='COLLECTED' or value.get('failure') is not None or value.get('actual_live_owned_PIDs'):
        raise ValueError('Completed owned CPU processes must exit before delivery')
    if value.get('complete',{}).get('status')!='ANALYZED' or value.get('launch',{}).get('commit')!=METHOD:
        raise ValueError('Fixed completed diagnostic required')
    s=value['summary'];cfg=d.configuration(METHOD)
    if s.get('configuration')!=cfg or value.get('config')!=cfg or s.get('run')!=d.RUN:
        raise ValueError('The fixed full-eight matrix and analytic coordinates required')
    expected=dict(status='ANALYZED',count=d.COUNT,views=list(d.VIEWS),groups=d.GROUPS,candidates=list(d.CANDIDATES),
        all_features_before_labels=True,target_truth_reads=1,target_truth_for_evaluation_only=True,
        F_access=0,F_weights_read=0,F_embeddings_logits_read=0,model_checkpoint_loads=0,extra_model_predictions=0,
        neural_training_updates=0,head_fit_updates=0,new_calibration_samples=0,parent_training_or_decisions_changed=False,
        source_selection_changed=False,new_source_thresholds=False,target_decisions_changed=False,selected_candidate=None)
    if any(s.get(k)!=v for k,v in expected.items()):raise ValueError('Independent postscore-only permissions required')
    expected=dict(status='FEATURES_COMPLETE',count=d.COUNT,views=list(d.VIEWS),groups=d.GROUPS,readouts=list(d.READOUTS),cells=d.TOTAL_CELLS,
        TX_labels_read=0,F_access=0,model_checkpoint_loads=0,extra_model_predictions=0,new_calibration_samples=0,
        reused_feature_cells=56,new_feature_cells=224,analytic_transform_samples=9408000)
    if any(value['feature_complete'].get(k)!=v for k,v in expected.items()):raise ValueError('Every fixed feature cell must precede truth')
    cal=value['calibration_reference']
    if (cal.get('run')!=d.REFERENCE or cal.get('commit')!=d.REFERENCE_COMMIT or cal.get('count')!=8192
        or cal.get('L_count',0)+cal.get('U_count',0)!=8192 or cal.get('new_calibration_samples')!=0 or cal.get('new_parameters_fitted')!=0):
        raise ValueError('Original sourceLU analytic calibration must be reused')
    smoke=value['smoke']
    if smoke.get('status')!='PASS' or smoke.get('source_Ls_IQ')!=2 or any(smoke.get(k)!=0 for k in ('model_checkpoint_loads','F_access','TX_labels_read','V_reads','target_reads','parameter_updates')):
        raise ValueError('Real source-only analytic no-query smoke required')
    groups={('overall','all'):168000,**{('TX',str(k)):28000 for k in range(6)}}
    rx={r['value'] for r in s['records'] if r['dimension']=='RX'};days={r['value'] for r in s['records'] if r['dimension']=='day'}
    if len(rx)!=7 or len(days)!=4:raise ValueError('Full seven-RX/four-day population required')
    groups.update({('RX',k):24000 for k in rx});groups.update({('day',k):42000 for k in days})
    expected={(a,g,v,dim,key) for a in d.CANDIDATES for g in d.GROUPS for v in d.VIEWS for dim,key in groups}
    actual=[(r['candidate'],r['group'],r['view'],r['dimension'],r['value']) for r in s['records']]
    if len(actual)!=d.EXPECTED_RECORDS or len(set(actual))!=len(actual) or set(actual)!=expected:
        raise ValueError('Every candidate/group/scene/TX/RX/day record required')
    for row in s['records']:
        n=groups[(row['dimension'],row['value'])]
        for name in ('baseline','restored','clean'):
            item=row[name];valid=item['valid_count']
            if item['count']!=n or not 0<=valid<=n or not 0<=item['zero_query_count']<=n:raise ValueError('Matched physical group counts required')
            values=[item.get(k) for k in ('mean_same_TX_cosine','mean_other_TX_cosine','mean_margin','positive_margin_percent')]
            if valid:
                if any(v is None or not math.isfinite(v) for v in values) or not 0<=values[-1]<=100:
                    raise ValueError('Finite descriptive similarities required')
                if not math.isclose(values[2],values[0]-values[1],rel_tol=1e-8,abs_tol=1e-10):raise ValueError('Same-minus-other margin must close')
            elif any(v is not None for v in values):raise ValueError('Missing reference/zero query cannot invent margins')
        if row['paired_descriptor_error']['count']!=n or not 0<=row['paired_delta']['count']<=n:
            raise ValueError('Paired descriptor and delta counts required')
        if any(not math.isfinite(row['paired_descriptor_error'][k]) or row['paired_descriptor_error'][k]<0 for k in ('raw','restored')):
            raise ValueError('Finite paired descriptor errors required')
    expected={(a,g,v,r) for a in d.CANDIDATES for g in d.GROUPS for v in d.VIEWS for r in ('raw','restored','clean')}
    controls=s['query_label_negative_controls'];keys=[(r['candidate'],r['group'],r['view'],r['readout']) for r in controls]
    if len(keys)!=d.EXPECTED_CONTROLS or len(set(keys))!=len(keys) or set(keys)!=expected:raise ValueError('All fixed conditional controls required')
    for row in controls:
        if row['permutations']!=199 or row['seed']!=2026102034 or not row['reference_labels_fixed'] or row['is_p_value']:
            raise ValueError('Fixed descriptive conditional controls required')
        if not 0<=row['count']<=168000:raise ValueError('Valid conditional control population required')
        if row['count']:
            if any(row[k] is None or not math.isfinite(row[k]) for k in ('observed_mean','null_mean','null_p05','null_p95')):
                raise ValueError('Finite conditional controls required')
            if row['null_p05']>row['null_p95'] or row['above_null_p95']!=(row['observed_mean']>row['null_p95']+1e-12):
                raise ValueError('Conditional control summary differs')
    pairs=s['comparisons'];keys=[(r['candidate'],r['control'],r['group'],r['view']) for r in pairs]
    expected={(a,b,g,v) for a,b in d.COMPARISONS for g in d.GROUPS for v in d.VIEWS}
    if len(keys)!=d.EXPECTED_COMPARISONS or len(set(keys))!=len(keys) or set(keys)!=expected:raise ValueError('All fixed paired comparisons required')
    for row in pairs:
        if not 0<=row['common_count']<=168000 or (row['common_count'] and any(row[k] is None or not math.isfinite(row[k]) for k in ('margin_change','descriptor_error_change'))):
            raise ValueError('Finite same-population paired comparisons required')
    return dict(s,verification=dict(status='VERIFIED',all_owned_CPU_processes_exited=True,formal_group_records=len(actual),
        conditional_controls=len(controls),paired_comparisons=len(pairs),arithmetic_checked=True,additional_truth_IQ_F_reads=0),
        execution_timing=value['timing'],feature_complete=value['feature_complete'],calibration_reference=cal,
        goal_complete=False,limitation='Postscore same-TX association;not classification accuracy or per-detail authenticity;no model selection')
