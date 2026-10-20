import copy
import pytest
from experiments.pamf_r46_cuda_graph import design as d
from tools import analyze_pamf_r46_cuda_graph_results as a


def profile(arm=d.ARMS[1]):
    names={'parent_R46_350521':350521}
    if arm!=d.ARMS[0]:names['candidate']=278180
    result={}
    for name,size in names.items():
        cases={}
        for mode in ('G_only','G_plus_F'):
            for batch in (1,32):
                for execution in ('native','graph'):
                    ms=(10. if execution=='native' else 1.)*(.8 if name=='parent_R46_350521' else 1.)
                    cases[mode+'_B'+str(batch)+'_'+execution]=dict(batch=batch,rounds=3,warmup_per_round=16 if batch==1 else 4,
                        iterations_per_round=64 if batch==1 else 16,source_V_unique_IQ=64,median_ms=ms,p90_ms=ms,p95_ms=ms,
                        throughput_samples_per_second=batch*1000/ms,peak_allocated_bytes=20000000,resident_before_forward_bytes=10000000,
                        incremental_peak_bytes=10000000,round_median_ms=[ms]*3)
        result[name]=dict(parameters=size,ConvLinear_MAC_per_IQ=39081004 if size==350521 else 29727404,cases=cases)
    return dict(status='PROFILE_COMPLETE',arm=arm,physical_GPU=d.GPUS[arm],TX_labels_read=0,target_reads=0,state_updates=0,
                training_F_access=False,includes_full_context_3prior_6CG=True,results=result)


def agreement(role):
    return dict(status='COMPARED',role=role,count=1800 if role=='source' else 168000,TX_truth_reads=0,
                native_predictions_reused=True,native_target_reruns=0,views={v:dict(iq=dict(max_abs_difference=0.,unequal_elements=0),
                features=dict(max_abs_difference=0.,unequal_elements=0),logits=dict(max_abs_difference=0.,unequal_elements=0),argmax_disagreements=0) for v in d.VIEWS})


def summary(role,wrong=False):
    n=1800 if role=='source' else 168000
    def record(dim,value,tx=None):
        matrix=[[0]*6 for _ in range(6)]
        for k in range(6) if tx is None else (tx,):matrix[k][(k+1)%6 if wrong else k]=n//6
        count=n if tx is None else n//6
        return dict(dimension=dim,value=str(value),readout='restored',confusion=matrix,count=count,accuracy=0. if wrong else 100.,
                    macro_f1=0. if wrong else 100. if tx is None else 100/6,baseline_accuracy=100.,repaired=0,damaged=count if wrong else 0,
                    net_repaired=-count if wrong else 0,iq_nmse=0.)
    records=[]
    for view in d.VIEWS:
        for r in [record('overall','all'),*[record('TX',k,k) for k in range(6)],record('RX','rx'),record('day','day')]:
            records.append(dict(r,view=view))
    return dict(status='ANALYZED',role=role,count=n,F_run=d.F_RUN,F_seed=d.F_SEED,original_default_head=True,target_access=role=='target',
                all_predictions_before_truth=True,training_F_access=False,training_TX_labels_read=0,head_fit_updates=0,records=records,
                views={v:dict(restored=dict(waveform=dict(iq_nmse=0.,prefix232_nmse=0.,tail24_nmse=0.,fft_log_power_L1=0.))) for v in d.VIEWS})


def collection():
    rows={}
    for arm in d.ARMS:
        r=dict(complete={'status':'ANALYZED'},failure=None,checkpoint_source=dict(source_arm=d.SOURCE_ROWS[arm],epoch=20,
            updates=d.SOURCE_BUDGET[arm][1],parameters=d.PARAMETERS[arm],new_training_updates=0,training_F_access=False,checkpoint_bytes=1482748),
            smoke=dict(status='PASS',real_Ls_IQ=2,batches={str(k):{} for k in (1,8,32,64,256)},source_checkpoint_weights_unchanged=True,native_fallback=False),
            profile=profile(arm),standalone_graph_memory={})
        for role in ('source','target'):
            r[role+'_summary']=summary(role);r['native_'+role+'_summary']=summary(role);r[role+'_agreement']=agreement(role)
            r[role+'_prediction_complete']=dict(status='PREDICTIONS_COMPLETE',kind=d.KIND,role=role,count=1800 if role=='source' else 168000,
                views=list(d.VIEWS),readouts=['restored'],TX_truth_reads=0,state_updates=0,head_fit_updates=0,F_run=d.F_RUN,F_seed=d.F_SEED,native_fallback=False,batch=256)
        rows[arm]=r
    return dict(run_id=d.RUN,method_commit=a.METHOD,failure=None,actual_live_owned_PIDs=[],complete=dict(status='ANALYZED',source_rows=list(d.ARMS),target_rows=list(d.ARMS)),
                rows=rows,baselines={k:{r:summary(r) for r in ('source','target')} for k in ('R17','R46')})


def test_reports_both_native_and_graph_parent_comparisons():
    c=a.profile_metrics(profile(),d.ARMS[1])['comparisons']['G_plus_F_B1']
    assert c['candidate_native_to_graph_median_speedup']==10
    assert c['parent_native_to_candidate_graph_median_speedup']==8
    assert c['parent_graph_to_candidate_graph_median_speedup']==.8


@pytest.mark.parametrize('bad',('GPU','parameters','case','rounds','memory'))
def test_invalid_matched_cost_rejected(bad):
    p=profile();c=p['results']['candidate'];row=c['cases']['G_plus_F_B1_graph']
    if bad=='GPU':p['physical_GPU']=0
    elif bad=='parameters':c['parameters']=91477
    elif bad=='case':del c['cases']['G_plus_F_B1_native']
    elif bad=='rounds':row['round_median_ms']=[1.]
    else:row['incremental_peak_bytes']=-1
    with pytest.raises(ValueError):a.profile_metrics(p,d.ARMS[1])


def test_numerical_difference_not_misreported_as_prediction_change():
    v=agreement('target');v['views']['clean']['logits']=dict(max_abs_difference=1e-6,unequal_elements=10)
    got=a.agreement_metrics(v,'target')
    assert got['all_argmax_equal'] and not got['all_values_bitwise_equal']


@pytest.mark.parametrize('bad',('count','truth','view','contradiction','argmax'))
def test_invalid_native_comparison_rejected(bad):
    v=agreement('target')
    if bad=='count':v['count']=100
    elif bad=='truth':v['TX_truth_reads']=1
    elif bad=='view':del v['views']['clean']
    elif bad=='contradiction':v['views']['clean']['iq']['unequal_elements']=1
    else:v['views']['clean']['argmax_disagreements']=1
    with pytest.raises(ValueError):a.agreement_metrics(v,'target')


def test_complete_collection_recomputes_all_three_models():
    got=a.completed_metrics(collection())
    assert got['accepted_rows']==list(d.ARMS)
    assert all(r['new_training_updates']==0 and r['agreement']['target']['all_argmax_equal'] for r in got['rows'].values())
    assert got['rows'][d.ARMS[1]]['target']['views']['clean']['correct']==168000


@pytest.mark.parametrize('bad',('live','missing_row','F_training','missing_view','contradictory_native'))
def test_incomplete_or_inconsistent_delivery_rejected(bad):
    raw=collection();row=raw['rows'][d.ARMS[0]]
    if bad=='live':raw['actual_live_owned_PIDs']=[1]
    elif bad=='missing_row':del raw['rows'][d.ARMS[-1]]
    elif bad=='F_training':row['checkpoint_source']['training_F_access']=True
    elif bad=='missing_view':row['target_prediction_complete']['views']=list(d.VIEWS[:-1])
    else:row['native_target_summary']=summary('target',wrong=True)
    with pytest.raises(ValueError):a.completed_metrics(raw)
