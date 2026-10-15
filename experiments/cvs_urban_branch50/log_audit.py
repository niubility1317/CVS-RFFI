"""Parse one immutable completed row's entire training channel trace."""
import gzip,json,math,time
from collections import Counter,defaultdict
from itertools import zip_longest
from pathlib import Path

def audit(base,row,reference_row,index_map=None):
    base=Path(base)
    source=base/row/'source'
    read=lambda p:json.loads(p.read_text())
    if not (source/'training_mechanism_complete.json').is_file():
        return dict(status='WAIT_ROW_TRAINING',row_id=row)
    started=time.perf_counter();inner=read(source/'inner_source_contract.json')
    allowed=set(map(str,inner['role_ids']['L_train']));developer=set(map(str,inner['role_ids']['L_dev']))
    assert len(allowed)==5580 and len(developer)==720 and not allowed&developer
    aliases={}
    if index_map is not None:
        assert index_map['source_role']=='L_s' and not index_map['query_read'] and index_map['U_label_tensor_reads']==0
        aliases={v['sample_id']:v for v in index_map['records']}
        assert len(aliases)==6300 and {v['physical_id'] for v in aliases.values()}==allowed|developer
    baseline=base/reference_row/'source/channel_training.jsonl.gz'
    own=source/'channel_training.jsonl.gz'
    fields=('sample_id','session_id','scene','state','lock','quality','added_noise_SNR','cfo','kernel_length','q')
    strata=defaultdict(lambda:dict(count=0,correct=0,eligible=0,q_sum=0.,added_noise_SNR_sum=0.,
        cfo_sum=0.,cfo_square_sum=0.,kernel_length_sum=0))
    ids=set();trace_ids=set();sessions=Counter();session_TX=defaultdict(set);coverage=Counter()
    physical_sessions={};mismatch=Counter();examples=[];steps=packets=0


    def quality_bin(value):
        return '<3' if value<3 else '3-8' if value<8 else '8-15' if value<15 else '>=15'


    with gzip.open(own,'rt',encoding='utf-8') as own_stream,gzip.open(baseline,'rt',encoding='utf-8') as reference_stream:
        for line,reference in zip_longest(own_stream,reference_stream):
            assert line is not None and reference is not None,'Incomplete matched trace length'
            item=json.loads(line);ref=json.loads(reference);steps+=1
            assert item['step']==ref['step']==steps and item['epoch']==ref['epoch']==(steps-1)//222+1
            assert len(item['records'])==len(ref['records'])==128
            for value,rv in zip(item['records'],ref['records']):
                packets+=1;trace_id=str(value['sample_id']);session=str(value['session_id'])
                entry=aliases.get(trace_id);pid=entry['physical_id'] if entry else trace_id
                assert pid in allowed and pid not in developer,'Non-training physical ID in actual training trace'
                ids.add(pid);trace_ids.add(trace_id);sessions[session]+=1
                if entry:
                    assert session==f"rx{entry['RX']}:day{entry['day']}:eq{entry['eq']}", 'Session must use RX/day/eq, not TX'
                    session_TX[session].add(entry['TX']);coverage[entry['TX'],entry['RX'],value['state']]+=1
                if pid in physical_sessions:assert physical_sessions[pid]==session,'RX hardware session changed for same source ID'
                else:physical_sessions[pid]=session
                for key in ('quality','added_noise_SNR','cfo','kernel_length','q','auxiliary_weight'):
                    assert math.isfinite(value[key]),key
                assert 0<=value['q']<=1 and 0<=value['auxiliary_weight']<=1
                if value['scene']=='practical_low_suburban':assert value['auxiliary_weight']==0
                for key in fields:
                    if value[key]!=rv[key]:
                        mismatch[key]+=1
                        if len(examples)<12:examples.append(dict(step=steps,packet=packets,field=key,value=value[key],reference=rv[key]))
                key=(value['scene'],value['state'],value['lock'],quality_bin(value['added_noise_SNR']))
                record=strata[key];record['count']+=1;record['correct']+=int(value['correct'])
                record['eligible']+=value['auxiliary_weight'];record['q_sum']+=value['q']
                record['added_noise_SNR_sum']+=value['added_noise_SNR'];record['cfo_sum']+=value['cfo']
                record['cfo_square_sum']+=value['cfo']**2;record['kernel_length_sum']+=value['kernel_length']
    assert steps==11100 and packets==1420800 and ids==allowed
    if aliases:
        assert all(v==set(range(6)) for v in session_TX.values())
        assert set(coverage)=={(tx,rx,state) for tx in range(6) for rx in (1,3,4,6,8) for state in ('LOS','PARTIAL','BLOCKED')}
    out=[]
    for key,v in sorted(strata.items()):
        n=v['count'];out.append(dict(scene=key[0],state=key[1],lock=key[2],added_noise_SNR_bin=key[3],
            count=n,accuracy=v['correct']/n,teacher_eligible_fraction=v['eligible']/n,mean_q=v['q_sum']/n,
            mean_added_noise_SNR_db=v['added_noise_SNR_sum']/n,mean_CFO_hz=v['cfo_sum']/n,
            RMS_CFO_hz=math.sqrt(v['cfo_square_sum']/n),mean_kernel_length=v['kernel_length_sum']/n))
    return dict(status='COMPLETE_CHANNEL_TRACE_AUDIT',row_id=row,steps=steps,packets=packets,
        actual_distinct_L_train_IDs=len(ids),actual_L_dev_or_other_IDs=0,session_count=len(sessions),
        session_counts=dict(sessions),physical_ID_session_stable=True,strata=out,
        distinct_trace_aliases=len(trace_ids),ID_namespace='native base_index translated to existing L_s opaque IDs' if aliases else 'existing opaque IDs',
        session_TX_support={key:sorted(v) for key,v in session_TX.items()},
        independent_TX_RX_state_coverage=[dict(TX=k[0],RX=k[1],state=k[2],count=n) for k,n in sorted(coverage.items())],
        matched_A0_channel_metadata_mismatch_counts=dict(mismatch),mismatch_examples=examples,
        compared_fields=list(fields),comparison_limit='Matched recorded source generator metadata, not waveform-byte or model-trajectory equality; output is posthoc reporting only',
        seconds=time.perf_counter()-started,source_only=True,query_read=False,optimizer_updates=0,model_forward_packets=0)


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-root',required=True,type=Path)
    parser.add_argument('--row',required=True)
    parser.add_argument('--reference-row',default='A0-s2026092701')
    parser.add_argument('--index-map',type=Path,help='Existing L-only native-index to opaque-ID metadata translation')
    args=parser.parse_args()
    mapping=json.loads(args.index_map.read_text()) if args.index_map else None
    print(json.dumps(audit(args.run_root,args.row,args.reference_row,mapping)))
