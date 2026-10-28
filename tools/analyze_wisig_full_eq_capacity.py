"""Read official count metadata only; no waveform, model, prediction or launch."""
import argparse
import io
import json
import pickle
from pathlib import Path
import numpy as np
try:
    from numpy._core.multiarray import _reconstruct
except ImportError:
    from numpy.core.multiarray import _reconstruct

OLD=['14-10','14-7','20-15','20-19','6-15','8-20']
NEW=['11-4','3-13','12-20','5-20','20-12','17-11','14-14','2-17','4-1','19-13','19-8','2-19','8-1','14-20','13-3','1-18','20-7','7-7','2-3','18-12']
TARGET_RX=['1-1','14-7','2-1','20-1','7-14','7-7','8-8']

class CountsOnly(pickle.Unpickler):
    def find_class(self,module,name):
        allowed={('numpy.core.multiarray','_reconstruct'):_reconstruct,
                 ('numpy','ndarray'):np.ndarray,('numpy','dtype'):np.dtype}
        if (module,name) not in allowed:raise pickle.UnpicklingError('Unexpected metadata class: '+module+'.'+name)
        return allowed[module,name]

def analyze(path):
    raw=Path(path).read_bytes()
    if len(raw)>2*1024**2:raise ValueError('Expected small official count metadata, not waveform data')
    data=CountsOnly(io.BytesIO(raw)).load()
    tx=list(map(str,data['tx_list']));rx=list(map(str,data['rx_list']));days=list(map(str,data['capture_date_list']))
    m=np.asarray(data['mat_date_eq'])
    if m.shape!=(len(days),len(rx),len(tx)) or m.dtype.kind not in 'ifu' or not np.isfinite(m).all() or (m<0).any() or not np.equal(m,np.floor(m)).all():
        raise ValueError(('Official equalized count layout differs',m.shape,len(days),len(rx),len(tx)))
    rows=[]
    for r in TARGET_RX:
        for t in OLD+NEW:
            counts=m[:,rx.index(r),tx.index(t)].astype(np.int64).tolist()
            rows.append(dict(receiver=r,transmitter=t,old=t in OLD,per_date=counts,total=sum(counts)))
    minimum_date=min(min(r['per_date']) for r in rows);minimum_total=min(r['total'] for r in rows)
    sizes=[]
    for q in (200,500,1000):
        for mode,nquery in [('eight_disjoint_query_sets',11),('eight_support_contexts_shared_mixed_query',4)]:
            required_total=11*40+nquery*q
            # 4 dates, 10 support candidates/pool/date. Every q is divisible by4.
            required_day=11*10+nquery*(q//4)
            short=[dict(receiver=r['receiver'],transmitter=r['transmitter'],per_date=r['per_date'],total=r['total']) for r in rows
                   if r['total']<required_total or min(r['per_date'])<required_day]
            sizes.append(dict(query_per_class_RX_evaluation_set=q,mode=mode,support_pool_per_context=40,
                required_per_RX_TX=required_total,required_per_date=required_day,
                target_physical_records=required_total*len(rows),balanced_date_short_pairs=len(short),balanced_date_feasible=not short))
    candidate=[]
    for i,t in enumerate(tx):
        if t in OLD:continue
        a=m[:,[rx.index(r) for r in TARGET_RX],i]
        candidate.append(dict(transmitter=t,minimum_date=int(a.min()),minimum_total=int(a.sum(0).min())))
    candidate.sort(key=lambda a:(-a['minimum_date'],-a['minimum_total'],a['transmitter']))
    chosen=candidate[:20];chosen_names=OLD+[r['transmitter'] for r in chosen]
    chosen_m=np.take(np.take(m,[rx.index(r) for r in TARGET_RX],axis=1),[tx.index(t) for t in chosen_names],axis=2)
    min_dates=chosen_m.min((1,2)).astype(np.int64)
    query_dates=np.asarray([124,110,129,137],np.int64)
    required_dates=4*(6+query_dates)
    proposal=dict(status='CANDIDATE_CLASS_SET_NOT_FINAL_ALLOCATION',rule='Among source-unseen TX: descending minimum packet count over all7RX×4dates, descending minimum4date_total over RX, lexical identity tie-break; no IQ/accuracy selection',
        proposed_new_classes=[r['transmitter'] for r in chosen],coverage=chosen,
        original_class_set_preserved_as='Historical/sparse-coverage comparison, not silently relabeled',
        query_domains=4,mixed_query_shared_across_support_episodes=True,mixed_support_repetitions=8,single_support_repetitions=5,
        support_pool_per_domain_RX_class=24,query_per_domain_RX_class=500,
        support_per_domain_per_date=[6]*4,query_per_domain_per_date=query_dates.tolist(),
        required_per_RX_TX_by_date=required_dates.tolist(),published_minimum_per_date=min_dates.tolist(),
        public_metadata_capacity_feasible=bool(np.all(required_dates<=min_dates)),
        total_records=4*(24+500)*7*26,unique_support_candidates=4*24*7*26,unique_query=4*500*7*26,
        note='Selecting different new TX and sharing query changes benchmark scope; source6/Rs/L/U/V and source models stay unchanged. User download counts/IDs/representation still pending; no launch.')
    return dict(source='https://github.com/WiSig-dataset/wisig-examples/blob/master/data_summary.pkl',
        status='OFFICIAL_METADATA_ONLY_NOT_LOCAL_FILE_VALIDATION',data_shape=list(m.shape),
        dates=days,selected_receivers=TARGET_RX,old_classes=OLD,new_classes=NEW,
        selected_pairs=len(rows),minimum_selected_pair_total=minimum_total,minimum_selected_pair_date=minimum_date,
        total_selected_eq_records=int(sum(r['total'] for r in rows)),
        total_range_by_class=[dict(transmitter=t,minimum_total=min(r['total'] for r in rows if r['transmitter']==t),
                                   minimum_date=min(min(r['per_date']) for r in rows if r['transmitter']==t)) for t in OLD+NEW],
        query_capacity_bounds=dict(disjoint_total_only=max(0,minimum_total//11-40),
            disjoint_balanced_dates=max(0,4*(minimum_date//11-10)),
            shared_mixed_query_total_only=max(0,(minimum_total-440)//4),
            shared_mixed_query_balanced_dates=max(0,4*((minimum_date-110)//4))),
        query_size_candidates=sizes,large_query_proposal=proposal,rows=rows,IQ_reads=0,query_truth_reads=0,model_reads=0,launches=0,
        caveat='Published Full WiSig metadata counts do not prove the user download is complete, has matching packet identities, or contains unused independent historical records')

def main():
    p=argparse.ArgumentParser();p.add_argument('--metadata',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    out=analyze(a.metadata);Path(a.output).write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in out.items() if k not in ('rows','total_range_by_class')},ensure_ascii=False))

if __name__=='__main__':main()
