"""Post-scoring source error overlap; no fitted selector, IQ values or model forwards."""
import argparse,json
from pathlib import Path
import numpy as np

VIEWS=('clean','practical_high','practical_mid','practical_low_suburban',
       'practical_high_urban','practical_mid_urban','practical_low_urban')
READOUTS=('identity','parent_R03','refined')


def vector(value,n):
    value=np.asarray(value)
    if value.shape!=(n,) or value.dtype.kind not in 'iu' or (value<0).any() or (value>5).any():
        raise ValueError('Matched six-class decision vectors required')
    return value


def overlap(labels,baseline,candidate,mask=None):
    labels=np.asarray(labels);n=len(labels)
    labels=vector(labels,n);baseline=vector(baseline,n);candidate=vector(candidate,n)
    mask=np.ones(n,bool) if mask is None else np.asarray(mask)
    if mask.shape!=(n,) or mask.dtype!=np.bool_:raise ValueError('Matched boolean source mask required')
    count=int(mask.sum());base=baseline==labels;new=candidate==labels
    values=dict(count=count,baseline_wrong=int((mask&~base).sum()),candidate_wrong=int((mask&~new).sum()),
        both_wrong=int((mask&~base&~new).sum()),both_correct=int((mask&base&new).sum()),
        corrected_vs_baseline=int((mask&~base&new).sum()),damaged_vs_baseline=int((mask&base&~new).sum()),
        decision_disagreements=int((mask&(baseline!=candidate)).sum()))
    values['net_corrected']=values['corrected_vs_baseline']-values['damaged_vs_baseline']
    values['baseline_error_overlap_fraction']=None if not values['baseline_wrong'] else values['both_wrong']/values['baseline_wrong']
    # Uses source truth after scoring. This is a diagnostic upper bound, never an inference rule.
    values['truth_oracle_coverage_upper_bound_percent']=None if not count else 100*(1-values['both_wrong']/count)
    return values


def read_packets(row,ids):
    folder=row/'prediction';done=json.loads((folder/'complete.json').read_text())
    expected=dict(status='SOURCE_PREDICTIONS_COMPLETE',count=len(ids),views=list(VIEWS),readouts=list(READOUTS),
                  role='source_V',target_reads=0,TX_label_reads=0,state_updates=0)
    if any(done.get(k)!=v for k,v in expected.items()):raise ValueError('All seven frozen source views required')
    if json.loads((row/'complete.json').read_text()).get('status')!='ANALYZED':raise ValueError('Already-scored source row required')
    packets={}
    for view in VIEWS:
        with np.load(folder/(view+'.npz'),allow_pickle=False) as packet:
            if set(packet.files)!={'ids',*READOUTS} or not np.array_equal(packet['ids'],ids):raise ValueError('Same physical source IDs required')
            packets[view]={name:vector(packet[name],len(ids)).copy() for name in READOUTS}
    return packets


def analyze(cache,source_cache,control,source_run):
    from experiments.pamf_fcrossrx_readout.source_diagnostic import source_reference
    ids,_,index=source_reference(cache/'validation',source_cache)
    rows={'posterior03':control/'posterior03','final_guidance':source_run/'final_guidance',
          'layerwise_guidance':source_run/'layerwise_guidance'}
    packets={name:read_packets(path,ids) for name,path in rows.items()}
    for view in VIEWS:
        for name in rows:
            if any(not np.array_equal(packets['posterior03'][view][key],packets[name][view][key]) for key in ('identity','parent_R03')):
                raise ValueError('Same source IQ and fixed F1 reference decisions required')
    # Labels only after all existing source decisions and row completion are checked.
    labels=np.load(source_cache/'V/labels.npy',allow_pickle=False)[index]
    clean_correct=packets['posterior03']['clean']['identity']==labels
    records=[]
    for view in VIEWS:
        for name in ('final_guidance','layerwise_guidance'):
            args=(labels,packets['posterior03'][view]['refined'],packets[name][view]['refined'])
            records.append(dict(view=view,candidate=name,all=overlap(*args),
                given_original_clean_correct=overlap(*args,mask=clean_correct)))
    pooled=[]
    for name in ('final_guidance','layerwise_guidance'):
        y=np.tile(labels,6);base=np.concatenate([packets['posterior03'][v]['refined'] for v in VIEWS[1:]])
        new=np.concatenate([packets[name][v]['refined'] for v in VIEWS[1:]])
        pooled.append(dict(candidate=name,all=overlap(y,base,new),
            given_original_clean_correct=overlap(y,base,new,mask=np.tile(clean_correct,6))))
    return dict(status='ANALYZED',source_only=True,count=len(ids),original_clean_correct=int(clean_correct.sum()),
        target_reads=0,new_model_forwards=0,parameter_updates=0,IQ_values_used=0,records=records,pooled_six_views=pooled,
        limitation='Post-hoc source error association. Oracle coverage uses source truth and is not deployable selection or target evidence. Six views repeat the same physical samples.')


def main():
    p=argparse.ArgumentParser()
    for key in ('cache','source-cache','control','source-run'):p.add_argument('--'+key,type=Path,required=True)
    args=p.parse_args()
    print(json.dumps(analyze(args.cache,args.source_cache,args.control,args.source_run),allow_nan=False))


if __name__=='__main__':main()
