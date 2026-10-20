"""Source-only, post-score cross-record/RX/day feature contrasts; no classifier."""
import argparse
import json
import os
from pathlib import Path
import numpy as np

RUN='20261020-diagnostic-restoration-source-cross-record-F50-s2026092701-r25'
F_RUN='20261016-phase1-F-supcon-e50-clean-manysig-m9-r02'
VIEWS=('clean','practical_high','practical_mid','practical_low_suburban','practical_high_urban','practical_mid_urban','practical_low_urban')
ROWS=(
    ('R17_unshared','20261020-phase1-restoration-unrolled-selfsup-s2026092701-m2-r17','unroll_unshared','cd97744884b5e460926e48c8b7efb7d78c6da0f4'),
    ('R19_time','20261020-phase1-restoration-alternating-tf-selfsup-s2026092701-m2-r19','time_time','5bbc0ccdc32de19e75cb6a01e2cda0352e39f36a'),
    ('R23_control','20261020-phase1-restoration-joint-coarse-selfsup-s2026092701-m2-r23','prior_only','d43dbab326d0b1e2cb6fb7faa23dc9a3ec25a025'),
    ('R24_TCN','20261020-phase1-restoration-fullresolution-priors-selfsup-s2026092701-m2-r24','tcn_prior','da5b9d52819ccd40f8b59dbfdefebff3bd3c2230'),
    ('R24_attention','20261020-phase1-restoration-fullresolution-priors-selfsup-s2026092701-m2-r24','attention_prior','da5b9d52819ccd40f8b59dbfdefebff3bd3c2230'))
REFERENCE='20261019-phase1-restoration-uncertainty-selfsup-s2026092701-m3-r10'


def unit_features(features):
    value=np.asarray(features,dtype=np.float64)
    if value.ndim!=2 or not len(value) or not value.shape[1] or not np.isfinite(value).all():raise ValueError('Finite nonempty feature matrix required')
    norm=np.linalg.norm(value,axis=1)
    if not np.isfinite(norm).all():raise ValueError('Finite feature norms required')
    valid=norm>1e-8
    return value/np.maximum(norm[:,None],1e-8),valid


def metadata(ids,labels,rx,day,count):
    ids=np.asarray(ids).astype(str);labels=np.asarray(labels);rx=np.asarray(rx).astype(str);day=np.asarray(day).astype(str)
    if (any(v.shape!=(count,) for v in (ids,labels,rx,day)) or len(set(ids.tolist()))!=count
            or labels.dtype.kind not in 'iu' or set(labels.tolist())!=set(range(6))):
        raise ValueError('Unique physical IDs and complete six-TX source metadata required')
    return ids,labels,rx,day


def reference_pool(clean,ids,labels,rx,day,cap=2):
    if cap!=2:raise ValueError('Preregistered cap2 per TX/RX/day cell required')
    unit,valid=unit_features(clean);ids,labels,rx,day=metadata(ids,labels,rx,day,len(unit))
    cells=[(r,d) for r in sorted(set(rx)) for d in sorted(set(day))];chosen=[];empty=0
    for r,d in cells:
        for tx in range(6):
            index=np.flatnonzero((rx==r)&(day==d)&(labels==tx))
            index=sorted(index,key=lambda i:ids[i])[:cap];empty+=int(not index);chosen.extend(index)
    index=np.asarray(chosen,dtype=np.int64)
    return dict(indices=index,unit=unit[index],nonzero=valid[index],labels=labels[index],rx=rx[index],day=day[index],
        cells=cells,cap=cap,empty_TX_cells=empty,reference_count=len(index),zero_reference_count=int((~valid[index]).sum()))


def eligible_reference_mask(reference,rx,day):return (reference['rx']!=str(rx))&(reference['day']!=str(day))


def contrasts(features,reference,labels,rx,day):
    unit,nonzero=unit_features(features);labels=np.asarray(labels);rx=np.asarray(rx).astype(str);day=np.asarray(day).astype(str)
    if (labels.shape!=(len(unit),) or rx.shape!=labels.shape or day.shape!=labels.shape
            or labels.dtype.kind not in 'iu' or not set(labels.tolist())<=set(range(6)) or unit.shape[1]!=reference['unit'].shape[1]):
        raise ValueError('Aligned finite source features and six-class metadata required')
    scores=unit@reference['unit'].T;positive=np.zeros(len(unit));negative=np.zeros(len(unit));valid_cells=np.zeros(len(unit),np.int64);missing=np.zeros_like(valid_cells)
    for r in sorted(set(rx)):
        for d in sorted(set(day)):
            query=np.flatnonzero((rx==r)&(day==d))
            if not len(query):continue
            for cr,cd in reference['cells']:
                if cr==r or cd==d:continue
                cell=(reference['rx']==cr)&(reference['day']==cd)&reference['nonzero']
                indices=[np.flatnonzero(cell&(reference['labels']==tx)) for tx in range(6)]
                if any(not len(index) for index in indices):missing[query]+=1;continue
                means=np.stack([scores[np.ix_(query,index)].mean(1) for index in indices],axis=1)
                own=means[np.arange(len(query)),labels[query]]
                positive[query]+=own;negative[query]+=(means.sum(1)-own)/5;valid_cells[query]+=1
    positive/=np.maximum(valid_cells,1);negative/=np.maximum(valid_cells,1)
    return dict(positive=positive,negative=negative,margin=positive-negative,valid=(valid_cells>0)&nonzero,
        zero_query=~nonzero,valid_cells=valid_cells,missing_cells=missing)


def summarize(before,after,clean,mask):
    def stats(value):
        valid=mask&value['valid'];count=int(valid.sum())
        return dict(count=int(mask.sum()),valid_count=count,zero_query_count=int((mask&value['zero_query']).sum()),
            eligible_reference_cells_total=int(value['valid_cells'][mask].sum()),missing_reference_cells_total=int(value['missing_cells'][mask].sum()),
            mean_same_TX_cosine=float(value['positive'][valid].mean()) if count else None,
            mean_other_TX_cosine=float(value['negative'][valid].mean()) if count else None,
            mean_margin=float(value['margin'][valid].mean()) if count else None,
            positive_margin_percent=float(100*(value['margin'][valid]>0).mean()) if count else None)
    common=mask&before['valid']&after['valid']&clean['valid'];count=int(common.sum())
    return dict(baseline=stats(before),restored=stats(after),clean=stats(clean),
        paired_delta=dict(count=count,mean_margin_change=float((after['margin'][common]-before['margin'][common]).mean()) if count else None,
            restored_minus_clean_margin=float((after['margin'][common]-clean['margin'][common]).mean()) if count else None))


def read(path):return json.loads(path.read_text(encoding='utf-8'))


def require_summary(value,kind):
    expected=dict(status='ANALYZED',kind=kind,role='source',count=1800,F_run=F_RUN,F_seed=2026092701,
        original_default_head=True,all_predictions_before_truth=True,training_TX_labels_read=0,
        training_F_access=False,head_fit_updates=0,target_access=False)
    if any(value.get(k)!=v for k,v in expected.items()) or tuple(value.get('views',{}))!=VIEWS:
        raise ValueError('Complete same-F source-only scored summary required before TX labels')


def load_features(folder,ids,kind,readout):
    pred=folder/'prediction';done=read(pred/'complete.json')
    readouts=['restored'] if kind=='native_map_prior' else ['identity','coarse','r03_reference','r07_native','global_floor']
    expected=dict(status='PREDICTIONS_COMPLETE',kind=kind,role='source',count=1800,views=list(VIEWS),
        readouts=readouts,TX_truth_reads=0,state_updates=0,head_fit_updates=0,F_run=F_RUN,F_seed=2026092701)
    if any(done.get(k)!=v for k,v in expected.items()) or not np.array_equal(np.load(pred/'physical_ids.npy',allow_pickle=False),ids):
        raise ValueError('All matching source predictions required before TX labels')
    result={}
    for view in VIEWS:
        value=np.load(pred/(view+'.'+readout+'.features.npy'),mmap_mode='r',allow_pickle=False)
        if value.shape!=(1800,160) or not np.isfinite(value).all():raise ValueError('Complete finite fixed F features required')
        result[view]=value
    return result


def analyze(args,*,label_loader=None):
    if os.environ.get('CUDA_VISIBLE_DEVICES','')!='':raise ValueError('Independent CPU source analysis required')
    folders={}
    for name,run,row,commit in ROWS:
        root=args.runs_root/run;folder=root/row/'native_map_prior'
        if read(root/'launch.json').get('commit')!=commit:raise ValueError('Registered source run/commit required')
        require_summary(read(folder/'summary.json'),'native_map_prior');folders[name]=folder
    baseline=args.runs_root/REFERENCE/'reference_controls'
    require_summary(read(baseline/'summary.json'),'reference_controls')
    from experiments.pamf_fcrossrx_readout.source_diagnostic import source_reference
    ids,source,index=source_reference(args.panel,args.source_cache)
    features={name:load_features(folder,ids,'native_map_prior','restored') for name,folder in folders.items()}
    raw=load_features(baseline,ids,'reference_controls','identity')
    # First source TX-label read, after every declared source prediction and score validates.
    labels=(np.load if label_loader is None else label_loader)(args.source_cache/'V/labels.npy',allow_pickle=False)[index]
    rx,day=source.rx[index],source.day[index];reference=reference_pool(raw['clean'],ids,labels,rx,day)
    clean=contrasts(raw['clean'],reference,labels,rx,day)
    groups=[('overall','all',np.ones(len(ids),bool))]
    for dim,values in (('TX',labels),('RX',rx),('day',day)):
        groups.extend((dim,str(value),values==value) for value in sorted(set(values.tolist())))
    records=[]
    for view in VIEWS:
        before=contrasts(raw[view],reference,labels,rx,day)
        for name in folders:
            after=contrasts(features[name][view],reference,labels,rx,day)
            for dimension,value,mask in groups:records.append(dict(candidate=name,view=view,dimension=dimension,value=value,
                **summarize(before,after,clean,mask)))
    return dict(schema='source_cross_record_v1',status='ANALYZED',role='source_V',count=1800,F_run=F_RUN,F_seed=2026092701,
        views=list(VIEWS),candidates=[name for name,*_ in ROWS],records=records,
        reference_pool={k:reference[k] for k in ('cap','empty_TX_cells','reference_count','zero_reference_count')},
        optimizer_updates=0,head_fit_updates=0,parameters_fitted=0,extra_model_predictions=0,target_reads=0,
        runtime_decisions_changed=False,source_selection_changed=False,source_labels_after_completed_scores=True,
        pairing='Reference RX and day both differ;cap2 fixed lowest opaque physical IDs per TX/RX/day;complete6TX nonzero reference cells;equal cell and other5TX weights',
        claim_scope='Source-only frozen same-F feature contrasts;no classifier predictions or training;cross-record/day does not establish decoded payload independence,independent physical families,newTX,multiseed,inorbit,or per-detail TX authenticity',goal_complete=False)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ('runs_root','panel','source_cache','output'):parser.add_argument('--'+key.replace('_','-'),type=Path,required=True)
    parser.add_argument('--expected-commit',required=True);args=parser.parse_args()
    from experiments.pamf_restoration.runner import code_commit
    if code_commit()!=args.expected_commit or args.output.name!=RUN or args.output.exists():raise ValueError('Pinned source-only code and new immutable diagnostic root required')
    value=analyze(args);value['analysis_commit']=args.expected_commit
    args.output.mkdir();out=args.output/'summary.json'
    with out.open('x',encoding='utf-8') as stream:json.dump(value,stream,ensure_ascii=False,indent=2,allow_nan=False)
    print(json.dumps(dict(status=value['status'],records=len(value['records']),count=value['count'],target_reads=0)),flush=True)


if __name__=='__main__':main()
