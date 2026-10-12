"""Historical sequential split, isolated from the current weak-label contract."""
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path
from . import design as d

def historical_split(dataset,*,labeled_ratio=.1,unlabeled_ratio=.7,source_val_ratio=.2):
    if (labeled_ratio,unlabeled_ratio,source_val_ratio)!=(.1,.7,.2):raise ValueError('Historical ratios changed')
    groups=defaultdict(list)
    for i,it in enumerate(dataset.index):groups[(it.tx_i,it.rx_i,it.day_i,it.eq_i)].append((it.sig_i,i))
    result=[[],[],[]]
    for key,pairs in sorted(groups.items()):
        ordered=[i for _,i in sorted(pairs)];n=len(ordered)
        if n!=1000:raise ValueError('Historical group must contain all1000 physical packets')
        if key[0] not in range(6) or key[1] not in range(7) or key[2] not in (0,1):raise ValueError('Historical source axes differ')
        for out,indices in zip(result,(ordered[:100],ordered[100:800],ordered[800:])):out.extend(indices)
    if len(groups)!=84:raise ValueError('Historical source group coverage differs')
    return tuple(sorted(x) for x in result)

@contextmanager
def historical_native(c,training=True):
    from experiments.cvs_feature_disentangle.runtime import installed
    with installed(c,training=training) as native:
        old=native.split_tx_rx_day_1_7_2
        native.split_tx_rx_day_1_7_2=historical_split
        try:yield native
        finally:native.split_tx_rx_day_1_7_2=old

def check_roles(roles):
    sets={k:set(v) for k,v in roles.items()}
    if set(sets)!={'L_s','U_s','V'} or {k:len(v) for k,v in sets.items()}!={'L_s':8400,'U_s':58800,'V':16800}:raise ValueError('Historical role sizes differ')
    if any(len(v)!=len(roles[k]) for k,v in sets.items()) or sets['L_s']&sets['U_s'] or sets['L_s']&sets['V'] or sets['U_s']&sets['V']:raise ValueError('Duplicate or overlapping source physical IDs')

def source_contract():
    import torch
    c=d.config(d.rows()[0]);a=d.make_args(c,'cpu')
    from cvsrffi.xuc_fusion.native import role_ids_from_native
    with historical_native(c) as native:
        ctx=native._build_ssdg_wisig_data(a,torch.device('cpu'))
        if ctx['named_test_loaders'] or ctx['num_domains']!=14:raise ValueError('Source preflight built target or wrong domains')
        roles=role_ids_from_native(ctx);check_roles(roles)
        classes=list(ctx['train_loader'].dataset.tx_list)
        if classes!=d.CLASSES:raise ValueError('Historical class order differs')
        sample=ctx['unlabeled_loader'].dataset[0]
        if sample[1]!=-1 or sample[3]:raise ValueError('U TX metadata leaked')
    value=dict(status='VALIDATED_ONCE',dataset=c['dataset'],historical_data=c['historical_data'],class_txs=classes,
        role_ids=roles,seed=392002,source_only=True,historical_exception='User explicitly requested ADV3B02 10/70/20;rho=.125')
    if Path(d.SOURCE).exists():raise FileExistsError('Source contract exists; reconcile')
    d.write(d.SOURCE,value)
    return dict(status='PASS',role_counts={k:len(v) for k,v in roles.items()},num_domains=14,query_read=False)

def build_capsule():
    """Builder process only, after source freeze; predictor gets IQ and opaque IDs."""
    import numpy as np
    from experiments.cvs_phase1_overlay.model import native_modules
    native_modules()
    from dataset_wisig import load_wisig_compact_pkl,WiSigCompactDataset
    from cvsrffi.game_tracking.data import opaque_id
    from .evaluate import frozen_source
    frozen_source();out=Path(d.CAPSULE)
    if out.exists() or Path(d.TRUTH).exists():raise FileExistsError('Existing capsule; reconcile rather than overwrite')
    raw=load_wisig_compact_pkl(d.config(d.rows()[0])['dataset'])
    if list(raw['tx_list'])!=d.CLASSES or len(raw['rx_list'])!=12 or len(raw['capture_date_list'])!=4:raise ValueError('ManySig dimensions changed')
    cells=[('unseen_day_seen_rx',list(range(7)),[2,3]),('seen_day_unseen_rx',list(range(7,12)),[0,1]),('unseen_day_unseen_rx',list(range(7,12)),[2,3])]
    datasets=[]
    for name,rxs,days in cells:
        ds=WiSigCompactDataset(raw,out_len=256,crop_mode='center',normalize=True,center=False,equalized=1,rx_keep=rxs,day_keep=days,domain='rx_day',seed=392002)
        if len(ds)!=d.SPLITS[name]:raise ValueError('Historical test cell count differs')
        datasets.append((name,ds))
    source=d.read(d.SOURCE);source_ids=set().union(*(set(v) for v in source['role_ids'].values()))
    out.mkdir(parents=True,exist_ok=False)
    array=np.lib.format.open_memmap(out/'clean.npy',mode='w+',dtype=np.float32,shape=(d.COUNT,2,256))
    ids=[];ys=[];rx=[];days=[];splits=[];cursor=0
    for name,ds in datasets:
        for i,it in enumerate(ds.index):
            x,y,_,meta=ds[i];pid=opaque_id(it)
            if pid in source_ids:raise ValueError('Source/query physical overlap')
            array[cursor]=x.tolist();ids.append(pid);ys.append(y);rx.append(it.rx_i);days.append(it.day_i);splits.append(name);cursor+=1
            if cursor%16384==0:print('CAPSULE BUILD '+str(cursor)+'/'+str(d.COUNT),flush=True)
    array.flush();del array
    if cursor!=d.COUNT or len(set(ids))!=d.COUNT:raise ValueError('Test physical coverage differs')
    np.savez(out/'index.npz',ids=np.asarray(ids))
    np.savez(d.TRUTH,ids=np.asarray(ids),y=np.asarray(ys),rx=np.asarray(rx),days=np.asarray(days),splits=np.asarray(splits))
    d.write(out/'manifest.json',dict(status='VALIDATED_ONCE',count=d.COUNT,splits=d.SPLITS,classes=d.CLASSES,source_query_disjoint=True,unique_ids=True,views=['clean'],preprocessing='equalized1;center256;packetRMS;centerFalse'))
    print('HISTORICAL CAPSULE COMPLETE '+str(cursor),flush=True)
