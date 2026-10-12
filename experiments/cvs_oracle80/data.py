from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path
from . import design as d

def supervised_split(dataset,*,labeled_ratio=.8,unlabeled_ratio=0.,source_val_ratio=.2):
    if (labeled_ratio,unlabeled_ratio,source_val_ratio)!=(.8,0.,.2):raise ValueError('Oracle ratios changed')
    groups=defaultdict(list)
    for i,it in enumerate(dataset.index):groups[(it.tx_i,it.rx_i,it.day_i,it.eq_i)].append((it.sig_i,i))
    train=[];val=[]
    for key,pairs in sorted(groups.items()):
        ordered=[i for _,i in sorted(pairs)]
        if len(ordered)!=1000 or key[0] not in range(6) or key[1] not in (1,3,4,6,8) or key[2] not in (1,2,3):raise ValueError('Source axes/count differ')
        train.extend(ordered[:800]);val.extend(ordered[800:])
    if len(groups)!=90:raise ValueError('Source group coverage differs')
    return sorted(train),[],sorted(val)

@contextmanager
def oracle_native(c,training=True):
    from experiments.cvs_feature_disentangle.runtime import installed
    with installed(c,training=training) as native:
        old_split=native.split_tx_rx_day_1_7_2;old_loader=native.make_loader
        def loader(ds,batch,shuffle,workers,device,drop_last,prefetch):
            # Complete training pass and valid empty-U loader. No U iteration.
            if 'labeled' in str(getattr(ds,'split_source','')) and 'unlabeled' not in str(getattr(ds,'split_source','')):drop_last=False
            return old_loader(ds,batch,shuffle,workers,device,drop_last,prefetch)
        native.split_tx_rx_day_1_7_2=supervised_split;native.make_loader=loader
        try:yield native
        finally:native.split_tx_rx_day_1_7_2=old_split;native.make_loader=old_loader

def check_roles(roles):
    sets={k:set(v) for k,v in roles.items()}
    if set(sets)!={'L_s','U_s','V'} or {k:len(v) for k,v in sets.items()}!={'L_s':72000,'U_s':0,'V':18000}:raise ValueError('Oracle role counts differ')
    if any(len(v)!=len(roles[k]) for k,v in sets.items()) or sets['L_s']&sets['V']:raise ValueError('Duplicate/overlapping physical IDs')

def source_contract():
    import torch
    from cvsrffi.xuc_fusion.native import role_ids_from_native
    c=d.config(d.rows()[0]);a=d.make_args(c,'cpu')
    with oracle_native(c) as n:
        ctx=n._build_ssdg_wisig_data(a,torch.device('cpu'))
        roles=role_ids_from_native(ctx);check_roles(roles)
        if ctx['named_test_loaders'] or ctx['num_domains']!=15 or len(ctx['train_loader'])!=563:raise ValueError('Source-only loader shape differs')
        classes=list(ctx['train_loader'].dataset.tx_list)
        if classes!=d.CLASSES:raise ValueError('Class order differs')
        # Reuse existing target observations, verifying new source roles are disjoint.
        import numpy as np
        manifest=d.read(Path(d.CAPSULE)/'manifest.json')
        if manifest.get('phase1_data_status',manifest.get('status'))!='VALIDATED_ONCE':raise ValueError('Existing capsule is not validated')
        with np.load(Path(d.CAPSULE)/'index.npz',allow_pickle=False) as ix:ids=ix['ids'].copy()
        if len(ids)!=d.COUNT or set(ids)&(set(roles['L_s'])|set(roles['V'])):raise ValueError('Source/target physical overlap')
    if Path(d.SOURCE).exists():raise FileExistsError('Existing source contract; reconcile')
    d.write(d.SOURCE,dict(status='VALIDATED_ONCE',dataset=c['dataset'],oracle_data=c['oracle_data'],class_txs=classes,
        role_ids=roles,source_only=True,source_target_disjoint=True,exception='User explicitly requested full-label80/20 supervised reference'))
    return dict(status='PASS',role_counts={k:len(v) for k,v in roles.items()},steps_per_epoch=563,query_read=False)
