"""A fixed physical L-only development fold; never opens U truth or target IQ."""
from collections import defaultdict
import hashlib

FOLD_SEED=2026101503
TRAIN_COUNT,DEV_COUNT=5580,720


def partition(index,opaque):
    cells=defaultdict(list)
    for i,item in enumerate(index):
        cells[(int(item.tx_i),int(item.rx_i),int(item.day_i))].append(i)
    expected={(tx,rx,day) for tx in range(6) for rx in (1,3,4,6,8) for day in (1,2,3)}
    if set(cells)!=expected or any(len(v)!=70 for v in cells.values()):
        raise ValueError('Expected original L6300: 90 physical TX/RX/day cells of 70')
    dev=[]
    for key in sorted(cells):
        # A deterministic random ordering of physical IDs, independent of model
        # seed/channel/loss. This is sampling, not a release hash or new seal.
        order=sorted(cells[key],key=lambda i:hashlib.sha256(
            (str(FOLD_SEED)+'/'+opaque(index[i])).encode('utf-8')).digest())
        dev.extend(order[:8])
    held=set(dev);train=[i for i in range(len(index)) if i not in held]
    if len(train)!=TRAIN_COUNT or len(dev)!=DEV_COUNT or len(held)!=DEV_COUNT:
        raise ValueError('Internal source fold size mismatch')
    return train,dev


def install(ctx,args,native,device):
    """Called after verifying the unchanged outer physical role contract."""
    from cvsrffi.game_tracking.data import opaque_id
    original=ctx['train_loader'].dataset
    train,dev=partition(original.index,opaque_id)
    if ctx['balanced_train_sampler'] is not None:
        raise ValueError('This registered fold retains the original shuffled L loader')
    train_ds=native.WiSigSubsetDataset(original.base,[int(original.selected[i]) for i in train],
        split_source='ssdg_labeled_tx_visible_inner_train')
    dev_ds=native.WiSigSubsetDataset(original.base,[int(original.selected[i]) for i in dev],
        split_source='ssdg_labeled_tx_visible_inner_dev_readonly')
    def loader(ds,batch,shuffle,drop):
        return native.make_loader(ds,batch,shuffle,int(args.num_workers),device,drop,int(args.prefetch_factor))
    ctx['train_loader']=loader(train_ds,int(args.batch_size),True,True)
    ctx['probe_train_loader']=loader(train_ds,int(args.eval_batch_size),False,False)
    ctx['inner_dev_loader']=loader(dev_ds,90,False,False)
    train_ids=sorted(opaque_id(i) for i in train_ds.index);dev_ids=sorted(opaque_id(i) for i in dev_ds.index)
    if set(train_ids)&set(dev_ids) or set(train_ids)|set(dev_ids)!=set(opaque_id(i) for i in original.index):
        raise ValueError('L internal physical separation failed')
    contract=dict(schema='phase1_L_internal_dev_v1',fold_seed=FOLD_SEED,
        counts=dict(L_train=TRAIN_COUNT,L_dev=DEV_COUNT,U_s=56700,V=27000),
        role_ids=dict(L_train=train_ids,L_dev=dev_ids),
        L_dev_updates=False,V_role='health_only',outer_roles_unchanged=True,
        initialization='scratch only; full-L historical checkpoints are not matching ancestors',
        historical_full_L_exposure=True)
    ctx['inner_source_contract']=contract
    ctx['split_info']=dict(ctx['split_info'],outer_labeled_size=6300,labeled_size=TRAIN_COUNT,
        inner_development_size=DEV_COUNT,inner_source_contract=contract)
    return contract


def panel(loader):
    """Materialize only the held-out legal L developer loader, without updates."""
    import torch
    from cvsrffi.game_tracking.data import opaque_id
    chunks=list(loader);ds=loader.dataset
    result=dict(x=torch.cat([v[0] for v in chunks]),y=torch.cat([v[1] for v in chunks]),
        rx=torch.cat([v[3]['rx_i'] for v in chunks]),day=torch.cat([v[3]['day_i'] for v in chunks]),
        ids=[opaque_id(i) for i in ds.index])
    if len(result['ids'])!=DEV_COUNT or len(set(result['ids']))!=DEV_COUNT:
        raise ValueError('L_dev loader must cover exactly 720 physical records')
    return result
