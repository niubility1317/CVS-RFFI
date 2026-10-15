"""Exactly matched TX/day counts, RX exclusion over the entire scratch lineage."""
from collections import defaultdict,Counter
import hashlib
from . import design as d
from experiments.cvs_urban_branch50.inner_source import partition,FOLD_SEED


def select(index,opaque,arm):
    if arm not in d.ARMS:raise ValueError('Registered source subset required')
    available,dev=partition(index,opaque);cells=defaultdict(list)
    for i in available:
        v=index[i];cells[int(v.tx_i),int(v.rx_i),int(v.day_i)].append(i)
    if any(len(v)!=62 for v in cells.values()):raise ValueError('Expected same5580 internal train fold')
    if arm=='EXCLUDE_RX1':chosen=[i for i in available if int(index[i].rx_i)!=d.HELD_RX]
    else:
        chosen=[]
        for group,(tx,day) in enumerate(( (tx,day) for tx in range(6) for day in (1,2,3) )):
            extra={(group*3+j)%5 for j in range(3)}
            for ri,rx in enumerate((1,3,4,6,8)):
                ordered=sorted(cells[tx,rx,day],key=lambda i:hashlib.sha256(
                    (str(d.SUBSET_SEED)+'/'+opaque(index[i])).encode()).digest())
                chosen+=ordered[:49+int(ri in extra)]
    chosen=sorted(chosen);unselected=sorted(set(available)-set(chosen))
    counts=Counter((int(index[i].tx_i),int(index[i].day_i)) for i in chosen)
    if len(chosen)!=4464 or len(unselected)!=1116 or set(chosen)&set(dev) or set(counts.values())!={248}:
        raise ValueError('Matched teacher source budget/physical roles failed')
    receivers={int(index[i].rx_i) for i in chosen}
    if receivers!=({3,4,6,8} if arm=='EXCLUDE_RX1' else {1,3,4,6,8}):raise ValueError('Teacher RX exposure differs')
    return chosen,dev,unselected


def install(ctx,args,native,device):
    from cvsrffi.game_tracking.data import opaque_id
    original=ctx['train_loader'].dataset
    row=next(v for v in d.rows() if v['row_id']==args.candidate_id)
    train,dev,unused=select(original.index,opaque_id,row['arm'])
    if ctx['balanced_train_sampler'] is not None:raise ValueError('Retain native shuffled L loader')
    def dataset(indices,role):
        return native.WiSigSubsetDataset(original.base,[int(original.selected[i]) for i in indices],split_source=role)
    train_ds=dataset(train,'ssdg_labeled_tx_visible_crossfit_teacher');dev_ds=dataset(dev,'ssdg_L_dev_readonly')
    def loader(ds,batch,shuffle,drop):
        return native.make_loader(ds,batch,shuffle,int(args.num_workers),device,drop,int(args.prefetch_factor))
    ctx['train_loader']=loader(train_ds,128,True,True);ctx['probe_train_loader']=loader(train_ds,int(args.eval_batch_size),False,False)
    ctx['inner_dev_loader']=loader(dev_ds,90,False,False)
    roles={key:sorted(opaque_id(original.index[i]) for i in values) for key,values in dict(L_train=train,L_dev=dev,L_unused=unused).items()}
    contract=dict(schema='phase1_RX1_teacher_source_v1',fold_seed=FOLD_SEED,subset_seed=d.SUBSET_SEED,arm=row['arm'],
        counts=dict(L_train=4464,L_dev=720,L_unused=1116,U_s=56700,V=27000),role_ids=roles,
        train_receivers=sorted({int(original.index[i].rx_i) for i in train}),
        TX_day_train_counts=[dict(TX=tx,day=day,count=248) for tx in range(6) for day in (1,2,3)],
        train_native_base_indices=[int(original.selected[i]) for i in train],
        L_dev_updates=False,V_role='health_only',U_iterator=False,U_forward_packets=0,
        initialization='scratch; no optimizer/EMA/model/feature-statistic inheritance',
        normalization='native per-record RMS only; no fitted cross-RX input statistic',outer_roles_unchanged=True)
    meta=[dict(physical_id=opaque_id(original.index[i]),rx_i=int(original.index[i].rx_i),day_i=int(original.index[i].day_i),
        eq_i=int(original.index[i].eq_i),sig_i=int(original.index[i].sig_i),base_index=int(original.selected[i])) for i in dev]
    d.write(__import__('pathlib').Path(args.output_dir)/'L_dev_native_metadata.json',dict(source_only=True,records=meta))
    ctx['inner_source_contract']=contract;ctx['split_info']=dict(ctx['split_info'],outer_labeled_size=6300,labeled_size=4464,
        inner_development_size=720,inner_source_contract=contract)
    return contract
