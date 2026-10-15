"""Metadata-only source filtering; held V never enters persistent updates."""
import copy
import torch
from . import design as d


def selected(dataset,role,held=False,held_rx=None):
    held_rx=d.HELD if held_rx is None else held_rx
    allowed={1,3,4,6,8}
    if held_rx not in allowed:raise ValueError('Only allowed source RX may be held out')
    expected={'L_s':6300,'U_s':56700,'V':27000}[role]
    if len(dataset)!=expected or len(dataset.index)!=expected or len(dataset.selected)!=expected or bool(getattr(dataset,'hide',False))!=(role=='U_s'):
        raise ValueError('Original source role or U masking differs')
    # Never inspect an unlabeled TX field, label or IQ to select this mask.
    positions=[i for i,item in enumerate(dataset.index) if (int(item.rx_i)==held_rx)==held]
    actual={int(dataset.index[i].rx_i) for i in positions}
    if len(positions)!=expected*(1 if held else 4)//5 or actual!=({held_rx} if held else allowed-{held_rx}):
        raise ValueError('Source RX selection size differs')
    return [int(dataset.selected[i]) for i in positions]


class RecordedLoader:
    def __init__(self,loader,role,record,steps=None):
        self.original=loader;self.role=role;self.record=record;self.steps=steps
    def __getattr__(self,name):return getattr(self.original,name)
    def __len__(self):return len(self.original) if self.steps is None else self.steps
    def __iter__(self):
        for batch in self.original:
            self.record(self.role,batch)
            yield batch


def seed_loader(loader,seed):
    loader.generator=torch.Generator().manual_seed(seed)
    if isinstance(loader.sampler,torch.utils.data.RandomSampler):
        loader.sampler.generator=torch.Generator().manual_seed(seed+1)
    return loader


def prepare(native,ctx,record,held_rx=None):
    held_rx=d.HELD if held_rx is None else held_rx
    fit=tuple(r for r in (1,3,4,6,8) if r!=held_rx)
    if ctx['named_test_loaders'] or ctx.get('balanced_train_sampler') is not None:
        raise ValueError('Only original unbalanced source carrier allowed')
    original_val=ctx['val_loader'];ds=original_val.dataset
    held_ds=native.WiSigSubsetDataset(ds.base,selected(ds,'V',held=True,held_rx=held_rx),split_source=ds.split_source,transform=None)
    held_loader=seed_loader(native.make_loader(held_ds,original_val.batch_size,False,0,torch.device('cuda:0'),False,2),20261016099)
    mapping=[('train_loader','L_s'),('probe_train_loader','L_s'),('unlabeled_loader','U_s'),('val_loader','V'),('source_calibration_loader','V')]
    for index,(name,role) in enumerate(mapping):
        loader=ctx[name];ds=loader.dataset
        filtered=native.WiSigSubsetDataset(ds.base,selected(ds,role,held_rx=held_rx),split_source=ds.split_source,transform=None)
        if bool(getattr(filtered,'hide',False))!=(role=='U_s'):raise ValueError('U masking changed after filtering')
        loader=native.make_loader(filtered,loader.batch_size,name in ('train_loader','unlabeled_loader'),0,torch.device('cuda:0'),name=='train_loader',2)
        loader=seed_loader(loader,20261016053+index*100)
        if name=='unlabeled_loader' and len(loader)!=178:raise ValueError('Four RX U loader must have178 batches')
        if name in ('train_loader','unlabeled_loader'):
            loader=RecordedLoader(loader,role,record,steps=222 if role=='U_s' else None)
        ctx[name]=loader
    info=copy.deepcopy(ctx['split_info']);info.pop('source_split_receipt',None)
    info.update(labeled_size=5040,unlabeled_size=45360,source_val_size=21600,source_calibration_size=21600,source_selection_size=21600,
        source_fold=dict(held_RX=held_rx,fit_RX=list(fit),actual_role_ids='source_contract.json',fixed_step_budget=222,U_observations_per_pseudo_epoch=56624))
    ctx['split_info']=info
    return held_loader
