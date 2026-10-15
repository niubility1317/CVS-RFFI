"""Filter only existing source physical roles before any model/teacher creation."""
import copy
import torch
from . import design as d
from experiments.cvs_u_schedule50.counterfactual import _preserve_rng

def selected_excluding(dataset,held,role):
    if role not in ('L_s','U_s','V') or held not in d.RX:raise ValueError('Original source role and fixed allowed RX required')
    if (role=='U_s')!=bool(getattr(dataset,'hide',False)):raise ValueError('Source U masking differs')
    expected={'L_s':6300,'U_s':56700,'V':27000}[role]
    if len(dataset)!=expected or len(dataset.index)!=expected or len(dataset.selected)!=expected:raise ValueError('Complete original source role required')
    # Do not read any U TX field or IQ to select the receiver mask.
    kept=[i for i,item in enumerate(dataset.index) if int(item.rx_i)!=held]
    if len(kept)!=expected*4//5 or {int(dataset.index[i].rx_i) for i in kept}!=set(d.FIT):raise ValueError('Four-RX source mask differs')
    return [int(dataset.selected[i]) for i in kept]

class BudgetU:
    def __init__(self,loader):self.original=loader;self.dataset=loader.dataset;self.sampler=loader.sampler
    def __len__(self):return d.STEPS
    def __iter__(self):return iter(self.original)

def prepare(native,ctx):
    if ctx['named_test_loaders'] or ctx.get('balanced_train_sampler') is not None:raise ValueError('Only original unbalanced source carrier is permitted')
    original=ctx['train_loader'].dataset
    cells={}
    for i,item in enumerate(original.index):cells.setdefault((int(item.rx_i),int(item.day_i),int(item.tx_i)),[]).append(i)
    if len(cells)!=90 or any(len(indices)!=70 for indices in cells.values()):raise ValueError('Original L physical cells differ')
    anchors=[i for key in sorted(cells) if key[0] in d.FIT for i in cells[key][:2]]
    held=[i for key in sorted(cells) if key[0]==d.HELD for i in cells[key]]
    def materialize(indices):
        x=[];y=[];rx=[];ids=[]
        with _preserve_rng():
            for index in indices:
                packet,label,domain,meta=original[index]
                identity=tuple(int(meta[k]) for k in ('eq_i','rx_i','day_i','sig_i','base_index'))
                if int(label)!=int(original.index[index].tx_i) or packet.shape!=(2,256) or not bool(torch.isfinite(packet).all()):raise ValueError('Visible source IQ or physical label differs')
                x.append(packet);y.append(int(label));rx.append(identity[1]);ids.append(identity)
        if len(set(ids))!=len(ids):raise ValueError('Duplicate source packet')
        return dict(x=torch.stack(x),y=torch.tensor(y),rx=torch.tensor(rx),ids=ids,role='L_s')
    anchor,evaluation=materialize(anchors),materialize(held)
    if len(anchor['y'])!=144 or len(evaluation['y'])!=1260 or set(anchor['ids'])&set(evaluation['ids']):raise ValueError('Anchor/evaluation physical separation differs')
    mapping=[('train_loader','L_s'),('probe_train_loader','L_s'),('unlabeled_loader','U_s'),('val_loader','V'),('source_calibration_loader','V')]
    for name,role in mapping:
        loader=ctx[name];ds=loader.dataset;selected=selected_excluding(ds,d.HELD,role)
        filtered=native.WiSigSubsetDataset(ds.base,selected,split_source=ds.split_source,transform=None)
        if bool(getattr(filtered,'hide',False))!=(role=='U_s'):raise ValueError('Filtered U must remain label-blind')
        ctx[name]=native.make_loader(filtered,loader.batch_size,name in ('train_loader','unlabeled_loader'),0,torch.device('cuda:0'),name=='train_loader',2)
    if len(ctx['unlabeled_loader'])!=178:raise ValueError('Filtered U must contain177 full256 batches and one48 batch')
    ctx['unlabeled_loader']=BudgetU(ctx['unlabeled_loader'])
    info=copy.deepcopy(ctx['split_info']);info.pop('source_split_receipt',None)
    info.update(labeled_size=5040,unlabeled_size=45360,source_val_size=21600,source_calibration_size=21600,source_selection_size=21600,
        source_fold=dict(held_RX=d.HELD,fit_RX=list(d.FIT),carrier_receipt_ref=d.SOURCE,actual_role_ids='source_contract.json',fixed_step_budget=222,U_observations_per_pseudo_epoch=56624))
    ctx['split_info']=info
    return anchor,evaluation
