"""Non-fitting source geometry on a fixed balanced V180, after final freeze."""
import torch
from experiments.cvs_sigtime_heldcredit.held_credit import physical_ids,fresh_parameter_caches


def sample(dataset):
    if len(dataset)!=27000 or getattr(dataset,'hide',True):
        raise ValueError('Actual visible source V27000 is required')
    cells={}
    for i,item in enumerate(dataset.index):
        key=(int(item.tx_i),int(item.rx_i),int(item.day_i))
        if key[0] not in range(6) or key[1] not in (1,3,4,6,8) or key[2] not in (1,2,3):
            raise ValueError('Source V physical roles differ')
        cells.setdefault(key,[]).append(i)
    if len(cells)!=90 or any(len(indices)!=300 for indices in cells.values()):
        raise ValueError('Fixed V90 cells x300 samples differ')
    xs=[];ys=[];rxs=[];days=[];ids=[]
    for key in sorted(cells):
        for ordinal in (0,1):
            i=cells[key][ordinal];x,y,_,meta=dataset[i]
            actual=tuple(int(meta[k]) for k in ('eq_i','rx_i','day_i','sig_i','base_index'))
            item=dataset.index[i]
            expected=(int(item.eq_i),int(item.rx_i),int(item.day_i),int(item.sig_i),int(dataset.selected[i]))
            if actual!=expected or int(y)!=key[0] or x.shape!=(2,256) or not torch.isfinite(x).all():
                raise ValueError('Source V IQ, class and physical metadata disagree')
            xs.append(x.detach().cpu());ys.append(key[0]);rxs.append(key[1]);days.append(key[2]);ids.append(actual)
    physical_ids(ids,180)
    return torch.stack(xs),torch.tensor(ys),torch.tensor(rxs),torch.tensor(days)


def geometry(z,y,rx,day):
    if z.shape!=(180,160) or not torch.isfinite(z).all():raise ValueError('Raw deployed V features differ')
    unit=torch.nn.functional.normalize(z.double(),dim=1);cos=unit@unit.T
    same_y=y[:,None]==y[None,:];same_rx=rx[:,None]==rx[None,:];same_day=day[:,None]==day[None,:]
    pos=same_y&~same_rx&same_day;neg=~same_y&same_rx&same_day;cross_neg=~same_y&~same_rx&same_day
    if int(pos.sum())!=1440 or int(neg.sum())!=1800 or int(cross_neg.sum())!=7200:
        raise ValueError('Fixed source relation pair counts differ')
    positive=float(cos[pos].mean());negative=float(cos[neg].mean());cross_negative=float(cos[cross_neg].mean())
    return dict(sameTX_crossRX_cosine=positive,differentTX_sameRX_cosine=negative,
        differentTX_crossRX_cosine=cross_negative,identity_margin=positive-negative,
        RX_excess_similarity=negative-cross_negative,mean_feature_norm=float(z.double().norm(dim=1).mean()))


def observe(model,initial,bank,selected_names):
    if model.training or any(p.requires_grad for p in model.parameters()):raise ValueError('Frozen eval observation required')
    if set(initial)!={'model','scratch_only'} or initial['scratch_only'] is not True:
        raise ValueError('Own initial_smoke schema differs')
    final={k:v.detach().clone() for k,v in model.state_dict().items()}
    x,y,rx,day=bank
    def extract():
        with fresh_parameter_caches(model),torch.no_grad(),torch.autocast(device_type='cpu',enabled=False):
            return model.id_backbone.encoder.features(x.float()).detach().clone()
    z_final=extract()
    try:
        model.load_state_dict(initial['model'],strict=True);z_initial=extract()
    finally:model.load_state_dict(final,strict=True)
    if any(not torch.equal(v,model.state_dict()[k]) for k,v in final.items()):raise ValueError('Frozen state was not restored')
    changes={name:float((final[name].double()-initial['model'][name].double()).norm()) for name in selected_names}
    return dict(initial=geometry(z_initial,y,rx,day),final=geometry(z_final,y,rx,day),
        initial_final_feature_RMS=float((z_final.double()-z_initial.double()).square().mean().sqrt()),
        selected_parameter_changes=changes,selected_tensors_changed=sum(v>0 for v in changes.values()),
        packet_network_forwards=360,head_fit_updates=0,optimizer_updates=0,source_V_only=True,
        scope='V180 balanced physical packets, known source RX/TX; descriptive geometry only, no unseen RX or new TX efficacy/selection claim')
