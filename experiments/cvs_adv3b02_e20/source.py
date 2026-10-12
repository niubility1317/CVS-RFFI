from pathlib import Path
from . import design as d
from .data import historical_native

def validation(model,loader,device):
    import torch
    model.eval();groups={};hits=total=0
    with torch.no_grad():
        for x,y,domain,meta in loader:
            ok=model(x.to(device)).argmax(1).cpu()==y
            for rx in meta['rx_i'].unique().tolist():
                mask=meta['rx_i']==rx;g=groups.setdefault(str(rx),[0,0]);g[0]+=int(ok[mask].sum());g[1]+=int(mask.sum())
            hits+=int(ok.sum());total+=len(y)
    if total!=16800 or set(groups)!=set(map(str,range(7))):raise ValueError('Historical V coverage differs')
    rates={k:h/n for k,(h,n) in groups.items()}
    return dict(source_val_count=total,source_val_accuracy=hits/total,source_val_rx_accuracy=rates,source_val_worst_rx=min(rates.values()))

def train(c):
    from experiments.cvs_ce_epoch2040 import source as shared
    # Private worker process: reuse the exact E20 training implementation with
    # only its immutable data/config entry points replaced for this experiment.
    shared.validate=d.validate;shared.make_args=d.make_args;shared.SOURCE=d.SOURCE
    shared.installed=historical_native;shared.source_validation=validation
    shared.train(c)
