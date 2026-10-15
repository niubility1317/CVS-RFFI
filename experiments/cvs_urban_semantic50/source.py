"""Same native E50/old-gated U joint optimizer path and physical L fold."""
from pathlib import Path
from . import design,model
from .observer import Observer
from experiments.cvs_urban_sigtime50.source import train as original_train
from experiments.cvs_urban_branch50.inner_source import install as inner_install


def adapter(ctx,args,native,device):
    contract=inner_install(ctx,args,native,device)
    design.write(Path(args.output_dir)/'native_train_indices.json',dict(source_only=True,
        base_indices=[int(i) for i in ctx['train_loader'].dataset.selected]))
    return contract


def train(c):
    observer=Observer(c);original_train(c,design=design,model_api=model,observer=observer,data_adapter=adapter)
    return observer.finish()
