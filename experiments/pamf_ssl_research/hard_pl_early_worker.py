"""E20 real checkpoint first;matched native10+20 and both seven-view readouts."""
import argparse
import gc
from pathlib import Path
from experiments.cric_dura_comparison.io import read,write
from .hard_pl_early import validate,METHODS
from .hard_pl_early_train import source_parents,objective,train


def smoke(rt):
    from .hard_pl_worker import smoke as original
    original(rt,validate_config=validate,max_rows=2,objective_cls=objective,
             source_parents_fn=source_parents,parent_epoch=20)
    from .hard_pl_mixstyle_source import run
    selected=dict(rt,rows=[row for row in rt['rows'] if row['method']==METHODS[1]])
    source=run(selected,validate_config=validate,source_parents_fn=source_parents,parent_epoch=20)
    if source['status']!='PASS':
        path=Path(rt['run_root'])/'smoke.json';value=read(path);value['status']='SCIENTIFIC_FALSIFIED'
        write(path,value,exclusive=False)
        raise RuntimeError('Predeclared E20 source identity falsification failed;no formal row started')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=('smoke','row','score'));p.add_argument('--config',required=True)
    a=p.parse_args();value=read(a.config)
    if a.action=='smoke':smoke(value)
    elif a.action=='row':
        from experiments.cric_dura_comparison.bootstrap import activate,numerical_policy
        c=read(value['train_config']);activate(c);numerical_policy();train(c);gc.collect()
        import torch
        torch.cuda.empty_cache()
        from .source_diagnostics import run
        run(c,Path(c['output_root'])/'source_V_diagnostic.json');gc.collect();torch.cuda.empty_cache()
        from experiments.cric_dura_comparison.evaluation import predict
        predict(read(value['prediction_config']))
    else:
        from experiments.cric_dura_comparison.evaluation import score
        score(value)
