"""Original F21 parents; accepted-only U style and complete dual-readout testing."""
import argparse
import gc
from pathlib import Path
from experiments.cric_dura_comparison.io import read,write
from .hard_pl_mixstyle import validate
from .hard_pl_mixstyle_objective import HardPLMixStyleObjective


def train(c,device='cuda:0'):
    from .hard_pl_train import train as original
    return original(c,device=device,validate_config=validate,objective_cls=HardPLMixStyleObjective)


def smoke(rt):
    from .hard_pl_worker import smoke as original
    original(rt,validate_config=validate,max_rows=1,objective_cls=HardPLMixStyleObjective)
    from .hard_pl_mixstyle_source import run
    source=run(rt)
    if source['status']!='PASS':
        path=Path(rt['run_root'])/'smoke.json';value=read(path);value['status']='SCIENTIFIC_FALSIFIED'
        write(path,value,exclusive=False)
        raise RuntimeError('Predeclared source identity-pair falsification failed;no formal row started')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=('smoke','row','score'));parser.add_argument('--config',required=True)
    a=parser.parse_args();value=read(a.config)
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
