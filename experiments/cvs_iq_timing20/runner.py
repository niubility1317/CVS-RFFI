"""Use the immutable, verified F code dependency; never an inherited model."""
import importlib.util
import json
import sys
import types
from pathlib import Path


def bootstrap():
    here=Path(__file__).resolve().parents[2]
    af=Path(json.loads((here/'runtime.json').read_text(encoding='utf8'))['AF'])
    request=json.loads((af/'experiments/cvs_rdrop_sinc_units/configs/matrix.json').read_text(encoding='utf8'))
    original,fold=(Path(request[k]) for k in ('dependency_release','fold_release'))
    sys.path[:0]=[str(original),str(original/'code'),str(original/'experiments/adv3b02_xuc/code')]
    for suffix,root in (('cvs_iq_timing20',here),('cvs_rdrop_sinc_units',af),('cvs_sigtime_rdrop_legacy',af),('cvs_rdrop_fold20',af),('cvs_sigtime_locked20',fold),('cvs_rx_u_increment20',fold)):
        name='experiments.'+suffix;module=types.ModuleType(name);module.__path__=[str(root/'experiments'/suffix)];sys.modules[name]=module
    name='experiments.cvs_feature_sequence50.source'
    spec=importlib.util.spec_from_file_location(name,fold/'experiments/cvs_feature_sequence50/source.py');module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module)
    return here


if __name__=='__main__':
    bootstrap()
    from experiments.cvs_iq_timing20 import design as d
    if len(sys.argv)==3 and sys.argv[1]=='--worker':
        row=next((r for r in d.rows() if r['row_id']==sys.argv[2]),None)
        if row is None:raise ValueError('Unregistered F timing worker')
        from experiments.cvs_iq_timing20.source import train
        train(d.config(row))
    elif len(sys.argv)==1:
        from experiments.cvs_iq_timing20.dispatch import run
        run()
    else:raise ValueError('Unsupported F timing command')
