import importlib.util,json,sys,types,gc
from pathlib import Path

def main():
    here=Path(__file__).resolve().parents[2];request=json.loads(Path(sys.argv[1]).read_text(encoding='utf-8'))
    original,fold=Path(request['dependency_release']),Path(request['fold_release'])
    sys.path[:0]=[str(original),str(original/'code'),str(original/'experiments/adv3b02_xuc/code')]
    for suffix,root in (('cvs_sigtime_cleanpath20',here),('cvs_sigtime_locked20',fold),('cvs_rx_u_increment20',fold)):
        name='experiments.'+suffix;package=types.ModuleType(name);package.__path__=[str(root/'experiments'/suffix)];sys.modules[name]=package
    name='experiments.cvs_feature_sequence50.source';spec=importlib.util.spec_from_file_location(name,fold/'experiments/cvs_feature_sequence50/source.py')
    module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module)
    from experiments.cvs_sigtime_cleanpath20 import design as d
    from experiments.cvs_sigtime_cleanpath20.source import train
    import torch
    if request['rows']!=[d.config(mode) for mode in d.MODES]:raise ValueError('Fixed new-seed three-mode matrix required')
    root=Path(d.PROJECT)/'runs'/d.RUN
    if root.exists():raise FileExistsError('Preserve existing source run')
    results={}
    for c in request['rows']:
        results[c['arm']]=train(c);gc.collect();torch.cuda.empty_cache()
    d.write(root/'matrix_result.json',dict(status='SOURCE_MATRIX_COMPLETE',results=results,qualification=d.qualify(results),
        actual_optimizer_steps=13320,target_access=False,goal84_completed=False))
if __name__=='__main__':main()
