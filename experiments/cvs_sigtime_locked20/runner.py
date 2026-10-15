import importlib.util,json,sys,types,gc
from pathlib import Path

def main():
    here=Path(__file__).resolve().parents[2];request=json.loads(Path(sys.argv[1]).read_text(encoding='utf-8'))
    previous=Path(request['dependency_release']);sys.path[:0]=[str(previous),str(previous/'code'),str(previous/'experiments/adv3b02_xuc/code')]
    for suffix in ('cvs_sigtime_locked20','cvs_rx_u_increment20'):
        name='experiments.'+suffix;package=types.ModuleType(name);package.__path__=[str(here/'experiments'/suffix)];sys.modules[name]=package
    name='experiments.cvs_feature_sequence50.source';spec=importlib.util.spec_from_file_location(name,here/'experiments/cvs_feature_sequence50/source.py')
    module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module)
    from experiments.cvs_sigtime_locked20 import design as d
    from experiments.cvs_sigtime_locked20.source import train
    import torch
    if request['rows']!=[d.config(mode) for mode in ('ELEMENT','NONE','LOCKED')]:raise ValueError('Fixed three-condition matrix required')
    root=Path(d.PROJECT)/'runs'/d.RUN
    if root.exists():raise FileExistsError('Preserve existing matrix, no automatic retry')
    results={}
    for c in request['rows']:
        results[c['arm']]=train(c);gc.collect();torch.cuda.empty_cache()
    locked=results['LOCKED'];controls=[results[key] for key in ('ELEMENT','NONE')]
    qualified=all(row['source_health'] for row in results.values()) and all(
        locked['exports']['FINAL']['errors']<control['exports']['FINAL']['errors'] and
        locked['exports']['FINAL']['mean_CE']<=control['exports']['FINAL']['mean_CE'] for control in controls)
    d.write(root/'matrix_result.json',dict(status='SOURCE_MATRIX_COMPLETE',results=results,primary_gate=d.GATE,
        primary_source_qualification=qualified,actual_optimizer_steps=13320,target_access=False,goal84_completed=False))

if __name__=='__main__':main()
