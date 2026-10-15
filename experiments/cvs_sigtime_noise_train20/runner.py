import importlib.util,json,sys,types
from pathlib import Path
def main():
    here=Path(__file__).resolve().parents[2];request=Path(sys.argv[1]).resolve();value=json.loads(request.read_text(encoding='utf-8'))
    original,mechanism,fold=(Path(value[key]) for key in ('dependency_release','mechanism_release','fold_release'))
    sys.path[:0]=[str(original),str(original/'code'),str(original/'experiments/adv3b02_xuc/code')]
    for suffix,root in (('cvs_sigtime_noise_train20',here),('cvs_sigtime_cleanpath20',mechanism),('cvs_sigtime_locked20',fold),('cvs_rx_u_increment20',fold)):
        name='experiments.'+suffix;package=types.ModuleType(name);package.__path__=[str(root/'experiments'/suffix)];sys.modules[name]=package
    name='experiments.cvs_feature_sequence50.source';spec=importlib.util.spec_from_file_location(name,fold/'experiments/cvs_feature_sequence50/source.py')
    module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module)
    from experiments.cvs_sigtime_noise_train20 import evaluate
    if len(sys.argv)==4 and sys.argv[2]=='--score':evaluate.score(sys.argv[3])
    elif len(sys.argv)==3 and sys.argv[2]=='--finish':evaluate.finish()
    elif len(sys.argv)==2:evaluate.run(request,Path(__file__).resolve())
    else:raise ValueError('Only fixed training/prediction pipeline or independent CPU scorer allowed')
if __name__=='__main__':main()
