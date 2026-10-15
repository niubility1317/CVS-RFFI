import json,sys,types
from pathlib import Path

def main():
    here=Path(__file__).resolve().parents[2];request=Path(sys.argv[1]).resolve();value=json.loads(request.read_text(encoding='utf-8'))
    previous=Path(value['dependency_release']);sys.path[:0]=[str(previous),str(previous/'code'),str(previous/'experiments/adv3b02_xuc/code')]
    name='experiments.cvs_sigtime_ema_export20';package=types.ModuleType(name);package.__path__=[str(here/'experiments/cvs_sigtime_ema_export20')];sys.modules[name]=package
    from experiments.cvs_sigtime_ema_export20 import evaluate
    if len(sys.argv)==4 and sys.argv[2]=='--score':evaluate.score(sys.argv[3])
    elif len(sys.argv)==3 and sys.argv[2]=='--finish':evaluate.finish()
    elif len(sys.argv)==2:evaluate.run(request,Path(__file__).resolve())
    else:raise ValueError('Only the fixed pipeline or independent row scorer is allowed')

if __name__=='__main__':main()
