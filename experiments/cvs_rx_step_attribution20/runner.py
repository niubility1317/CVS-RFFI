import json,sys,types
from pathlib import Path

def main():
    request=json.loads(Path(sys.argv[1]).read_text(encoding='utf-8'))
    here=Path(__file__).resolve().parents[2];previous=Path(request['source_release'])
    sys.path[:0]=[str(previous),str(previous/'code'),str(previous/'experiments/adv3b02_xuc/code')]
    package=types.ModuleType('experiments.cvs_rx_step_attribution20')
    package.__path__=[str(here/'experiments/cvs_rx_step_attribution20')];sys.modules[package.__name__]=package
    from experiments.cvs_rx_step_attribution20.source_probe import run
    run(request)

if __name__=='__main__':main()
