"""Source-only immutable entry; unchanged native data builder is reused."""
import json,sys,types
from pathlib import Path

def main():
    request=json.loads(Path(sys.argv[1]).read_text(encoding='utf-8'))
    here=Path(__file__).resolve().parents[2];previous=Path(request['source_release'])
    sys.path[:0]=[str(previous),str(previous/'code')]
    parent=types.ModuleType('experiments.cvs_repetition_feasibility')
    parent.__path__=[str(here/'experiments/cvs_repetition_feasibility')]
    sys.modules[parent.__name__]=parent
    from experiments.cvs_repetition_feasibility.source import run
    run(request)

if __name__=='__main__':main()
