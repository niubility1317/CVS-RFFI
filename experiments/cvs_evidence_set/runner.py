"""Immutable source-analysis entry point using an unchanged PA release."""
import json,sys,types
from pathlib import Path

def main():
    request=json.loads(Path(sys.argv[1]).read_text(encoding='utf-8'))
    here=Path(__file__).resolve().parents[2]
    previous=Path(request['source_release'])
    sys.path[:0]=[str(previous),str(previous/'code')]
    # Reuse all historical source builders without editing their release;
    # inject only this committed source-analysis package into the process.
    parent=types.ModuleType('experiments.cvs_evidence_set');parent.__path__=[str(here/'experiments/cvs_evidence_set')]
    sys.modules[parent.__name__]=parent
    from experiments.cvs_evidence_set.proxy import run
    result=run(request)
    print(json.dumps(dict(status=result['status'],qualification=result['qualification'],
                         overall=result['records']['overall'],wall_seconds=result['wall_seconds'])),flush=True)

if __name__=='__main__':main()
