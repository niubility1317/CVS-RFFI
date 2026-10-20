"""Route only future evaluation execs; an already-running trainer stays intact."""
import json
import os
from pathlib import Path
import runpy
import sys


def main():
    here=Path(__file__).resolve()
    route=json.loads((here.parent/'evaluation_handoff.json').read_text())
    mode=sys.argv[sys.argv.index('--mode')+1] if '--mode' in sys.argv else None
    if mode in ('predict','score'):
        target=Path(route['evaluation_runner'])
        os.execve(sys.executable,[sys.executable,'-u',str(target),*sys.argv[1:]],dict(os.environ))
    else:
        runpy.run_path(route['original_runner'],run_name='__main__')


if __name__=='__main__':main()
