"""Reuse fixed source release; install only current diagnostic modules."""
import json
import sys
import types
from pathlib import Path


def main():
    request=json.loads(Path(sys.argv[1]).read_text(encoding='utf-8'))
    here=Path(__file__).resolve().parents[2]
    previous=Path(request['source_release'])
    sys.path[:0]=[str(previous),str(previous/'code')]
    parent=types.ModuleType('experiments.cvs_crossrx_credit50')
    parent.__path__=[str(here/'experiments/cvs_crossrx_credit50')]
    sys.modules[parent.__name__]=parent
    for name in ('cvs_semantic_order50','cvs_u_schedule50'):
        package=types.ModuleType('experiments.'+name)
        package.__path__=[str(here/'experiments'/name)]
        sys.modules[package.__name__]=package
    import importlib.util
    name='experiments.cvs_source_semantic_geometry'
    spec=importlib.util.spec_from_file_location(name,here/'experiments/cvs_source_semantic_geometry.py')
    module=importlib.util.module_from_spec(spec)
    sys.modules[name]=module
    spec.loader.exec_module(module)
    from experiments.cvs_crossrx_credit50.source_probe import run
    run(request)


if __name__=='__main__':main()
