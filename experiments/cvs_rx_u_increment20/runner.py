"""Use the immutable R02 runtime and this committed narrow source overlay."""
import importlib.util,json,sys,types
from pathlib import Path
def main():
    here=Path(__file__).resolve().parents[2];request=json.loads(Path(sys.argv[1]).read_text(encoding='utf-8'))
    previous=Path(request['dependency_release'])
    sys.path[:0]=[str(previous),str(previous/'code'),str(previous/'experiments/adv3b02_xuc/code')]
    package=types.ModuleType('experiments.cvs_rx_u_increment20');package.__path__=[str(here/'experiments/cvs_rx_u_increment20')]
    sys.modules[package.__name__]=package
    name='experiments.cvs_feature_sequence50.source'
    spec=importlib.util.spec_from_file_location(name,here/'experiments/cvs_feature_sequence50/source.py')
    module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module)
    from experiments.cvs_rx_u_increment20.source import train
    train(request['config'])
if __name__=='__main__':main()
