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
    import torch
    from experiments.cvs_rx_u_increment20.geometry import project
    projected=project(torch.tensor([2.,3.],dtype=torch.float64),torch.tensor([[1.,0.]],dtype=torch.float64))['projected']
    if not torch.allclose(projected,torch.tensor([0.,3.],dtype=torch.float64),atol=1e-8,rtol=1e-8):raise ValueError('Projection backend smoke failed')
    print('GEOMETRY_SMOKE PASS source_IQ_reads=0 query_access=false',flush=True)
    train(request['config'])
if __name__=='__main__':main()
