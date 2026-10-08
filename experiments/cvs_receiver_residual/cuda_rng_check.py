"""CPU-built modules must leave the already initialized CUDA RNG unchanged."""
import argparse
import torch
from experiments.cvs_reference_identity.model import build
from experiments.cvs_receiver_residual import design as d


def check():
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA RNG check requires CUDA')
    results=[]
    for arm in d.ARMS:
        torch.manual_seed(37);build('reference_response')
        expected_cpu=torch.random.get_rng_state().clone()
        expected_cuda=torch.cuda.get_rng_state().clone()
        torch.manual_seed(37);d.build_model(arm)
        cpu=torch.equal(expected_cpu,torch.random.get_rng_state())
        cuda=torch.equal(expected_cuda,torch.cuda.get_rng_state())
        if not cpu or not cuda: raise AssertionError('Initialization changed common RNG: '+arm)
        results.append(dict(arm=arm,cpu_rng_preserved=cpu,cuda_rng_preserved=cuda))
    return dict(status='PASS',rows=results,real_dataset_read=False,query_read=False)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args()
    result=check();d.write(a.output,result);print(result)
