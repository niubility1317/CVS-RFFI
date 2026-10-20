"""Read-only original R17 kernel factorization costs, without IQ or forwards."""
import argparse
import json
import math
import sys
from pathlib import Path
import torch
from torch import nn


def factorization_structure(priors,frozen_parameters):
    if len(priors)!=3:raise ValueError('Three original independent R17 priors required')
    layers=[]
    for stage,prior in enumerate(priors):
        for name,module in prior.named_modules():
            if not isinstance(module,nn.Conv1d) or module.kernel_size[0]<=1:continue
            if module.groups!=1 or module.padding_mode!='zeros':
                raise ValueError('Ordinary zero-padded dense kernels required')
            matrix=module.weight.detach().cpu().double().flatten(1)
            singulars=torch.linalg.svdvals(matrix);energies=singulars.square()
            total=float(energies.sum());cumulative=energies.cumsum(0)
            energy_ranks={str(level):1 if total==0 else int(torch.searchsorted(cumulative,cumulative[-1]*level))+1
                          for level in (.99,.999)}
            rank=max(1,min(matrix.shape)//2)
            cost=rank*sum(matrix.shape);eligible=cost<matrix.numel()
            layers.append(dict(stage=stage,name=name,outer=matrix.shape[0],inner=matrix.shape[1],
                kernel_parameters=matrix.numel(),energy=total,energies=energies,
                half_rank=rank,half_saves_parameters=matrix.numel()-cost if eligible else 0,
                half_relative_Frobenius_error=math.sqrt(float(energies[rank:].sum())/max(total,1e-30)) if eligible else 0.,
                energy_ranks=energy_ranks))
    if not layers:raise ValueError('Long convolution kernels required')
    prior_parameters=sum(p.numel() for prior in priors for p in prior.parameters())
    original=frozen_parameters+prior_parameters;total_energy=sum(v['energy'] for v in layers)
    strategies={
        'half_all':lambda v:max(1,min(v['outer'],v['inner'])//2),
        'half_output_ge64':lambda v:max(1,min(v['outer'],v['inner'])//2) if v['outer']>=64 else None,
        'half_output_ge96':lambda v:max(1,min(v['outer'],v['inner'])//2) if v['outer']>=96 else None,
        'two_thirds':lambda v:max(1,2*min(v['outer'],v['inner'])//3),
        'energy99':lambda v:v['energy_ranks']['0.99'],
        'energy999':lambda v:v['energy_ranks']['0.999'],
    }
    comparisons=[]
    for name,choose in strategies.items():
        saved=changed=0;tail=0.;by_stage=[dict(saved_parameters=0,changed_layers=0,error_energy=0.) for _ in range(3)]
        for layer in layers:
            rank=choose(layer)
            if rank is None:continue
            saving=layer['kernel_parameters']-rank*(layer['outer']+layer['inner'])
            if saving<=0:continue
            error=float(layer['energies'][rank:].sum());stage=by_stage[layer['stage']]
            saved+=saving;changed+=1;tail+=error
            stage['saved_parameters']+=saving;stage['changed_layers']+=1;stage['error_energy']+=error
        comparisons.append(dict(strategy=name,full_parameters=original-saved,prior_parameters=prior_parameters-saved,
            saved_parameters=saved,parameter_reduction_percent=100*saved/original,changed_long_layers=changed,
            relative_long_kernel_Frobenius_error=math.sqrt(tail/max(total_energy,1e-30)),by_stage=by_stage))
    public_layers=[{k:v for k,v in layer.items() if k!='energies'} for layer in layers]
    return dict(status='DATA_FREE_ORIGINAL_KERNEL_FACTORIZATION_ANALYSIS',original_full_parameters=original,
        original_prior_parameters=prior_parameters,frozen_parameters=frozen_parameters,long_layers=len(layers),
        hypothetical_costs=comparisons,layers=public_layers,
        algebra='Each kernel flattened out_channels x in_channels*kernel;rank(r) cost=r*(outer+inner);original biases/GN/1x1 retained;apply only for positive storage savings;float64 SVD',
        limitation='Kernel storage/error only;not implemented candidates,activation/identity errors,performance,latency or feasibility proof;no active model/budget/selection changes',
        physical_IQ_reads=0,TX_F_V_target_reads=0,optimizer_updates=0,forward_calls=0)


def main():
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
    from experiments.pamf_unrolled_restore.runtime import load_checkpoint
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();torch.set_num_threads(2)
    if args.output.exists():raise FileExistsError(args.output)
    model,value=load_checkpoint(args.checkpoint,'cpu')
    if (value['candidate'],value['epoch'],value['updates'])!=('unroll_unshared',20,4940):
        raise ValueError('Original R17 final unshared source baseline required')
    result=factorization_structure(model.prior,sum(p.numel() for p in model.frozen.parameters()))
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x',encoding='utf-8') as stream:json.dump(result,stream,indent=2)
    print(json.dumps({k:v for k,v in result.items() if k!='layers'}))


if __name__=='__main__':main()
