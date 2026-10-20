"""Read-only, data-free storage analysis of original R17 stage specialization."""
import argparse
import json
import sys
from pathlib import Path
import torch
from torch import nn


def weight_structure(priors,frozen_parameters):
    if len(priors)!=3:raise ValueError('Three original independent R17 priors required')
    maps=[dict(p.named_modules()) for p in priors]
    names=[key for key,module in maps[0].items() if isinstance(module,nn.Conv1d) and module.kernel_size[0]>1]
    if not names:raise ValueError('Long convolution kernels required')
    layers=[];weight_energy=delta_energy=0.;rank99=[]
    for name in names:
        modules=[values[name] for values in maps]
        if any(not isinstance(v,nn.Conv1d) or v.groups!=1 or v.weight.shape!=modules[0].weight.shape for v in modules):
            raise ValueError('Matching ordinary convolution structure required')
        matrices=[v.weight.detach().cpu().double().flatten(1) for v in modules]
        center=torch.stack(matrices).mean(0)
        singulars=[torch.linalg.svdvals(value-center) for value in matrices]
        energy=sum(float(value.square().sum()) for value in matrices)
        residual=sum(float(value.square().sum()) for value in singulars)
        weight_energy+=energy;delta_energy+=residual
        for values in singulars:
            cumulative=values.square().cumsum(0)
            rank99.append(0 if float(cumulative[-1])<1e-30 else int(torch.searchsorted(cumulative,cumulative[-1]*.99))+1)
        layers.append(dict(parameters=matrices[0].numel(),inner=matrices[0].shape[1],outer=matrices[0].shape[0],singulars=singulars))
    original_prior=sum(p.numel() for prior in priors for p in prior.parameters())
    untouched=original_prior-3*sum(v['parameters'] for v in layers)
    comparisons=[]
    for requested in (0,1,2,4,8,16):
        storage=untouched;error=0.;shared_layers=0
        for layer in layers:
            rank=min(requested,layer['inner'],layer['outer'])
            compressed=layer['parameters']+3*rank*(layer['inner']+layer['outer'])
            if compressed>=3*layer['parameters']:
                storage+=3*layer['parameters']
            else:
                storage+=compressed;shared_layers+=1
                error+=sum(float(values[rank:].square().sum()) for values in layer['singulars'])
        total=storage+frozen_parameters;original=original_prior+frozen_parameters
        comparisons.append(dict(residual_rank=requested,full_parameters=total,prior_parameters=storage,
            parameter_reduction_percent=100*(1-total/original),shared_long_layers=shared_layers,
            relative_kernel_Frobenius_error=(error/max(weight_energy,1e-30))**.5,
            unrepresented_stage_delta_energy_fraction=error/max(delta_energy,1e-30)))
    return dict(status='DATA_FREE_ORIGINAL_WEIGHT_ANALYSIS',original_full_parameters=frozen_parameters+original_prior,
        original_prior_parameters=original_prior,frozen_parameters=frozen_parameters,long_layers_per_stage=len(layers),
        long_kernel_parameters_per_stage=sum(v['parameters'] for v in layers),
        stage_kernel_relative_Frobenius_difference=(delta_energy/max(weight_energy,1e-30))**.5,
        delta_rank99_min=min(rank99),delta_rank99_median=float(torch.quantile(torch.tensor(rank99,dtype=torch.float64),.5)),delta_rank99_max=max(rank99),
        hypothetical_shared_base_plus_independent_lowrank_deltas=comparisons,
        algebra='Float64 kernel matrix analysis;all biases,GN,1x1 and other parameters remain independent across stages;only use shared+delta storage when smaller than3densekernels',
        limitation='Storage/error arithmetic only,not an implemented model or IQ/identity/latency result;extra delta branches can increase compute;does not change the two active candidates',
        physical_IQ_reads=0,TX_F_V_target_reads=0,optimizer_updates=0,forward_calls=0)


def main():
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
    from experiments.pamf_unrolled_restore.runtime import load_checkpoint
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();torch.set_num_threads(2)
    model,value=load_checkpoint(args.checkpoint,'cpu')
    if (value['candidate'],value['epoch'],value['updates'])!=('unroll_unshared',20,4940):
        raise ValueError('Original R17 final unshared source baseline required')
    result=weight_structure(model.prior,sum(p.numel() for p in model.frozen.parameters()))
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x',encoding='utf-8') as stream:json.dump(result,stream,indent=2)
    print(json.dumps(result))


if __name__=='__main__':main()
