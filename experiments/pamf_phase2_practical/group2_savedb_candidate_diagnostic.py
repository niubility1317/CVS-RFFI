"""Compare paper MoPC with a labelled clipping variant on a saved B/support only."""
import argparse
import json
import time
from unittest.mock import patch
import torch

from . import group2_cil_bridge as cil
from . import group2_cross_fit as cross
from . import group2_savedb_diagnostic as diagnostic
from .vendor_native.comparison_suite import registration as native
from .io import read, require


def register_candidate(prepared,x,y,*,seed,methods,clip):
    require(tuple(methods)==('mopc_hr',) and set(prepared)=={'mopc_hr'}, 'Wrong diagnostic methods')
    return {'mopc_hr':cil.register(prepared['mopc_hr'],x,y,seed=seed,
                                   mopc_semantics='paper',mopc_max_grad_norm=clip)}


def historical_record(trace,model,terms,callback,start,**fields):
    """Preserve r15's FP32 gradient-norm check, including failure diagnostics."""
    gradients=[p.grad.detach().norm().square() for p in model.parameters() if p.grad is not None]
    row={**fields,**{key:float(value.detach()) if isinstance(value,torch.Tensor) else value
                     for key,value in terms.items()},
         'gradient_norm':float(terms['gradient_norm']) if 'gradient_norm' in terms else
                         (float(torch.stack(gradients).sum().sqrt()) if gradients else 0.0),
         'elapsed_seconds':time.perf_counter()-start,
         'source_validation':None,'source_validation_reason':'Phase2 does not select on validation'}
    nonfinite=[key for key,value in row.items() if isinstance(value,float) and not torch.isfinite(torch.tensor(value))]
    if nonfinite:
        context={key:fields[key] for key in ('phase','method','stage','epoch','iteration') if key in fields}
        if 'gradient_norm' in nonfinite and 'gradient_norm' not in terms:
            named=[(name,param.grad.detach()) for name,param in model.named_parameters() if param.grad is not None]
            bad=[name for name,value in named if not torch.isfinite(value).all()]
            context['nonfinite_gradient_tensors']=bad
            if not bad and named:
                context['gradient_norm_float64']=float(torch.stack([v.double().norm() for _,v in named]).norm())
                context['gradient_absmax_float64']=float(torch.stack([v.double().abs().max() for _,v in named]).max())
        raise FloatingPointError(f"nonfinite registration training term: {', '.join(nonfinite)}; {context}")
    trace.append(row)
    if callback:callback(dict(row))


def run(rt,job,k,n,device,clip,guard='current_fp64'):
    require(clip is None or clip==1., 'Unknown diagnostic variant')
    require(guard in ('current_fp64','historical_fp32'), 'Unknown diagnostic guard')
    name=('paper_no_clip_' if clip is None else 'paper_diagnostic_clip_1_')+guard
    def emit(event):
        print(json.dumps(dict(variant=name,**event),allow_nan=False),flush=True)
    def register(prepared,x,y,*,seed,methods):
        return register_candidate(prepared,x,y,seed=seed,methods=methods,clip=clip)
    try:
        with patch.object(cross,'register_c',register):
            if guard=='historical_fp32':
                with patch.object(native,'_record',historical_record):
                    diagnostic.run(rt,job,k,n,device,emit)
            else:diagnostic.run(rt,job,k,n,device,emit)
        status='COMPLETE';error=None
    except FloatingPointError as exc:
        status='NONFINITE';error=str(exc)
    emit(dict(event='DIAGNOSTIC_RESULT',status=status,error=error,semantics='paper',
              guard=guard,max_grad_norm=clip,optimizer_changed=clip is not None,
              query_read=False,truth_read=False,predictions_written=False))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runtime',required=True);p.add_argument('--job',required=True)
    p.add_argument('--k',type=int,required=True);p.add_argument('--n',type=int,required=True)
    p.add_argument('--device',choices=['cuda:0'],required=True)
    p.add_argument('--variant',choices=['original','clip1'],required=True)
    p.add_argument('--guard',choices=['current_fp64','historical_fp32'],default='current_fp64')
    a=p.parse_args();run(read(a.runtime),read(a.job),a.k,a.n,a.device,None if a.variant=='original' else 1.,a.guard)


if __name__=='__main__':main()
