"""Compare paper MoPC with a labelled clipping variant on a saved B/support only."""
import argparse
import json
from unittest.mock import patch

from . import group2_cil_bridge as cil
from . import group2_cross_fit as cross
from . import group2_savedb_diagnostic as diagnostic
from .io import read, require


def register_candidate(prepared,x,y,*,seed,methods,clip):
    require(tuple(methods)==('mopc_hr',) and set(prepared)=={'mopc_hr'}, 'Wrong diagnostic methods')
    return {'mopc_hr':cil.register(prepared['mopc_hr'],x,y,seed=seed,
                                   mopc_semantics='paper',mopc_max_grad_norm=clip)}


def run(rt,job,k,n,device,clip):
    require(clip is None or clip==1., 'Unknown diagnostic variant')
    name='paper_no_clip_current_guard' if clip is None else 'paper_diagnostic_clip_1_current_guard'
    def emit(event):
        print(json.dumps(dict(variant=name,**event),allow_nan=False),flush=True)
    def register(prepared,x,y,*,seed,methods):
        return register_candidate(prepared,x,y,seed=seed,methods=methods,clip=clip)
    try:
        with patch.object(cross,'register_c',register):
            diagnostic.run(rt,job,k,n,device,emit)
        status='COMPLETE';error=None
    except FloatingPointError as exc:
        status='NONFINITE';error=str(exc)
    emit(dict(event='DIAGNOSTIC_RESULT',status=status,error=error,semantics='paper',
              max_grad_norm=clip,optimizer_changed=clip is not None,
              query_read=False,truth_read=False,predictions_written=False))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runtime',required=True);p.add_argument('--job',required=True)
    p.add_argument('--k',type=int,required=True);p.add_argument('--n',type=int,required=True)
    p.add_argument('--device',choices=['cuda:0'],required=True)
    p.add_argument('--variant',choices=['original','clip1'],required=True)
    a=p.parse_args();run(read(a.runtime),read(a.job),a.k,a.n,a.device,None if a.variant=='original' else 1.)


if __name__=='__main__':main()
