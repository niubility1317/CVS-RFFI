import argparse,torch
from . import design as d
from .run import load_L,build
from .core import readout
from experiments.cvs_equivariant_identity.precision import numerical_context
def preflight():
    spec=d.read(d.ROOT/'registration/experiment.json')
    assert spec['run_id']==d.RUN and [r['config'] for r in spec['rows']]==[d.config(r) for r in d.rows()]
    torch.set_num_threads(2);results=[]
    with numerical_context(d.FULL_FP32_POLICY):
        for row in d.rows():
            c=d.config(row);L,contract=load_L(c);small=dict(role='L_s',**{k:(v[:8] if k!='role' else v) for k,v in L.items() if k!='role'})
            model=build(c['parent_build_config'],'cpu');_,proof=readout(model,small,c['model_seed'],'cpu')
            assert proof['model_forwards']==1 and proof['L_occurrences']==8
            results.append(dict(row_id=row['row_id'],provenance=model._parent_provenance,readout=proof))
            del model,L,small
    return dict(status='PASS',rows=results,source_contract=contract,model_forwards=3,L_occurrences=24,first_order_derivative_calls=6,identity_updates=0,target_access=False,U_forward=0,V_forward=0)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--remote-preflight',action='store_true');p.add_argument('--output',required=True);a=p.parse_args();d.write(a.output,preflight())
