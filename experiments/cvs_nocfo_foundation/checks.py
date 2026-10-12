import argparse,contextlib,io,json,tempfile
from pathlib import Path
from unittest.mock import patch
import torch
from . import design as d
from .model import build,groups
from .objective import objective,BASE_WEIGHTED_KEYS
from .update import take_update
from .preflight import local
from experiments.cvs_source_identity_readout.input_policy import NoExplicitCFO
from experiments.cvs_equivariant_identity.precision import numerical_context

def run(device):
    from experiments.cvs_early_geometry.checks import fixture
    from . import source
    L,_=fixture();results=[];torch.set_num_threads(2)
    with numerical_context(d.FULL_FP32_POLICY):
        for row in d.rows():
            c=d.config(row);model=build(c,device);pg,_=groups(model,c);model.train();optimizer=torch.optim.AdamW(pg,weight_decay=c['optimizer']['weight_decay']);idx=torch.arange(128)
            response=next(m for m in model.modules() if isinstance(m,NoExplicitCFO));assert not response.contract()['explicit_CFO_identity_input'] and torch.count_nonzero(response(L['x'][idx].to(device))[:,27])==0
            def closure():return objective(c,model,None,L,idx,None,None,1333,device)
            loss,parts,w,detail,_,norm,update=take_update(c,model,optimizer,closure)
            assert torch.isfinite(loss) and torch.equal(loss,parts['ce']) and torch.equal(loss,sum(w[k] for k in BASE_WEIGHTED_KEYS))
            assert not update['sam_applied'] and detail['U_packets']==detail['teacher_forwards']==detail['auxiliary_forwards']==0
            assert all(int(v['step'])==1 for v in optimizer.state.values()) and all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
            class Stop(Exception):pass
            with tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp);(root/'release_commit.txt').write_text('synthetic-only',encoding='utf-8');c['output_root']=str(root/'source')
                def boundary(config,model,optimizer,closure,audit):
                    resolved=d.read(root/'source/resolved_config.json');assert resolved['execution']['explicit_CFO_identity_input'] is False and not resolved['execution']['U_iterator'] and resolved['fishr_calibration']['status']=='NOT_APPLICABLE'
                    raise Stop
                with contextlib.redirect_stdout(io.StringIO()),patch.object(d,'validate',lambda c:c),patch.object(d,'ROOT',root),patch.object(source,'load_source',return_value=(L,None,L,{})),patch.object(source,'take_update',side_effect=boundary):
                    try:source.train(c,'cpu')
                    except Stop:pass
                    else:raise AssertionError('Actual source entry not reached')
            results.append(dict(row_id=row['row_id'],finite_CE_backward=True,exact_CE_only=True,one_AdamW_step=True,CFO_slot_zero=True,no_U_teacher_aux_forward=True,actual_source_entry=True))
    return dict(status='PASS',device=device,synthetic_only=True,rows=results,checkpoint_checks=local(),target_access=False)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--device',default='cpu');p.add_argument('--output',type=Path,required=True);a=p.parse_args();v=run(a.device);d.write(a.output,v);print(json.dumps(dict(status=v['status'],rows=3,device=a.device)))
