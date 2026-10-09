"""Focused role, identity/no-update and actual model integration checks."""
import argparse
from pathlib import Path
import torch
from experiments.cvs_multi_action_audit import design as d
from experiments.cvs_multi_action_audit.runner import role_split,gradient_audit,gradient_indices,channel_view

def channel_bridge_check(device='cpu'):
    from unittest.mock import patch
    data=dict(x=torch.randn(2,2,256,device=device),rx=torch.tensor([1,3],device=device),ids=['fit0','audit0'])
    expected=channel_view(data)
    with patch.object(torch.Tensor,'numpy',side_effect=RuntimeError('unsafe numpy bridge')),\
         patch.object(torch,'from_numpy',side_effect=RuntimeError('unsafe numpy bridge')):
        actual=channel_view(data)
    assert torch.equal(expected['x'],actual['x']) and not torch.equal(actual['x'],data['x'])
    assert all('\\' not in d.config(r)['checkpoint'] and '\\' not in d.config(r)['output_root'] for r in d.rows())
    return dict(status='PASS',unsafe_tensor_numpy_bridge_absent=True,output_deterministic=True,remote_paths_posix=True)

def checks(device='cpu'):
    torch.set_num_threads(2)
    records=[dict(id=str(i),y=i//6,rx=i%2,role='L_s') for i in range(12)]
    ids=[r['id'] for r in records]
    a,b=role_split(records,ids,2,1)
    a2,b2=role_split(list(reversed(records)),ids,2,1)
    assert a==a2 and b==b2 and not {r['id'] for r in a}&{r['id'] for r in b}
    gy=torch.arange(6).repeat_interleave(5*32);gr=torch.arange(5).repeat_interleave(32).repeat(6)
    gi,gc=gradient_indices(dict(x=torch.zeros(960,2,256),y=gy,rx=gr))
    assert len(gi)==30 and len(set(map(tuple,gc)))==30
    for bad in ([*records,records[0]],[dict(r,role='U_s') for r in records]):
        try:role_split(bad,ids,2,1)
        except ValueError:pass
        else:raise AssertionError('Bad roles accepted')
    try:d.validate(dict(d.config(d.rows()[0]),target_access=True))
    except ValueError:pass
    else:raise AssertionError('Target access accepted')
    from experiments.cvs_multi_disentangle.checks import identity
    from experiments.cvs_multi_action_audit.actions import fit_and_audit,composition_audit
    c=d.parent_config(d.SEEDS[0]);model=identity(c,device);m=model.id_backbone;m.eval()
    saved={k:v.clone() for k,v in m.state_dict().items()}
    x=torch.randn(24,2,256,device=device);y=torch.arange(24,device=device)%6
    r=fit_and_audit(m,x[:12],y[:12],x[12:],y[12:],torch.Generator().manual_seed(4),
        fit_ids=list(map(str,range(12))),audit_ids=list(map(str,range(12,24))),steps=2,batch_size=6)
    composition=composition_audit(m,x[12:],y[12:],torch.Generator().manual_seed(5),r['state_dicts'])
    assert len(composition['modes'])==4
    g=gradient_audit(m,x[:6],y[:6],torch.Generator().manual_seed(3))
    assert len(g['rows'])==20 and not g['actual_identity_update']
    assert all(torch.equal(v,m.state_dict()[k]) for k,v in saved.items())
    channel_bridge_check(device)
    return dict(status='PASS',device=device,physical_roles_disjoint=True,negative_roles_rejected=True,
        actual_CVS_fit_and_audit=True,composition=True,gradient_audit=True,identity_unchanged=True,
        source_LEO_deterministic=True,target_read=False)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--device',default='cpu');p.add_argument('--output',required=True);a=p.parse_args()
    result=checks(a.device);d.write(a.output,result);print(result)
