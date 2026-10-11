"""Bounded local source-only readout and native-gradient regression checks."""
import argparse
import inspect
import json
from pathlib import Path
import torch
from torch import nn
from .audit import collect_predictions,paired_readout,gradient_audit,modes,_validate,_groups


def run(device='cpu'):
    torch.set_num_threads(2)
    class Probe(nn.Module):
        def __init__(self):
            super().__init__();self.drop=nn.Dropout(.5);self.head=nn.Linear(6,6);self.calls=[]
        def forward(self,x):
            self.calls.append((torch.is_grad_enabled(),self.training,self.drop.training));return self.head(self.drop(x))
    toy=Probe();toy.train();toy.drop.eval()
    L=dict(x=torch.eye(6),y=torch.arange(6),rx=torch.ones(6,dtype=torch.long),day=torch.zeros(6,dtype=torch.long),ids=list(range(6)))
    before=[m.training for m in toy.modules()]
    r=collect_predictions(toy,L,'cpu',batch=2,role='V')
    assert all(not any(call) for call in toy.calls) and not r['logits'].requires_grad
    assert before==[m.training for m in toy.modules()]
    try:collect_predictions(toy,L,'cpu',role='U')
    except ValueError:pass
    else:raise AssertionError('U readout truth accepted')
    # Four exact initial confidence/correctness groups; one fix, one break.
    old=torch.zeros(4,6);old[0,0]=10;old[1,1]=10;old[2,0]=1;old[3,1]=1
    new=old.clone();new[0]=0;new[0,1]=10;new[1]=0;new[1,0]=10
    first=dict(role='V',ids=[10,11,12,13],y=torch.zeros(4,dtype=torch.long),rx=torch.tensor([1,1,3,3]),day=torch.zeros(4,dtype=torch.long),logits=old)
    second=dict(first,logits=new)
    paired=paired_readout(first,second)
    assert all(g['count']==1 for g in paired['initial_groups'].values())
    assert paired['overall']['fixed']==1 and paired['overall']['broken']==1 and paired['overall']['persistent_error']==1
    assert sum(r['count'] for r in paired['calibration']['initial']['bins'])==4
    for bad in (dict(second,ids=[11,10,12,13]),dict(second,y=torch.ones(4,dtype=torch.long))):
        try:paired_readout(first,bad)
        except ValueError:pass
        else:raise AssertionError('Invalid paired PID/truth accepted')
    bn=nn.Sequential(nn.BatchNorm1d(6),nn.Dropout(.5));bn.train();running=bn[0].running_mean.clone()
    with modes(bn,True) as frozen:
        assert frozen==['0'] and not bn[0].training and bn[1].training
        bn(torch.randn(10,6))
    assert torch.equal(running,bn[0].running_mean) and bn[0].training
    from experiments.cvs_early_geometry import design as old_design
    from experiments.cvs_early_geometry.model import build
    from experiments.cvs_e20_budget_fishr.multilevel import MultiLevel
    from experiments.cvs_e20_budget_fishr import design
    model=build(old_design.config(next(row for row in old_design.rows() if row['arm']=='PHYS')),device)
    model.multilevel=MultiLevel(model.base,seed=design.SEEDS[0]).to(device)
    groups=_groups(model)
    encoder_ids={id(p) for p in groups['encoder']};classifier_ids={id(p) for p in groups['classifier']}
    dense_ids={id(p) for n,p in model.named_parameters() if p.requires_grad and 'id_backbone' in n and not n.endswith(('low_hz_','band_hz_'))}
    assert not encoder_ids&classifier_ids and encoder_ids|classifier_ids==dense_ids
    assert all(id(p) in encoder_ids for n,p in model.named_parameters() if p.requires_grad and 'id_backbone' in n and any(token in n for token in ('t_proj','f_proj','pa_proj','fusion','gate','reference_proj','response_proj','response_gain')))
    parameter_counts={name:sum(p.numel() for p in values) for name,values in groups.items()}
    generator=torch.Generator().manual_seed(1122)
    L=dict(x=torch.randn(180,2,256,generator=generator),y=torch.arange(6).repeat_interleave(10).repeat(3),
        rx=torch.tensor([1,3,4,6,8]).repeat_interleave(2).repeat(18),day=torch.arange(3).repeat_interleave(60),ids=list(range(180)))
    class Hidden(dict):
        def __getitem__(self,key):
            if key=='y':raise AssertionError('U y accessed')
            return super().__getitem__(key)
    U=Hidden(role='U_s',x=torch.randn(180,2,256,generator=generator),rx=L['rx'].clone(),day=L['day'].clone(),ids=list(range(1000,1180)))
    _validate(dict(L,role='L_s'),'L');_validate(dict(L,role='V'),'V');_validate(U,'U')
    for source,expected,bad_role in [(U,'U','L_s'),(U,'U','V'),(U,'U','query'),(L,'L','U_s'),(L,'L','V'),(L,'V','L_s'),(L,'V','U_s')]:
        try:_validate(dict(source,role=bad_role),expected)
        except ValueError:pass
        else:raise AssertionError('Mixed/unknown role accepted')
    c=design.config(next(r for r in design.rows() if r['arm']=='JOINT'))
    signature=inspect.signature(gradient_audit)
    assert 'V' not in signature.parameters
    try:gradient_audit(model,L,dict(U,y=object()),c,rounds=1,device=device)
    except ValueError:pass
    else:raise AssertionError('Poison U accepted')
    saved={n:t.clone() for n,t in model.state_dict().items()};state=torch.random.get_rng_state().clone()
    cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
    mode_before=[m.training for m in model.modules()]
    result=gradient_audit(model,L,U,c,rounds=1,device=device)
    assert len(result['rows'])==8 and all(torch.equal(saved[n],t) for n,t in model.state_dict().items())
    assert torch.equal(state,torch.random.get_rng_state()) and mode_before==[m.training for m in model.modules()]
    assert all(torch.equal(a,b) for a,b in zip(cuda,torch.cuda.get_rng_state_all())) if cuda else True
    for row in result['rows']:
        assert row['tasks']['ce']['groups']['classifier']['norm']>0
        assert row['tasks']['ce']['groups']['classifier']['cosine'] is not None
        assert all(not (set(row[a])&set(row[b])) for a,b in [('main_L_ids','reference_L_ids'),('aux_L_ids','reference_L_ids'),('main_L_ids','aux_L_ids')])
    for row in result['rows']:
        if row['arm'] in ('FISHR_RAW','FISHR_BUDGET','JOINT'):
            assert row['tasks']['fishr']['groups']['shallow']['norm']>0 and row['tasks']['fishr']['groups']['middle']['norm']>0
    json.dumps(result,allow_nan=False)
    return dict(status='PASS',device=device,target_access=False,readout_no_grad=True,all_module_modes_restored=True,
        paired_PID_truth_rejected=True,all_four_groups_checked=True,repair_break_counts_checked=True,V_gradient_argument_absent=True,
        U_truth_poison_rejected=True,U_truth_never_read=True,native_U_s_role_accepted=True,source_role_mixing_rejected=True,all_eight_arms_native_gradient_valid=True,
        model_state_bitwise_unchanged=True,CPU_CUDA_RNG_unchanged=True,running_statistics_frozen_dropout_retained=True,
        encoder_complete_dense_identity_except_classifier=True,encoder_classifier_disjoint=True,parameter_counts=parameter_counts,
        native_fixture_round=result)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--device',default='cpu');parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();value=run(args.device);args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in value.items() if k!='native_fixture_round'},allow_nan=False))
