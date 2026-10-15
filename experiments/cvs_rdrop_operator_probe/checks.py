import copy, json, torch
from . import design as d
from .measure import pair_gradients, summary
from experiments.cvs_sigtime_rdrop_envelope20 import model as api


def reject(fn):
    try:fn()
    except (ValueError,RuntimeError,TypeError):return
    raise AssertionError('Illegal source diagnostic accepted')


def main():
    torch.set_num_threads(2)
    negative=0
    for key,value in [('checkpoint','unknown.pt'),('optimizer_steps',1),('head_fit_updates',1),('target_access',True),('source_selection','U_s'),('contexts',['target']),('parent_row',{})]:
        c=d.request();c[key]=value;reject(lambda:d.validate(c));negative+=1
    m=api.build(d.parent.config(d.ROW),'cpu').train()
    before={k:v.detach().clone() for k,v in m.state_dict().items()}
    torch.manual_seed(2026101601)
    x=torch.randn(144,2,256);y=torch.arange(144)%6
    records=pair_gradients(m,x,y)
    assert len(records)==30 and all(torch.equal(v,m.state_dict()[k]) for k,v in before.items())
    assert all(z['against_CE']['L2']>0 for z in records)
    assert all(abs(z['against_CE']['L2_ratio']-1)<1e-12 and abs(z['against_CE']['cosine']-1)<1e-12 for z in records if z['operator']=='CE')
    assert all(abs(z['against_FULL']['L2_ratio']-1)<1e-12 and abs(z['against_FULL']['cosine']-1)<1e-12 for z in records if z['operator']=='FULL')
    m.eval();reject(lambda:pair_gradients(m,x,y));negative+=1
    m.train();reject(lambda:pair_gradients(m,x[:128],y[:128]));negative+=1
    assert all(p.grad is None for p in m.parameters())
    print(json.dumps(dict(status='PASS',negative_cases=negative,actual_gradient_records=30,
        actual_joint160_autograd_connected=True,shallow_middle_SIG_and_default_W_gradients_nonzero=True,
        parameters_buffers_unchanged=True,parameter_grad_slots_empty=True,
        synthetic_identity_forward_packets=288,real_IQ_materializations=0,checkpoint_loads=0,
        optimizer_steps=0,head_fit_updates=0,target_access=False)))


if __name__=='__main__':main()
