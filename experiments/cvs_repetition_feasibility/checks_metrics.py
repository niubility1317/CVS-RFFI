"""Independent tiny arithmetic checks, without reading IQ or fitting anything."""
import json
import torch
from .metrics import summarize

def check():
    valid=torch.tensor([True,True,False,True]);target=torch.zeros(4,24,dtype=torch.double);target[0]=1;target[1]=2;target[3]=3
    predictions={'known_reference':torch.zeros_like(target),'visible_mean':target*.5,'copy_cycle1':target.clone()}
    labels=torch.tensor([0,1,0,1]);rx=torch.tensor([1,1,3,3]);day=torch.tensor([1,1,2,2])
    result=summarize(valid,target,predictions,labels,rx,day);r=result['records']['overall']
    assert r['count']==4 and r['valid']==3 and r['invalid']==1
    assert r['target_energy_sum']==14 and r['arms']['visible_mean']['error_sum']==3.5
    assert r['arms']['visible_mean']['improvement_over_reference']==.75
    assert r['arms']['visible_mean']['MSE_valid']==3.5/3
    assert r['arms']['visible_mean']['MSE_full_group_zero_invalid']==3.5/4
    assert result['records']['RX/3']['valid']==1 and result['records']['RX/3']['count']==2
    assert sum(x['count'] for k,x in result['records'].items() if k.startswith('TX/'))==4
    empty=summarize(torch.zeros(4,dtype=torch.bool),target,predictions,labels,rx,day)['records']['overall']
    assert empty['arms']['visible_mean']['MSE_valid'] is None
    assert empty['arms']['visible_mean']['improvement_over_reference'] is None
    assert empty['target_energy']['mean'] is None
    try:summarize(valid,target.requires_grad_(),predictions,labels,rx,day)
    except ValueError:pass
    else:raise AssertionError('Live target gradient accepted')
    return dict(status='VERIFIED',independent_closed_form_error_counts=True,valid_and_full_denominators_explicit=True,
                empty_metrics_null=True,metadata_downstream_only=True,source_or_target_access=False)

if __name__=='__main__':print(json.dumps(check()))
