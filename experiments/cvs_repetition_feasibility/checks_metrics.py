"""Independent tiny arithmetic checks, without reading IQ or fitting anything."""
import json
import torch
from .metrics import summarize

def _raises(callback):
    try:callback()
    except ValueError:return
    raise AssertionError('Invalid observation kind/dimension/reference accepted')

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
    assert set(result)=={'records','packet_target_energy','packet_MSE','normalization'}
    assert result['normalization']=='Per-packet mean over24 real residual coordinates; ratio of pooled sums on identical visible-valid packets. Invalid packets explicitly counted; never selected by hidden-target energy.'
    explicit24=summarize(valid,target,predictions,labels,rx,day,observation_kind='physical_residual24')
    assert explicit24['records']==result['records'] and explicit24['normalization']==result['normalization']
    assert torch.equal(explicit24['packet_target_energy'],result['packet_target_energy'])
    for arm in predictions:assert torch.equal(explicit24['packet_MSE'][arm],result['packet_MSE'][arm])
    # A true 14-coordinate census: per-packet mean is over14, never padded to24.
    target14=torch.zeros(4,14,dtype=torch.double);target14[0]=1;target14[1]=2;target14[3]=3
    predictions14={'known_reference':torch.zeros_like(target14),'visible_mean':target14*.5,'copy_cycle1':target14.clone()}
    null=summarize(valid,target14,predictions14,labels,rx,day,observation_kind='null_reference14')
    assert null['records']==result['records']
    assert torch.equal(null['packet_target_energy'],result['packet_target_energy'])
    assert null['observation_kind']=='null_reference14' and null['dimension']==14
    assert null['reference_baseline']=='fixed zero null-tone observation (external zero_reference mapped to known_reference)'
    assert null['packet_MSE']['known_reference'][3].item()==9
    empty14=summarize(torch.zeros_like(valid),target14,predictions14,labels,rx,day,
                      observation_kind='null_reference14')['records']['overall']
    assert empty14['arms']['visible_mean']['MSE_valid'] is None
    _raises(lambda:summarize(valid,target14,predictions14,labels,rx,day))
    _raises(lambda:summarize(valid,target,predictions,labels,rx,day,observation_kind='null_reference14'))
    _raises(lambda:summarize(valid,target,predictions,labels,rx,day,observation_kind='arbitrary24'))
    _raises(lambda:summarize(valid,target14,predictions14,labels,rx,day,observation_kind=14))
    nonzero_reference=dict(predictions14);nonzero_reference['known_reference']=torch.ones_like(target14)
    _raises(lambda:summarize(valid,target14,nonzero_reference,labels,rx,day,observation_kind='null_reference14'))
    unmapped=dict(predictions14);unmapped['zero_reference']=unmapped.pop('known_reference')
    _raises(lambda:summarize(valid,target14,unmapped,labels,rx,day,observation_kind='null_reference14'))
    empty=summarize(torch.zeros(4,dtype=torch.bool),target,predictions,labels,rx,day)['records']['overall']
    assert empty['arms']['visible_mean']['MSE_valid'] is None
    assert empty['arms']['visible_mean']['improvement_over_reference'] is None
    assert empty['target_energy']['mean'] is None
    try:summarize(valid,target.requires_grad_(),predictions,labels,rx,day)
    except ValueError:pass
    else:raise AssertionError('Live target gradient accepted')
    return dict(status='VERIFIED',independent_closed_form_error_counts=True,valid_and_full_denominators_explicit=True,
                empty_metrics_null=True,metadata_downstream_only=True,source_or_target_access=False,
                residual24_default_output_compatible=True,null14_closed_form_no_padding=True,
                fixed_kind_dimension_and_zero_reference_checked=True)

if __name__=='__main__':print(json.dumps(check()))
