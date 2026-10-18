"""Focused source/data/physics permission negatives, no models or IQ opened."""
from copy import deepcopy
from . import design as d

def run():
    d.validate_config(d.config())
    negatives=0
    for key,value in dict(source_role='U_s',target_access=True,head_fit_updates=1,
            training_updates=1,source_RX=[1,3,4,6],full_model_examples=32400,
            diagnostic_fc_hz=2462000000.,rows=['As'],U_truth_read=True).items():
        c=deepcopy(d.config());c[key]=value
        try:d.validate_config(c)
        except ValueError:negatives+=1
        else:raise AssertionError('Unauthorized source contract accepted: '+key)
    physics=dict(EQ_off_G_equals_one=True,formal_test_changed=False,actual_configs={s:dict(
        fs_hz=25000000.,fc_hz=2450000000.,reference_snr_db=30.,mode='post_sync',
        processing_route='residual',equalization_enabled=False,boundary='reflect',
        quality_adaptation_enabled=True,phase_tracking_enabled=True,iq_compensation_enabled=True) for s in d.SCENES})
    d.validate_physics(physics)
    for key,value in dict(equalization_enabled=True,fc_hz=2462000000.,mode='pre_sync',
            processing_route='full',reference_snr_db=40.,boundary='zero',
            quality_adaptation_enabled=False,phase_tracking_enabled=False,iq_compensation_enabled=False).items():
        bad=deepcopy(physics);bad['actual_configs'][d.SCENES[0]][key]=value
        try:d.validate_physics(bad)
        except ValueError:negatives+=1
        else:raise AssertionError('Changed physics accepted: '+key)
    assert d.NN==2*(1800+2*2*2*1800+2)
    assert d.EXTRA_HEAD==2*(16200+12*7200+4*1800+26)
    return dict(status='PASS',checks=20,negative_cases=negatives,
        source_only=True,NN=0,IQ=0,CP=0,training_updates=0)

if __name__=='__main__':
    import json
    print(json.dumps(run()))
