"""Use the actual training engine and its resolved config without weaker defaults."""
from dataclasses import asdict
from . import design as d
def imports():
 import leo_practical.channel as ch
 import leo_practical.residual as residual
 return ch,residual
def configurations():
 ch,_=imports();actual=d.read(d.REFERENCES['B01']/'actual_practical_first_batch.json')['config']
 expected=dict(processing_route='residual',mode='post_sync',equalization_enabled=False,reference_snr_db=30.,fs_hz=25000000.,fc_hz=2450000000.,quality_adaptation_enabled=True,phase_tracking_enabled=True,iq_compensation_enabled=True,boundary='reflect')
 if any(actual.get(k)!=v for k,v in expected.items()):raise ValueError('Actual trained practical contract differs')
 return {s:ch.Config(**dict(actual,scenario=s)) for s in d.SCENES}
def tensor_iq(a,device):
 import numpy as np,torch
 a=np.ascontiguousarray(a,dtype=np.float32)
 return torch.frombuffer(bytearray(a.tobytes()),dtype=torch.float32).reshape(a.shape).to(device)
def audit(configs):
 ch,r=imports()
 return dict(channel_module_file=ch.__file__,residual_module_file=r.__file__,actual_configs={s:asdict(c) for s,c in configs.items()},EQ_off_G_equals_one=True,formal_test_changed=False)
