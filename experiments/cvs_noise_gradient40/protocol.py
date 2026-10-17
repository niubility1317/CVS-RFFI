"""Source role/shape checks only;no new data validator or hash layer."""
from . import design as d
def validate_role(role,ids,contract,rx,day,labels):
 if role!='L_s' or len(ids)!=6300 or len(set(ids))!=6300 or set(ids)!=set(contract['role_ids']['L_s']):raise ValueError('Only original full L6300 may provide gradients')
 if set(ids)&(set(contract['role_ids']['U_s'])|set(contract['role_ids']['V'])):raise ValueError('L cannot overlap hidden U or V')
 if len(rx)!=6300 or len(day)!=6300 or len(labels)!=6300 or set(rx)!=set(d.RX) or set(day)!=set(d.DAYS) or set(labels)!=set(range(6)):raise ValueError('All5 original RX/all3 days/all6TX required;no held RX')
def validate_config(c):
 expected=dict(processing_route='residual',mode='post_sync',equalization_enabled=False,reference_snr_db=30.,fs_hz=25000000.,fc_hz=2462000000.,boundary='reflect',quality_adaptation_enabled=True,phase_tracking_enabled=True,iq_compensation_enabled=True)
 if any(c.get(k)!=v for k,v in expected.items()):raise ValueError('Actual original L practical configuration required;G1/no weaker defaults')
