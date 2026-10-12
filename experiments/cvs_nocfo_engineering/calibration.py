"""Explicitly inactive: engineering arms build no sensitivity graphs."""
from pathlib import Path
from . import design as d

def calibrate(c,model,L,device,out):
    if c['multilevel']['enabled'] or hasattr(model,'multilevel'):raise ValueError('Engineering arm unexpectedly contains auxiliary module')
    p=Path(out)/'calibration.json'
    if p.exists():raise FileExistsError('No-calibration record already exists')
    value=dict(status='NOT_APPLICABLE',fixed_coefficients={},pairs=0,extra_source_readouts=0,model_forwards=0,seconds=0.,state_unchanged=True,
        calibration_reference='no_auxiliary_module',identity_updates=0,head_updates=0,source_role='no source read',target_access=False)
    d.write(p,value);return value
