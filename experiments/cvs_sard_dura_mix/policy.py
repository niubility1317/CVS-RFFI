"""Balanced source-only view membership, independent of labels and RNG."""
import torch

def restored_mask(count, step, device):
    if count!=64 or not isinstance(step,int) or step<1:
        raise ValueError('Only the fixed64 labeled As slots and positive optimizer step')
    return (torch.arange(count,device=device)+step).remainder(2)!=0

def mix(engine, satellite, step):
    if satellite.shape!=(64,2,256) or satellite.dtype!=torch.float32 or not bool(torch.isfinite(satellite).all()):
        raise ValueError('Actual finite FP32 As64 required')
    mask=restored_mask(len(satellite),step,satellite.device)
    with torch.inference_mode(False):
        restored=engine(satellite[mask].detach())[0]
        restored=restored.clone().detach()
        if restored.shape!=satellite[mask].shape or restored.dtype!=satellite.dtype or restored.device!=satellite.device:
            raise ValueError('No changed restored batch,shape,precision or device')
        if not bool(torch.isfinite(restored).all()):
            raise FloatingPointError('Nonfinite DURA output;no fallback')
        output=satellite.detach().clone()
        output[mask]=restored
    return output,mask
