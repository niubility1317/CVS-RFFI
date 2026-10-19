import math
import torch
from torch import nn
from . import design as d
from .losses import posterior_distance
from experiments.pamf_feature_refine.models import feature_distance
from experiments.pamf_restoration.models import complex_iq,pair
from experiments.pamf_reception_restore.models import Core,ComplexTFUNet,mirror_spectrum
from experiments.pamf_residual_restore.models import ResidualDeterministic
from experiments.pamf_change_preserve.runner import load_restorer


class BasisFilter(nn.Module):
    def __init__(self,arm):
        super().__init__()
        if arm not in d.ARMS:raise ValueError('Registered current-frame filter basis required')
        self.arm=arm;self.core=Core(incoming=4,outgoing=4,widths=d.FILTER_WIDTHS)
        self.training_calls=0;self.training_IQ_examples=0;self.training_coefficient_positions=0;self.training_FFTs=0

    def forward(self,received,restored):
        if (received.ndim!=3 or received.shape[1:]!=(2,256) or received.shape!=restored.shape or
            received.dtype!=torch.float32 or restored.dtype!=torch.float32 or received.device!=restored.device):
            raise ValueError('Matching current-frame FP32 received/restored IQ Bx2x256 required')
        observed=complex_iq(received);base=complex_iq(restored)
        spectral=self.arm==d.ARMS[1]
        if spectral:
            observed=torch.fft.fft(observed,norm='ortho');base=torch.fft.fft(base,norm='ortho')
            features=torch.cat((pair(torch.fft.fftshift(observed,dim=-1)),pair(torch.fft.fftshift(base,dim=-1))),1)
        else:features=torch.cat((received,restored),1)
        parameters=self.core(features)
        if spectral:parameters=torch.fft.ifftshift(parameters,dim=-1)
        # Subtract identity before applying the transform: zero output is exactly
        # the inherited waveform, including FFT rounding at initialization.
        delta=torch.exp(torch.complex(math.log(d.MAIN_GAIN)*parameters[:,0].tanh(),d.PHASE_LIMIT*parameters[:,1].tanh()))-1
        image=d.IMAGE_LIMIT*torch.complex(parameters[:,2].tanh(),parameters[:,3].tanh())
        correction=delta*base+image*(mirror_spectrum(base) if spectral else base.conj())
        if spectral:correction=torch.fft.ifft(correction,norm='ortho')
        if self.training:
            self.training_calls+=1;self.training_IQ_examples+=len(received)
            self.training_coefficient_positions+=len(received)*256;self.training_FFTs+=len(received)*3*int(spectral)
        return restored+pair(correction)


class JointBasis(ResidualDeterministic):
    def __init__(self,coarse,scale,arm):
        super().__init__(coarse,scale);self.filter=BasisFilter(arm);self.requires_grad_(True)

    def context(self,received):
        base=self.coarse(received)
        return base,torch.cat((received,base),1)

    def train(self,mode=True):return nn.Module.train(self,mode)

    def forward(self,received):return self.filter(received,super().forward(received))


def initialize(parent_checkpoint,device,arm):
    parent=load_restorer(parent_checkpoint);model=JointBasis(ComplexTFUNet(),float(parent.residual_scale),arm)
    state=model.state_dict();old=parent.state_dict()
    if set(state)-set(old)!={'filter.'+k for k in model.filter.state_dict()} or not set(old).issubset(state):
        raise ValueError('All original R03 state plus exactly the new filter required')
    state.update(old);model.load_state_dict(state,strict=True)
    return model.to(device).eval()


def save_checkpoint(path,model,optimizer,arm,epoch,updates,filter_state=None):
    torch.save(dict(schema='basis_filter_checkpoint_v1',config=d.configuration(arm),arm=arm,epoch=epoch,updates=updates,
        model=model.state_dict(),optimizer=None if optimizer is None else optimizer.state_dict(),scale=float(model.residual_scale),
        parent_run=d.PARENT_RUN,parent_commit=d.PARENT_COMMIT,parent_corrector_epochs=20,parent_coarse_epochs=20,
        TX_labels_read=0,target_training=False,F_optimizer_updates=0,head_fit_updates=0,coarse_updates=updates,
        filter_updates=updates,filter_epochs=epoch,filter_state=filter_state),path)


def load_checkpoint(path,arm,device):
    value=torch.load(path,map_location='cpu',weights_only=False)
    if (value['schema']!='basis_filter_checkpoint_v1' or value['config']!=d.configuration(arm) or value['arm']!=arm or
        value['parent_run']!=d.PARENT_RUN or value['parent_commit']!=d.PARENT_COMMIT or value['parent_corrector_epochs']!=20 or
        value['parent_coarse_epochs']!=20 or value['TX_labels_read']!=0 or value['target_training'] or value['F_optimizer_updates']!=0 or
        value['head_fit_updates']!=0 or value['coarse_updates']!=value['updates'] or value['filter_updates']!=value['updates'] or
        value['filter_epochs']!=value['epoch'] or not 0<=value['epoch']<=d.EPOCHS or not 0<=value['updates']<=d.UPDATES or
        not math.isfinite(value['scale']) or value['scale']<=0):raise ValueError('Filter checkpoint budget/permissions/provenance mismatch')
    model=JointBasis(ComplexTFUNet(),value['scale'],arm).to(device);model.load_state_dict(value['model'],strict=True)
    return model.eval().requires_grad_(False),value
