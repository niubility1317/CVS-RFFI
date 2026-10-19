import math
import torch
from torch import nn
from torch.nn import functional as F
from . import design as d
from .losses import posterior_distance
from experiments.pamf_feature_refine.models import feature_distance
from experiments.pamf_reception_restore.models import Core,ComplexTFUNet
from experiments.pamf_change_preserve.runner import load_restorer


class FeedbackCore(Core):
    def __init__(self):
        super().__init__(incoming=4)
        self.widths=(32,64,96,128)
        self.modulation=nn.Sequential(nn.Linear(160,d.MODULATION_HIDDEN),nn.SiLU(),nn.Linear(d.MODULATION_HIDDEN,640))
        nn.init.zeros_(self.modulation[-1].weight);nn.init.zeros_(self.modulation[-1].bias)

    def forward(self,value,code):
        if code.shape!=(len(value),160):raise ValueError('One per-frame 160D conditioning code required')
        features=self.encoder(value);gamma,beta=self.modulation(code).chunk(2,1)
        gamma=gamma.split(self.widths,1);beta=beta.split(self.widths,1)
        features=[feature*(1+d.MODULATION_BOUND*g.tanh()[:,:,None])+d.MODULATION_BOUND*b.tanh()[:,:,None]
                  for feature,g,b in zip(features,gamma,beta)]
        value=features[-1]
        for block,skip in zip(self.decode,features[-2::-1]):
            value=block(torch.cat((F.interpolate(value,size=skip.shape[-1],mode='linear',align_corners=False),skip),1))
        return self.out(value)


class JointFeedback(nn.Module):
    def __init__(self,coarse,scale,arm,runtime_reader=None):
        super().__init__()
        if arm not in d.ARMS or not math.isfinite(float(scale)) or float(scale)<=0:raise ValueError('Registered feedback arm and source scale required')
        self.arm=arm;self.coarse=coarse.requires_grad_(True);self.correction=FeedbackCore()
        self.register_buffer('residual_scale',torch.tensor(float(scale),dtype=torch.float32))
        if arm==d.ARMS[1]:
            if runtime_reader is None:raise ValueError('Verified original frozen F1 required at runtime')
            self.runtime_reader=runtime_reader.eval().requires_grad_(False)

    def train(self,mode=True):
        super().train(mode)
        if hasattr(self,'runtime_reader'):self.runtime_reader.eval()
        return self

    def forward(self,received):
        if received.ndim!=3 or received.shape[1:]!=(2,256):raise ValueError('Current frame IQ only,shape N2x256 required')
        base=self.coarse(received);context=torch.cat((received,base),1)
        code=(self.runtime_reader.features(base) if self.arm==d.ARMS[1] else F.adaptive_avg_pool1d(context,40).flatten(1))
        code=F.normalize(code,dim=1)
        return base+self.residual_scale*self.correction(context,code)


def initialize(parent_checkpoint,device,arm,runtime_reader=None):
    parent=load_restorer(parent_checkpoint)
    model=JointFeedback(ComplexTFUNet(),float(parent.residual_scale),arm,runtime_reader)
    model.coarse.load_state_dict(parent.coarse.state_dict(),strict=True)
    status=model.correction.load_state_dict(parent.correction.state_dict(),strict=False)
    expected={'modulation.'+key for key in model.correction.modulation.state_dict()}
    if set(status.missing_keys)!=expected or status.unexpected_keys:raise ValueError('Only new zero-FiLM parameters may be missing from original R03')
    return model.to(device).eval()


def save_checkpoint(path,model,optimizer,arm,epoch,updates):
    torch.save(dict(schema='semantic_feedback_checkpoint_v1',config=d.configuration(arm),arm=arm,epoch=epoch,updates=updates,
        model=model.state_dict(),optimizer=None if optimizer is None else optimizer.state_dict(),scale=float(model.residual_scale),
        parent_run=d.PARENT_RUN,parent_commit=d.PARENT_COMMIT,parent_corrector_epochs=20,parent_coarse_epochs=20,
        runtime_F_seed=d.configuration(arm)['runtime_F_seed'],F_optimizer_updates=0,head_fit_updates=0,
        TX_labels_read=0,target_training=False,coarse_updates=updates,modulation_updates=updates),path)


def load_checkpoint(path,arm,device,runtime_reader=None):
    value=torch.load(path,map_location='cpu',weights_only=False)
    if (value['schema']!='semantic_feedback_checkpoint_v1' or value['config']!=d.configuration(arm) or value['arm']!=arm or
        value['parent_run']!=d.PARENT_RUN or value['parent_commit']!=d.PARENT_COMMIT or value['parent_corrector_epochs']!=20 or
        value['parent_coarse_epochs']!=20 or value['runtime_F_seed']!=d.configuration(arm)['runtime_F_seed'] or
        value['TX_labels_read']!=0 or value['target_training'] or value['F_optimizer_updates']!=0 or value['head_fit_updates']!=0 or
        value['coarse_updates']!=value['updates'] or value['modulation_updates']!=value['updates'] or
        not 0<=value['epoch']<=d.EPOCHS or not 0<=value['updates']<=d.UPDATES or
        not math.isfinite(value['scale']) or value['scale']<=0):raise ValueError('Feedback checkpoint budget/permissions/provenance mismatch')
    model=JointFeedback(ComplexTFUNet(),value['scale'],arm,runtime_reader).to(device)
    model.load_state_dict(value['model'],strict=True)
    return model.eval().requires_grad_(False),value
