"""Frozen-teacher-conditioned, bounded source restoration correction."""
import torch
from torch import nn
from torch.nn import functional as F


def source_targets(labeled,clean_logits,*,threshold=0.95):
    """L_s labels take precedence; U_s uses only confident clean pseudo-labels."""
    if clean_logits.ndim!=2 or labeled.shape!=(len(clean_logits),) or not 0<=threshold<=1:
        raise ValueError('Aligned source labels and clean teacher logits required')
    if ((labeled < -1) | (labeled >= clean_logits.shape[1])).any():
        raise ValueError('Only valid L_s labels or -1 for U_s allowed')
    confidence,pseudo=F.softmax(clean_logits.detach(),dim=1).max(1)
    known=labeled>=0
    return torch.where(known,labeled,pseudo), known | (confidence>=threshold)


def weighted_margin(logits,target,valid,baseline,*,hard_weight=8.0,margin=0.5):
    if logits.ndim!=2 or target.shape!=valid.shape or target.shape!=baseline.shape or target.shape!=(len(logits),):
        raise ValueError('Aligned logits, labels, validity and baseline required')
    if valid.dtype!=torch.bool or hard_weight<1:
        raise ValueError('Boolean target validity and hard weight required')
    own=logits.gather(1,target[:,None]).squeeze(1)
    other=logits.masked_fill(F.one_hot(target,num_classes=logits.shape[1]).bool(),-torch.inf).max(1).values
    per_row=F.softplus(other-own+margin)
    weights=torch.where(baseline!=target,hard_weight,1.0).to(logits)*valid.to(logits)
    return (weights*per_row).sum()/weights.sum().clamp_min(1)


class FeatureConditionedCorrector(nn.Module):
    """Single-pass IQ correction conditioned on frozen teacher features/logits."""
    def __init__(self,*,feature_dim=160,classes=6,width=64,layers=3,heads=4,epsilon=0.1):
        super().__init__()
        if not 0<epsilon<=1:raise ValueError('Positive bounded radius required')
        self.feature_dim=feature_dim;self.classes=classes;self.epsilon=float(epsilon)
        self.input=nn.Conv1d(4,width,4,stride=4)
        self.position=nn.Parameter(torch.randn(1,64,width)*.01)
        self.condition=nn.Sequential(nn.Linear(feature_dim+classes,width*2),nn.SiLU(),nn.Linear(width*2,width))
        block=nn.TransformerEncoderLayer(width,heads,4*width,dropout=0.,activation='gelu',batch_first=True,norm_first=True)
        self.encoder=nn.TransformerEncoder(block,layers,enable_nested_tensor=False)
        self.output=nn.Linear(width,8)
        nn.init.zeros_(self.output.weight);nn.init.zeros_(self.output.bias)

    def forward(self,received,anchor,features,logits):
        if received.ndim!=3 or received.shape[1:]!=(2,256) or anchor.shape!=received.shape:
            raise ValueError('Aligned received and anchor IQ windows required')
        if features.shape!=(len(received),self.feature_dim) or logits.shape!=(len(received),self.classes):
            raise ValueError('One frozen teacher feature and score vector per record required')
        context=torch.cat((F.normalize(features.detach(),dim=1),F.softmax(logits.detach(),dim=1)),dim=1)
        h=self.input(torch.cat((anchor,received),dim=1)).transpose(1,2)
        h=h+self.position+self.condition(context)[:,None,:]
        proposed=self.output(self.encoder(h)).reshape(len(h),64,2,4).permute(0,2,1,3).reshape_as(received)
        radius=self.epsilon*anchor.flatten(1).norm(dim=1)
        magnitude=proposed.flatten(1).norm(dim=1).clamp_min(1e-12)
        scale=(radius/magnitude).clamp(max=1)
        return anchor+proposed*scale[:,None,None]
