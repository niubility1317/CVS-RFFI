"""Unified H1--H5 head. All persistent updates are explicit training/support calls."""
from dataclasses import dataclass, asdict
from typing import Mapping
import json
import math

import torch
from torch import nn
import torch.nn.functional as F

from .evidence_observation import extract_evidence
from .evidence_conditions import received_conditions, ObservationErrorModel
from .partial_gaussian_head import gaussian_scores, lowrank_gaussian_scores
from .conditional_response import ConditionalResponse, fit_support_response
from .pairwise_evidence import SharedPairResidual, project_pairwise


@dataclass(frozen=True)
class EvidenceConfig:
    variant: str = "H2"
    layout: str = "joint"
    covariance_rank: int = 4
    observation_floor: float = 1e-5
    observation_ceiling: float = 1.0
    response_variance: float = .02
    state_error_variance: float = .01
    prior_precision: float = 1.0
    pair_strength: float = .1
    pair_anchor: float = 1.0
    nll_weight: float = .01
    response_regularization: float = .001
    support_weight: float = .1
    partial_dropout: float = .15

    def __post_init__(self):
        if self.variant not in {"H1","H2","H3","H4","H5"}:
            raise ValueError("variant must be H1--H5; omit evidence config for original H0")
        if self.layout not in {"joint","blocks"}:
            raise ValueError("invalid evidence layout")
        if type(self.covariance_rank) is not int or self.covariance_rank < 0:
            raise ValueError("covariance_rank must be a nonnegative integer")
        if self.variant == "H1" and self.covariance_rank != 0:
            raise ValueError("H1 is diagonal: set covariance_rank=0 explicitly")
        if self.variant != "H1" and self.covariance_rank <= 0:
            raise ValueError("H2--H5 require a nonzero correlation rank")
        for key,value in asdict(self).items():
            if key not in {"variant","layout","covariance_rank","partial_dropout"}:
                if type(value) not in (int,float) or not math.isfinite(value) or value<=0:
                    raise ValueError(f"{key} must be finite and positive")
        if self.observation_floor >= self.observation_ceiling:
            raise ValueError("invalid observation bounds")
        if not 0<=self.partial_dropout<1:
            raise ValueError("partial_dropout must be in [0,1)")

    @classmethod
    def parse(cls, value):
        if isinstance(value, cls):
            return value
        if isinstance(value,str):
            value=json.loads(value)
        if not isinstance(value, Mapping):
            raise ValueError("evidence config must be an embedded JSON object")
        return cls(**dict(value))  # unknown fields are errors, never silently ignored


class EvidenceHead(nn.Module):
    def __init__(self, num_classes, feature_dim, config):
        super().__init__()
        self.config = EvidenceConfig.parse(config)
        self.num_classes,self.feature_dim = int(num_classes),int(feature_dim)
        self.stage=int(self.config.variant[1:])
        self.response=ConditionalResponse(num_classes,feature_dim,2)
        self.error=ObservationErrorModel(feature_dim,self.config.observation_floor,self.config.observation_ceiling)
        init=math.log(math.expm1(self.config.response_variance))
        self.raw_diagonal=nn.Parameter(torch.full((feature_dim,),init))
        self.factor=nn.Parameter(torch.randn(feature_dim,self.config.covariance_rank)*.01)
        self.register_buffer("state_error",torch.eye(2)*self.config.state_error_variance)
        if self.stage<3:
            # Only the constant response is relevant; inactive slope parameters must
            # not look like enabled trainable components in optimizer/activation reports.
            for name,p in self.response.named_parameters():
                if "mu" not in name and "mean" not in name:
                    p.requires_grad_(False)
        self.pair=(SharedPairResidual(feature_dim,2) if self.stage==5 else None)

    def observation(self, x, aux, observed=None):
        conditions=received_conditions(x)
        obs=extract_evidence(aux,self.config.layout,observed)
        mask=obs.observed & conditions["valid"][:,None]
        z=torch.where(mask,obs.z,0.)
        return z,mask,conditions

    def forward(self,x,aux,observed=None,*,detach_parameters=False):
        if observed is None and self.training and not detach_parameters and self.config.partial_dropout>0:
            # Feature-domain stress training only, not a claim of physical
            # phase/PA intervention. Supplied real masks always take precedence.
            observed=torch.rand((len(x),self.feature_dim),device=x.device)>=self.config.partial_dropout
        z,mask,cond=self.observation(x,aux,observed)
        state=cond["state"] if self.stage>=3 else torch.zeros_like(cond["state"])
        # No state/quality is obtained from labels, receiver IDs, or other queries.
        means=self.response(state)
        diag=F.softplus(self.raw_diagonal)+1e-6
        factor=self.factor
        if detach_parameters:
            means,diag,factor=means.detach(),diag.detach(),factor.detach()
        obs_var=self.error(cond["quality"])
        diagonal=diag[None,:]+obs_var
        b,c,d=means.shape
        response_cov=torch.diag_embed(diag)+factor@factor.T
        covariance=response_cov[None,None,:,:]+torch.diag_embed(obs_var)[:,None,:,:]
        state_cov=torch.zeros_like(covariance.expand(b,c,d,d))
        if self.stage>=3:
            state_cov=self.response.state_uncertainty(state,self.state_error.expand(b,2,2))
            domain=self.response.domain_diagnostics(state)
            # Coverage penalty is a conservative modeling policy, not physical
            # Fisher uncertainty. Never let clamping claim improved certainty.
            outside_var=domain["outside_distance"].square()*self.config.response_variance
            state_cov=state_cov+outside_var[:,None,None,None]*torch.eye(d,device=z.device)
            if detach_parameters:
                state_cov=state_cov.detach()
            covariance=covariance+state_cov
            result=gaussian_scores(z,means,covariance,mask)
        else:
            result=lowrank_gaussian_scores(z,means,diagonal,factor.expand(b,-1,-1),mask)
        base_scores=result["scores"]
        scores=base_scores
        residual=base_scores.new_zeros(b,c,c)
        if self.pair is not None:
            residual=self.config.pair_strength*self.pair(z,means,state,obs_var,covariance.expand(b,c,d,d).diagonal(dim1=-2,dim2=-1),mask)
            if detach_parameters:
                residual=residual.detach()
            differences=base_scores[:,:,None]-base_scores[:,None,:]+residual
            weights=torch.ones_like(differences)-torch.eye(c,device=z.device)[None]
            scores=project_pairwise(base_scores,differences,weights,self.config.pair_anchor)
        # An expert must not invent evidence for fully missing observations.
        scores=torch.where(mask.any(-1,keepdim=True),scores,torch.zeros_like(scores))
        return {**result,"scores":scores,"base_scores":base_scores,"z":z,"observed":mask,
                "state":state,"quality":cond["quality"],"means":means,
                "covariance":covariance.expand(b,c,d,d),"observation_variance":obs_var,
                "state_covariance":state_cov,"pair_residual":residual,
                "response_covariance":response_cov,
                "plateau":cond["plateau"],
                "state_in_domain":self.response.domain_diagnostics(state)["in_domain"] if self.stage>=3 else torch.ones(b,device=z.device,dtype=torch.bool),
                "state_outside_distance":self.response.domain_diagnostics(state)["outside_distance"] if self.stage>=3 else z.new_zeros(b)}

    def extra_loss(self,result,labels,physical_ids=None):
        labels=labels.long()
        valid=result["observed"].any(-1)
        # NLL estimates uncertainty in the current coordinate system; it must
        # not reward the joint backbone simply for shrinking that system.
        density=gaussian_scores(result["z"].detach(),result["means"],result["covariance"],result["observed"])
        nll=-density["scores"][torch.arange(len(labels),device=labels.device),labels]
        nll=nll[valid].mean() if valid.any() else result["scores"].sum()*0
        reg=sum(p.square().mean() for p in self.response.parameters())
        episode=result["scores"].sum()*0
        pairs=0
        competing_classes=0
        if self.stage>=4:
            if physical_ids is None:
                raise ValueError("H4/H5 source episodes require physical IDs")
            from .evidence_observation import physical_support_ids
            physical_support_ids(physical_ids)
            support=[]; held=[]
            for y in range(self.num_classes):
                ids=torch.where((labels==y)&valid)[0]
                if len(ids)>=2:
                    support.extend(ids[::2].tolist()); held.extend(ids[1::2].tolist())
            competing_classes=len(set(labels[support].tolist()))
            if held and competing_classes>=2:
                # Only episode-registered classes compete. These episodes are
                # source training, never evidence of truly unseen target identity.
                episode_classes=sorted(set(labels[support].tolist()))
                phi=self.response.basis(result["state"])
                posterior_scores=[]
                for y in episode_classes:
                    take=[i for i in support if int(labels[i])==y]
                    post=fit_support_response(result["z"][take],phi[take],result["covariance"][take,y],
                         result["observed"][take],[physical_ids[i] for i in take],
                         prior_mean=self.response.coefficients()[y],prior_precision=self.config.prior_precision)
                    mean,pvar=post.predict(phi[held])
                    cov=result["covariance"][held,y]+pvar
                    likelihood=gaussian_scores(result["z"][held],mean[:,None,:],cov[:,None,:,:],result["observed"][held])
                    posterior_scores.append(likelihood["scores"][:,0])
                targets=torch.tensor([episode_classes.index(int(labels[i])) for i in held],device=labels.device)
                episode=F.cross_entropy(torch.stack(posterior_scores,-1),targets)
                pairs=len(held)
        total=self.config.nll_weight*nll+self.config.response_regularization*reg+self.config.support_weight*episode
        return total,{"nll":nll.detach(),"response_regularization":reg.detach(),"support_loss":episode.detach(),"support_queries":pairs,
                      "support_competing_classes":competing_classes,"support_effective_queries":pairs,
                      "observed_fraction":float(result["observed"].float().mean().detach()),
                      "outside_state_fraction":float((~result["state_in_domain"]).float().mean().detach()),
                      "observation_active":float(result["observation_variance"].mean().detach()),
                      "correlation_energy":float(self.factor.square().sum().detach()),
                      "state_covariance_energy":float(result["state_covariance"].square().sum().detach()),
                      "pair_residual_energy":float(result["pair_residual"].square().sum().detach())}
