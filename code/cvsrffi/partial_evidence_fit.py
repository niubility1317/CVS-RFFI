"""Independent shared-covariance E: source moments, separate observation fit."""
from dataclasses import asdict
import json
from pathlib import Path
import torch
from torch import nn
from .evidence_conditions import ObservationErrorModel
from .partial_gaussian_head import lowrank_gaussian_scores
from .mask_pattern_calibration import PatternSpec


class PartialEvidenceHead(nn.Module):
    def __init__(self,means,diagonal,factor,error,pattern_spec):
        super().__init__();d=sum(pattern_spec.block_sizes)
        if means.ndim!=2 or means.shape[1]!=d or diagonal.shape!=(d,) or factor.ndim!=2 or factor.shape[0]!=d:
            raise ValueError('invalid partial evidence dimensions')
        if not torch.isfinite(means).all() or not torch.isfinite(diagonal).all() or (diagonal<=0).any() or not torch.isfinite(factor).all():
            raise ValueError('nonfinite/invalid E parameters')
        self.register_buffer('means',means.detach().clone());self.register_buffer('diagonal',diagonal.detach().clone());self.register_buffer('factor',factor.detach().clone())
        self.error=error;self.pattern_spec=pattern_spec;self.requires_grad_(False).eval()

    def forward(self,z,observed,quality):
        if quality.shape!=(len(z),3) or not torch.isfinite(quality).all():raise ValueError('finite quality [N,3] required')
        z=z.to(self.means);quality=quality.to(self.means);observed=observed.to(self.means.device)
        diagonal=self.diagonal[None]+self.error(quality)
        result=lowrank_gaussian_scores(z,self.means,diagonal,self.factor.expand(len(z),-1,-1),observed)
        result['solver_calls']=1
        result['density_nll_per_dimension']=-result['scores']/result['observed_count'].clamp_min(1)[:,None]
        return result

    def export_state(self):
        return dict(kind='partial_shared_v1',means=self.means.cpu().clone(),diagonal=self.diagonal.cpu().clone(),
                    factor=self.factor.cpu().clone(),error={k:v.cpu().clone() for k,v in self.error.state_dict().items()},
                    observation_floor=self.error.floor,observation_ceiling=self.error.ceiling,
                    block_names=list(self.pattern_spec.block_names),block_sizes=list(self.pattern_spec.block_sizes))

    @classmethod
    def from_export(cls,state):
        expected={'kind','means','diagonal','factor','error','observation_floor','observation_ceiling','block_names','block_sizes'}
        if set(state)!=expected or state['kind']!='partial_shared_v1':raise ValueError('invalid E export schema')
        spec=PatternSpec(tuple(state['block_names']),tuple(state['block_sizes']))
        error=ObservationErrorModel(sum(spec.block_sizes),state['observation_floor'],state['observation_ceiling'])
        error.load_state_dict(state['error'],strict=True)
        if not bool(error.calibrated) or not torch.isfinite(error.coefficients).all() or (error.coefficients<0).any():raise ValueError('invalid E observation model')
        return cls(state['means'],state['diagonal'],state['factor'],error,spec)


@torch.no_grad()
def fit_partial_evidence(cache,pattern_spec,*,rank=4,shrinkage=.1,observation_floor=1e-5,observation_ceiling=1.,output=None):
    cache.validate()
    if cache.identity.role!='L_s':raise ValueError('E fit accepts only L_s')
    d=sum(pattern_spec.block_sizes)
    if cache.h.shape[1]!=d or not cache.observed.all() or not 0<=shrinkage<=1 or not 1<=rank<d:
        raise ValueError('invalid E full-block cache/rank/shrinkage')
    clean=[i for i,v in enumerate(cache.view_ids) if v=='clean'];degraded=[i for i,v in enumerate(cache.view_ids) if v!='clean']
    by_id={cache.physical_ids[i]:i for i in clean}
    if len(by_id)!=len(clean) or len(degraded)<4 or set(by_id)!={cache.physical_ids[i] for i in degraded}:raise ValueError('E needs paired physical clean/degraded records')
    reference=cache.h[clean].double();labels=cache.labels[clean];classes=cache.baseline_inference_logits.shape[1]
    if sorted(labels.unique().tolist())!=list(range(classes)):raise ValueError('E source classes incomplete')
    if len(clean)<=classes:raise ValueError('insufficient within-class residual support')
    means=torch.stack([reference[labels==y].mean(0) for y in range(classes)])
    residual=reference-means[labels];cov=residual.T@residual/(len(reference)-classes)
    cov=(1-shrinkage)*cov+shrinkage*torch.diag(cov.diagonal())
    eigen,vectors=torch.linalg.eigh(cov);factor=vectors[:,-rank:]*eigen[-rank:].clamp_min(0).sqrt()
    diagonal=(cov.diagonal()-factor.square().sum(-1)).clamp_min(observation_floor)
    error=ObservationErrorModel(d,observation_floor,observation_ceiling).to(cache.h.device)
    pairs=torch.tensor([by_id[cache.physical_ids[i]] for i in degraded])
    # Exactly one LEO view per physical sample in the default cache. If expanded,
    # average per-physical squared errors before fitting rather than upweighting.
    if len(degraded)!=len(clean):raise ValueError('E v1 requires exactly one paired degradation per physical record')
    error.fit(cache.quality[degraded],cache.h[pairs],cache.h[degraded],role='L_s')
    model=PartialEvidenceHead(means.float(),diagonal.float(),factor.float(),error,pattern_spec)
    if output is not None:
        out=Path(output);out.mkdir(parents=True,exist_ok=False);torch.save(model.export_state(),out/'partial_evidence.pt')
        (out/'fit_manifest.json').write_text(json.dumps(dict(role='L_s',cache_identity=asdict(cache.identity),physical_count=len(clean),
            rank=rank,shrinkage=shrinkage,block_sizes=pattern_spec.block_sizes,shared_covariance=True,state_response=False,
            support_used=False,density_ce_joint_optimization=False,observation_bias_norm=float(error.calibration_bias.norm()),
            covariance_interpretation='source residual and controlled-degradation proxy; error independence not physically verified'),indent=2)+'\n',encoding='utf-8')
    return model
