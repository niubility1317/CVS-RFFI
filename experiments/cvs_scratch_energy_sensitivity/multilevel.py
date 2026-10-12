"""Head-free raw CE/margin gradients scaled by one frozen source RMS per layer."""
import math
import torch
from torch.nn import functional as F
from experiments.cvs_e20_boundary_style.multilevel import BoundarySensitivity as CommonSensitivity,identity_margin
from experiments.cvs_e20_cross_style.multilevel import fixed_fishr
from experiments.cvs_e20_budget_fishr.multilevel import LEVELS

ESTIMATOR='finite_L_corrected_cross_AB_energy_sensitivity'

def _positive(values):
    return isinstance(values,dict) and set(values)==set(LEVELS) and all(type(v) in (float,int) and math.isfinite(v) and v>0 for v in values.values())


class BoundarySensitivity(CommonSensitivity):
    """Reuse native identity hooks and independent A/B sampling, never probe heads."""
    def __init__(self,base,seed,objective):
        super().__init__(base,seed)
        if objective not in ('ce','margin'):raise ValueError('Unknown main identity objective')
        self.objective=objective

    def raw_sensitivities(self,model,x,labels,levels=LEVELS,create_graph=True,perturb=None):
        with self.capture(perturb) as (probes,shapes):logits=model(x)
        margin=identity_margin(logits,labels)
        per_sample=F.cross_entropy(logits,labels,reduction='none') if self.objective=='ce' else margin
        leaves=tuple(p for level in levels for p in probes[level])
        gradients=torch.autograd.grad(per_sample.sum(),leaves,create_graph=create_graph,retain_graph=create_graph)
        raw={level:torch.cat((gradients[2*i].squeeze(-1),gradients[2*i+1].squeeze(-1)),1) for i,level in enumerate(levels)}
        if not all(bool(torch.isfinite(g).all()) for g in raw.values()):raise FloatingPointError('Nonfinite raw identity sensitivity')
        return logits,raw,shapes

    def sensitivities(self,model,x,labels,levels=LEVELS,eps=1e-6):
        if eps!=1e-6 or not _positive(getattr(self,'fixed_scales',None)):raise ValueError('Fixed source scales required before active sensitivities')
        logits,raw,shapes=self.raw_sensitivities(model,x,labels,levels)
        scores={};stats={};margin=identity_margin(logits,labels)
        for level,g in raw.items():
            scale=float(self.fixed_scales[level]);score=g/scale;rms=g.detach().square().mean(1).sqrt()
            scores[level]=score
            stats[level]=dict(raw_sensitivity_rms_mean=float(rms.mean()),raw_sensitivity_rms_min=float(rms.min()),raw_sensitivity_rms_max=float(rms.max()),
                sample_raw_rms=rms.tolist(),low_norm_rate=float((rms<=eps).float().mean()),
                normalized_score_rms_mean=float(score.detach().square().mean(1).sqrt().mean()),normalized_score_l2_mean=float(score.detach().norm(dim=1).mean()),
                sensitivity_eps=eps,normalization='fixed_layer_source_RMS',fixed_scale=scale,objective=self.objective,
                shape=shapes[level],coordinates=score.shape[1],score_requires_grad=score.requires_grad,
                main_margin_mean=float(margin.detach().mean()),main_margin_min=float(margin.detach().min()),main_margin_max=float(margin.detach().max()),
                main_CE_mean=float(F.cross_entropy(logits.detach(),labels)))
        return logits,scores,stats

    def _pair(self,model,source,step,device,config,calibration=False):
        if (config['statistic']!='raw_sensitivity' or config['sensitivity_objective']!=self.objective
            or config['sensitivity_normalization']!='fixed_layer_source_RMS'):
            raise ValueError('Raw sensitivity mode differs from registered config')
        # The inherited routine supplies only shared sampling/moment reduction.
        # Its legacy tag guard is adapted locally; self.sensitivities above is
        # the actual derivative implementation. Emitted metadata is replaced.
        result=super()._pair(model,source,step,device,dict(config,statistic='margin'),calibration)
        detail=result[2];ia,ib,_,_=self.pair_indices(source,step,calibration)
        detail.update(statistic='raw_sensitivity',estimator=ESTIMATOR,sensitivity_objective=self.objective,
            sensitivity_normalization='fixed_layer_source_RMS',fixed_scales=dict(self.fixed_scales))
        for level,info in detail['levels'].items():
            if not info['fishr_computed']:continue
            info['estimator']=ESTIMATOR
            for split,index in (('A',ia),('B',ib)):
                stats=info['sensitivity_'+split];values=stats.pop('sample_raw_rms');cells={}
                for value,y,rx in zip(values,source['y'][index].tolist(),source['rx'][index].tolist()):cells.setdefault(f'{rx}:{y}',[]).append(value)
                stats['raw_RMS_by_RX_TX']={key:dict(n=len(a),mean=sum(a)/len(a),rms=math.sqrt(sum(x*x for x in a)/len(a))) for key,a in cells.items()}
        return result

    def measure_scales(self,model,L,device):
        if hasattr(self,'fixed_scales'):raise ValueError('Source scales cannot be measured twice')
        if L.get('role','L_s') not in ('L_s','L'):raise ValueError('Only labeled source L can set scales')
        sums={k:0. for k in LEVELS};counts={k:0 for k in LEVELS};records=[]
        with self._evaluation(model):
            for i in range(12):
                ia,ib,day,_=self.pair_indices(L,4*(i+1),True)
                row=dict(pair=i,day=day,A_ids=[L['ids'][j] for j in ia.tolist()],B_ids=[L['ids'][j] for j in ib.tolist()],sums={},coordinates={})
                for index in (ia,ib):
                    _,raw,_=self.raw_sensitivities(model,L['x'][index].to(device),L['y'][index].to(device),create_graph=False)
                    for level,g in raw.items():
                        squared=float(g.detach().to(torch.float64).square().sum());n=g.numel()
                        sums[level]+=squared;counts[level]+=n
                        row['sums'][level]=row['sums'].get(level,0.)+squared;row['coordinates'][level]=row['coordinates'].get(level,0)+n
                records.append(row)
        scales={level:math.sqrt(sums[level]/counts[level]) for level in LEVELS}
        if not _positive(scales):raise FloatingPointError('Nonpositive/nonfinite source RMS; no default or resampling allowed')
        self.fixed_scales=scales
        return dict(status='FIXED_LAYER_RMS_COMPLETE',objective=self.objective,fixed_scales=scales,sums=sums,coordinate_counts=counts,records=records,
            pairs=12,model_forwards=24,source_L_readouts=1440,identity_updates=0,source_role='L_s',target_access=False,scale_detached=True,training_updates=False)

    def export_state(self):
        if not _positive(getattr(self,'fixed_scales',None)) or not _positive(getattr(self,'fixed_coefficients',None)):raise ValueError('Missing fixed sensitivity state')
        if any(v>10000 for v in self.fixed_coefficients.values()):raise ValueError('Coefficient cap violated')
        return dict(schema='fixed_energy_sensitivity_v1',objective=self.objective,normalization='fixed_layer_source_RMS',after_identity_updates=444,
            fixed_scales=dict(self.fixed_scales),fixed_coefficients=dict(self.fixed_coefficients))

    def restore_state(self,state):
        if hasattr(self,'fixed_scales') or hasattr(self,'fixed_coefficients'):raise ValueError('Existing fixed sensitivity state cannot be overwritten')
        expected=dict(schema='fixed_energy_sensitivity_v1',objective=self.objective,normalization='fixed_layer_source_RMS',after_identity_updates=444)
        if (not isinstance(state,dict) or set(state)!=set(expected)|{'fixed_scales','fixed_coefficients'}
            or any(state.get(k)!=v for k,v in expected.items()) or not _positive(state.get('fixed_scales')) or not _positive(state.get('fixed_coefficients'))
            or any(v>10000 for v in state['fixed_coefficients'].values())):raise ValueError('Invalid or incompatible fixed sensitivity state')
        self.fixed_scales=dict(state['fixed_scales']);self.fixed_coefficients=dict(state['fixed_coefficients'])
