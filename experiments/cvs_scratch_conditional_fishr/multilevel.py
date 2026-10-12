"""Live class-conditioned mixed derivatives at shallow and middle native maps."""
import math
import torch
from torch.nn import functional as F
from experiments.cvs_scratch_energy_sensitivity.multilevel import BoundarySensitivity as EnergySensitivity,_positive,identity_margin,fixed_fishr,LEVELS
from experiments.cvs_e20_moment_audit.fishr_stats import unbiased_variances,penalty
from .design import ESTIMATOR

def interaction_means(scores,y,rx):
    if scores.ndim!=2 or len(scores)!=60 or set(y.tolist())!=set(range(6)) or set(rx.tolist())!={1,3,4,6,8}:
        raise ValueError('Full balanced labeled 6TX/5RX pair required')
    cells=[]
    for tx in range(6):
        row=[]
        for receiver in (1,3,4,6,8):
            mask=(y==tx)&(rx==receiver)
            if int(mask.sum())!=2:raise ValueError('Exactly two samples per TX/RX cell required')
            row.append(scores[mask].mean(0))
        cells.append(torch.stack(row))
    m=torch.stack(cells)
    b=m-m.mean(1,keepdim=True)-m.mean(0,keepdim=True)+m.mean((0,1),keepdim=True)
    return b,m

def components_valid(values,components):
    return isinstance(values,dict) and set(values)==set(LEVELS) and all(
        isinstance(row,dict) and set(row)==set(components) and all(type(v) in (int,float) and math.isfinite(v) and 0<v<=10000 for v in row.values()) for row in values.values())

class BoundarySensitivity(EnergySensitivity):
    def __init__(self,base,seed,objective,components):
        super().__init__(base,seed,objective)
        if list(components) not in (['interaction'],['moment','interaction']):raise ValueError('Unknown conditional components')
        self.components=tuple(components)

    def component_pair(self,model,source,step,device,config,calibration=False):
        if (config['statistic']!='raw_sensitivity' or config['sensitivity_objective']!=self.objective
            or config['sensitivity_normalization']!='fixed_layer_source_RMS' or tuple(config['conditional_components'])!=self.components
            or config['fishr_estimator']!=ESTIMATOR or tuple(config['fishr_levels'])!=LEVELS
            or config['batch_size']!=120 or config['pair_batch_size']!=60 or config['pair_quota']!=2
            or config['fishr_fixed_denominator']!=2 or source.get('role','L_s') not in ('L_s','L')):
            raise ValueError('Conditional sensitivity contract differs')
        ia,ib,day,sizes=self.pair_indices(source,step,calibration)
        packs=[]
        with self._evaluation(model):
            for index in (ia,ib):
                y=source['y'][index].to(device);rx=source['rx'][index].to(device)
                logits,scores,stats=self.sensitivities(model,source['x'][index].to(device),y)
                for info in stats.values():
                    values=info.pop('sample_raw_rms');cells={}
                    for value,tx,receiver in zip(values,y.tolist(),rx.tolist()):cells.setdefault(f'{receiver}:{tx}',[]).append(value)
                    info['raw_RMS_by_RX_TX']={k:dict(n=len(v),mean=sum(v)/len(v),rms=math.sqrt(sum(x*x for x in v)/len(v))) for k,v in cells.items()}
                packs.append(dict(y=y,rx=rx,logits=logits,scores=scores,stats=stats))
            terms={};layer_detail={}
            for level in LEVELS:
                bm=[interaction_means(p['scores'][level],p['y'],p['rx']) for p in packs]
                interaction=(bm[0][0]*bm[1][0]).mean()
                terms[level]={'interaction':interaction}
                if 'moment' in self.components:
                    variances=[unbiased_variances(p['scores'][level],p['y'],p['rx'],sizes) for p in packs]
                    terms[level]['moment']=penalty(*variances)
                if not all(bool(torch.isfinite(t)) for t in terms[level].values()):raise FloatingPointError('Nonfinite conditional term')
                layer_detail[level]=dict(shape=packs[0]['stats'][level]['shape'],fishr_computed=True,
                    component_values={k:float(v.detach()) for k,v in terms[level].items()},
                    interaction_mean_square_A=float(bm[0][0].detach().square().mean()),interaction_mean_square_B=float(bm[1][0].detach().square().mean()),
                    sensitivity_A=packs[0]['stats'][level],sensitivity_B=packs[1]['stats'][level],estimator=ESTIMATOR,
                    gradient_path='live main-identity mixed second derivatives; double centering preserves additive TX and RX means')
            logits=torch.cat([p['logits'] for p in packs]);labels=torch.cat([p['y'] for p in packs]);zero=logits.sum()*0
            ce=F.cross_entropy(logits,labels) if calibration else None
        detail=dict(aux_forwards=2,auxiliary_packets=120,aux_ce=0.,fishr_steps=1,levels=layer_detail,
            day=day,source_RX=[1,3,4,6,8],distinct_PID=len(set(ia.tolist())|set(ib.tolist())),
            pair_A_distinct_PID=60,pair_B_distinct_PID=60,cross_batch_PID_overlap=len(set(ia.tolist())&set(ib.tolist())),
            aux_ce_to_backbone=False,fishr_levels=list(LEVELS),fishr_active_levels=list(LEVELS),fishr_fixed_denominator=2,
            estimator=ESTIMATOR,statistic='raw_sensitivity',auxiliary_heads=0,head_Fishr_gradient=False,EMA=False,
            native_CPU_CUDA_RNG_preserved=True,independent_AB=True,calibration_sampling=calibration,sensitivity_eps=1e-6,
            virtual_affine='a=1,b=0 per packet/channel; std(H) detached; no registered parameters',
            main_margin_mean=float(identity_margin(logits.detach(),labels).mean()),sensitivity_objective=self.objective,
            sensitivity_normalization='fixed_layer_source_RMS',fixed_scales=dict(self.fixed_scales),
            conditional_components=list(self.components),component_mean_denominator=len(self.components))
        return zero,terms,detail,ce

    def _pair(self,model,source,step,device,config,calibration=False):
        if not components_valid(getattr(self,'fixed_component_coefficients',None),self.components):raise ValueError('Missing component calibration')
        zero,components,detail,ce=self.component_pair(model,source,step,device,config,calibration)
        terms={}
        for level in LEVELS:
            terms[level]=sum(self.fixed_component_coefficients[level][k]*components[level][k] for k in self.components)/len(self.components)
            detail['levels'][level].update(raw_fishr=float(terms[level].detach()),raw_loss_negative=bool(terms[level].detach()<0),
                term_units='mean of individually calibrated components, before ramp/layer mean/joint cap',component_coefficients=dict(self.fixed_component_coefficients[level]))
        detail.update(fixed_component_coefficients=self.fixed_component_coefficients,fishr=float((sum(terms.values())/2).detach()))
        return zero,terms,detail,ce

    def export_state(self):
        if (not _positive(getattr(self,'fixed_scales',None)) or getattr(self,'fixed_coefficients',None)!={k:1. for k in LEVELS}
            or not components_valid(getattr(self,'fixed_component_coefficients',None),self.components)):raise ValueError('Invalid conditional fixed state')
        return dict(schema='fixed_conditional_sensitivity_v1',objective=self.objective,components=list(self.components),normalization='fixed_layer_source_RMS',
            after_identity_updates=444,fixed_scales=dict(self.fixed_scales),fixed_coefficients=dict(self.fixed_coefficients),
            fixed_component_coefficients={k:dict(v) for k,v in self.fixed_component_coefficients.items()})

    def restore_state(self,state):
        if any(hasattr(self,k) for k in ('fixed_scales','fixed_coefficients','fixed_component_coefficients')):raise ValueError('Existing calibration cannot be overwritten')
        expected=dict(schema='fixed_conditional_sensitivity_v1',objective=self.objective,components=list(self.components),normalization='fixed_layer_source_RMS',after_identity_updates=444)
        if (not isinstance(state,dict) or set(state)!=set(expected)|{'fixed_scales','fixed_coefficients','fixed_component_coefficients'}
            or any(state.get(k)!=v for k,v in expected.items()) or not _positive(state.get('fixed_scales'))
            or state.get('fixed_coefficients')!={k:1. for k in LEVELS} or not components_valid(state.get('fixed_component_coefficients'),self.components)):
            raise ValueError('Invalid conditional checkpoint state')
        self.fixed_scales=dict(state['fixed_scales']);self.fixed_coefficients=dict(state['fixed_coefficients'])
        self.fixed_component_coefficients={k:dict(v) for k,v in state['fixed_component_coefficients'].items()}
