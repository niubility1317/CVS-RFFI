"""Source-frozen Phase1 deployment. No target coordinator or training rows."""
import hashlib
import json
import math
from pathlib import Path
from types import SimpleNamespace
import torch
from .anchored_cache import IdentityAnchor, normal_cpu
from .anchored_geometry import load_angle_head
from .anchored_calibration import FinalProbabilityCalibrator
from .anchored_fusion import UtilityGate, realize_actions, utility_features
from .anchored_fit import FitConfig

STAGES=('cache','fit','oof','fuse','calibrate','export','profile')
CANDIDATES=('A0','A1','A2','A3','A4','A5-0','A5-25','A5-50','A5-75','A5-100','A6','C_angle','C_angle_keep','P1')
ARCH_KEYS=('num_classes','num_domains','model_size','dataset','input_len','sample_rate_hz','model_variant',
           'branch_ablation','domain_branch_ablation','domain_enhancer','domain_enhancer_strength',
           'id_feature_key','dom_feature_key','arch_family','representation_mode')


def state_identity(value):
    """Local parameter binding, not an extra data-validation or receipt gate."""
    digest=hashlib.sha256()
    def feed(x):
        if isinstance(x,torch.Tensor):
            t=x.detach().cpu().contiguous();digest.update(str((str(t.dtype),tuple(t.shape))).encode());digest.update(t.numpy().tobytes())
        elif isinstance(x,dict):
            for key in sorted(x):feed(key);feed(x[key])
        elif isinstance(x,(list,tuple)):
            digest.update(b'[')
            for item in x:feed(item)
            digest.update(b']')
        else:digest.update(json.dumps(x,sort_keys=True,allow_nan=False).encode())
    feed(value);return digest.hexdigest()


def validate_config(config):
    keys={'schema','stages','source_receivers','source_days','split_seed','view_seed','head_seeds','candidates',
          'fit','fusion','calibration','partial','cache','profile','deferred'}
    if set(config)!=keys or config['schema']!='core90_anchored_v1':raise ValueError('unknown/missing configuration field')
    if config['stages']!=list(STAGES) or config['candidates']!=list(CANDIDATES):raise ValueError('source stages/matrix must be explicit and complete')
    if config['source_receivers']!=[1,3,4,6,8] or config['source_days']!=[1,2,3] or config['split_seed']!=392005:raise ValueError('CORE90 contract mismatch')
    if config['view_seed']!=392005 or config['head_seeds']!=[392005,392006,392007]:raise ValueError('preregistered seed mismatch')
    fit=FitConfig.parse(config['fit'])
    if fit.rank!=4 or abs(fit.rho-math.log(2)/2)>1e-12 or fit.epochs!=80:raise ValueError('v1 fixed geometry/budget mismatch')
    expected={'fusion':{'lambda_h','ridge','utility_margin','protection_q'},'calibration':{'target_coverage'},
              'partial':{'rank','shrinkage','variance_floor','variance_ceiling','quality_bins','min_group_samples','consistency_quantile','min_confidence'},
              'cache':{'batch_size','shard_rows','preprocessing_version'},'profile':{'warmup','repeats'},
              'deferred':{'class_direction','state_response','support_posterior','support_slope','joint_backbone','adaptive_budget'}}
    for name,fields in expected.items():
        if set(config[name])!=fields:raise ValueError('unknown/missing '+name+' field')
    if any(config['deferred'].values()):raise ValueError('deferred branches are disabled in v1')
    if config['fusion']!={'lambda_h':2.0,'ridge':.001,'utility_margin':0.0,'protection_q':.9}:raise ValueError('v1 fusion definition changed')
    if config['calibration']['target_coverage']!=.9:raise ValueError('v1 coverage changed')
    if min(config['cache']['batch_size'],config['cache']['shard_rows'],config['profile']['repeats'])<1 or config['profile']['warmup']<0:raise ValueError('invalid runtime budget')
    if config['cache']['preprocessing_version']!='received_iq_v1':raise ValueError('unknown preprocessing')
    from .mask_pattern_calibration import PatternCalibrator
    p=config['partial'];PatternCalibrator(**{k:p[k] for k in ('quality_bins','min_group_samples','consistency_quantile','min_confidence')})
    if p['rank']!=4 or not 0<=p['shrinkage']<=1 or not 0<p['variance_floor']<=p['variance_ceiling']:raise ValueError('invalid E configuration')
    return config


class FrozenAnchoredSystem:
    def __init__(self,anchor,expert_state,*,fixed_action=0,gate_state=None,protection_threshold=1.,architecture=None,class_labels=None,mode='mixture'):
        if mode not in {'expert','mixture'} or (mode=='expert' and gate_state is not None):raise ValueError('invalid prediction mode')
        self.mode=mode
        if fixed_action not in range(5):raise ValueError('invalid fixed action')
        if not math.isfinite(protection_threshold) or not 0<=protection_threshold<=1:raise ValueError('invalid protection threshold')
        self.anchor=anchor;self.head=load_angle_head(expert_state).requires_grad_(False).eval()
        self.gate=UtilityGate.from_state_dict(gate_state) if gate_state is not None else None
        self.fixed_action=int(fixed_action);self.protection_threshold=float(protection_threshold)
        self.architecture={k:v for k,v in (architecture or {}).items() if k in ARCH_KEYS}
        self.class_labels=tuple(class_labels or range(len(expert_state.get('w0',expert_state.get('weight')))))
        if len(set(self.class_labels))!=len(self.class_labels) or len(self.class_labels)!=len(expert_state.get('w0',expert_state.get('weight'))):raise ValueError('duplicate/mismatched classes')
        self.calibrator=None
        self.identity=state_identity(self._core_state())
        self._frozen_versions=self._versions()

    def _versions(self):
        tensors=list(self.head.parameters())+list(self.head.buffers())
        if self.anchor is not None:tensors+=list(self.anchor.original_model.id_backbone.parameters())+list(self.anchor.original_model.id_backbone.buffers())
        if self.gate is not None:tensors+=[self.gate.mean,self.gate.scale,self.gate.coefficients]
        return (tuple((id(t),t._version) for t in tensors),self.mode,self.fixed_action,self.protection_threshold,
                None if self.gate is None else (self.gate.ridge,self.gate.utility_margin),self.head.tau0,tuple(self.class_labels))

    def _core_state(self):
        return dict(schema='anchored_system_v1',stage='Phase1',architecture=dict(self.architecture),class_labels=list(self.class_labels),
                    identity_backbone=None if self.anchor is None else {k:normal_cpu(v) for k,v in self.anchor.original_model.id_backbone.state_dict().items()},
                    expert=self.head.export_state(),gate=None if self.gate is None else self.gate.state_dict(),
                    fixed_action=self.fixed_action,protection_threshold=self.protection_threshold,mode=self.mode)

    def attach_calibrator(self,calibrator):
        # Validate live state once at the freeze/export boundary, not per sample.
        if calibrator.system_identity!=self.identity or state_identity(self._core_state())!=self.identity:
            raise ValueError('final calibration system identity mismatch')
        if self.calibrator is not None:raise RuntimeError('system already calibrated')
        self.calibrator=calibrator
        return self

    @torch.no_grad()
    def predict_features(self,h,s0,quality,valid):
        """Technical/cache interface; no labels or sample metadata accepted."""
        if self._versions()!=self._frozen_versions:raise ValueError('frozen system parameters changed')
        h=h.cpu()
        sg=None if self.mode=='mixture' and self.gate is None and self.fixed_action==0 else self.head(h).cpu()
        return self.predict_scores(s0,sg,quality,valid)

    @torch.no_grad()
    def predict_scores(self,s0,sg,quality,valid):
        """Frozen-score execution also allows separate fusion/calibration timing."""
        if self._versions()!=self._frozen_versions:raise ValueError('frozen system parameters changed')
        s0=s0.cpu();quality=quality.cpu();valid=valid.cpu();n=len(s0)
        if s0.shape!=(n,len(self.class_labels)) or valid.shape!=(n,) or valid.dtype!=torch.bool:raise ValueError('invalid predictor schema')
        p0=s0.double().log_softmax(-1)
        if self.mode=='expert':
            logp=sg.cpu().double().log_softmax(-1);choice=torch.full((n,),4,dtype=torch.long)
            alpha=torch.ones(n,dtype=torch.double);reason=torch.zeros(n,dtype=torch.long)
        elif self.gate is None and self.fixed_action==0:
            # Exact original H0 decision, no alternate readout on the fallback.
            logp=p0;choice=torch.zeros(n,dtype=torch.long);alpha=torch.zeros(n,dtype=torch.double);reason=torch.zeros(n,dtype=torch.long)
        else:
            sg=sg.cpu();actions=realize_actions(p0,sg.double().log_softmax(-1),protection_threshold=self.protection_threshold,valid=valid)
            safe_sg=torch.where(torch.isfinite(sg).all(-1)[:,None],sg,s0)
            choice=(self.gate.choose(utility_features(s0,safe_sg,quality,torch.ones(n)))['action'] if self.gate is not None else torch.full((n,),self.fixed_action,dtype=torch.long))
            index=torch.arange(n);logp=actions['log_probabilities'][index,choice]
            alpha=actions['alpha'][index,choice];reason=actions['reason'][index,choice]
        result=(self.calibrator.predict(logp,system_identity=self.identity) if self.calibrator is not None else
                dict(log_probabilities=logp,probabilities=logp.exp(),confidence=logp.exp().amax(-1),accepted=valid.clone(),top_class=logp.argmax(-1)))
        result['accepted']&=valid
        result.update(action=choice,realized_alpha=alpha,protection_reason=reason,baseline_top_class=s0.argmax(-1))
        result['predicted_labels']=[self.class_labels[i] for i in result['top_class'].tolist()]
        return result

    @torch.no_grad()
    def predict(self,x):
        if self.anchor is None:raise RuntimeError('IQ predictor requires frozen identity backbone')
        from .evidence_conditions import received_conditions
        h,s0,valid=self.anchor.extract(x)
        return self.predict_features(h,s0,received_conditions(x)['quality'],valid)

    def export_state(self):
        core=self._core_state()
        if state_identity(core)!=self.identity:raise ValueError('frozen system parameters changed after binding')
        return dict(**core,system_identity=self.identity,calibrator=None if self.calibrator is None else self.calibrator.state_dict())

    def save(self,path):
        if self.calibrator is None:raise RuntimeError('calibrate frozen system on V before deployment export')
        with Path(path).open('xb') as f:torch.save(self.export_state(),f)

    @classmethod
    def from_state(cls,state):
        expected={'schema','stage','architecture','class_labels','identity_backbone','expert','gate','fixed_action','protection_threshold','system_identity','calibrator','mode'}
        if set(state)!=expected or state['schema']!='anchored_system_v1' or state['stage']!='Phase1':raise ValueError('invalid deployment whitelist')
        if set(state['architecture'])-set(ARCH_KEYS):raise ValueError('invalid architecture whitelist')
        anchor=None
        if state['identity_backbone'] is not None:
            from post_stage_common import build_baseline_model
            model=build_baseline_model(SimpleNamespace(**state['architecture']),torch.device('cpu'))
            model.id_backbone.load_state_dict(state['identity_backbone'],strict=True);anchor=IdentityAnchor(model)
        obj=cls(anchor,state['expert'],fixed_action=state['fixed_action'],gate_state=state['gate'],protection_threshold=state['protection_threshold'],architecture=state['architecture'],class_labels=state['class_labels'],mode=state['mode'])
        if obj.identity!=state['system_identity']:raise ValueError('frozen system identity mismatch')
        if state['calibrator'] is not None:obj.attach_calibrator(FinalProbabilityCalibrator.from_state_dict(state['calibrator']))
        return obj
