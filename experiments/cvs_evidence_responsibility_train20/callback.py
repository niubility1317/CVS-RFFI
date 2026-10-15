"""Visible-L responsibility loss, evaluated on one fixed current bank at a time."""
from pathlib import Path
import copy
import json

import torch

from . import design as d
from experiments.cvs_evidence_responsibility20.objective import losses
from experiments.cvs_evidence_recomposition20.capture import forward as capture, HEAD_PATH
from experiments.cvs_u_schedule50.counterfactual import _preserve_rng


SOURCE_RX = (1,3,4,6,8)
CURRENT_ORDINALS = (2,19,37,54)
BASELINES = ('BASE_CE','BASE_SSL')


def protected_names(model):
    encoder = model.id_backbone.encoder
    head = model.get_submodule(HEAD_PATH)
    objects = (head.weight,head.gain,encoder.response_gain,encoder.sequence_pool.gain_raw)
    if (any(not isinstance(p,torch.nn.Parameter) for p in objects)
            or tuple(objects[0].shape)!=(6,160)
            or any(tuple(p.shape)!=(160,) for p in objects[1:])
            or len({id(p) for p in objects})!=4):
        raise ValueError('Original default W and three actual fusion gains required')
    names = tuple(name for name,p in model.named_parameters() if any(p is q for q in objects))
    if len(names)!=4: raise ValueError('Protected default W/three gains must have unique native names')
    return names


def selected_names(model):
    protected = set(protected_names(model))
    names = tuple(name for name,p in model.named_parameters()
                  if name.startswith('id_backbone.encoder.') and name not in protected)
    named = dict(model.named_parameters())
    if len(names)!=68 or any(not named[name].requires_grad for name in names):
        raise ValueError('Actual trainable encoder feature scope must be68 tensors excluding W/three gains')
    return names


class Callback:
    def __init__(self,native,config,bank,*,default_weight_name=None,metrics_path=None):
        d.validate(config)
        self.native,self.c,self.bank = native,config,bank
        self.mode = 'ZERO' if config['arm'] in BASELINES else config['arm']
        if self.mode not in ('ZERO','JOINT_GAP','ROUTED_GAP'): raise ValueError('Unknown responsibility mode')
        if self.mode=='ZERO':
            if bank is not None: raise ValueError('Baseline must not construct or retain an auxiliary bank')
        elif (not isinstance(bank,dict) or set(bank.get('current',{}))!=set(CURRENT_ORDINALS)):
            raise ValueError('Four caller-validated visible L180 banks required')
        self.default_weight_name = default_weight_name
        self.metrics_path = None if metrics_path is None else Path(metrics_path)
        self.model = None; self.selected_names = None; self.last_step = None; self.last_record = None
        self.counts = dict(calls=0,L_packets=0,inactive_calls=0,scheduled_calls=0,eligible_calls=0,
            extra_IQ_forwards=0,extra_model_forward_calls=0,extra_L_packets=0,held_L_packets=0)
        self.by_current_ordinal = {str(key):0 for key in CURRENT_ORDINALS}

    def _bind(self,model):
        if self.model is not None:
            if model is not self.model: raise ValueError('Callback cannot bind a different student')
            return
        names = selected_names(model)
        head = model.get_submodule(HEAD_PATH)
        actual_W = [name for name,p in model.named_parameters() if p is head.weight]
        if len(actual_W)!=1 or (self.default_weight_name is not None and self.default_weight_name!=actual_W[0]):
            raise ValueError('Callback default W differs from the native head')
        self.model,self.selected_names,self.default_weight_name = model,names,actual_W[0]

    def _zero(self):
        return dict(self.model.named_parameters())[self.selected_names[0]].reshape(-1)[0]*0.

    def _record(self,record,loss):
        record['weighted_loss'] = float(loss.detach())
        self.last_record = record
        if self.metrics_path is not None:
            with self.metrics_path.open('a',encoding='utf-8') as stream:
                stream.write(json.dumps(record,allow_nan=False)+'\n')
        return loss,{'evidence_responsibility/scheduled':float(record['scheduled']),
            'evidence_responsibility/eligible':float(record['status']=='executed'),
            'evidence_responsibility/loss':record['weighted_loss'],
            'evidence_responsibility/extra_IQ_forwards':float(record['extra_IQ_forwards'])}

    def __call__(self,*,model,labeled_output,labels,extra,domains,optimizer_step,
                 identity_loss,labeled_count,labeled_inputs=None,role='L_s',**compat_kwargs):
        if role!='L_s' or not model.training: raise ValueError('Only the actual training student visible L_s is permitted')
        if type(optimizer_step) is not int or optimizer_step<1 or (self.last_step is not None and optimizer_step<=self.last_step):
            raise ValueError('Increasing actual optimizer step required')
        if type(labeled_count) is not int or labeled_count!=128: raise ValueError('Original main L128 batch required')
        self._bind(model)
        self.last_step = optimizer_step; self.counts['calls']+=1; self.counts['L_packets']+=labeled_count
        scheduled = self.mode!='ZERO' and d.active(optimizer_step)
        record = dict(step=optimizer_step,mode=self.mode,source_count=labeled_count,scheduled=scheduled,
            extra_IQ_forwards=0,extra_model_forward_calls=0,by_RX={})
        if not scheduled:
            self.counts['inactive_calls']+=1; record['status']='inactive'
            return self._record(record,self._zero())
        ordinal = CURRENT_ORDINALS[(self.counts['scheduled_calls'])%len(CURRENT_ORDINALS)]
        bank = self.bank['current'][ordinal]
        if len(bank['x'])!=180: raise ValueError('Fixed complete L180 population required')
        device = next(model.parameters()).device
        current = {key:value.to(device=device) if isinstance(value,torch.Tensor) else value for key,value in bank.items()}
        flags = [(module,module.training) for module in model.modules()]
        main_counter = model._feature_l_steps
        try:
            with _preserve_rng():
                model.eval()
                components = capture(model,current['x'])
                encoder = model.id_backbone.encoder; head = model.get_submodule(HEAD_PATH)
                values,summaries = losses(components,current,head.weight,
                    (head.gain,encoder.response_gain,encoder.sequence_pool.gain_raw),source_role='L_s')
                loss = values[self.mode]
                facts = copy.deepcopy(summaries[self.mode])
        finally:
            for module,flag in flags: module.training=flag
        if model._feature_l_steps!=main_counter: raise ValueError('Extra source eval changed the native main L counter')
        if not torch.is_tensor(loss) or loss.ndim!=0 or not bool(torch.isfinite(loss.detach())):
            raise ValueError('Responsibility auxiliary must be one finite scalar')
        for row in facts['by_RX'].values(): row['positive_gap']=row['mean_positive_burden']
        self.counts['scheduled_calls']+=1; self.counts['eligible_calls']+=1
        self.counts['extra_IQ_forwards']+=180; self.counts['extra_L_packets']+=180
        self.counts['extra_model_forward_calls']+=1; self.by_current_ordinal[str(ordinal)]+=1
        record.update(status='executed',bank_count=180,current_ordinal=ordinal,
            extra_IQ_forwards=180,extra_model_forward_calls=1,by_RX=facts['by_RX'],objective=facts,
            held_bank_used_for_main_gradient=False,eval_flags_RNG_caches_restored=True,
            native_L_counter_unchanged=True)
        return self._record(record,loss)

    __call__._accepts_labeled_inputs = False

    def summary(self):
        return dict(self.counts,mode=self.mode,frequency=d.FREQUENCY,
            scheduled_calls_expected=d.scheduled(self.c),bank_count=0 if self.bank is None else 720,
            bank_source_IQ_reads=0 if self.bank is None else 900,
            held_bank_IQ_reads=0 if self.bank is None else 180,held_bank_used_for_main_gradient=False,
            by_current_ordinal=dict(self.by_current_ordinal),selected_count=None if self.selected_names is None else len(self.selected_names),
            default_weight_name=self.default_weight_name,head_fit_updates=0,extra_actual_optimizer_steps=0,
            EMA_updates=0,permanent_forward_hooks=0,original_sampler_unchanged=True,
            default_head_joint_main_training=True,U_truth_read=False,target_access=False)


Observer = Callback
