"""Original six forwards and KL; replace only As two-mask CE aggregation."""
from pathlib import Path
from types import MethodType

import torch

KEYS = ('eq_i','rx_i','day_i','sig_i','base_index')


def _labeled_hook(self, **kwargs):
    from experiments.cvs_quality_risk70 import model as original
    from . import objective
    if not self._quality_pool_bound or self._quality_method != 'POOL_RANDOM':
        raise ValueError('Bind original-order observations with no quality reweighting')
    labels, domains = kwargs['labels'], kwargs['domains']
    extra, step = kwargs['extra'], int(kwargs['optimizer_step'])
    raw, ordinary = kwargs['labeled_inputs'], self._legacy_actual_L_input
    if (labels.shape != (128,) or labels.dtype != torch.long or domains is None
            or domains.shape != labels.shape or raw.shape != (128,2,256)
            or ordinary is None or ordinary.shape != raw.shape
            or not isinstance(extra,(tuple,list)) or len(extra)<2):
        raise ValueError('Original visible L128/raw/ordinary/domain tuple required')
    smask = (torch.arange(128,device=labels.device)+step).remainder(2) != 0
    slots = torch.nonzero(smask,as_tuple=False).flatten().cpu().tolist()
    columns = {}
    for key in KEYS:
        value=extra[1][key]
        columns[key]=value.detach().cpu().tolist() if hasattr(value,'detach') else list(value)
        if len(columns[key])!=128:
            raise ValueError('All original five-key metadata must align to L128')
    ids=[[columns[key][slot] for key in KEYS] for slot in slots]
    origin=self._full_channel.validate_origin(step,(step-1)//222+1,
        labels=labels[smask],domains=domains[smask],physical_ids=ids)

    # Capture existing calls only.  Snapshots precede each call so an accidental
    # in-place IQ change cannot make two references appear falsely identical.
    forward=self.forward_identity_only
    had_forward='forward_identity_only' in vars(self)
    old_forward=vars(self).get('forward_identity_only')
    frames=[]
    def captured(x,*args,**kw):
        if len(frames)>=5 or x.shape!=(128,2,256):
            raise ValueError('Exactly five existing extra L128 forwards required')
        d=kw.get('domain_labels')
        if not torch.is_tensor(d) or not torch.equal(d,domains):
            raise ValueError('All original forward domains must remain unchanged')
        iq=x.detach().clone()
        output=forward(x,*args,**kw)
        logits=output.get('tx_logits') if isinstance(output,dict) else None
        if logits is None or logits.shape!=(128,6) or not bool(torch.isfinite(logits).all()):
            raise ValueError('Finite default-head128x6 logits required')
        frames.append(dict(iq=iq,logits=logits))
        return output
    before=len(self._quality_records)
    self.forward_identity_only=captured
    try:
        delta,info=original._labeled_hook(self,**kwargs)
    finally:
        if had_forward:self.forward_identity_only=old_forward
        else:delattr(self,'forward_identity_only')
    if (len(frames)!=5 or len(self._quality_records)!=before+1
            or not torch.equal(frames[0]['iq'],ordinary)
            or not torch.equal(frames[1]['iq'],frames[2]['iq'])
            or not torch.equal(frames[1]['iq'][~smask],ordinary[~smask])
            or not torch.equal(frames[3]['iq'],raw)
            or not torch.equal(frames[4]['iq'],raw)):
        raise ValueError('Native ordinary/mixed/raw paired IQ or call sequence changed')
    q=self._quality_records[-1]
    quality_groups=torch.as_tensor(q['groups'],device=labels.device,dtype=torch.long)
    difference,diagnostic=objective.terms(frames[1]['logits'][smask],frames[2]['logits'][smask],
        labels[smask],self._marginal_mode,quality_groups=quality_groups)
    if not difference.requires_grad:
        raise ValueError('Both original As CE branches must retain default-head gradients')
    correction=self._marginal_coefficient*difference
    d=float(difference.detach());change=float(correction.detach())
    native=self._full_records[-1]
    base_total=native['total_L'];base_mixed_ce=native['extra_CE'];base_mixed_rd=native['extra_RD']
    base_before_shape=native['base_L_before_shape']
    native.update(extra_CE=base_mixed_ce+.5*d,extra_RD=base_mixed_rd+.5*d,
        total_L=base_total+change,base_L_before_shape=base_before_shape+change,
        base_L_before_marginal=base_total,As_CE_aggregation=self._marginal_mode,
        As_CE_correction=change)
    q.update(order='ORIGINAL_GENERATION_ORDER',origin_verified=True,
        selected_raw_indices=slots,physical_ids=origin['physical_ids'],
        source_labels=origin['labels'],source_domains=origin['domains'],no_quality_weighting=True,
        As_CE_weighted=diagnostic['used_CE'],As_CE_used=diagnostic['used_CE'],
        As_CE_probability_mean=diagnostic['probmean_CE'],As_CE_aggregation=self._marginal_mode,
        used_total_L=q['used_total_L']+change,CE_correction_to_total=q['CE_correction_to_total']+change,
        group_probability_CE=[r['probmean_CE'] for r in diagnostic['group_stats']],
        group_used_CE=[r['used_CE'] for r in diagnostic['group_stats']])
    record=dict(diagnostic,method=self._marginal_method,step=step,
        cumulative_step=step+self._shape_offset,epoch=(step-1)//222+1,
        cumulative_epoch=20+(step-1)//222+1,coefficient=self._marginal_coefficient,
        satellite_CE_delta_unscaled=d,total_CE_correction=change,
        base_total_L=base_total,used_total_L=native['total_L'],
        base_mixed_CE=base_mixed_ce,used_mixed_CE=native['extra_CE'],
        base_mixed_RD=base_mixed_rd,used_mixed_RD=native['extra_RD'],
        base_L_before_shape=base_before_shape,used_L_before_shape=native['base_L_before_shape'],
        As_KL_unweighted=q['As_KL_unweighted'],mixed_KL=native['extra_KL'],KL_unchanged=True,
        forward_count=5,total_L_forwards=6,L_NN=768,all_As_CE_supervised=64,
        same_IQ_each_RD_pair=True,ordinary_raw_IQ_unchanged=True,
        As_origin_verified=True,selected_raw_indices=slots,used_pool_indices=q['used_pool_indices'],
        groups=q['groups'],physical_ids=origin['physical_ids'],extra_NN=0,new_channel_calls=0,
        prediction_changes_are_training_view_diagnostics=True,
        diagnostics_do_not_gate_or_weight=True,not_inference_ensemble=True)
    self._marginal_records.append(record)
    info.update({'concat/total_L':native['total_L'],
        'quality/As_CE':diagnostic['used_CE'],'quality/CE_correction':q['CE_correction_to_total'],
        'marginal/mean_CE':diagnostic['mean_CE'],'marginal/probmean_CE':diagnostic['probmean_CE'],
        'marginal/Jensen_gap':diagnostic['Jensen_gap'],'marginal/correction':change})
    return delta+correction,info


_labeled_hook._accepts_labeled_inputs=True


def classifier(*args,**kwargs):
    from experiments.cvs_quality_risk70 import model as original
    return original.classifier(*args,**kwargs)


def source_domain_count(*args,**kwargs):
    from experiments.cvs_quality_risk70 import model as original
    return original.source_domain_count(*args,**kwargs)


def _attach(m,c):
    from . import design as d
    d.validate(c)
    if m._quality_method!='POOL_RANDOM' or m._quality_pool_bound or m._quality_records:
        raise ValueError('Only untouched original six-forward/default-head installation')
    m._marginal_mode=c['as_marginalization']['mode']
    m._marginal_method=c['method'];m._marginal_coefficient=c['as_marginalization']['coefficient']
    m._marginal_records=[]
    m._source_labeled_feature_hook=MethodType(_labeled_hook,m)
    m._feature_learning_report['as_marginal73']=dict(c['as_marginalization'])
    return m


def install(m,c):
    from . import design as d
    from experiments.cvs_quality_risk70 import model as original
    d.validate(c);return _attach(original.install(m,d.nominal_config()),c)


def build(c,device='cpu'):
    from . import design as d
    from experiments.cvs_quality_risk70 import model as original
    d.validate(c);return _attach(original.build(d.nominal_config(),device),c)


def bind_pool(m,c,directory):
    from . import design as d,ordered_pool
    d.validate(c)
    if (m._quality_pool_bound or m._full_calls or m._quality_records
            or Path(directory).resolve()!=Path(c['ordered_pool']['root']).resolve()):
        raise ValueError('Bind only the existing exact original-order source pool')
    m._full_channel=ordered_pool.load(directory,seed=c['quality_pool']['allocation_seed'])
    m._quality_pool_bound=True
    return m


def own_model(c,device='cpu'):
    from . import design as d
    from .source import parent_payload
    from experiments.cvs_quality_risk70.protocol import final_metadata,checkpoint
    d.validate(c);parent=parent_payload(c);out=Path(c['output_root'])
    final_metadata(c,d.read(out/'completion.json'),d.read(out/'source_frozen.json'),
        d.read(out/'initialization.json'),d.read(out/'source_contract.json'),d.read(d.SOURCE)['role_ids'])
    cp=torch.load(out/'final_ssdg.pth',map_location='cpu',weights_only=False);checkpoint(c,cp)
    if set(cp['model'])!=set(parent['model']) or set(cp['ema_model'])!=set(parent['ema_model']):
        raise ValueError('Original joint student/default-head and ownEMA keys differ')
    m=build(c,device);m.load_state_dict(cp['model'],strict=True)
    return m.eval().requires_grad_(False)
