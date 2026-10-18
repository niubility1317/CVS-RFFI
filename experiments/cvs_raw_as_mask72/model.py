"""Original six L forwards, exact raw-As origin, one dropout correlation change."""
from pathlib import Path
from types import MethodType

import torch

KEYS = ('eq_i','rx_i','day_i','sig_i','base_index')


def _labeled_hook(self, **kwargs):
    from experiments.cvs_quality_risk70 import model as original
    from . import masks
    if not self._quality_pool_bound or self._quality_method != 'POOL_RANDOM':
        raise ValueError('Bind original-order As storage with unchanged flat CE/KL')
    labels, domains = kwargs['labels'], kwargs['domains']
    extra, step = kwargs['extra'], int(kwargs['optimizer_step'])
    if (labels.shape != (128,) or labels.dtype != torch.long
            or domains is None or domains.shape != labels.shape
            or not isinstance(extra,(tuple,list)) or len(extra)<2):
        raise ValueError('One real original visible L128 batch required for pairing')
    smask = (torch.arange(128,device=labels.device)+step).remainder(2) != 0
    slots = torch.nonzero(smask,as_tuple=False).flatten().cpu().tolist()
    meta = extra[1]
    columns = {}
    for key in KEYS:
        value = meta[key]
        columns[key] = value.detach().cpu().tolist() if hasattr(value,'detach') else list(value)
        if len(columns[key]) != 128:
            raise ValueError('All five original physical fields must align to L128')
    ids = [[columns[key][slot] for key in KEYS] for slot in slots]
    origin = self._full_channel.validate_origin(step,(step-1)//222+1,
        labels=labels[smask],domains=domains[smask],physical_ids=ids)
    before = len(self._quality_records)
    with masks.scope(self,step=step,cumulative_step=step+self._shape_offset,
                     method=self._mask_mode,smask=smask) as recorder:
        result = original._labeled_hook(self,**kwargs)
    if len(self._quality_records) != before+1 or self._full_channel.last_origin is None:
        raise ValueError('Exactly one validated six-forward step per mask scope')
    q = self._quality_records[-1]
    q.update(order='ORIGINAL_GENERATION_ORDER',origin_verified=True,
             selected_raw_indices=slots,physical_ids=origin['physical_ids'],
             source_labels=origin['labels'],source_domains=origin['domains'],
             no_quality_weighting=True)
    record = dict(recorder.summary(),logical_method=self._mask_logical_method,
        epoch=(step-1)//222+1,cumulative_epoch=20+(step-1)//222+1,
        As_origin_verified=True,As_origin_count=64,
        selected_raw_indices=slots,used_pool_indices=q['used_pool_indices'],
        physical_ids=origin['physical_ids'],source_labels=origin['labels'],
        source_domains=origin['domains'],
        same_IQ_each_RD_pair=True,As_redraws=0,new_channel_calls=0,
        CE_KL_unchanged=True,extra_NN=0,persistent_mask_state=0)
    self._mask_records.append(record)
    return result


_labeled_hook._accepts_labeled_inputs = True


def classifier(*args,**kwargs):
    from experiments.cvs_quality_risk70 import model as original
    return original.classifier(*args,**kwargs)


def source_domain_count(*args,**kwargs):
    from experiments.cvs_quality_risk70 import model as original
    return original.source_domain_count(*args,**kwargs)


def _attach(m,c):
    from . import design as d
    d.validate(c)
    if m._quality_method != 'POOL_RANDOM' or m._quality_pool_bound or m._quality_records:
        raise ValueError('Start from the unchanged six-forward flat CE/KL code')
    m._mask_mode = c['mask_coupling']['mode']
    m._mask_logical_method = c['method']
    m._mask_records = []
    m._source_labeled_feature_hook = MethodType(_labeled_hook,m)
    m._feature_learning_report['raw_as_mask72'] = dict(c['mask_coupling'],
        logical_method=c['method'],As_order='Original generation order with exact current raw origin')
    return m


def install(m,c):
    from . import design as d
    from experiments.cvs_quality_risk70 import model as original
    d.validate(c)
    return _attach(original.install(m,d.nominal_config()),c)


def build(c,device='cpu'):
    from . import design as d
    from experiments.cvs_quality_risk70 import model as original
    d.validate(c)
    return _attach(original.build(d.nominal_config(),device),c)


def bind_pool(m,c,directory):
    from . import design as d,ordered_pool
    d.validate(c)
    if (m._quality_pool_bound or m._full_calls or m._quality_records
            or Path(directory).resolve()!=Path(c['ordered_pool']['root']).resolve()):
        raise ValueError('Bind only the fixed existing original-order pool before updating')
    m._full_channel = ordered_pool.load(directory,seed=c['quality_pool']['allocation_seed'])
    m._quality_pool_bound = True
    return m


def own_model(c,device='cpu'):
    from . import design as d
    from .source import parent_payload
    from experiments.cvs_quality_risk70.protocol import final_metadata,checkpoint
    d.validate(c);parent=parent_payload(c);out=Path(c['output_root'])
    final_metadata(c,d.read(out/'completion.json'),d.read(out/'source_frozen.json'),
        d.read(out/'initialization.json'),d.read(out/'source_contract.json'),d.read(d.SOURCE)['role_ids'])
    cp=torch.load(out/'final_ssdg.pth',map_location='cpu',weights_only=False)
    checkpoint(c,cp)
    if set(cp['model'])!=set(parent['model']) or set(cp['ema_model'])!=set(parent['ema_model']):
        raise ValueError('Final joint default-head student/ownEMA state keys differ')
    m=build(c,device);m.load_state_dict(cp['model'],strict=True)
    return m.eval().requires_grad_(False)
