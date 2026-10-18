"""Wrap the exact P76 C00 hook; only the As64 CE target entropy changes."""
from copy import deepcopy
from pathlib import Path
from types import MethodType
from weakref import WeakKeyDictionary
import torch
from . import design as d,entropy
from experiments.cvs_prop_contrast76 import model as control

pair=control.pair
classifier,source_domain_count=control.classifier,control.source_domain_count
_PHYSICS_SINKS=WeakKeyDictionary()


def physics_sink(model,sink=None):
    if sink is None:_PHYSICS_SINKS.pop(model,None)
    else:_PHYSICS_SINKS[model]=sink


def hook(self,**kw):
    mode=self._entropy_arm;step=kw['optimizer_step'];labels=kw['labels']
    if mode not in ('R','Q') or self._prop_arm!='C00' or self._prop_share or self._prop_contrast:
        raise ValueError('Only fixed Q/R over exact P76 C00 may run')
    mask=(torch.arange(128,device=labels.device)+step).remainder(2)!=0
    old_grouped,old_terms=control.grouped_view,pair.as_ce_terms
    previous_sink=control._PHYSICS_SINKS.get(self)
    mappings=[];losses=[];physical=[]

    def observed_grouped(primary,raw,meta,namespace,share):
        if mappings or share is not False:raise ValueError('Exactly one unchanged C00 native As64 generation')
        value=old_grouped(primary,raw,meta,namespace,share)
        if len(value[1])!=64 or any('signal_to_added_noise_db' not in row or row.get('namespace')!=namespace for row in value[1]):
            raise ValueError('Actual per-packet native added-noise metadata required')
        mappings.append(entropy.from_records(value[1],namespace,mode))
        return value

    def observed_terms(z0,z1,y,actual_mask,terms):
        if len(mappings)!=1 or losses or not torch.equal(actual_mask,mask) or not torch.equal(y,labels):
            raise ValueError('One aligned original mixed As CE pair required')
        result=old_terms(z0,z1,y,actual_mask,terms)
        epsilon=torch.tensor(mappings[0]['epsilon_assigned'],device=z0.device,dtype=z0.dtype)
        losses.append(entropy.paired_delta(z0[mask],z1[mask],y[mask],epsilon))
        # Keep the old nominal .01 scalar decomposition.  The exact As-only
        # correction is added once after the complete unchanged C00 hook.
        return result

    def capture(record):
        if physical:raise ValueError('One original C00 physical payload per L step')
        physical.append(record)

    try:
        control.grouped_view=observed_grouped;pair.as_ce_terms=observed_terms;control.physics_sink(self,capture)
        delta,info=control.hook(self,**kw)
        if len(mappings)!=len(losses) or len(losses)!=1 or len(physical)!=1:
            raise ValueError('Every actual As64 must supply epsilon, two CE logits and physical context')
        correction=losses[0]['weighted_delta'];native=self._full_records[-1];base_total=native['total_L']
        if not losses[0]['all_default']:delta=delta+correction
        native['total_L']=base_total+float(correction.detach())
        scalar=entropy.summary(mappings[0],losses[0],base_total,native['total_L'])
        record=self._prop_records[-1]
        record.update(method=mode,control_arm='C00',entropy=scalar,
            entropy_weighted_delta=scalar['weighted_delta'],entropy_epsilon_mean=scalar['epsilon_mean'],
            entropy_epsilon_min=scalar['epsilon_min'],entropy_epsilon_max=scalar['epsilon_max'])
        native.update(quality_entropy77=scalar,entropy_weighted_delta=scalar['weighted_delta'],
            entropy_epsilon_mean=scalar['epsilon_mean'],entropy_epsilon_min=scalar['epsilon_min'],entropy_epsilon_max=scalar['epsilon_max'])
        payload=dict(physical[0],arm=mode,control_arm='C00',entropy=dict(mappings[0],**{k:v for k,v in scalar.items() if k not in mappings[0]}))
        sink=_PHYSICS_SINKS.get(self)
        if sink is not None:sink(payload)
        info.update({'entropy77/weighted_delta':scalar['weighted_delta'],'entropy77/epsilon_mean':scalar['epsilon_mean']})
        return delta,info
    finally:
        control.grouped_view=old_grouped;pair.as_ce_terms=old_terms
        control.physics_sink(self,previous_sink)


hook._accepts_labeled_inputs=True


def attach(m,c):
    d.validate(c)
    if m._prop_arm!='C00' or m._prop_share or m._prop_contrast:
        raise ValueError('Exact already-installed P76 C00 control required')
    m._entropy_arm=c['arm'];m._entropy_records=m._prop_records
    m._source_labeled_feature_hook=MethodType(hook,m)
    m._feature_learning_report['quality_entropy77']=dict(deepcopy(c['quality_entropy']),
        control_arm='C00',permutation_seed=entropy.PERMUTATION_SEED,permutation_part=entropy.PERMUTATION_PART,
        old_As_CE_telemetry='Nominal .01; entropy correction is logged separately')
    return m


def install(m,c):return attach(control.install(m,d.control_config()),c)
def build(c,device='cpu'):return attach(control.build(d.control_config(),device),c)


def own_model(c,device='cpu'):
    from .source import parent_payload
    from .protocol import final_metadata,checkpoint
    d.validate(c);parent=parent_payload(c);out=Path(c['output_root'])
    final_metadata(c,d.read(out/'completion.json'),d.read(out/'source_frozen.json'),d.read(out/'initialization.json'),
        d.read(out/'source_contract.json'),d.read(d.SOURCE)['role_ids'])
    cp=torch.load(out/'final_ssdg.pth',map_location='cpu',weights_only=False);checkpoint(c,cp)
    if set(cp['model'])!=set(parent['model']) or set(cp['ema_model'])!=set(parent['ema_model']):
        raise ValueError('Own joint default-head/student/EMA keys differ')
    m=build(c,device);m.load_state_dict(cp['model'],strict=True);return m.eval().requires_grad_(False)
