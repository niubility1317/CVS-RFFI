"""Four fixed cells reuse six L forwards; one native As64 and optional As contrast."""
from collections import Counter
from copy import deepcopy
from pathlib import Path
from types import MethodType
from weakref import WeakKeyDictionary
import torch
from . import design as d,relations
from .channel import grouped_view
from experiments.cvs_full_channel_pair68 import model as pair

classifier,source_domain_count=pair.classifier,pair.source_domain_count
PHYSICAL_FIELDS=pair.PHYSICAL_FIELDS
_PHYSICS_SINKS=WeakKeyDictionary()


def physics_sink(model,sink=None):
    if sink is None:_PHYSICS_SINKS.pop(model,None)
    else:_PHYSICS_SINKS[model]=sink


def metadata(values,n,device):
    result={}
    for key in ('rx_i','day_i','eq_i'):
        value=values.get(key)
        if (not torch.is_tensor(value) or value.dtype!=torch.long or value.shape!=(n,)):
            raise ValueError('Complete visible L integer RX/day/eq metadata required')
        result[key]=value.to(device=device)
    dummy=torch.zeros(n,dtype=torch.long,device=device)
    relations.masks(dummy,result['rx_i'],result['day_i'],result['eq_i'])
    return result


def group_statistics(records,labels,share):
    grouped={}
    for obs,label in zip(records,labels):
        pid=obs['sample_id'];group=tuple((int(pid[1]),int(pid[2]),int(pid[0])))
        cell=grouped.setdefault(group,dict(count=0,TX=Counter(),scenes=set(),states=set(),locks=set()))
        cell['count']+=1;cell['TX'][int(label)]+=1;cell['scenes'].add(obs['scene'])
        cell['states'].add(obs['state_end']);cell['locks'].add(obs['receiver_lock_status'])
    return [dict(RX=key[0],day=key[1],eq=key[2],count=v['count'],TX_counts=dict(v['TX']),
        scenes=sorted(v['scenes']),states=sorted(v['states']),locks=sorted(v['locks']),share_propagation=share)
        for key,v in sorted(grouped.items())]


def hook(self,**kw):
    labels,step=kw['labels'],kw['optimizer_step'];raw=kw['labeled_inputs']
    if (self._prop_arm not in d.ARMS or labels.dtype!=torch.long or labels.shape!=(128,)
            or not bool(((labels>=0)&(labels<6)).all()) or raw.shape!=(128,2,256)
            or raw.dtype!=torch.float32 or not bool(torch.isfinite(raw).all())
            or type(step)is not int or step not in range(1,2221)
            or not isinstance(kw['extra'],(tuple,list)) or len(kw['extra'])<2
            or not isinstance(kw['extra'][1],dict)):
        raise ValueError('Only the fixed true L128/source E21..30 hook')
    full_meta=metadata(kw['extra'][1],128,labels.device)
    mask=(torch.arange(128,device=labels.device)+step).remainder(2)!=0
    selected={key:value[mask] for key,value in full_meta.items()}
    channel=self._full_channel;primary=channel.obj
    old_primary=primary.apply;had_primary='apply' in vars(primary);old_primary_override=vars(primary).get('apply')
    old_view=pair.independent_view;old_penalty=pair.base.relations.penalty
    originals=[];second_records=[];as_values=[];as_details=[];penalty_calls=0

    def first_apply(inputs,meta,*args,**kwargs):
        if args:
            if len(args)!=2 or kwargs:raise ValueError('Original primary epoch/step signature required')
            epoch,cumulative_step=args
        else:
            epoch,cumulative_step=kwargs['epoch'],kwargs['step']
        if (originals or cumulative_step!=step+4440 or epoch!=(cumulative_step-1)//222+1
                or inputs.shape!=(64,2,256)):
            raise ValueError('Exactly one original cumulative As64 generation required')
        namespace=f'FcrossRX/train/e{epoch}/step{cumulative_step}/L_S'
        value=grouped_view(primary,inputs,meta,namespace,self._prop_share)
        if value[0].shape!=(64,2,256) or len(value[1])!=64:
            raise ValueError('One complete grouped native As64 required')
        originals.append((value[0].detach().clone(),deepcopy(value[1])))
        return value

    def repeated_view(actual_primary,inputs,meta,first,hardware):
        if actual_primary is not primary or len(originals)!=1 or second_records or len(hardware)!=64:
            raise ValueError('Copy the existing first As64 only after its native generation')
        records=deepcopy(first)
        for obs in records:obs.update(view_index=1,view_kind='same_IQ_reuse',channel_generated=False,agc_recomputed=False)
        second_records.append(records)
        return originals[0][0].clone(),records

    def observed_penalty(ordinary,satellite,y,rx,actual_mask,mode):
        nonlocal penalty_calls
        if (mode!='ORD128' or not torch.equal(actual_mask,mask) or not torch.equal(y,labels)
                or not torch.equal(rx,full_meta['rx_i'])):
            raise ValueError('Original ORD128 policy and actual As positions required')
        value,details=old_penalty(ordinary,satellite,y,rx,actual_mask,mode)
        if self._prop_contrast:
            maps={key:satellite[key][mask] for key in relations.LEVELS}
            as_value,stats=relations.penalty(maps,y[mask],selected['rx_i'],selected['day_i'],selected['eq_i'])
            as_values.append(as_value);as_details.extend(dict(phase=penalty_calls,**v) for v in stats)
        penalty_calls+=1
        return value,details

    try:
        primary.apply=first_apply;pair.independent_view=repeated_view;pair.base.relations.penalty=observed_penalty
        delta,info=pair.hook(self,**kw)
        if len(originals)!=1 or len(second_records)!=1 or penalty_calls!=2:
            raise ValueError('Exactly two original mixed graph captures and one actual As64 required')
        if self._prop_contrast:
            if len(as_values)!=2:raise ValueError('Both existing As map graphs required')
            auxiliary=torch.stack(as_values).mean();weighted=.05*auxiliary
            delta=delta+weighted
            raw_value=float(auxiliary.detach());weighted_value=float(weighted.detach())
        else:raw_value=weighted_value=0.
        native=self._full_records[-1]
        if native.get('SupCon_weight')!=.1:
            raise ValueError('Keep the original actual tail ordinary SupCon coefficient .1')
        native['total_L']+=weighted_value
        record=self._pair_records[-1]
        group_stats=group_statistics(originals[0][1],labels[mask].tolist(),self._prop_share)
        if any(len(v['scenes'])!=1 for v in group_stats):raise ValueError('Scene grouped in every cell')
        record.update(method=self._prop_arm,share_propagation=self._prop_share,As_contrast=self._prop_contrast,
            generated_As0=64,generated_As1=0,As_generated_packets=64,As_forward_packets=128,
            actual_residual_snapshots=64,logical_second_slots=64,As_same_IQ=True,
            As_loss_raw=raw_value,As_loss_weighted=weighted_value,As_loss_evaluated=self._prop_contrast,
            ordinary_SupCon_coefficient=.1,ordinary_SupCon_loss=native['SupCon'],
            As_SupCon_coefficient=.05 if self._prop_contrast else 0.,As_details=as_details,
            As_valid_anchors=sum(v['valid_anchors'] for v in as_details)/4 if as_details else 0.,
            As_zero_anchors=sum(v['zero_anchors'] for v in as_details)/4 if as_details else 0.,
            As_positive_pairs=sum(v['positive_pairs'] for v in as_details)/4 if as_details else 0.,
            As_negative_pairs=sum(v['negative_pairs'] for v in as_details)/4 if as_details else 0.,
            group_count=len(group_stats),group_stats=group_stats,all_labeled_CE_retained=True,
            two_separate_CE_mean=True,cross_channel_KL=False)
        native.update(same_IQ_each_pair=True,same_IQ_ordinary_raw_pairs=True,As_generated_packets=64,
            As_forward_packets=128,As_loss_raw=raw_value,As_loss_weighted=weighted_value,
            ordinary_SupCon_coefficient=.1,As_SupCon_coefficient=record['As_SupCon_coefficient'],prop_contrast76=record)
        self._prop_records.append(record);self._prop_generated+=64;self._prop_slots+=128
        for view,records in enumerate((originals[0][1],second_records[0])):
            for label,obs in zip(labels[mask].tolist(),records):
                self._prop_coverage[(view,int(label),int(obs['sample_id'][1]),obs['scene'],obs['state_end'],obs['receiver_lock_status'])]+=1
        sink=_PHYSICS_SINKS.get(self)
        if sink is not None:sink(dict(step=step,cumulative_step=step+4440,arm=self._prop_arm,
            share_propagation=self._prop_share,As_contrast=self._prop_contrast,
            view0=originals[0][1],view1=second_records[0],labels_L_As=labels[mask].tolist(),
            group_stats=group_stats,actual_residual_snapshots=64,logical_As_forward_slots=128))
        info.update({'prop76/As_loss_raw':raw_value,'prop76/As_loss_weighted':weighted_value})
        return delta,info
    finally:
        pair.independent_view=old_view;pair.base.relations.penalty=old_penalty
        if had_primary:primary.apply=old_primary_override
        else:delattr(primary,'apply')


hook._accepts_labeled_inputs=True


def attach(m,c):
    d.validate(c);pair.attach(m,d.reference_config('INDEP_CE6'))
    if m._sat_engine.mode!='AS_CE':raise ValueError('Keep native dropout and As KL off')
    m._prop_arm=c['arm'];m._prop_share=c['share_propagation'];m._prop_contrast=c['As_contrast']
    m._prop_records=[];m._prop_generated=m._prop_slots=0;m._prop_coverage=Counter()
    m._source_labeled_feature_hook=MethodType(hook,m)
    m._feature_learning_report['prop_contrast76']=deepcopy(c['prop_contrast'])
    return m


def install(m,c):return attach(pair.original.install(m,d.reference_config('AS_CE')),c)
def build(c,device='cpu'):return attach(pair.original.build(d.reference_config('AS_CE'),device),c)


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
