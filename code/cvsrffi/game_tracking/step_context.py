from __future__ import annotations
from dataclasses import dataclass, field
from collections import defaultdict
import math
import torch
import torch.nn.functional as F
from cvsrffi.eval import apply_sat_channel_for_scenario
from .legacy.objective import labeled_terms
from .legacy.options import _threshold_mask

def slice_batch(value, start, end, total):
    if torch.is_tensor(value):
        return value[start:end] if value.ndim and value.shape[0] == total else value
    if isinstance(value, dict):
        return {k:slice_batch(v,start,end,total) for k,v in value.items()}
    if isinstance(value, list):
        return [slice_batch(v,start,end,total) for v in value]
    if isinstance(value, tuple):
        return tuple(slice_batch(v,start,end,total) for v in value)
    return value

def temporal_mask(pseudo, conf, meta, args):
    # Preserve the historical RX/day/eq + adjacent sample rule without TX metadata.
    mask = torch.zeros_like(conf, dtype=torch.bool)
    groups = defaultdict(list)
    for i in range(len(pseudo)):
        groups[tuple(int(meta[k][i]) for k in ('rx_i','day_i','eq_i'))].append(i)
    for rows in groups.values():
        for i in rows:
            if float(conf[i]) < args.pseudo_temporal_min_conf: continue
            for j in rows:
                if i == j or float(conf[j]) < args.pseudo_temporal_min_conf: continue
                if (int(pseudo[i]) == int(pseudo[j]) and
                    abs(int(meta['sig_i'][i])-int(meta['sig_i'][j])) <= args.pseudo_temporal_window and
                    abs(int(meta['base_index'][i])-int(meta['base_index'][j])) <= args.pseudo_temporal_window):
                    mask[i] = True
                    break
    return mask

def satellite_stage(epoch, capability_level=None):
    if capability_level is not None:
        level = float(capability_level)
        return (['leo_clear_weak'], .30) if level < 1/3 else ((['leo_low_elev_weak','leo_rain_weak'],.60) if level < 2/3 else (['leo_clear_weak','leo_low_elev_weak','leo_rain_weak'],.80))
    if epoch <= 40: return ['leo_clear_weak'], .30
    if epoch <= 90: return ['leo_low_elev_weak','leo_rain_weak'], .60
    return ['leo_clear_weak','leo_low_elev_weak','leo_rain_weak'], .80


def satellite_policy(level):
    level=float(level)
    if not math.isfinite(level) or not 0 <= level <= 1:
        raise ValueError('capability level must be finite inside [0,1]')
    return dict(probability=.30+.50*level,weights=[1.-2.*level/3.,level/3.,level/3.],
                scenarios=['leo_clear_weak','leo_low_elev_weak','leo_rain_weak'],
                physical_config_version='leo_weak_unchanged_v1')

@dataclass
class StepContext:
    x: torch.Tensor
    satellite: torch.Tensor
    y: torch.Tensor
    domain: torch.Tensor
    weights: dict
    epoch: int
    batch_index: int
    sample_ids: list
    satellite_scenario: str
    satellite_mask_count: int
    strong: torch.Tensor | None = None
    pseudo: torch.Tensor | None = None
    base_mask: torch.Tensor | None = None
    strong_mask: torch.Tensor | None = None
    origin_features: torch.Tensor | None = None
    origin_terms: dict = field(default_factory=dict)
    forward_calls: int = 0
    pseudo_selected: int = 0
    satellite_selected_mask: torch.Tensor | None = None
    satellite_channel_seed: int | None = None
    satellite_probability: float | None = None
    satellite_policy_weights: list | None = None
    capability_level: float | None = None

def prepare_context(batch, unlabeled_batch, model, ema, args, epoch, batch_index, weights, gen, augmentor=None, capability_level=None, *, exposure_record=None):
    device = next(model.parameters()).device
    x,y,d,meta = batch
    x,y,d = x.to(device),y.to(device),d.to(device)
    x_main = augmentor(x,labels=y,no_pa=not bool(args.aug_enable_pa_normal)) if augmentor is not None else x
    scenarios, probability = satellite_stage(epoch,capability_level)
    scenario = scenarios[(epoch+batch_index-2)%len(scenarios)]
    policy_weights = None
    version2 = getattr(args,'game_evidence_version',1) == 2
    if version2 and capability_level is not None:
        policy = satellite_policy(capability_level)
        probability, policy_weights = policy['probability'], policy['weights']
        selected_scenario = int(torch.multinomial(torch.tensor(policy_weights,device=device),1,generator=gen))
        scenario = policy['scenarios'][selected_scenario]
    channel_seed = None
    with torch.no_grad():
        if version2:
            channel_seed = int(torch.randint(0,2**31-1,(),device=device,generator=gen))
            selected = torch.rand(len(x),device=device,generator=gen) < probability
            if exposure_record is not None:
                if exposure_record.get('sample_ids') != list(meta['sample_id']) or exposure_record.get('epoch') != epoch:
                    raise ValueError('exposure replay sample stream/epoch mismatch')
                mask = exposure_record.get('selected_mask')
                if not isinstance(mask,list) or len(mask)!=len(x) or any(type(v) is not bool for v in mask):
                    raise ValueError('exposure replay selected mask mismatch')
                scenario, channel_seed = exposure_record.get('scenario'), exposure_record.get('channel_seed')
                if scenario not in satellite_policy(1.)['scenarios'] or type(channel_seed) is not int or channel_seed<0:
                    raise ValueError('exposure replay scenario/channel seed invalid')
                selected = torch.tensor(mask,device=device,dtype=torch.bool)
            channel_gen = torch.Generator(device=device).manual_seed(channel_seed)
            sat,_ = apply_sat_channel_for_scenario(x,scenario,args,gen=channel_gen,return_meta=False)
        else:
            if exposure_record is not None: raise ValueError('exposure replay requires V2')
            sat,_ = apply_sat_channel_for_scenario(x,scenario,args,gen=gen,return_meta=False)
            selected = torch.rand(len(x),device=device,generator=gen) < probability
        sat = torch.where(selected[:,None,None],sat,x)
    ctx = StepContext(x_main.detach(),sat.detach(),y,d,dict(weights),epoch,batch_index,list(meta['sample_id']),scenario,int(selected.sum()))
    ctx.satellite_selected_mask=selected.detach().clone()
    ctx.satellite_channel_seed=channel_seed;ctx.satellite_probability=probability
    ctx.satellite_policy_weights=policy_weights;ctx.capability_level=capability_level
    if unlabeled_batch is not None:
        ux,hidden,ud,umeta = unlabeled_batch
        if not bool((hidden == -1).all()) or any('tx' in k.lower() for k in umeta):
            raise ValueError('U_s hidden TX escaped the data boundary')
        ux,ud = ux.to(device),ud.to(device)
        teacher = ema if ema is not None else model
        was_training = teacher.training
        teacher.eval()
        with torch.no_grad():
            out = teacher(ux,return_aux=True)
            ctx.forward_calls += 1
            conf,pseudo = out['tx_logits'].softmax(1).max(1)
            mask = _threshold_mask(conf,ud,args)
            if args.pseudo_domain_gate: mask &= out['dom_logits'].argmax(1)==ud
            if args.pseudo_temporal_gate: mask &= temporal_mask(pseudo,conf,umeta,args)
        teacher.train(was_training)
        ctx.strong = (ux + torch.randn_like(ux)*args.strong_noise_std).detach()
        ctx.pseudo,ctx.base_mask = pseudo.detach(),mask.detach()
    return ctx

class Core90Objective:
    def __init__(self,model,args,proto_bank):
        self.model,self.args,self.proto = model,args,proto_bank
    def prepare_reusable_graph(self,ctx):
        from .head_lookahead import Core90ReusableGraph
        return Core90ReusableGraph(self,ctx)
    def __call__(self,ctx):
        n = len(ctx.y)
        separate = self.args.game_head_scale == 'separate_head_scale'
        grl = ctx.weights['adv'] if separate else 1.
        combined = self.model(torch.cat((ctx.x,ctx.satellite)),y_tx=torch.cat((ctx.y,ctx.y)),
                              grl_lambda=grl,return_aux=True,domain_labels=torch.cat((ctx.domain,ctx.domain)))
        ctx.forward_calls += 1
        out = slice_batch(combined,0,n,2*n)
        sat = slice_batch(combined,n,2*n,2*n)
        weights = dict(ctx.weights)
        if separate:
            weights['adv'] = 1. if ctx.weights['adv'] > 0 else 0.
        loss,terms = labeled_terms(out,ctx.y,ctx.domain,self.args,ctx.epoch,ctx.batch_index,weights,self.proto)
        sat_ce = F.cross_entropy(sat['tx_logits'],ctx.y) if ctx.epoch>=self.args.sat_cons_start_epoch else out['tx_logits'].sum()*0
        loss = loss + ctx.weights['sat_cls']*sat_ce
        terms['sat_cls'] = sat_ce
        if ctx.strong is not None:
            strong = self.model(ctx.strong,return_aux=True)
            ctx.forward_calls += 1
            if ctx.strong_mask is None:
                agreement = strong['tx_logits'].detach().argmax(1)==ctx.pseudo if self.args.pseudo_strong_agreement else torch.ones_like(ctx.base_mask)
                ctx.strong_mask = (ctx.base_mask & agreement).detach()
            selected = ctx.strong_mask
            ce = F.cross_entropy(strong['tx_logits'][selected],ctx.pseudo[selected]) if selected.any() else strong['tx_logits'].sum()*0
            prob = strong['tx_logits'].softmax(1)
            ent = -(prob*prob.clamp_min(1e-8).log()).sum(1).mean()
            loss = loss + self.args.lambda_u*ce + self.args.lambda_ent*ent
            terms.update(unlabeled_ce=ce,unlabeled_entropy=ent)
            ctx.pseudo_selected = int(selected.sum())
        if ctx.origin_features is None:
            ctx.origin_features = out['z_id'].detach().clone()
            ctx.origin_terms = {k:float(v.detach()) for k,v in terms.items()}
        return loss
