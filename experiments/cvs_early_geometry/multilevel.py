"""Training-only shallow/middle linear probes, real parameter-gradient Fishr.

The original identity inference path is untouched outside call-local contexts.
Both selected maps are real outputs after learned convolution/ReLU, not raw IQ.
"""
from contextlib import contextmanager
import weakref

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

LEVELS = ('time_down', 't2.pool')
SOURCE_RX = (1, 3, 4, 6, 8)


def linear_sample_gradients(features, head, labels):
    """Actual individual CE gradients wrt Linear.weight AND Linear.bias.

    [B,6,C+1], with feature/probability graphs retained. This is neither logit
    gradients nor activation-gradient alignment. No EMA or gradient unit norm.
    """
    if head.bias is None or head.out_features != 6:
        raise ValueError('Fishr requires the registered biased six-class linear head')
    delta = head(features).softmax(1) - F.one_hot(labels, 6).to(features)
    augmented = torch.cat((features, torch.ones_like(features[:, :1])), 1)
    return delta[:, :, None] * augmented[:, None, :]


def fishr_variance(features, head, labels, receivers):
    gradients = linear_sample_gradients(features, head, labels)
    domains = receivers.unique()
    if len(domains) < 2:
        raise ValueError('At least two actual source RX required; absent domains cannot be zero-filled')
    variance = []
    for rx in domains:
        member = receivers == rx
        if int(member.sum()) < 2:
            raise ValueError('Each actual RX must have at least two samples')
        g = gradients[member]
        variance.append((g-g.mean(0, keepdim=True)).square().mean(0))
    stack = torch.stack(variance)
    loss = (stack-stack.mean(0, keepdim=True)).square().mean()
    return loss, dict(domains=domains.detach().cpu().tolist(), coordinates=6*(features.shape[1]+1),
                     variance_mean=float(stack.detach().mean()), variance_max=float(stack.detach().max()),
                     variance_gap_mse=float(loss.detach()), variance_denominator='n',
                     parameter_coordinates='linear weight and bias', variance_normalized=False,
                     per_sample_gradient_normalized=False)


def pooled_real_map(value):
    if value.ndim != 3 or value.is_complex():
        raise ValueError('Registered time_down/t2.pool are real [B,C,T], not IQ or complex maps')
    return value.mean(-1)


def weak_mixstyle(value, labels, receivers, days, ids, seed, p=.5, alpha=.1, eta=.2):
    """Detached channel moments; legal within-TX, cross-RX, same-day donors."""
    if not 0 <= p <= 1 or not 0 <= eta <= 1 or alpha <= 0:
        raise ValueError('Invalid registered MixStyle settings')
    pooled_real_map(value)
    y, rx, day = [v.detach().cpu().tolist() for v in (labels, receivers, days)]
    if len(ids) != len(value) or len(set(ids)) != len(ids):
        raise ValueError('MixStyle requires distinct visible source-L physical packets')
    rng = np.random.default_rng(int(seed))
    eligible = [[j for j in range(len(ids)) if y[j] == y[i] and rx[j] != rx[i]
                 and day[j] == day[i] and ids[j] != ids[i]] for i in range(len(ids))]
    detail = dict(enabled=False, eligible_packets=sum(bool(v) for v in eligible),
                  changed_packets=0, changed_ratio=0., movement_mean=0., delta_ratio=0., donors=[])
    if rng.random() >= p:
        return value, detail
    donor = [int(rng.choice(v)) if v else i for i, v in enumerate(eligible)]
    movement = eta * (1-rng.beta(alpha, alpha, len(ids)))
    movement = np.where(np.asarray([bool(v) for v in eligible]), movement, 0.)
    # Keep the remote NumPy 2 / PyTorch 2.1 boundary independent of ndarray ABI.
    a = torch.tensor(movement.tolist(), dtype=value.dtype, device=value.device)[:, None, None]
    mu = value.mean(-1, keepdim=True).detach()
    sigma = (value.var(-1, unbiased=False, keepdim=True)+1e-6).sqrt().detach()
    output = (value-mu)/sigma * ((1-a)*sigma+a*sigma[donor]) + (1-a)*mu+a*mu[donor]
    active = a != 0
    output = torch.where(active, output, value)
    ratio = (output.detach()-value.detach()).flatten(1).norm(dim=1)/value.detach().flatten(1).norm(dim=1).clamp_min(1e-6)
    detail.update(enabled=True, changed_packets=int((ratio>0).sum()), changed_ratio=float((ratio>0).float().mean()),
                  movement_mean=float(a.mean()), movement_max=float(a.max()), delta_ratio=float(ratio.mean()), donors=donor)
    return output, detail


class MultiLevel(nn.Module):
    """Only auxiliary readout parameters are registered; base is a weak reference."""
    def __init__(self, base, seed=2026092701):
        super().__init__()
        self._base = weakref.ref(base)
        self.seed = int(seed)
        candidates = [(n,m) for n,m in base.named_modules()
                      if n.startswith('id_backbone.') and n.endswith('time_down')]
        if len(candidates) != 1:
            raise ValueError('Exactly one native time_down required')
        prefix = candidates[0][0].rsplit('.',1)[0]
        modules = dict(base.named_modules())
        self.paths = {level: prefix+'.'+level for level in LEVELS}
        if any(p not in modules for p in self.paths.values()):
            raise ValueError('Missing native shallow/middle maps')
        channels = {'time_down': modules[prefix+'.t1'].dw.in_channels,
                    't2.pool': modules[prefix+'.t2'].pw.out_channels}
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(self.seed+88561)
            self.heads = nn.ModuleDict({self.key(level): nn.Linear(channels[level],6) for level in LEVELS})
        self.to(next(base.parameters()).device)
        self._active = False
        self._source_cache = None

    @staticmethod
    def key(level):
        return level.replace('.', '_')

    @contextmanager
    def capture(self, *, mix=False, labels=None, receivers=None, days=None, ids=None,
                step=0, config=None):
        if self._active:
            raise RuntimeError('Nested auxiliary feature capture prohibited')
        base = self._base()
        if base is None:
            raise RuntimeError('Identity base expired')
        self._active = True
        handles, maps, styles = [], {}, {}
        modules = dict(base.named_modules())
        # Dedicated statistics use deterministic real maps; no dropout noise.
        drops = [(m,m.training) for m in base.modules() if isinstance(m,nn.modules.dropout._DropoutNd)]
        for module,_ in drops:
            module.training = False
        try:
            for index,level in enumerate(LEVELS):
                def hook(module, inputs, output, level=level, index=index):
                    if level in maps:
                        raise RuntimeError('Multiple identity forwards in one capture context')
                    pooled_real_map(output)
                    if mix and base.training and self.training:
                        output, detail = weak_mixstyle(output,labels,receivers,days,ids,
                            self.seed+int(step)*701+index*1709,
                            p=(config or {}).get('mixstyle_p',.5), alpha=(config or {}).get('mixstyle_alpha',.1),
                            eta=(config or {}).get('mixstyle_strength',.3))
                        styles[level] = detail
                    else:
                        styles[level] = dict(enabled=False, changed_packets=0, changed_ratio=0.)
                    maps[level] = output
                    return output
                handles.append(modules[self.paths[level]].register_forward_hook(hook))
            yield maps, styles
        finally:
            for handle in handles:
                handle.remove()
            for module,mode in drops:
                module.training = mode
            self._active = False

    def balanced_indices(self, source, step, interval=4):
        """Independent step-private L sampling, same-day 6 TX x 5 RX x 2 PID."""
        if self._source_cache is None or self._source_cache[0] is not source:
            y,rx,days = (source[k].cpu() for k in ('y','rx','day'))
            if set(y.tolist()) != set(range(6)) or set(rx.tolist()) != set(SOURCE_RX):
                raise ValueError('Expected visible source L labels and five registered receivers')
            if len(set(source['ids'])) != len(y):
                raise ValueError('Source L contains repeated physical packets')
            members = {}
            for day in sorted(days.unique().tolist()):
                cells = [torch.where((y==tx)&(rx==receiver)&(days==day))[0]
                         for tx in range(6) for receiver in SOURCE_RX]
                if all(len(cell)>=2 for cell in cells):
                    members[int(day)] = cells
            if not members:
                raise ValueError('No source day supports 6 TX x 5 RX x 2 distinct PID')
            self._source_cache = (source,members)
        members = self._source_cache[1]
        days = sorted(members)
        day = days[(int(step)//interval-1)%len(days)]
        generator = torch.Generator().manual_seed(self.seed+int(step)*99991)
        indices = torch.cat([cell[torch.randperm(len(cell),generator=generator)[:2]] for cell in members[day]])
        if len(indices) != 60 or len(set(indices.tolist())) != 60:
            raise ValueError('Invalid auxiliary balanced subset')
        return indices

    def loss(self, model, source, step, device, config):
        # Auxiliary work must not advance native dropout or any native CUDA RNG.
        devices = sorted({p.device.index for p in model.parameters() if p.device.type=='cuda'})
        with torch.random.fork_rng(devices=devices):
            return self._loss(model, source, step, device, config)

    def _loss(self, model, source, step, device, config):
        zero = next(self.heads.parameters()).sum()*0
        empty = dict(aux_forwards=0, auxiliary_packets=0, fishr_steps=0, levels={})
        if not config.get('enabled',False) or not model.training or step%int(config.get('interval',4)):
            return zero, zero, empty
        if int(config.get('batch_size',60)) != 60 or tuple(config.get('levels',LEVELS)) != LEVELS:
            raise ValueError('Auxiliary batch/layers differ from fixed design')
        index = self.balanced_indices(source,step,int(config.get('interval',4)))
        x=source['x'][index].to(device); y=source['y'][index].to(device)
        rx=source['rx'][index].to(device); day=source['day'][index].to(device)
        ids=[source['ids'][i] for i in index.tolist()]
        with self.capture(step=step) as (maps,_):
            model(x)
        clean = {level:pooled_real_map(value) for level,value in maps.items()}
        if set(clean) != set(LEVELS):
            raise RuntimeError('Identity bypassed registered auxiliary maps')
        ces = {level:F.cross_entropy(self.heads[self.key(level)](f),y) for level,f in clean.items()}
        detail = dict(aux_forwards=1, auxiliary_packets=60, distinct_PID=60, day=int(day[0]),
                      source_RX=rx.unique().cpu().tolist(), fishr_steps=int(config.get('fishr',False)),
                      style_role='additional labeled auxiliary forward only',
                      pooling='real channel temporal mean; neither abs(IQ) nor final embedding',
                      dropout_disabled_for_auxiliary_statistics=True,
                      native_CPU_CUDA_RNG_preserved=True, levels={})
        fishrs=[]
        for level,feature in clean.items():
            record = dict(clean_ce=float(ces[level].detach()),shape=list(maps[level].shape))
            if config.get('fishr',False):
                loss,fishr_record=fishr_variance(feature,self.heads[self.key(level)],y,rx)
                fishrs.append(loss); record.update(fishr_record)
            detail['levels'][level]=record
        if config.get('mixstyle',False):
            with self.capture(mix=True,labels=y,receivers=rx,days=day,ids=ids,step=step,config=config) as (mixed,styles):
                model(x)
            detail['aux_forwards']+=1
            for level,value in mixed.items():
                mixed_ce=F.cross_entropy(self.heads[self.key(level)](pooled_real_map(value)),y)
                ces[level]=.5*(ces[level]+mixed_ce)
                detail['levels'][level].update(mixed_ce=float(mixed_ce.detach()),style=styles[level])
        auxiliary_ce = torch.stack(list(ces.values())).mean()
        fishr = torch.stack(fishrs).mean() if fishrs else zero
        detail.update(aux_ce=float(auxiliary_ce.detach()),fishr=float(fishr.detach()),
                      averaging='fixed mean of two levels; five RX equally weighted; coordinate mean',
                      EMA=False,identity_classifier_used_for_fishr=False)
        return auxiliary_ce, fishr, detail
