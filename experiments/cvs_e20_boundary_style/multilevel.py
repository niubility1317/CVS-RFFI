"""Head-free Fishr on normalized main-identity margin sensitivities.

The virtual per-packet affine coordinates are ephemeral autograd leaves. They
are never parameters, optimizer variables, stored states, or inference modules.
Both sensitivity graphs remain live through the cross-batch moment penalty.
"""
from contextlib import contextmanager
import json
import math
from pathlib import Path
import time
import weakref
import torch
from torch import nn
from torch.nn import functional as F
from experiments.cvs_e20_cross_style.multilevel import MultiLevel as ProbeMultiLevel, fixed_fishr
from experiments.cvs_e20_cross_style.style import _paths
from experiments.cvs_e20_budget_fishr.multilevel import LEVELS, _parameter_groups, _norm
from experiments.cvs_e20_moment_audit.fishr_stats import unbiased_variances, penalty


def normalize_scores(sensitivity, eps=1e-6):
    """Per-packet RMS normalization, finite with exact zero sensitivity."""
    if eps != 1e-6:
        raise ValueError('Registered sensitivity epsilon must equal 1e-6')
    rms = sensitivity.square().mean(1, keepdim=True).sqrt()
    score = sensitivity/(sensitivity.square().mean(1, keepdim=True)+eps**2).sqrt()
    return score, dict(raw_sensitivity_rms_mean=float(rms.detach().mean()),
        raw_sensitivity_rms_min=float(rms.detach().min()), raw_sensitivity_rms_max=float(rms.detach().max()),
        low_norm_rate=float((rms.detach() <= eps).float().mean()),
        normalized_score_rms_mean=float(score.detach().square().mean(1).sqrt().mean()),
        normalized_score_l2_mean=float(score.detach().norm(dim=1).mean()),
        sensitivity_eps=eps, normalization='per-packet RMS over concatenated a/b coordinates')


def identity_margin(logits, labels):
    if logits.ndim != 2 or logits.shape[1] != 6 or logits.shape[0] != len(labels) or not torch.isfinite(logits).all():
        raise ValueError('Finite six-class main identity logits required')
    other = logits.masked_fill(F.one_hot(labels, 6).bool(), -torch.inf)
    return logits.gather(1, labels[:, None]).squeeze(1)-torch.logsumexp(other, dim=1)


class BoundarySensitivity(nn.Module):
    """Zero-parameter virtual affine sensitivity sampler on native maps."""
    pair_indices = ProbeMultiLevel.pair_indices

    def __init__(self, base, seed=2026092701):
        super().__init__()
        self._base = weakref.ref(base)
        self.seed = int(seed)
        _, self.paths = _paths(base)
        self._active = False
        self._pair_source_cache = None

    @contextmanager
    def capture(self, perturb=None):
        """Identity hooks; perturb is exclusively for bounded derivative checks."""
        if self._active:
            raise RuntimeError('Nested boundary sensitivity capture prohibited')
        base = self._base()
        if base is None:
            raise RuntimeError('Identity base expired')
        modules = dict(base.named_modules()); handles=[]; probes={}; shapes={}
        self._active = True
        try:
            for level in LEVELS:
                def hook(module, inputs, h, level=level):
                    if level in probes:
                        raise RuntimeError('Repeated sensitivity layer in one forward')
                    if h.ndim != 3 or h.is_complex() or not h.is_floating_point():
                        raise ValueError('Expected real native [B,C,T] feature map')
                    a = torch.ones(h.shape[:2]+(1,), device=h.device, dtype=h.dtype, requires_grad=True)
                    b = torch.zeros_like(a, requires_grad=True)
                    std = h.detach().std(dim=-1, unbiased=False, keepdim=True)
                    probes[level] = (a, b); shapes[level] = list(h.shape)
                    da, db = (perturb or {}).get(level, (0., 0.))
                    return h*(a+da)+std*(b+db)
                handles.append(modules[self.paths[level]].register_forward_hook(hook))
            yield probes, shapes
            if tuple(probes) != LEVELS:
                raise RuntimeError('Native affine hook order differs')
        finally:
            for handle in handles:
                handle.remove()
            self._active = False

    @contextmanager
    def _evaluation(self, model):
        modes=[(m,m.training) for m in model.modules()]
        # Do not copy unchanged parameter tensors after building live graphs:
        # doing so would increment autograd versions before the optimizer backward.
        buffers={n:v.detach().clone() for n,v in model.named_buffers()}
        devices=list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
        try:
            model.eval()
            with torch.random.fork_rng(devices=devices):
                yield
            changed=[n for n,v in model.named_buffers() if not torch.equal(v,buffers[n])]
            if changed:
                with torch.no_grad():
                    for n,v in model.named_buffers():
                        if n in changed:v.copy_(buffers[n])
                raise AssertionError('Sensitivity readout changed model buffers: '+','.join(changed))
        finally:
            for module, mode in modes:
                module.training=mode

    def sensitivities(self, model, x, labels, levels=LEVELS, eps=1e-6):
        """Return live normalized margin derivatives for one eval forward."""
        with self.capture() as (probes, shapes):
            logits=model(x)
        margin=identity_margin(logits,labels)
        leaves=tuple(p for level in levels for p in probes[level])
        derivatives=torch.autograd.grad(margin.sum(),leaves,create_graph=True,retain_graph=True)
        scores={};stats={}
        for i,level in enumerate(levels):
            sensitivity=torch.cat((derivatives[2*i].squeeze(-1),derivatives[2*i+1].squeeze(-1)),1)
            score,info=normalize_scores(sensitivity,eps)
            if not torch.isfinite(score).all():raise FloatingPointError('Nonfinite boundary sensitivity')
            scores[level]=score
            stats[level]=dict(info,shape=shapes[level],coordinates=score.shape[1],
                score_requires_grad=score.requires_grad,main_margin_mean=float(margin.detach().mean()),
                main_margin_min=float(margin.detach().min()), main_margin_max=float(margin.detach().max()))
        return logits,scores,stats

    def _pair(self,model,source,step,device,config,calibration=False):
        levels=tuple(config['fishr_levels'])
        if not levels or levels != tuple(k for k in LEVELS if k in levels):
            raise ValueError('Unique registered active margin levels required')
        if config.get('statistic','margin') != 'margin' or config.get('sensitivity_eps',1e-6) != 1e-6:
            raise ValueError('Boundary sensitivity statistic differs')
        if config['batch_size']!=120 or config['pair_batch_size']!=60 or config['pair_quota']!=2 or config['fishr_fixed_denominator']!=2:
            raise ValueError('Boundary sensitivity pair budget differs')
        if source.get('role','L_s') not in ('L_s','L'):
            raise ValueError('Only visible source L allowed')
        ia,ib,day,sizes=self.pair_indices(source,step,calibration)
        packs=[]
        with self._evaluation(model):
            for index in (ia,ib):
                y=source['y'][index].to(device);rx=source['rx'][index].to(device)
                logits,scores,stats=self.sensitivities(model,source['x'][index].to(device),y,levels)
                packs.append(dict(y=y,rx=rx,logits=logits,scores=scores,stats=stats))
            terms={};layer_detail={}
            for level in LEVELS:
                if level not in levels:
                    layer_detail[level]=dict(fishr_computed=False,reason='inactive_by_design')
                    continue
                variances=[unbiased_variances(pack['scores'][level],pack['y'],pack['rx'],sizes) for pack in packs]
                term=penalty(variances[0],variances[1]);terms[level]=term
                if not torch.isfinite(term):raise FloatingPointError('Nonfinite margin Fishr moment')
                layer_detail[level]=dict(shape=packs[0]['stats'][level]['shape'],fishr_computed=True,
                    raw_fishr=float(term.detach()),raw_loss_negative=bool(term.detach()<0),
                    variance_mean_A=float(variances[0].detach().mean()),variance_mean_B=float(variances[1].detach().mean()),
                    sensitivity_A=packs[0]['stats'][level],sensitivity_B=packs[1]['stats'][level],
                    estimator='finite_L_corrected_cross_AB_margin_sensitivity',
                    gradient_path='live main-identity mixed second derivatives; all upstream/downstream identity dependencies retained')
            logits=torch.cat([p['logits'] for p in packs]);labels=torch.cat([p['y'] for p in packs])
            main_ce=F.cross_entropy(logits,labels) if calibration else None
            zero=logits.sum()*0
        detail=dict(aux_forwards=2,auxiliary_packets=120,aux_ce=0.,fishr_steps=1,levels=layer_detail,
            day=day,source_RX=[1,3,4,6,8],distinct_PID=len(set(ia.tolist())|set(ib.tolist())),
            pair_A_distinct_PID=60,pair_B_distinct_PID=60,cross_batch_PID_overlap=len(set(ia.tolist())&set(ib.tolist())),
            aux_ce_to_backbone=False,fishr_levels=list(levels),fishr_active_levels=list(levels),
            fishr_fixed_denominator=2,estimator='finite_L_corrected_cross_AB_margin_sensitivity',
            statistic='margin',auxiliary_heads=0,head_Fishr_gradient=False,EMA=False,
            native_CPU_CUDA_RNG_preserved=True,independent_AB=True,calibration_sampling=calibration,
            sensitivity_eps=1e-6,virtual_affine='a=1,b=0 per packet/channel; std(H) detached; no registered parameters',
            main_margin_mean=float(identity_margin(logits.detach(),labels).mean()),
            fishr=float((sum(terms.values())/2).detach()))
        return zero,terms,detail,main_ce

    def loss(self,model,source,step,device,config):
        if not config['enabled'] or not model.training or step%config['interval']:
            zero=next(model.base.parameters()).sum()*0
            return zero,{},dict(aux_forwards=0,auxiliary_packets=0,aux_ce=0.,fishr_steps=0,levels={},auxiliary_heads=0)
        ce,terms,detail,_=self._pair(model,source,step,device,config)
        return ce,terms,detail

    _loss = loss


def calibrate(c,model,L,device,out):
    """Fixed initial own-margin-term norm calibration, not old probe calibration."""
    path=Path(out)/'calibration.json'
    if path.exists():raise FileExistsError('Fixed calibration already exists')
    mc=c['multilevel'];started=time.perf_counter()
    if not mc['enabled']:
        result=dict(status='NOT_APPLICABLE',fixed_coefficients={},pairs=0,extra_source_readouts=0,model_forwards=0,
            seconds=0.,state_unchanged=True,calibration_reference='own_margin_term')
    else:
        if not isinstance(model.multilevel,BoundarySensitivity):raise ValueError('Own margin calibration needs zero-head margin module')
        if mc['fishr_calibration_pairs']!=12 or mc['fishr_ratio']!=.2 or mc['fishr_cap']!=10000.:
            raise ValueError('Fixed margin calibration protocol differs')
        before={n:v.detach().clone() for n,v in model.state_dict().items()}
        modes=[(m,m.training) for m in model.modules()];groups=_parameter_groups(model)
        selected=dict(mc,fishr_levels=list(LEVELS));rows=[]
        devices=list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
        try:
            model.eval()
            with torch.random.fork_rng(devices=devices):
                for i in range(12):
                    tick=time.perf_counter()
                    _,terms,detail,ce=model.multilevel._pair(model,L,4*(i+1),device,selected,calibration=True)
                    values={level:dict(main_ce_gradient_norm=_norm(ce,groups[level]),
                        term_gradient_norm=_norm(terms[level],groups[level]),raw_term=float(terms[level].detach())) for level in LEVELS}
                    if str(device).startswith('cuda'):torch.cuda.synchronize(device)
                    rows.append(dict(pair=i,day=detail['day'],cross_batch_PID_overlap=detail['cross_batch_PID_overlap'],
                        levels=values,main_ce=float(ce.detach()),seconds=time.perf_counter()-tick))
            if any(not torch.equal(v,before[n]) for n,v in model.state_dict().items()):raise AssertionError('Margin calibration changed model state')
        finally:
            for module,mode in modes:module.training=mode
        coefficients={};summary={}
        for level in LEVELS:
            ce=sum(r['levels'][level]['main_ce_gradient_norm'] for r in rows)/12
            norm=sum(r['levels'][level]['term_gradient_norm'] for r in rows)/12
            if not math.isfinite(ce+norm) or min(ce,norm)<=mc['fishr_eps']:raise ValueError('Invalid initial margin calibration gradient norm')
            uncapped=.2*ce/(norm+mc['fishr_eps']);value=min(mc['fishr_cap'],uncapped)
            coefficients[level]=value
            summary[level]=dict(mean_CE_gradient_norm=ce,mean_term_gradient_norm=norm,
                uncapped_coefficient=uncapped,coefficient=value,cap_active=uncapped>mc['fishr_cap'])
        model.multilevel.fixed_coefficients=coefficients
        result=dict(status='FIXED_CALIBRATION_COMPLETE',calibration_reference='own_margin_term',
            fixed_coefficients=coefficients,levels=summary,pairs=12,records=rows,extra_source_readouts=1440,
            model_forwards=24,seconds=time.perf_counter()-started,state_unchanged=True,RNG_preserved=True,
            source_role='L_s only',target_access=False,identity_updates=0,head_updates=0,auxiliary_heads=0,
            all_layers_calibrated=True,coefficient_formula='min(cap,0.2*mean(pair_main_CE_gradient_norm)/(mean(own_margin_term_gradient_norm)+eps))',
            sampling_namespace='same source-private calibration pairs as probe; own-margin statistic, not old-pair probe term')
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding='utf-8') as stream:
        json.dump(result,stream,indent=2,allow_nan=False);stream.write('\n')
    return result
