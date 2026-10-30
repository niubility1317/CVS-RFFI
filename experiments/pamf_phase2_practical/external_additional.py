"""Native additional external-method bridges on the fixed Phase2 source contract.

This module never accepts a query loader, target pool, truth path or scorer.
Source fitting uses L_s and read-only V; target fitting uses only tensors passed
as currently opened support. Runtime source roots are explicit, never PAMF.
The independent ports retain method objectives and full registered budgets.
"""
from __future__ import annotations

import copy
import json
import math
import random
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from .vendor_extra.metric_models import ISSLResNet18, LoRaExtractor
from .vendor_extra.hu.model import FeatureSeparationNet
from .vendor_extra.hu.preprocess import preprocess_iq
from .vendor_extra.hu.augmentation import augment_iq
from .vendor_extra.hu.losses import feature_separation_loss
from .vendor_extra.tweak.model import TweakEncoder
from .vendor_extra.tweak.triplet import batch_hard_triplet_loss


_SPECS = {
    'issl_fixed_v1': dict(source_epochs=300, source_batch=128, source_optimizer='Adam', source_lr=.03,
        B_epochs=10, B_lr=.0001, B_batch=32, C_SSL_epochs=2000, C_transfer_epochs=300,
        C_SSL_lr=.00001, C_transfer_lr=.03, native_C=True, source_U=False,
        implementation='explicit_repaired_v1_independent_topology_port',
        claim='repaired ISSL extension; not author-behavior parity'),
    'issl_author_behavior': dict(source_epochs=300, source_batch=128, source_optimizer='Adam', source_lr=.03,
        B_epochs=10, B_lr=.0001, B_batch=32, C_SSL_epochs=2000, C_transfer_epochs=300,
        C_SSL_lr=.00001, C_transfer_lr=.03, native_C=True, source_U=False,
        implementation='explicit_author_semantics_independent_topology_port',
        claim='detached KD, accumulated SSL gradients, reshape queue and parameter-only momentum retained; B10 is project extension'),
    'lora_triplet_knn': dict(source_epochs=1000, source_batch=32, source_optimizer='RMSprop', source_lr=.001,
        source_early_patience=20, source_lr_patience=10, source_lr_factor=.2,
        B_epochs=0, native_C=True, source_U=False, knn_k=15,
        implementation='independent_Torch_LoRa_256IQ_STFT64_overlap32',
        claim='Torch port with short-IQ STFT extension; TF2.1 RMSprop numerical parity unverified'),
    'poster_ft': dict(source_epochs=200, source_batch=128, source_optimizer='Adam', source_lr=.001,
        B_epochs=10, B_lr=.001, B_batch=256, native_C=False, source_U=False,
        implementation='existing_repo_POSTER_Homegrown_port', claim='author dense-layer-reset FT'),
    'radionet_df_ft': dict(source_epochs=200, source_batch=128, source_optimizer='Adam', source_lr=.001,
        B_epochs=30, B_lr=.001, B_batch=128, native_C=False, source_U=False,
        implementation='existing_repo_RadioNet_DF_port', claim='author final-layer-reset FT'),
    'radionet_triplet': dict(source_epochs=30, source_batch=128, source_optimizer='Adam', source_lr=.001,
        B_epochs=0, native_C=True, source_U=False,
        implementation='DF64_cosine_triplet_semihard_Torch_port',
        claim='all positive pairs, epochwise source-only semihard mining, cosine distance-weighted kNN; N-MEV support mean'),
    'feature_separation_hu2024': dict(source_epochs=200, source_batch=256, source_optimizer='Adam', source_lr=.005,
        source_early_patience=20, source_lr_patience=10, source_lr_factor=.1,
        B_epochs=100, B_lr=.005, B_batch=256, native_C=False, source_U=False,
        implementation='existing_repo_Hu2024_independent_reproduction',
        claim='shared attention encoder; TX/RX CE, Frobenius separation, both entropy terms; target RX frozen'),
    'ra_collab_single_ft': dict(source_epochs=100, source_batch=64, source_optimizer='SGD', source_lr=.001,
        source_early_patience=20, source_lr_patience=10, source_lr_factor=.2,
        B_epochs=20, B_lr=.00001, B_batch=32, native_C=False, source_U=False,
        implementation='existing_repo_RA_Collab_GRL_single_node', claim='single-node FT extension; no cross-query fusion'),
    'tweak_m1': dict(source_epochs=100, source_batch=64, source_optimizer='SGD',
        source_lr_grid=(.01, .001, .0001, .00001, .000001), B_epochs=0, native_C=True, source_U=False,
        implementation='existing_repo_Tweak_12D_batch_hard',
        claim='first 128 samples of each 256-IQ received window, M=1 explicit extension; centroid and mean L2 radius'),
    'tweak_m10': dict(source_alias='tweak_m1', native_C=False, blocked=True,
        reason='Native M=10 needs ten legally linked same-transmitter observations per query track. The current capsule has one fixed 256-IQ window per opaque physical ID and no predeclared same-TX track ID. Query truth, reshape and random grouping cannot supply those tracks.'),
}


def profiles():
    return copy.deepcopy(_SPECS)


def source_training_spec(profile):
    if profile not in _SPECS:
        raise ValueError('Unknown additional profile: ' + str(profile))
    return copy.deepcopy(_SPECS[profile])


def _log(logger, **row):
    if logger is None:
        return
    if hasattr(logger, 'log'):
        logger.log(**row)
    else:
        logger(row)


def _seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def _write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def _iq(x):
    if not torch.is_tensor(x) or x.ndim != 3 or tuple(x.shape[1:]) != (2, 256) or not torch.isfinite(x).all():
        raise ValueError('Expected finite support/source IQ [N,2,256]')
    return x.float()


def _labels(y, n, classes=None):
    if not torch.is_tensor(y) or y.ndim != 1 or y.dtype != torch.long or len(y) != n:
        raise ValueError('Expected aligned LongTensor support labels')
    actual = tuple(sorted(set(y.detach().cpu().tolist())))
    if classes is not None and actual != tuple(classes):
        raise ValueError('Support must cover exactly the currently opened classes')
    return actual


def _issl_input(x, crop=None):
    x = _iq(x)
    if crop is not None:
        starts = torch.randint(0, x.shape[-1] - crop + 1, (len(x),), device=x.device)
        x = x.gather(2, (starts[:, None] + torch.arange(crop, device=x.device))[..., None].transpose(1, 2).expand(-1, 2, -1))
    return (x - x.mean((1, 2), keepdim=True)) / x.std((1, 2), correction=0, keepdim=True).clamp_min(1e-12)


def channel_spectrogram(x):
    """Exact boxcar/no-boundary adjacent STFT ratio, adapted 64/32 for 256 IQ."""
    x = _iq(x)
    z = torch.complex(x[:, 0], x[:, 1])
    rms = z.abs().square().mean(1, keepdim=True).sqrt()
    if bool((rms <= 0).any()):
        raise ValueError('Zero IQ cannot form a channel-independent spectrogram')
    z = z / rms
    s = torch.stft(z, n_fft=64, hop_length=32, win_length=64,
        window=torch.ones(64, device=x.device, dtype=x.dtype), center=False,
        normalized=False, onesided=False, return_complex=True)
    s = torch.fft.fftshift(s, dim=1)
    v = (s[:, :, 1:] / s[:, :, :-1]).abs().square().log10()[:, round(64*.3):round(64*.7)]
    if not torch.isfinite(v).all():
        raise ValueError('Nonfinite LoRa adjacent-frame ratio; no silent clipping')
    return v.unsqueeze(1)


class NativeModel(nn.Module):
    def __init__(self, profile, classes=6, receivers=5):
        super().__init__()
        self.profile = profile
        if profile.startswith('issl_'):
            self.network = ISSLResNet18(classes)
        elif profile == 'lora_triplet_knn':
            self.network = LoRaExtractor()
        elif profile in ('poster_ft', 'radionet_df_ft', 'radionet_triplet'):
            from .vendor_extra.li_backbones import PosterHomegrown, RadioNetDF
            self.network = PosterHomegrown(classes) if profile == 'poster_ft' else RadioNetDF(64 if profile == 'radionet_triplet' else classes)
        elif profile == 'feature_separation_hu2024':
            self.network = FeatureSeparationNet(num_tx=classes, num_rx=receivers)
        elif profile == 'ra_collab_single_ft':
            from .vendor_extra.ra_model import RACollabRFFI
            self.network = RACollabRFFI(classes, receivers)
        elif profile == 'tweak_m1':
            self.network = TweakEncoder()
        else:
            raise ValueError('No executable model for ' + str(profile))

    def raw(self, x, crop=None):
        p = self.profile
        if p.startswith('issl_'):
            return self.network(_issl_input(x, crop))
        if p == 'lora_triplet_knn':
            return self.network(channel_spectrogram(x))
        if p == 'feature_separation_hu2024':
            return self.network(preprocess_iq(_iq(x)))
        if p == 'ra_collab_single_ft':
            from .vendor_extra.spectrogram import iq_to_log_spectrogram
            return self.network(iq_to_log_spectrogram(_iq(x),64,32,64,normalize='zscore'))
        if p == 'tweak_m1':
            # A deterministic per-record crop; no second physical observation.
            return self.network(_iq(x)[..., :128])
        return self.network(_iq(x))

    def forward(self, x):
        return native_logits(self, x)


def native_features(model, x):
    p = model.profile
    if p.startswith('issl_'):
        return model.raw(x)[1]
    if p in ('poster_ft', 'radionet_df_ft'):
        return model.network.embedding(_iq(x))
    out = model.raw(x)
    if p == 'feature_separation_hu2024':
        return out['tx_features']
    if p == 'ra_collab_single_ft':
        return out['feature']
    return out


def native_logits(model, x):
    out = model.raw(x)
    if model.profile.startswith('issl_'):
        return out[0]
    if isinstance(out, dict):
        return out['tx_logits']
    return out


def _device(model):
    return next(model.parameters()).device


@torch.no_grad()
def _extract(model, x, batch=256):
    model.eval()
    return torch.cat([native_features(model, part.to(_device(model))).detach().cpu() for part in x.split(batch)])


def _source_data(rt, role):
    from experiments.cric_dura_comparison.f_data import SourceDataset
    contract = _read(rt['source_contract'])
    if contract.get('counts') != {'L_s': 6300, 'U_s': 56700, 'V': 27000}:
        raise ValueError('CHECKPOINT_DATA_CONTRACT_MISMATCH: source counts')
    if len(contract['source_rxs']) != 5 or len(contract['classes']) != 6:
        raise ValueError('Expected original six-class, five-receiver source contract')
    if role not in ('L_s', 'V'):
        raise ValueError('These profiles never open U_s TX labels')
    data = SourceDataset(rt['source_cache'], role, contract)
    x = data.x.float()
    y = torch.tensor(data.metadata['y'].tolist(), dtype=torch.long)
    rx = torch.tensor(data.metadata['rx_i'].tolist(), dtype=torch.long)
    mapping = {int(v): i for i, v in enumerate(contract['source_rxs'])}
    if set(rx.tolist()) - set(mapping):
        raise ValueError('Source receiver outside matching contract')
    rx = torch.tensor([mapping[v] for v in rx.tolist()], dtype=torch.long)
    return x, y, rx


def _indices(n, batch, generator=None, shuffle=True, drop_last=False):
    indices = torch.randperm(n, generator=generator) if shuffle else torch.arange(n)
    for ix in indices.split(batch):
        if not drop_last or len(ix) == batch:
            yield ix


def _checked_step(loss, optimizer):
    if not torch.isfinite(loss):
        raise FloatingPointError('Nonfinite native training loss')
    loss.backward()
    if any(p.grad is not None and not torch.isfinite(p.grad).all() for g in optimizer.param_groups for p in g['params']):
        raise FloatingPointError('Nonfinite native gradient')
    optimizer.step()


def _train_ce(model, x, y, epochs, batch, lr, seed, logger=None, optimizer_kind='Adam', crop=None, stage='B', parameters=None):
    parameters = list(model.parameters()) if parameters is None else list(parameters)
    optimizer = (torch.optim.SGD(parameters, lr=lr, momentum=.9) if optimizer_kind == 'SGD'
                 else torch.optim.Adam(parameters, lr=lr, eps=1e-7 if model.profile in ('poster_ft','radionet_df_ft') else 1e-8))
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 100, .5) if model.profile.startswith('issl_') else None
    gen = torch.Generator().manual_seed(seed)
    dev = _device(model)
    for epoch in range(epochs):
        values = []
        model.train()
        if model.profile == 'feature_separation_hu2024':
            model.network.rx_branch.eval(); model.network.rx_classifier.eval()
        for ix in _indices(len(y), batch, gen):
            xx, yy = x[ix].to(dev), y[ix].to(dev)
            optimizer.zero_grad(set_to_none=True)
            scores = model.raw(xx, crop)[0] if model.profile.startswith('issl_') else native_logits(model, xx)
            loss = F.cross_entropy(scores, yy)
            _checked_step(loss, optimizer)
            values.append(float(loss.detach()))
        if scheduler is not None:
            scheduler.step()
        _log(logger, stage=stage, epoch=epoch+1, loss=sum(values)/len(values), lr=optimizer.param_groups[0]['lr'], query_fit=False)
    model.eval()


def _triplets(y, batch, generator):
    classes = torch.unique(y).tolist()
    groups = {c: torch.where(y == c)[0] for c in classes}
    a, p, n = [], [], []
    for _ in range(batch):
        c = classes[int(torch.randint(len(classes), (), generator=generator))]
        other = [v for v in classes if v != c]
        d = other[int(torch.randint(len(other), (), generator=generator))]
        picks = torch.randint(len(groups[c]), (2,), generator=generator)
        a.append(int(groups[c][picks[0]])); p.append(int(groups[c][picks[1]]))
        n.append(int(groups[d][torch.randint(len(groups[d]), (), generator=generator)]))
    return [torch.tensor(v, dtype=torch.long) for v in (a, p, n)]


def _awgn(x, generator):
    # Author LoRa one-time source augmentation: SNR uniform [20,79].
    power = x.square().sum(1).mean(1, keepdim=True).unsqueeze(1)
    snr = 20 + 59 * torch.rand((len(x),1,1), generator=generator)
    scale = (power / (10 ** (snr / 10)) / 2).sqrt()
    return x + scale * torch.randn(x.shape, generator=generator)


@torch.no_grad()
def _validation(model, x, y, batch=256):
    model.eval(); correct=0; loss=0.
    for ix in _indices(len(y), batch, shuffle=False):
        yy = y[ix].to(_device(model)); scores = native_logits(model, x[ix].to(_device(model)))
        correct += int((scores.argmax(1) == yy).sum())
        loss += float(F.cross_entropy(scores, yy, reduction='sum'))
    return correct/len(y), loss/len(y)


def _source_supervised(model, data, validation, spec, seed, logger):
    x,y,rx=data; vx,vy,_=validation; dev=_device(model)
    p=model.profile; gen=torch.Generator().manual_seed(seed)
    parameters=list(model.parameters())
    optimizer=(torch.optim.SGD(parameters, lr=spec['source_lr'], momentum=.9) if spec['source_optimizer']=='SGD'
        else torch.optim.Adam(parameters, lr=spec['source_lr'], eps=1e-7 if p in ('poster_ft','radionet_df_ft') else 1e-8))
    mode='max' if p=='feature_separation_hu2024' else 'min'
    scheduler=(torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode=mode,
        factor=spec['source_lr_factor'], patience=spec['source_lr_patience']) if 'source_lr_patience' in spec
        else torch.optim.lr_scheduler.StepLR(optimizer, 100, .5) if p.startswith('issl_') else None)
    best=-math.inf if mode=='max' else math.inf; best_state=None; stale=0; selected=0
    ra_aug=None
    if p=='ra_collab_single_ft':
        from .vendor_extra.rf_augmentation import OnlineRFChannelAugment
        ra_aug=OnlineRFChannelAugment(sample_rate=1_000_000., max_taps=4, p=1.)
    for epoch in range(spec['source_epochs']):
        model.train(); losses=[]
        for ix in _indices(len(y), spec['source_batch'], gen, drop_last=p.startswith('issl_')):
            xx,yy,rr=x[ix].to(dev),y[ix].to(dev),rx[ix].to(dev)
            optimizer.zero_grad(set_to_none=True)
            if p=='feature_separation_hu2024':
                # Existing independent reproduction preprocesses before augmentation;
                # representation then consumes those values without a second norm.
                outputs=model.network(augment_iq(preprocess_iq(xx)))
                loss,_=feature_separation_loss(outputs,yy,rr)
            elif p=='ra_collab_single_ft':
                out=model.raw(ra_aug(xx)); loss=F.cross_entropy(out['tx_logits'],yy)+F.cross_entropy(out['rx_logits'],rr)
            else:
                loss=F.cross_entropy(native_logits(model,xx),yy)
            _checked_step(loss,optimizer);losses.append(float(loss.detach()))
        acc,val_loss=_validation(model,vx,vy)
        metric=acc if mode=='max' else val_loss
        improved=metric>best if mode=='max' else metric<best
        if improved:
            best=metric;best_state=copy.deepcopy(model.state_dict());stale=0;selected=epoch+1
        else:stale+=1
        if isinstance(scheduler,torch.optim.lr_scheduler.ReduceLROnPlateau):scheduler.step(metric)
        elif scheduler is not None:scheduler.step()
        _log(logger,stage='source',epoch=epoch+1,loss=sum(losses)/len(losses),source_V_accuracy=acc,
            source_V_loss=val_loss,lr=optimizer.param_groups[0]['lr'],target_contact=False)
        if spec.get('source_early_patience') and stale>=spec['source_early_patience']:break
    # Fixed-budget POSTER/DF/ISSL retain final epoch, rather than source-V cherry selection.
    if 'source_early_patience' in spec:model.load_state_dict(best_state)
    else:selected=epoch+1
    return dict(executed_epochs=epoch+1,selected_epoch=selected,selection='source_V' if 'source_early_patience' in spec else 'fixed_final_epoch')


def _source_lora(model,data,validation,spec,seed,logger):
    x,y,_=data;vx,vy,_=validation;dev=_device(model);gen=torch.Generator().manual_seed(seed)
    # Cached spectral tensors preserve author one-time augmentation, not renewed per epoch.
    sx=channel_spectrogram(_awgn(x,gen)).cpu();sv=channel_spectrogram(_awgn(vx,gen)).cpu()
    opt=torch.optim.RMSprop(model.parameters(),lr=.001,alpha=.9,eps=1e-7)
    scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(opt,mode='min',factor=.2,patience=10)
    best=math.inf;stale=0;best_state=None;selected=0
    for epoch in range(1000):
        model.train();values=[]
        for _ in range(len(y)//32):
            ix=_triplets(y,32,gen);z=[model.network(sx[v].to(dev)) for v in ix]
            loss=F.relu((z[0]-z[1]).square().sum(1)-(z[0]-z[2]).square().sum(1)+.1).mean()
            opt.zero_grad(set_to_none=True);_checked_step(loss,opt);values.append(float(loss.detach()))
        model.eval();valid=[]
        with torch.no_grad():
            for _ in range(len(vy)//32):
                ix=_triplets(vy,32,gen);z=[model.network(sv[v].to(dev)) for v in ix]
                valid.append(float(F.relu((z[0]-z[1]).square().sum(1)-(z[0]-z[2]).square().sum(1)+.1).mean()))
        val=sum(valid)/len(valid);scheduler.step(val)
        if val<best:best=val;best_state=copy.deepcopy(model.state_dict());stale=0;selected=epoch+1
        else:stale+=1
        _log(logger,stage='source_triplet',epoch=epoch+1,loss=sum(values)/len(values),source_V_loss=val,lr=opt.param_groups[0]['lr'],target_contact=False)
        if stale>=20:break
    model.load_state_dict(best_state)
    return dict(executed_epochs=epoch+1,selected_epoch=selected,selection='source_V_triplet_loss',max_epochs=1000,early_patience=20,lr_patience=10)


def _source_tweak(model,data,validation,spec,seed,logger):
    x,y,_=data;vx,vy,_=validation;dev=_device(model);initial=copy.deepcopy(model.state_dict())
    best=math.inf;best_state=None;selected={};total=0
    # Retain the full existing reproduction's source-only decade LR search.
    for lr in spec['source_lr_grid']:
        model.load_state_dict(initial);opt=torch.optim.SGD(model.parameters(),lr=lr,momentum=.9)
        gen=torch.Generator().manual_seed(seed)
        for epoch in range(100):
            model.train();values=[]
            for _ in range(math.ceil(len(y)/64)):
                # Balanced 2+ per class makes every anchor's hard positive legal.
                ix=_balanced_batch(y,64,gen);opt.zero_grad(set_to_none=True)
                loss=batch_hard_triplet_loss(native_features(model,x[ix].to(dev)),y[ix].to(dev))
                _checked_step(loss,opt);values.append(float(loss.detach()))
            model.eval();v=[]
            with torch.no_grad():
                for _ in range(math.ceil(len(vy)/64)):
                    ix=_balanced_batch(vy,64,gen)
                    v.append(float(batch_hard_triplet_loss(native_features(model,vx[ix].to(dev)),vy[ix].to(dev))))
            val=sum(v)/len(v);total+=1
            if val<best:best=val;best_state=copy.deepcopy(model.state_dict());selected=dict(selected_epoch=epoch+1,selected_lr=lr)
            _log(logger,stage='source_tweak',epoch=epoch+1,grid_lr=lr,loss=sum(values)/len(values),source_V_loss=val,target_contact=False)
    model.load_state_dict(best_state)
    return dict(executed_epochs=total,epochs_per_lr=100,lr_grid=list(spec['source_lr_grid']),selection='source_V_triplet_loss',**selected)


def _balanced_batch(y,batch,gen):
    classes=torch.unique(y);n=math.ceil(batch/len(classes));rows=[]
    for c in classes:
        ix=torch.where(y==c)[0]
        if len(ix)<2:raise ValueError('Triplet source requires two physical examples per class')
        rows.append(ix[torch.randperm(len(ix),generator=gen)[:max(2,n)]])
    return torch.cat(rows)


def _source_radio_triplet(model,data,spec,seed,logger):
    """RadioNet all-positive-pair cosine triplets, epochwise semihard selection."""
    x,y,_=data;dev=_device(model);gen=torch.Generator().manual_seed(seed)
    pairs=torch.cat([torch.combinations(torch.where(y==c)[0],r=2) for c in torch.unique(y)])
    opt=torch.optim.Adam(model.parameters(),lr=.001,eps=1e-7)
    similarities=None
    for epoch in range(30):
        if epoch:
            z=F.normalize(_extract(model,x),dim=1);similarities=z@z.T
        model.train();values=[]
        for step,ix in enumerate(_indices(len(pairs),128,gen,drop_last=True)):
            a,p=pairs[ix].unbind(1);neg=[]
            for ai,pi in zip(a.tolist(),p.tolist()):
                candidates=torch.where(y!=y[ai])[0]
                if similarities is not None:
                    hard=candidates[similarities[ai,candidates]+.1>similarities[ai,pi]]
                    if len(hard):candidates=hard
                neg.append(int(candidates[torch.randint(len(candidates),(),generator=gen)]))
            z=[F.normalize(native_features(model,x[v].to(dev)),dim=1) for v in (a,p,torch.tensor(neg))]
            loss=F.relu((z[0]*z[2]).sum(1)-(z[0]*z[1]).sum(1)+.1).mean()
            opt.zero_grad(set_to_none=True);_checked_step(loss,opt);values.append(float(loss.detach()))
            if step%128==0:
                _log(logger,stage='source_radio_triplet',epoch=epoch+1,step=step+1,
                    expected_steps=len(pairs)//128,loss=values[-1],target_contact=False)
        _log(logger,stage='source_radio_triplet',epoch=epoch+1,loss=sum(values)/len(values),positive_pairs=len(pairs),target_contact=False)
    return dict(executed_epochs=30,selected_epoch=30,selection='fixed_final_epoch',positive_pairs=len(pairs))


def train_source(rt,profile,seed,output,device,logger=None):
    spec=source_training_spec(profile)
    if spec.get('blocked'):raise ValueError(spec['reason'])
    _seed(seed);out=Path(output);out.mkdir(parents=True,exist_ok=False)
    contract=_read(rt['source_contract']);data=_source_data(rt,'L_s');validation=_source_data(rt,'V')
    model=NativeModel(profile,len(contract['classes']),len(contract['source_rxs'])).to(device)
    if profile=='lora_triplet_knn':history=_source_lora(model,data,validation,spec,seed,logger)
    elif profile=='tweak_m1':history=_source_tweak(model,data,validation,spec,seed,logger)
    elif profile=='radionet_triplet':history=_source_radio_triplet(model,data,spec,seed,logger)
    else:history=_source_supervised(model,data,validation,spec,seed,logger)
    model.eval();x,y,_=data
    native={}
    if profile in ('lora_triplet_knn','tweak_m1','radionet_triplet'):
        native=dict(source_enrollment_features=_extract(model,x),source_enrollment_y=y.clone())
    if profile.startswith('issl_'):
        native=dict(source_replay_x=x.clone(),source_replay_y=y.clone())
    provenance=dict(schema='phase2_external_additional_source_v1',profile=profile,model_seed=seed,
        source_contract=contract,initialization='scratch',checkpoint_sources=[],target_contact=False,
        U_s_TX_access=False,source_roles_used=['L_s','V'],source_spec=spec,history=history,
        selection_role='V' if history['selection'].startswith('source_V') else None,
        source_replay_exception=profile.startswith('issl_'),claim=spec['claim'])
    torch.save(dict(model=model.state_dict(),native_state=native,provenance=provenance),out/'source.pt')
    _write(out/'source_provenance.json',provenance)
    _write(out/'source_contract.json',contract)
    _write(out/'completion.json',dict(status='SOURCE_TRAINED',target_evaluated=False,**history))
    return load_source({**rt,'additional_sources':{profile:{str(seed):str(out)}}},profile,seed,device)


def _source_path(rt,profile,seed):
    alias='issl_fixed_v1' if profile=='issl_author_behavior' else profile
    for key in ('additional_sources','external_sources','external_source_roots'):
        table=rt.get(key,{})
        for name in (profile,alias):
            entry=table.get(name)
            if isinstance(entry,dict):entry=entry.get(str(seed),entry.get(seed))
            if entry:return Path(entry)
    raise FileNotFoundError('No explicit compliant native source root for '+profile+' seed '+str(seed))


def _contract_match(actual,expected):
    for key in ('role_ids','source_rxs','source_days','counts','classes','ratios','split_seed'):
        if actual.get(key)!=expected.get(key):raise ValueError('CHECKPOINT_DATA_CONTRACT_MISMATCH: '+key)


def _legacy_source(rt,profile,seed,root,device):
    """Reuse the eight already-trained native E200 POSTER/DF Phase1 rows."""
    if profile not in ('poster_ft','radionet_df_ft'):
        raise ValueError('CHECKPOINT_PROVENANCE_UNVERIFIED: no compatible legacy loader for '+profile)
    init=_read(root/'initialization.json');complete=_read(root/'completion.json')
    actual=_read(root/'source_contract.json');cfg=_read(root/'resolved_config.json')
    expected=_read(rt['source_contract']);_contract_match(actual,expected)
    if init.get('scratch_only') is not True or init.get('checkpoint_sources')!=[] or init.get('target_contact') is not False:
        raise ValueError('CHECKPOINT_PROVENANCE_UNVERIFIED')
    if (complete.get('status')!='SOURCE_TRAINED' or complete.get('epoch')!=200
            or complete.get('target_evaluated') is not False):
        raise ValueError('CHECKPOINT_TARGET_CONTAMINATED_OR_INCOMPLETE')
    native_method='poster' if profile=='poster_ft' else 'radionet'
    if cfg.get('method')!=native_method or cfg.get('model_seed')!=seed or init.get('model_seed')!=seed:
        raise ValueError('CHECKPOINT_PROVENANCE_UNVERIFIED: source row')
    if actual.get('physical_roles')!='EXACT_MATCH' or actual.get('equalized')!=1 or actual.get('out_len')!=256 or actual.get('normalize') is not True:
        raise ValueError('CHECKPOINT_DATA_CONTRACT_MISMATCH: source input representation')
    options=cfg.get('options',{})
    if options.get('--epochs')!=200 or options.get('--author_backbone')!=native_method or '--source_only' not in options:
        raise ValueError('CHECKPOINT_PROVENANCE_UNVERIFIED: native E200 source options')
    path=root/'last.pt';payload=torch.load(path,map_location='cpu',weights_only=False)
    if payload.get('epoch')!=200:raise ValueError('CHECKPOINT_PROVENANCE_UNVERIFIED: incomplete E200 state')
    model=NativeModel(profile,len(expected['classes']),len(expected['source_rxs'])).to(device)
    model.network.load_state_dict(payload['model'],strict=True);model.eval()
    provenance=dict(schema='legacy_native_E200_source_reuse',profile=profile,model_seed=seed,
        source_contract=actual,initialization=init,completion=complete,resolved_config=cfg,
        selection='fixed_final_epoch_200',checkpoint_sources=[],target_contact=False,
        note='Original source-only clean/satellite augmentation and costs retained; no new source training.')
    return dict(model=model,features_fn=native_features,score_fn=native_logits,classes=tuple(range(len(expected['classes']))),
        resources=dict(profile=profile,source_checkpoint=str(path),source_provenance=provenance,source_reused=True,
            source_spec=source_training_spec(profile),target_fit=False,parameters=sum(v.numel() for v in model.parameters())),
        native_state={},profile=profile,source_path=str(path),provenance=provenance)


def load_source(rt,profile,seed,device):
    spec=source_training_spec(profile)
    if spec.get('blocked'):raise ValueError(spec['reason'])
    root=_source_path(rt,profile,seed);path=root if root.suffix=='.pt' else root/'source.pt'
    if not path.is_file() and root.is_dir():return _legacy_source(rt,profile,seed,root,device)
    payload=torch.load(path,map_location='cpu',weights_only=False);p=payload.get('provenance',{})
    source_profile=p.get('profile')
    if source_profile!=profile and not (profile=='issl_author_behavior' and source_profile=='issl_fixed_v1'):
        raise ValueError('CHECKPOINT_PROVENANCE_UNVERIFIED: wrong native method')
    if p.get('schema')!='phase2_external_additional_source_v1' or p.get('model_seed')!=seed or p.get('initialization')!='scratch' or p.get('checkpoint_sources')!=[]:
        raise ValueError('CHECKPOINT_PROVENANCE_UNVERIFIED')
    if p.get('target_contact') is not False or p.get('U_s_TX_access') is not False:
        raise ValueError('CHECKPOINT_TARGET_CONTAMINATED')
    expected=_read(rt['source_contract']);_contract_match(p['source_contract'],expected)
    trained_spec=p.get('source_spec',{})
    for key in ('source_epochs','source_batch','source_optimizer','source_lr','source_lr_grid'):
        if trained_spec.get(key)!=spec.get(key) and not (key=='source_lr_grid' and list(trained_spec.get(key,[]))==list(spec.get(key,[]))):
            raise ValueError('CHECKPOINT_PROVENANCE_UNVERIFIED: source budget '+key)
    history=p.get('history',{});executed=history.get('executed_epochs',0)
    if profile=='tweak_m1':required=100*len(spec['source_lr_grid'])
    else:required=spec['source_epochs']
    if 'source_early_patience' not in spec and executed!=required:
        raise ValueError('CHECKPOINT_PROVENANCE_UNVERIFIED: incomplete native source budget')
    if 'source_early_patience' in spec and not spec['source_early_patience']<executed<=required:
        raise ValueError('CHECKPOINT_PROVENANCE_UNVERIFIED: source-only early-stop budget')
    model=NativeModel(profile,len(expected['classes']),len(expected['source_rxs'])).to(device)
    model.load_state_dict(payload['model'],strict=True);model.eval()
    state=dict(model=model,features_fn=native_features,score_fn=native_logits,classes=tuple(range(len(expected['classes']))),
        resources=dict(profile=profile,source_checkpoint=str(path),source_provenance=p,source_spec=spec,
            target_fit=False,parameters=sum(v.numel() for v in model.parameters())),
        native_state=payload.get('native_state',{}),profile=profile,source_path=str(path),provenance=p)
    if profile in ('lora_triplet_knn','tweak_m1','radionet_triplet'):
        n=state['native_state'];state=_metric_state(state,n['source_enrollment_features'],n['source_enrollment_y'],source=True)
    return state


def _head(model):
    p=model.profile;n=model.network
    if p in ('poster_ft','radionet_df_ft'):return n.head
    if p=='feature_separation_hu2024':return n.tx_classifier
    if p=='ra_collab_single_ft':return n.tx_classifier.net[-1]
    if p.startswith('issl_'):return n.fc
    raise ValueError('No native affine output in '+p)


def _set_head(model,head):
    p=model.profile;n=model.network
    if p in ('poster_ft','radionet_df_ft'):n.head=head
    elif p=='feature_separation_hu2024':n.tx_classifier=head
    elif p=='ra_collab_single_ft':n.tx_classifier.net[-1]=head
    elif p.startswith('issl_'):n.fc=head
    else:raise ValueError('No native affine output')


def _prehead(model,x):
    if model.profile=='ra_collab_single_ft':
        z=native_features(model,x)
        for layer in list(model.network.tx_classifier.net.children())[:-1]:z=layer(z)
        return z
    return native_features(model,x)


def expand_head(state,all_support_x,all_support_y,class_count,seed):
    """Project native-head control: preserve every B pre-head/old output row."""
    _seed(seed);model=copy.deepcopy(state['model']);model.eval();old=_head(model)
    if class_count<old.out_features:raise ValueError('Cannot remove B registered classes')
    _labels(all_support_y,len(all_support_x),range(class_count));_iq(all_support_x)
    head=nn.Linear(old.in_features,class_count,bias=old.bias is not None).to(old.weight.device)
    with torch.no_grad():
        head.weight[:old.out_features].copy_(old.weight)
        if old.bias is not None:head.bias[:old.out_features].copy_(old.bias)
        for c in range(old.out_features,class_count):
            z=_prehead(model,all_support_x[all_support_y==c].to(_device(model))).mean(0)
            head.weight[c].copy_(F.normalize(z,dim=0)*old.weight.norm(dim=1).mean())
            if head.bias is not None:head.bias[c].zero_()
    _set_head(model,head)
    for parameter in model.parameters():parameter.requires_grad_(False)
    for parameter in head.parameters():parameter.requires_grad_(True)
    return dict(model=model,logits_fn=native_logits,trainable_parameters=list(head.parameters()))


def _state(bundle,model,support_x,support_y,**resources):
    result=dict(bundle);result['model']=model;result['features_fn']=native_features;result['score_fn']=native_logits
    result['native_state']=dict(bundle.get('native_state',{}),support_x=support_x.detach().cpu().clone(),support_y=support_y.detach().cpu().clone())
    result['resources']=dict(bundle['resources'],target_fit=True,**resources)
    result['classes']=tuple(sorted(set(support_y.cpu().tolist())))
    if model.profile in ('poster_ft','radionet_df_ft','feature_separation_hu2024','ra_collab_single_ft'):result['expand_head']=expand_head
    return result


def _metric_state(state,features,labels,source=False):
    p=state['profile'];classes=tuple(sorted(set(labels.cpu().tolist())))
    f=features.detach().cpu().clone();y=labels.detach().cpu().clone()
    n=dict(state.get('native_state',{}),enrollment_features=f,enrollment_y=y)
    if p=='tweak_m1':
        centroids=torch.stack([f[y==c].mean(0) for c in classes])
        radii=torch.stack([(f[y==c]-centroids[i]).norm(dim=1).mean() for i,c in enumerate(classes)])
        n.update(centroids=centroids,radii=radii)
        def score(model,x):
            distance=torch.cdist(native_features(model,x),centroids.to(x.device))
            radius=radii.to(x.device)[None,:]
            inside=distance<radius
            # Algorithm 2: nearest contained class; when none contains the
            # point, choose minimum distance outside each class's own radius.
            contained=torch.where(inside,-distance,torch.full_like(distance,-1e30))
            return torch.where(inside.any(1,keepdim=True),contained,radius-distance)
    elif p=='radionet_triplet':
        # Native N-MEV averages legal enrollment vectors per registered identity.
        means=torch.stack([f[y==c].mean(0) for c in classes]);n.update(signature_means=means)
        def score(model,x):
            d=(1-F.normalize(native_features(model,x),dim=1)@F.normalize(means.to(x.device),dim=1).T).clamp_min(0)
            selected=d.argsort(dim=1,stable=True)[:,:min(19,len(means))]
            included=torch.zeros_like(d,dtype=torch.bool).scatter_(1,selected,True)
            # Author kNN uses 19 neighbors, distance weights and cosine metric;
            # min(19,C) is explicit for this six-old-class benchmark.
            zero=(d<=1e-7)&included
            weights=torch.where(included,1/d.clamp_min(1e-12),torch.zeros_like(d))
            weights=torch.where(zero.any(1,keepdim=True),zero.to(d.dtype),weights)
            return weights/weights.sum(1,keepdim=True).clamp_min(1e-12)
    else:
        # The frozen enrollment gallery is reused for every query batch.
        # Keep one device copy instead of transferring the same gallery on
        # every call; retain CPU originals for the method's native state.
        gallery_device=f.to(_device(state['model']))
        labels_device=y.to(_device(state['model']))
        n.update(enrollment_device_features=gallery_device,enrollment_device_y=labels_device)
        def score(model,x):
            z=native_features(model,x);distance=torch.cdist(z,gallery_device.to(x.device));k=min(15,len(f))
            # Uniform 15-neighbor native LoRa vote; class ties choose lowest class.
            ix=distance.argsort(dim=1,stable=True)[:,:k];neighbors=labels_device.to(x.device)[ix]
            return torch.stack([(neighbors==c).sum(1).to(z.dtype) for c in classes],1)/k
    result=dict(state,score_fn=score,classes=classes,native_state=n)
    result['resources']=dict(state['resources'],enrollment_count=len(y),source_enrollment=source,
        knn_k=min(15,len(y)) if p=='lora_triplet_knn' else None,query_fit=False)
    if p=='lora_triplet_knn':
        result['resources'].update(enrollment_cache_device=str(gallery_device.device),
            enrollment_cache_bytes=gallery_device.numel()*gallery_device.element_size()+labels_device.numel()*labels_device.element_size())
    return result


def fit_B(rt,bundle,support_x,support_y,seed,logger=None):
    p=bundle['profile'];spec=source_training_spec(p)
    _iq(support_x);_labels(support_y,len(support_x),bundle['classes']);_seed(seed)
    model=copy.deepcopy(bundle['model']);model.requires_grad_(True)
    if p in ('lora_triplet_knn','tweak_m1','radionet_triplet'):
        state=_state(bundle,model,support_x,support_y,B_epochs=0,B_rule='target_support_enrollment_only')
        return _metric_state(state,_extract(model,support_x),support_y)
    parameters=None
    if p in ('poster_ft','radionet_df_ft'):
        model.network.author_finetune(len(bundle['classes']))
        parameters=[v for v in model.parameters() if v.requires_grad]
    elif p=='feature_separation_hu2024':
        for module in (model.network.rx_branch,model.network.rx_classifier):module.requires_grad_(False);module.eval()
        parameters=[v for v in model.parameters() if v.requires_grad]
    _train_ce(model,support_x,support_y,spec['B_epochs'],min(spec['B_batch'],len(support_y)),spec['B_lr'],seed,logger,
        'SGD' if p=='ra_collab_single_ft' else 'Adam',parameters=parameters)
    return _state(bundle,model,support_x,support_y,B_epochs=spec['B_epochs'],B_rule=spec['claim'])


def _issl_expand(model,classes):
    old=model.network.fc
    if classes<=old.out_features:raise ValueError('ISSL C requires new opened classes')
    head=nn.Linear(old.in_features,classes,bias=False).to(old.weight.device)
    with torch.no_grad():head.weight[:old.out_features].copy_(old.weight)
    model.network.fc=head


def _kd(student,teacher,temperature):
    return -(F.softmax(teacher.detach()/temperature,1)*F.log_softmax(student/temperature,1)).sum(1).mean()


def _issl_ssl(model,x,old_classes,spec,seed,logger):
    dev=_device(model);author=model.profile=='issl_author_behavior';key=copy.deepcopy(model).eval().requires_grad_(False)
    gen=torch.Generator().manual_seed(seed)
    size=random.randint(1,min(1000,len(x)))
    with torch.no_grad():queue=key.raw(x[torch.randperm(len(x),generator=gen)[:size]].to(dev),218)[0].detach()
    if not author:queue=F.normalize(queue,dim=1,eps=1e-12)
    opt=torch.optim.Adam(model.parameters(),lr=spec['C_SSL_lr']);scheduler=torch.optim.lr_scheduler.StepLR(opt,100,.8)
    for epoch in range(spec['C_SSL_epochs']):
        values=[];model.train()
        # Partial support batches are retained: the formal K=1 extension cannot drop every batch.
        for ix in _indices(len(x),64,gen):
            xx=x[ix].to(dev);qx=_issl_input(xx,218);kx=_issl_input(xx,218);q=model.network(qx)[0]
            with torch.no_grad():k=key.network(kx)[0];t=key.network(qx)[0]
            kd=_kd(q[:,:old_classes],t[:,:old_classes],20)
            if author:kd=kd.detach().requires_grad_(True)
            q=F.normalize(q,dim=1,eps=1e-12);k=F.normalize(k,dim=1,eps=1e-12)
            negative=q@(queue.reshape(queue.shape[1],queue.shape[0]) if author else queue.T)
            contrast=F.cross_entropy(torch.cat(((q*k).sum(1,keepdim=True),negative),1)/5,
                torch.zeros(len(q),dtype=torch.long,device=dev))
            loss=.5*kd+.5*contrast
            if not author:opt.zero_grad(set_to_none=True)
            _checked_step(loss,opt);values.append(float(loss.detach()))
            queue=torch.cat((queue,k.detach()),0)
            queue=queue[64:] if author and len(queue)>1000 else queue[-1000:]
            if not author:queue=F.normalize(queue,dim=1,eps=1e-12)
            with torch.no_grad():
                for kp,mp in zip(key.parameters(),model.parameters()):kp.mul_(.99).add_(mp,alpha=.01)
                if not author:
                    for kb,mb in zip(key.buffers(),model.buffers()):
                        if kb.is_floating_point():kb.mul_(.99).add_(mb,alpha=.01)
                        else:kb.copy_(mb)
        scheduler.step()
        _log(logger,stage='C_ssl',epoch=epoch+1,loss=sum(values)/len(values),queue_size=len(queue),
            kd_student_gradient=not author,zero_grad=not author,query_fit=False)


def native_C(rt,bundle,Bstate,new_x,new_y,new_ids,seed,logger=None):
    p=bundle['profile'];spec=source_training_spec(p)
    if not spec.get('native_C'):raise ValueError('This closed-set native method has no C; use its explicitly named registered control')
    _iq(new_x);old_count=len(Bstate['classes']);new_classes=_labels(new_y,len(new_x))
    if not new_classes or new_classes!=tuple(range(old_count,max(new_classes)+1)):
        raise ValueError('C may open only the current contiguous new classes')
    if tuple(new_ids)!=new_classes:raise ValueError('New class IDs must match the currently opened contiguous new classes')
    old=Bstate['native_state'];sx=torch.cat((old['support_x'],new_x.detach().cpu()));sy=torch.cat((old['support_y'],new_y.detach().cpu()))
    _seed(seed);model=copy.deepcopy(Bstate['model']).requires_grad_(True)
    if p in ('lora_triplet_knn','tweak_m1','radionet_triplet'):
        state=_state(Bstate,model,sx,sy,C_rule='append_current_new_support_enrollment')
        # Frozen encoder means old enrollment values and old centroid/radius remain exact.
        return _metric_state(state,torch.cat((old['enrollment_features'],_extract(model,new_x))),sy)
    _issl_expand(model,len(set(sy.tolist())))
    replay=bundle['native_state'];rx,ry=replay['source_replay_x'],replay['source_replay_y']
    transfer_x=torch.cat((rx,sx));transfer_y=torch.cat((ry,sy))
    ssl_x=torch.cat((rx,transfer_x))  # Author repeats old labeled source records.
    _issl_ssl(model,ssl_x,old_count,spec,seed,logger)
    _train_ce(model,transfer_x,transfer_y,spec['C_transfer_epochs'],128,spec['C_transfer_lr'],seed,logger,
        crop=218,stage='C_transfer')
    return _state(Bstate,model,sx,sy,C_SSL_epochs=2000,C_transfer_epochs=300,
        C_source_replay_count=len(rx),C_source_old_repeated=True,C_claim=spec['claim'])
