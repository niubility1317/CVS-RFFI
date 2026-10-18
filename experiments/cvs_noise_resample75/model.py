"""Reuse the exact six-L hook; change only the second existing As IQ slot."""
from collections import Counter
from copy import deepcopy
from types import MethodType
from weakref import WeakKeyDictionary
import torch
from . import design as d
from .channel import second_noise_view
from experiments.cvs_full_channel_pair68 import model as pair

classifier, source_domain_count = pair.classifier, pair.source_domain_count
PHYSICAL_FIELDS = pair.PHYSICAL_FIELDS
_PHYSICS_SINKS = WeakKeyDictionary()


def physics_sink(model, sink=None):
    """Keep writers outside model state, checkpoints and EMA deepcopy graphs."""
    if sink is None:
        _PHYSICS_SINKS.pop(model, None)
    else:
        _PHYSICS_SINKS[model] = sink


def hook(self, **kw):
    arm, step = self._noise_arm, kw['optimizer_step']
    if arm not in ('C01', 'N01'):
        raise ValueError('Only same-IQ and conditional independent-noise arms')
    channel, originals = self._full_channel, []
    old_view, apply = pair.independent_view, channel.apply
    had_apply, old_apply = 'apply' in vars(channel), vars(channel).get('apply')
    observations = []

    def first_apply(*args, **kwargs):
        value = apply(*args, **kwargs)
        if originals or len(value[0]) != 64 or len(value[1]) != 64:
            raise ValueError('Exactly one original As64 generation per L step')
        originals.append((value[0].detach().clone(), deepcopy(value[1])))
        return value

    def choose_view(primary, raw, meta, first, hardware):
        if primary is not channel.obj or len(originals) != 1 or observations:
            raise ValueError('Second As must follow this model original first view')
        if len(first) != 64 or len(hardware) != 64:
            raise ValueError('Exactly64 first-view metadata and RX objects required')
        if arm == 'N01':
            iq, records = second_noise_view(primary, raw, meta, first, hardware)
        else:
            iq, records = originals[0][0].clone(), deepcopy(first)
        if iq.shape != raw.shape or len(records) != 64 or not bool(torch.isfinite(iq).all()):
            raise ValueError('Every labeled second-view slot must contain finite IQ')
        for original, record in zip(first, records):
            if any(record.get(k) != original.get(k) for k in
                   ('sample_id', 'session_id', 'scene', 'state_end', 'receiver_lock_status')):
                raise ValueError('Conditional views changed physical sample or channel state')
            record.update(view_index=1, view_kind='same_IQ_reuse' if arm == 'C01' else 'independent_noise',
                channel_generated=arm == 'N01', full_physical_latent_redrawn=False,
                same_RX_hardware=True, agc_recomputed=arm == 'N01')
        if arm == 'C01' and not torch.equal(iq, originals[0][0]):
            raise ValueError('Same-IQ control changed its original satellite input')
        observations.append(records)
        return iq, records

    try:
        channel.apply, pair.independent_view = first_apply, choose_view
        result = pair.hook(self, **kw)
        if len(originals) != 1 or len(observations) != 1 or not self._pair_records:
            raise ValueError('One original As64 and one existing second As64 slot required')
        record = self._pair_records[-1]
        actual_second = 64 if arm == 'N01' else 0
        record.update(method='NOISE_CE6' if arm == 'N01' else 'SAME_CE6', noise_arm=arm,
            As_same_IQ=arm == 'C01', generated_As0=64, generated_As1=actual_second,
            As_generated_packets=64 + actual_second, As_forward_packets=128,
            As_supervised=128, As_unique_physical_samples=64,
            actual_residual_snapshots=64 + actual_second,
            actual_secondary_snapshots=actual_second, physical_latent_draws=64,
            secondary_noise_redraws=actual_second, same_H_and_receiver_state=True, receiver_state_excludes_AGC=True,
            same_AGC=arm == 'C01', per_view_native_AGC_recomputed=arm == 'N01',
            first_namespaces=sorted({r['namespace'] for r in originals[0][1]}),
            second_namespaces=sorted({r['namespace'] for r in observations[0]}),
            pair_generated_is_logical_second_slots=True,
            two_separate_CE_mean=True, cross_channel_KL=False)
        native = self._full_records[-1]
        native.update(same_IQ_each_pair=arm == 'C01', same_IQ_ordinary_raw_pairs=True,
            As_generated_packets=64 + actual_second, As_forward_packets=128,
            satellite_subnetwork=record, full_as_pair=record, noise_resample75=record)
        self._noise_records.append(record)
        self._noise_generated += 64 + actual_second
        self._noise_secondary_generated += actual_second
        self._noise_latent_draws += 64
        self._noise_slots += 128
        labels = kw['labels']
        mask = (torch.arange(128, device=labels.device) + step).remainder(2) != 0
        for view, records in enumerate((originals[0][1], observations[0])):
            for label, actual in zip(labels[mask].tolist(), records):
                rx = int(actual['sample_id'][1])
                self._noise_coverage[(view, int(label), rx, actual['scene'], actual['state_end'], actual['receiver_lock_status'])] += 1
        sink = _PHYSICS_SINKS.get(self)
        if sink is not None:
            sink(dict(step=step, cumulative_step=step + 4440, noise_arm=arm,
                view0=originals[0][1], view1=observations[0],
                actual_residual_snapshots=64 + actual_second,
                physical_latent_draws=64, secondary_noise_redraws=actual_second))
        return result
    finally:
        pair.independent_view = old_view
        if had_apply:
            channel.apply = old_apply
        else:
            delattr(channel, 'apply')


hook._accepts_labeled_inputs = True


def attach(m, c):
    d.validate(c)
    pair.attach(m, d.reference_config('INDEP_CE6'))
    if m._sat_engine.mode != 'AS_CE':
        raise ValueError('Keep original satellite dropout and CE-only rule')
    m._noise_arm = c['arm']
    m._noise_records, m._noise_coverage = [], Counter()
    m._noise_generated = m._noise_secondary_generated = m._noise_latent_draws = m._noise_slots = 0
    m._source_labeled_feature_hook = MethodType(hook, m)
    m._feature_learning_report['noise_resample75'] = dict(arm=c['arm'], L_NN_per_step=768,
        As_forward_packets_per_step=128, new_losses=0, extra_U_NN=0,
        actual_snapshots_per_step=64 if c['arm'] == 'C01' else 128,
        unique_latents_per_step=64, As_KL=False, native_AGC_each_view=c['arm'] == 'N01')
    return m


def install(m, c):
    return attach(pair.original.install(m, d.old_config()), c)


def build(c, device='cpu'):
    return attach(pair.original.build(d.old_config(), device), c)


def own_model(c,device='cpu'):
    from pathlib import Path
    from .source import parent_payload
    from .protocol import final_metadata,checkpoint
    d.validate(c);parent=parent_payload(c);out=Path(c['output_root'])
    final_metadata(c,d.read(out/'completion.json'),d.read(out/'source_frozen.json'),
        d.read(out/'initialization.json'),d.read(out/'source_contract.json'),d.read(d.SOURCE)['role_ids'])
    cp=torch.load(out/'final_ssdg.pth',map_location='cpu',weights_only=False);checkpoint(c,cp)
    if set(cp['model'])!=set(parent['model']) or set(cp['ema_model'])!=set(parent['ema_model']):
        raise ValueError('Own default-head/student/EMA state keys differ')
    m=build(c,device);m.load_state_dict(cp['model'],strict=True);return m.eval().requires_grad_(False)
