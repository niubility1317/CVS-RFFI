"""Original F50 inference graph; no restorer training or head fitting.

The identity encoder, triple SIG_TIME pool, centered Sinc leaves, and original
default head are copied from the declared checkpoint. Domain/EMA training
branches are never inference inputs. Historical cleanF/AsF remain separate.
"""
from pathlib import Path
import torch
from torch import nn

from experiments.pamf_restoration.frozen import read, check_contract, TripleSignaturePool
from experiments.standard_ce_baselines.sigtime import SIGTimeCVS

REFERENCE_RUN = '20261016-phase1-F-supcon-e50-clean-manysig-m9-r02'
REFERENCE_COMMIT = '4df7b94d0fd8ce29056bd8236364e45afa92e07f'
SEEDS = (2026092701, 2026092702, 2026092703)
RECIPE_KEYS = ('seed', 'epochs', 'label_epochs', 'pseudo_epochs', 'label_smoothing',
               'use_aug', 'use_ema_teacher', 'use_concat_sat_channel_aug',
               'baseline_ckpt', 'teacher_ckpt', 'batch_size', 'max_grad_norm',
               'pseudo_temporal_mode', 'phase1_lr_schedule', 'labeled_ratio', 'unlabeled_ratio',
               'source_val_ratio')


class FrozenCenteredSinc(nn.Module):
    """Preserve the native p=p0+25000*q coordinates and kernel evaluation.

    Same implementation boundary as cvs_rdrop_sinc_units/coordinate.py at the
    reference commit. Parameters remain leaves, so the native kernel cache
    observes their actual versions rather than computed-tensor version zero.
    """
    def __init__(self, original):
        super().__init__()
        if (type(original).__name__ != 'SincConv1d' or original.low_hz_.shape != (24, 1)
                or original.kernel_size != 79 or original.sample_rate != 25e6):
            raise ValueError('Original F identity Sinc required')
        self._native_type = type(original)
        self.scale_hz = 25000.
        for name in ('out_channels', 'kernel_size', 'sample_rate', 'dataset', 'input_len',
                     'pad_crop_mode', 'min_low_hz', 'min_band_hz'):
            setattr(self, name, getattr(original, name))
        for short in ('low', 'band'):
            value = getattr(original, short+'_hz_').detach().clone()
            self.register_parameter(short+'_coordinate', nn.Parameter(torch.zeros_like(value)))
            self.register_buffer(short+'_origin_hz', value)
        for name in ('t_', 'window_'):
            self.register_buffer(name, getattr(original, name).detach().clone())
        self._filter_cache_key = self._filter_cache = self._coordinate_cache_key = None

    @property
    def low_hz_(self):
        return self.low_origin_hz + self.scale_hz*self.low_coordinate

    @property
    def band_hz_(self):
        return self.band_origin_hz + self.scale_hz*self.band_coordinate

    def _filters(self, device, dtype):
        version = (device.type, device.index, self.low_coordinate.dtype,
                   self.low_coordinate._version, self.band_coordinate._version,
                   self.low_origin_hz._version, self.band_origin_hz._version,
                   self.t_._version, self.window_._version)
        if version != self._coordinate_cache_key:
            self._filter_cache_key = self._filter_cache = None
            self._coordinate_cache_key = version
        # Native Sinc reads physical-tensor versions. Creating p0+s*q inside
        # inference_mode would produce tensors without a version counter.
        # This reader is frozen; synthesize its constant filters in no_grad.
        with torch.inference_mode(False), torch.no_grad():
            return self._native_type._filters(self, device, dtype)

    def forward(self, x):
        return self._native_type.forward(self, x)

    def forward_iq_pair(self, x):
        return self._native_type.forward_iq_pair(self, x)


def validate_metadata(initial, done, freeze, actual, auxiliary, seed):
    if seed not in SEEDS:
        raise ValueError('Registered original F50 seed required')
    config = done.get('config', {})
    required_config = dict(run_id=REFERENCE_RUN, row_id=f'CROSS_RX-s{seed}',
                           model_seed=seed, arm='CROSS_RX', epochs=50,
                           label_epochs=33, pseudo_epochs=17, updates=11100,
                           checkpoint_selection='final_only', separate_head_training=False)
    if any(config.get(k) != v for k, v in required_config.items()):
        raise ValueError('CHECKPOINT_PROVENANCE_UNVERIFIED: original F+CROSS_RX50 config')
    if (initial.get('scratch_only') is not True or initial.get('checkpoint_sources') != []
            or initial.get('ancestors') != [] or initial.get('target_contact') is not False
            or initial.get('source_roles') != 'EXACT_MATCH' or initial.get('seed') != seed):
        raise ValueError('CHECKPOINT_PROVENANCE_UNVERIFIED: scratch lineage')
    required_done = dict(status='SOURCE_TRAINED', epoch=50, label_epochs=33, pseudo_epochs=17,
                         logged_steps=11100, optimizer_steps=11100, head_fit_updates=0,
                         checkpoint_sources=[], ancestors=[], initialization='scratch',
                         target_access=False, target_evaluated=False)
    if any(done.get(k) != v for k, v in required_done.items()):
        raise ValueError('CHECKPOINT_PROVENANCE_UNVERIFIED: source trajectory')
    if (freeze.get('status') != 'SOURCE_FROZEN' or freeze.get('config') != config
            or freeze.get('epoch') != 50 or freeze.get('selection') != 'final_only'
            or freeze.get('target_access') is not False
            or freeze.get('checkpoint') != done.get('checkpoint')):
        raise ValueError('CHECKPOINT_PROVENANCE_UNVERIFIED: fixed final50')
    args = actual.get('args', actual)
    required_args = dict(seed=seed, epochs=50, label_epochs=33, pseudo_epochs=17,
                         label_smoothing=.01, use_aug=True, use_ema_teacher=True,
                         use_concat_sat_channel_aug=False, baseline_ckpt='', teacher_ckpt='',
                         batch_size=128, max_grad_norm=5., pseudo_temporal_mode='batch_neighbor',
                         phase1_lr_schedule='cosine', labeled_ratio=.07, unlabeled_ratio=.63,
                         source_val_ratio=.3)
    if any(args.get(k) != v for k, v in required_args.items()):
        raise ValueError('CHECKPOINT_PROVENANCE_UNVERIFIED: actual original recipe')
    expected_aux = dict(status='VERIFIED', mode='CROSS_RX', calls=11100,
                        applied_calls=9990, head_fit_updates=0, U_truth_read=False,
                        default_head_jointly_trained=True)
    if any(auxiliary.get(k) != v for k, v in expected_aux.items()):
        raise ValueError('CHECKPOINT_PROVENANCE_UNVERIFIED: original contrastive/default head')
    return config, args


def validate_checkpoint(payload, config, args):
    expected = dict(run_id=REFERENCE_RUN, candidate_id=config['row_id'], epoch=50,
                    checkpoint_selection='final_only', a1_ema_successful_updates=11100)
    if (any(payload.get(k) != v for k, v in expected.items())
            or any(payload.get('args', {}).get(k) != args[k] for k in RECIPE_KEYS)):
        raise ValueError('CHECKPOINT_PROVENANCE_UNVERIFIED: payload identity/recipe')


def from_state(state, seed):
    prefix = 'id_backbone.encoder.'
    encoder_state = {k[len(prefix):]: v for k, v in state.items() if k.startswith(prefix)}
    model = SIGTimeCVS(seed)
    model.sequence_pool = TripleSignaturePool(model.core.id_backbone.t2.pw.out_channels)
    model.core.id_backbone.sinc = FrozenCenteredSinc(model.core.id_backbone.sinc)
    # Includes the original jointly trained head; missing/unexpected keys fail.
    model.load_state_dict(encoder_state, strict=True)
    if model.core.id_backbone.cls_head.weight.shape != (6, 160):
        raise ValueError('Original default head shape differs')
    return model.eval().requires_grad_(False)


def load_reader(folder, source_cache, device='cpu', seed=2026092701):
    folder = Path(folder)
    check_contract(folder, source_cache)
    config, args = validate_metadata(*[read(folder/n) for n in (
        'initialization.json', 'completion.json', 'source_frozen.json',
        'actual_native_args.json', 'supcon_training_observation.json')], seed)
    payload = torch.load(folder/'final_ssdg.pth', map_location='cpu', weights_only=False)
    validate_checkpoint(payload, config, args)
    return from_state(payload['model'], seed).to(device)
