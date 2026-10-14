"""Fixed-final RAW_HZ/CENTER_25K chart and saved AdamW-state mapping.

CoordinateSinc below preserves the implementation at commit
0f81373c25ba69a6cb9deec5c7ca652880fd62af. This is an evaluation hypothesis
at the same frozen physical state, not a historical CENTER training trajectory.
For functional_call branches, the caller must clear parameter-derived eval
caches before each branch and restore them afterwards: distinct temporary
tensors can have identical _version values. The historical cache rule is kept.
"""
import copy
import math
from collections.abc import Mapping

import torch
from torch import nn

PATH = 'id_backbone.encoder.core.id_backbone.sinc'
PREFIX = PATH
SCALE_HZ = 25000.
MODES = ('RAW_HZ', 'CENTER_25K')


class CoordinateSinc(nn.Module):
 def __init__(self,original,mode,scale_hz=25000.):
  super().__init__()
  if type(original).__name__!='SincConv1d' or mode not in ('RAW_HZ','CENTER_25K') or scale_hz!=25000.:raise ValueError('Unregistered Sinc coordinate scheme')
  if original.low_hz_.shape!=(24,1) or original.band_hz_.shape!=(24,1) or original.kernel_size!=79 or original.sample_rate!=25e6:raise ValueError('Unexpected native identity Sinc')
  self.mode=mode;self.scale_hz=float(scale_hz);self._native_type=type(original)
  for name in ('out_channels','kernel_size','sample_rate','dataset','input_len','pad_crop_mode','min_low_hz','min_band_hz'):setattr(self,name,getattr(original,name))
  low=original.low_hz_.detach().clone();band=original.band_hz_.detach().clone()
  self.low_coordinate=nn.Parameter(low.clone() if mode=='RAW_HZ' else torch.zeros_like(low))
  self.band_coordinate=nn.Parameter(band.clone() if mode=='RAW_HZ' else torch.zeros_like(band))
  self.register_buffer('low_origin_hz',low);self.register_buffer('band_origin_hz',band)
  self.register_buffer('t_',original.t_.detach().clone());self.register_buffer('window_',original.window_.detach().clone())
  self._filter_cache_key=None;self._filter_cache=None;self._coordinate_cache_key=None
 @property
 def low_hz_(self):return self.low_coordinate if self.mode=='RAW_HZ' else self.low_origin_hz+self.scale_hz*self.low_coordinate
 @property
 def band_hz_(self):return self.band_coordinate if self.mode=='RAW_HZ' else self.band_origin_hz+self.scale_hz*self.band_coordinate
 def _filters(self,device,dtype):
  # Computed low_hz_/band_hz_ tensors have version0. Track their actual leaves.
  version=(device.type,device.index,self.low_coordinate.dtype,self.low_coordinate._version,self.band_coordinate._version,self.low_origin_hz._version,self.band_origin_hz._version,self.t_._version,self.window_._version)
  if version!=self._coordinate_cache_key:
   self._filter_cache_key=None;self._filter_cache=None;self._coordinate_cache_key=version
  return self._native_type._filters(self,device,dtype)
 def forward(self,x):return self._native_type.forward(self,x)
 def forward_iq_pair(self,x):return self._native_type.forward_iq_pair(self,x)


def _mode(mode):
    if mode not in MODES:
        raise ValueError('Only fixed RAW_HZ/CENTER_25K coordinates are supported')


def _finite_tensor(value, shape, name):
    if (not isinstance(value, torch.Tensor) or value.layout != torch.strided or value.shape != shape
            or not value.is_floating_point() or not bool(torch.isfinite(value).all())):
        raise ValueError(f'{name} must be a finite floating tensor of shape {shape}')


def _native(sinc):
    if isinstance(sinc, CoordinateSinc):
        _mode(sinc.mode)
        if sinc.scale_hz != SCALE_HZ:
            raise ValueError('Only the original 25k chart scale is supported')
        kind = sinc._native_type
    elif type(sinc).__name__ == 'SincConv1d':
        kind = type(sinc)
    else:
        raise ValueError('Exact native identity Sinc or this CoordinateSinc required')
    if sinc.out_channels != 24 or sinc.kernel_size != 79 or sinc.sample_rate != 25e6:
        raise ValueError('Only native 24x1, kernel79, Fs25M identity Sinc is supported')
    for name in ('low_hz_', 'band_hz_'):
        _finite_tensor(getattr(sinc, name), (24, 1), name)
    for name in ('t_', 'window_'):
        _finite_tensor(getattr(sinc, name), (1, 79), name)
    if sinc.low_hz_.dtype != sinc.band_hz_.dtype or sinc.low_hz_.device != sinc.band_hz_.device:
        raise ValueError('Sinc coordinate dtype/device mismatch')
    return kind(sinc.out_channels, sinc.kernel_size, sample_rate=sinc.sample_rate,
                min_low_hz=sinc.min_low_hz, min_band_hz=sinc.min_band_hz,
                dataset=sinc.dataset, input_len=sinc.input_len, pad_crop_mode=sinc.pad_crop_mode).to(
                    device=sinc.low_hz_.device, dtype=sinc.low_hz_.dtype)


def convert_model(raw_model, initial_physical, mode='CENTER_25K'):
    """Deepcopy a model, changing ONLY the declared identity Sinc chart.

    initial_physical has low_hz_/band_hz_ [24,1] tensors from the original native
    constructor. The supplied values must match that constructor; they never
    come from the frozen final frequencies. CENTER coordinate=(frozen-origin)/
    25000. RAW control uses the same physical values in raw Hz. Non-Sinc state,
    training flags, requires_grad flags and the existing default head are kept.
    A CoordinateSinc input is accepted for the explicit physical roundtrip.
    """
    _mode(mode)
    if not isinstance(raw_model, nn.Module):
        raise ValueError('raw_model must be an existing native model')
    try:
        original = raw_model.get_submodule(PATH)
    except AttributeError as error:
        raise ValueError('Declared native identity Sinc path is absent') from error
    native_initial = _native(original)
    if not isinstance(initial_physical, Mapping) or set(initial_physical) != {'low_hz_', 'band_hz_'}:
        raise ValueError('initial_physical must supply exactly constructor low_hz_/band_hz_')
    origins = {}
    for name in ('low_hz_', 'band_hz_'):
        value = initial_physical[name]
        _finite_tensor(value, (24, 1), 'initial_physical.' + name)
        physical = getattr(original, name).detach()
        if value.dtype != physical.dtype:
            raise ValueError('Constructor origin dtype differs from the native physical parameters')
        origins[name] = value.detach().to(device=physical.device).clone()
        if not torch.equal(origins[name], getattr(native_initial, name).detach()):
            raise ValueError('Origin must be the original native constructor frequency, not a fitted/frozen replacement')
    with torch.no_grad():
        native_initial.low_hz_.copy_(original.low_hz_.detach())
        native_initial.band_hz_.copy_(original.band_hz_.detach())
        native_initial.t_ = original.t_.detach().clone()
        native_initial.window_ = original.window_.detach().clone()
    converted = copy.deepcopy(raw_model)
    replacement = CoordinateSinc(native_initial, mode)
    with torch.no_grad():
        for part in ('low', 'band'):
            origin = origins[part + '_hz_']
            physical = getattr(original, part + '_hz_').detach()
            getattr(replacement, part + '_origin_hz').copy_(origin)
            getattr(replacement, part + '_coordinate').copy_(
                physical if mode == 'RAW_HZ' else (physical - origin) / SCALE_HZ)
    replacement.low_coordinate.requires_grad_(original.low_coordinate.requires_grad if isinstance(original, CoordinateSinc) else original.low_hz_.requires_grad)
    replacement.band_coordinate.requires_grad_(original.band_coordinate.requires_grad if isinstance(original, CoordinateSinc) else original.band_hz_.requires_grad)
    replacement.train(original.training)
    parent_path, name = PATH.rsplit('.', 1)
    setattr(converted.get_submodule(parent_path), name, replacement)
    return converted


def _renamed(name):
    return {PATH + '.low_hz_': PATH + '.low_coordinate',
            PATH + '.band_hz_': PATH + '.band_coordinate'}.get(name, name)


def convert_optimizer(saved_state, raw_named_parameters, converted_model, mode='CENTER_25K', *,
                      source_mode='RAW_HZ'):
    """Load the real one-group AdamW state by exact named traversal.

    raw_named_parameters is the ordered full native optimizer traversal as
    (name, parameter) pairs. Canonical state IDs 0..N-1 and the mapped converted
    model traversal must match. Present entries require complete real moments;
    absent never-used non-Sinc entries remain absent. Both Sinc states must
    exist. m scales by chartJacobian(target)/chartJacobian(source), v/max_v by
    its square. Step, beta, LR, wd, eps and all other state remain unchanged.
    source_mode is explicit for a reverse CENTER->RAW roundtrip; names alone
    cannot distinguish a RAW CoordinateSinc from a CENTER CoordinateSinc.
    """
    _mode(mode); _mode(source_mode)
    if not isinstance(saved_state, Mapping) or set(saved_state) != {'state', 'param_groups'}:
        raise ValueError('A standard complete saved AdamW state_dict is required')
    groups = saved_state['param_groups']
    if not isinstance(groups, list) or len(groups) != 1 or not isinstance(groups[0], Mapping):
        raise ValueError('Only the actual native single optimizer group is supported')
    pairs = list(raw_named_parameters.items()) if isinstance(raw_named_parameters, Mapping) else list(raw_named_parameters)
    if not pairs or any(not isinstance(pair, (tuple, list)) or len(pair) != 2
                        or not isinstance(pair[0], str) or not isinstance(pair[1], nn.Parameter) for pair in pairs):
        raise ValueError('Exact ordered (name, native parameter) traversal required')
    names = [name for name, p in pairs]
    if len(set(names)) != len(names) or len({id(p) for name, p in pairs}) != len(pairs):
        raise ValueError('Duplicate native parameter name/object is unsupported')
    target_sinc = converted_model.get_submodule(PATH)
    if not isinstance(target_sinc, CoordinateSinc) or target_sinc.mode != mode or target_sinc.scale_hz != SCALE_HZ:
        raise ValueError('Converted model chart does not match requested optimizer chart')
    mapped = [_renamed(name) for name in names]
    target_pairs = list(converted_model.named_parameters())
    if mapped != [name for name, p in target_pairs]:
        raise ValueError('Native saved-group/named-parameter order differs from the exact converted traversal')
    for (old_name, old), (new_name, new) in zip(pairs, target_pairs):
        if old.shape != new.shape or old.dtype != new.dtype or old.device != new.device:
            raise ValueError('Native/converted parameter shape dtype or device differs')
        if not old_name.startswith(PATH + '.') and not torch.equal(old.detach(), new.detach()):
            raise ValueError('Non-Sinc parameter changed during coordinate conversion')
    sinc_indices = [i for i, name in enumerate(mapped) if name in (PATH + '.low_coordinate', PATH + '.band_coordinate')]
    if len(sinc_indices) != 2 or (source_mode == 'CENTER_25K' and any(names[i].endswith('_hz_') for i in sinc_indices)):
        raise ValueError('Both source Sinc coordinates and explicit source chart must match')
    group = groups[0]
    required = {'params', 'lr', 'betas', 'eps', 'weight_decay', 'amsgrad', 'maximize',
                'foreach', 'capturable', 'differentiable', 'fused'}
    if (not required.issubset(group) or not isinstance(group.get('params'), list)
            or any(type(i) is not int for i in group['params'])
            or group['params'] != list(range(len(pairs)))):
        raise ValueError('Saved native group options/parameter IDs/order are incomplete or unsupported')
    if group['differentiable'] or (group['capturable'] and target_pairs[0][1].device.type == 'cpu'):
        raise ValueError('Differentiable or CPU-capturable AdamW is unsupported')
    for key in ('lr', 'eps', 'weight_decay'):
        value = group[key]
        if isinstance(value, torch.Tensor):
            if value.numel() != 1 or not bool(torch.isfinite(value)) or bool(value < 0):
                raise ValueError('Invalid saved AdamW scalar option')
        elif isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise ValueError('Invalid saved AdamW scalar option')
    if (not isinstance(group['betas'], (tuple, list)) or len(group['betas']) != 2
            or any(isinstance(b, bool) or not isinstance(b, (int, float)) or not math.isfinite(b) or not 0 <= b < 1 for b in group['betas'])):
        raise ValueError('Invalid saved AdamW betas')
    for key in ('amsgrad', 'maximize', 'capturable', 'differentiable'):
        if not isinstance(group[key], bool):raise ValueError('Malformed saved AdamW boolean option')
    for key in ('foreach', 'fused'):
        if group[key] is not None and not isinstance(group[key], bool):raise ValueError('Malformed saved AdamW backend option')
    if 'param_names' in group and group['param_names'] != names:
        raise ValueError('Saved explicit parameter names disagree with supplied traversal')
    states = saved_state['state']
    if not isinstance(states, Mapping) or any(isinstance(i, bool) or not isinstance(i, int) or i not in group['params'] for i in states):
        raise ValueError('Saved state parameter IDs differ from the original group')
    if any(i not in states for i in sinc_indices):
        raise ValueError('Saved actual Sinc moments are absent; do not fabricate/reset them')
    transformed = copy.deepcopy(saved_state)
    ratio = (SCALE_HZ if mode == 'CENTER_25K' else 1.) / (SCALE_HZ if source_mode == 'CENTER_25K' else 1.)
    required_state = {'step', 'exp_avg', 'exp_avg_sq'} | ({'max_exp_avg_sq'} if group['amsgrad'] else set())
    for index, state in states.items():
        if (not isinstance(state, Mapping) or not required_state.issubset(state)
                or set(state) - {'step', 'exp_avg', 'exp_avg_sq', 'max_exp_avg_sq'}):
            raise ValueError('Present saved AdamW state is incomplete/unsupported')
        step = state['step']
        if not isinstance(step, torch.Tensor) or step.ndim != 0 or not bool(torch.isfinite(step)) or bool(step < 0) or float(step) != int(step):
            raise ValueError('A real finite integral saved AdamW scalar step tensor is required')
        param = pairs[index][1]
        for key in ('exp_avg', 'exp_avg_sq', 'max_exp_avg_sq'):
            if key not in state:continue
            value = state[key]
            _finite_tensor(value, param.shape, f'state[{index}].{key}')
            if value.dtype != param.dtype or (key != 'exp_avg' and bool((value < 0).any())):
                raise ValueError('Saved moment dtype/value differs from native AdamW')
            if index in sinc_indices:
                transformed['state'][index][key] = value.detach().clone() * (ratio if key == 'exp_avg' else ratio**2)
                if not bool(torch.isfinite(transformed['state'][index][key]).all()):
                    raise ValueError('Fixed coordinate moment conversion overflowed')
    if 'param_names' in transformed['param_groups'][0]:
        transformed['param_groups'][0]['param_names'] = mapped
    target_group = copy.deepcopy(transformed['param_groups'][0])
    target_group['params'] = [p for name, p in target_pairs]
    optimizer = torch.optim.AdamW([target_group])
    optimizer.load_state_dict(transformed)
    return optimizer
