"""Explicit compatibility helpers; upstream sources remain untouched."""
import copy
import importlib.util
import subprocess
from pathlib import Path
import numpy as np
import torch
from scipy import signal

PINS = {'ISSL': 'f58eda144063c7f150fbb7c0d0f31aaccb9d3bbe',
        'LoRa_RFFI': '00948769c05487315b88b7edf40d903fcf8e5b7d',
        'LoRa_RFFI_Torch': 'cc6e0e68c52d06869c8382cec0684a4cf651ee76'}

def verify_source(path, method):
    path = Path(path)
    head = subprocess.check_output(['git', '-C', str(path), 'rev-parse', 'HEAD'], text=True).strip()
    dirty = subprocess.check_output(['git', '-C', str(path), 'status', '--porcelain', '--untracked-files=no'], text=True)
    if head != PINS[method] or dirty.strip():
        raise ValueError(f'Unexpected or modified {method} source: {head}')

def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def expand_head(model, new_classes):
    # increment_model.increment_classes: keep default Linear initialization and
    # bias=False. Its kaiming_normal_init(Parameter) is a no-op, preserved here.
    if new_classes <= 0: raise ValueError('new_classes must be positive')
    teacher = copy.deepcopy(model)
    old = model.fc
    model.fc = torch.nn.Linear(old.in_features, old.out_features + new_classes,
                               bias=False).to(old.weight.device)
    with torch.no_grad(): model.fc.weight[:old.out_features].copy_(old.weight)
    return model, teacher

def validate_split(train_ids, query_ids):
    if len(set(train_ids)) != len(train_ids) or len(set(query_ids)) != len(query_ids):
        raise ValueError('Duplicate physical IDs')
    if set(train_ids) & set(query_ids): raise ValueError('Training/query physical-ID overlap')

def channel_spectrogram(iq, window=256, overlap=128):
    # Same operations as author preprocessing, with explicit short-IQ adaptation.
    iq = np.asarray(iq)
    if iq.ndim != 3 or iq.shape[-1] != 2: raise ValueError('Expected N,L,2 IQ')
    if not 0 <= overlap < window or iq.shape[1] < 2 * window - overlap:
        raise ValueError('Need at least two STFT frames for adjacent-frame ratio')
    x = iq[..., 0] + 1j * iq[..., 1]
    rms = np.sqrt(np.mean(np.abs(x)**2, axis=1, keepdims=True))
    if not np.isfinite(x).all() or (rms <= 0).any(): raise ValueError('Invalid IQ')
    x = x / rms
    _, _, spec = signal.stft(x, window='boxcar', nperseg=window,
        noverlap=overlap, nfft=window, return_onesided=False, padded=False, boundary=None)
    spec = np.fft.fftshift(spec, axes=1)
    with np.errstate(divide='ignore', invalid='ignore'):
        values = np.log10(np.abs(spec[..., 1:] / spec[..., :-1])**2)
    values = values[:, round(window*.3):round(window*.7), :, None]
    if not np.isfinite(values).all(): raise ValueError('Non-finite spectrogram; no silent clipping')
    return values.astype(np.float32)
