"""Runtime support for local repairs; no target data or training policy."""
import csv
import json
import math
import time
from pathlib import Path

import torch


def resolve_device(device):
    if isinstance(device, int):
        device = f'cuda:{device}' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError(f'Requested device unavailable: {device}')
    return device


def save_state(model, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)


def load_state(model, path, device, allow_legacy=False):
    # Default uses safe state_dict loading. Legacy whole-model loading is opt-in.
    state = torch.load(path, map_location=device, weights_only=not allow_legacy)
    if isinstance(state, torch.nn.Module):
        state = state.state_dict()
    if not isinstance(state, dict):
        raise TypeError('Expected a state_dict or an explicitly allowed legacy model')
    model.load_state_dict(state, strict=True)
    return model.to(device)


def gradient_norm(model):
    terms = [p.grad.detach().abs().square().sum().item()
             for p in model.parameters() if p.grad is not None]
    return math.sqrt(sum(terms))


class Metrics:
    """Keep measured step/epoch metrics as text, JSONL and compact CSV."""
    def __init__(self, directory, config):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.step_path = self.directory / 'steps.jsonl'
        self.epoch_path = self.directory / 'epochs.jsonl'
        self.csv_path = self.directory / 'epochs.csv'
        self.text_path = self.directory / 'training.log'
        # Reusing a completed log would mix runs; caller must choose a fresh run.
        if any(p.exists() for p in (self.step_path, self.epoch_path, self.csv_path, self.text_path)):
            raise FileExistsError(f'Metrics already exist: {self.directory}')
        (self.directory / 'resolved_config.json').write_text(
            json.dumps(config, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
        self.emit('CONFIG ' + json.dumps(config, ensure_ascii=False, allow_nan=False))

    def emit(self, message):
        print(message, flush=True)
        with self.text_path.open('a', encoding='utf-8') as f:
            f.write(message + '\n')

    def step(self, record):
        with self.step_path.open('a', encoding='utf-8') as f:
            f.write(json.dumps(record, allow_nan=False) + '\n')
        self.emit('STEP ' + json.dumps(record, allow_nan=False))

    def epoch(self, record):
        with self.epoch_path.open('a', encoding='utf-8') as f:
            f.write(json.dumps(record, allow_nan=False) + '\n')
        first = not self.csv_path.exists()
        with self.csv_path.open('a', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(record))
            if first:
                writer.writeheader()
            writer.writerow(record)
        self.emit('EPOCH ' + json.dumps(record, allow_nan=False))


def optional_writer():
    try:
        from torch.utils.tensorboard import SummaryWriter
        return SummaryWriter
    except ImportError:
        class NullWriter:
            def __init__(self, *args, **kwargs):
                pass
            def add_scalar(self, *args, **kwargs):
                pass
            def close(self):
                pass
        return NullWriter


def timed_start(device):
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    return time.perf_counter()


def elapsed(start, device):
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    return time.perf_counter() - start
