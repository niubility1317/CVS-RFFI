"""Opt-in bounded profiler for actual training; no RNG or model changes."""
from contextlib import nullcontext
from contextvars import ContextVar
from functools import wraps
import json
from pathlib import Path
import time
import warnings

import torch

_active = ContextVar('cvs_execution_profiler', default=False)
_owners = ContextVar('cvs_execution_profile_owners', default=None)


def close_profiles_on_exit(function):
    @wraps(function)
    def wrapped(*args, **kwargs):
        owners=[]
        token=_owners.set(owners)
        try:
            return function(*args, **kwargs)
        finally:
            # Never replace a training exception with a trace-export exception.
            for owner in owners:
                try:
                    owner.close()
                except Exception as error:
                    warnings.warn(f'Execution profile close failed: {error}')
            _owners.reset(token)
    return wrapped


def stage(name):
    return torch.profiler.record_function('cvs/' + name) if _active.get() else nullcontext()


class TrainingProfile:
    def __init__(self, output, epoch=1, start=1, steps=0):
        self.output = Path(output) / 'execution_profile'
        self.epoch, self.start, self.steps = int(epoch), int(start), int(steps)
        self.profiler = None
        self.token = None
        self.count = 0
        self.finished = False
        if self.epoch < 1 or self.start < 1 or self.steps < 0:
            raise ValueError('Invalid execution profiling window')
        if _owners.get() is not None:
            _owners.get().append(self)

    def begin(self, epoch, step):
        if self.finished or self.steps <= 0 or int(epoch) != self.epoch or int(step) < self.start:
            return
        if self.profiler is None:
            self.output.mkdir(parents=True, exist_ok=False)
            activities = [torch.profiler.ProfilerActivity.CPU]
            if torch.cuda.is_available():
                activities.append(torch.profiler.ProfilerActivity.CUDA)
            self.profiler = torch.profiler.profile(activities=activities, record_shapes=True)
            self.profiler.__enter__()
            self.token = _active.set(True)
            self.started = time.perf_counter()

    def end(self):
        if self.profiler is None:
            return
        self.profiler.step()
        self.count += 1
        if self.count >= self.steps:
            self.close()

    def close(self):
        if self.profiler is None:
            return
        profiler, self.profiler = self.profiler, None
        try:
            profiler.__exit__(None, None, None)
        finally:
            _active.reset(self.token)
        profiler.export_chrome_trace(str(self.output / 'trace.json'))
        rows = [dict(name=e.key, count=e.count, cpu_total_us=e.cpu_time_total,
                     cpu_self_us=e.self_cpu_time_total) for e in profiler.key_averages()]
        (self.output / 'summary.json').write_text(json.dumps(dict(
            epoch=self.epoch, start_step=self.start, steps=self.count,
            wall_seconds=time.perf_counter()-self.started,
            scope='Actual training window; profiling overhead included; GPU timing in trace.json',
            rows=rows), indent=2)+'\n', encoding='utf-8')
        self.finished = True
