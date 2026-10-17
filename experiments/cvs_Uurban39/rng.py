from contextlib import contextmanager
import torch
DROPOUT_SEED=20261018442
@contextmanager
def private_rng(x,step):
 devices=[x.device.index if x.device.index is not None else torch.cuda.current_device()] if x.is_cuda else []
 with torch.random.fork_rng(devices=devices):
  torch.random.default_generator.manual_seed(DROPOUT_SEED+step)
  if x.is_cuda:torch.cuda.manual_seed(DROPOUT_SEED+step)
  yield
