"""Source-only eval diagnostics without perturbing training RNG or mode."""
import random
import numpy as np
import torch
def preserving_rng(model,loader,device,evaluate):
 mode=model.training;python_rng=random.getstate();numpy_rng=np.random.get_state()
 generator=getattr(loader,'generator',None);loader_rng=generator.get_state() if generator is not None else None
 devices=[torch.cuda.current_device()] if str(device).startswith('cuda') else []
 try:
  with torch.random.fork_rng(devices=devices):return evaluate(model,loader,device)
 finally:
  random.setstate(python_rng);np.random.set_state(numpy_rng)
  if generator is not None:generator.set_state(loader_rng)
  model.train(mode)
