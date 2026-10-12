import torch
from experiments.cvs_report_ssl.sampling import DomainStream
class RotatingStreams:
 def __init__(self,data,seed,labeled=False):
  self.indices={r:(data['rx']==r).nonzero().flatten() for r in sorted(data['rx'].unique().tolist())}
  # DomainStream only sorts integer strata. Day then TX interleaves six TXs,
  # so every32 L draw covers eachTX5or6times. U never reads a TX label.
  self.stream={r:DomainStream(data['day'][ix],data['y'][ix],seed+101*r) if labeled else DomainStream(data['rx'][ix],data['day'][ix],seed+101*r) for r,ix in self.indices.items()}
 def batch(self,r,count):
  idx,passes=self.stream[r].batch(count);return self.indices[r][idx],passes
