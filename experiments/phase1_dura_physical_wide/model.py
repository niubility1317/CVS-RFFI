"""DURA/physical fusion with doubled complex correction range."""
from experiments.phase1_dura_physical_fusion.model import PhysicalFusion


class WidePhysicalFusion(PhysicalFusion):
 def forward(self,received,packed,log_variance):
  anchor=packed[:,:2]
  standard=super().forward(received,packed,log_variance)
  return anchor+2*(standard-anchor)
