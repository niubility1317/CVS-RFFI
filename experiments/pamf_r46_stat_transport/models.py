from . import design as d
from .normalization import convert_normalization
from experiments.pamf_r46_fold_compensate.models import CompactSuccessor as Existing

def CompactSuccessor(arm):
    if arm not in d.ARMS:raise ValueError('Registered statistic-transport arm required')
    model=convert_normalization(Existing('fold_full'));model.arm=arm
    if sum(p.numel() for p in model.parameters())!=d.PARAMETERS[arm]:raise ValueError('Full statistic-transport parameter count differs')
    return model
