from . import design as d
from .losses import objective
from experiments.pamf_r46_anchor_refine.models import CompactSuccessor as Skeleton,compact_from_parent as prune


def CompactSuccessor(arm):
    if arm not in d.ARMS:raise ValueError('Fixed reconstruction-tail arm required')
    model=Skeleton('prune_decay');model.arm=arm;return model


def compact_from_parent(parent,arm):
    if arm not in d.ARMS:raise ValueError('Fixed tail arm required before source initialization')
    model=prune(parent,'prune_decay');model.arm=arm
    model.initialization_report.update(mode=d.MODE[arm],tail_fraction=d.FRACTION[arm],tail_mix=d.MIX,
        ranking_source='current source reconstruction only',external_student_checkpoint_reads=0,new_calibration_IQ=0,new_trainable_parameters=0)
    return model
