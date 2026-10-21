"""Reuse the exact existing static compression graph; no new restorer family."""
from . import design as d
from experiments.pamf_r46_anchor_refine.models import CompactSuccessor as Skeleton,compact_from_parent as prune
from experiments.pamf_r46_multipoint_fast.models import objective,spectral_error


def CompactSuccessor(arm):
    if arm not in (*d.ARMS,d.SOURCE_ARM):raise ValueError('Fixed source path or final readout required')
    model=Skeleton('prune_decay');model.arm=arm;return model


def source_from_parent(parent):
    model=prune(parent,'prune_decay');model.arm=d.SOURCE_ARM
    model.initialization_report.update(mode='original_M35_static_prune_shared_source_path',
        fixed_original_maps=True,parameter_mean_readouts=list(d.ARMS),external_student_checkpoint_reads=0,
        new_calibration_IQ=0,new_trainable_parameters=0)
    return model
