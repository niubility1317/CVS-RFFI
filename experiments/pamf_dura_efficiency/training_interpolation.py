"""Use the verified fixed-grid implementation only while constructing training G."""
from .linear2x import install_switch


def train_with_slices(base,trainer,args,paths):
    native_factory=base.scratch_model
    def factory(seed):
        model=native_factory(seed)
        install_switch(model)('slice_linear2x')
        model.initialization_report['training_execution']='12Corefixed_halfpixel_linear2x_slices;nativefactoryrestoredaftertrain'
        return model
    base.scratch_model=factory
    try:return trainer(args,paths)
    finally:base.scratch_model=native_factory
