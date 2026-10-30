"""Fixed twofold, half-pixel linear interpolation for the four DURA Core paths."""
import torch


def linear2x(value):
    # Each original decoder doubles a power-of-two time axis. Preserve the
    # clamped first/last samples and both interior half-pixel weights.
    even=torch.cat((value[...,:1],.25*value[...,:-1]+.75*value[...,1:]),-1)
    odd=torch.cat((.75*value[...,:-1]+.25*value[...,1:],value[...,-1:]),-1)
    return torch.stack((even,odd),-1).flatten(-2)


def install_switch(model):
    from experiments.pamf_reception_restore.models import Core

    class SlicedCore(Core):
        def forward(self,value,condition=None,time=None):
            if self.conditional or condition is not None or time is not None:
                raise ValueError('Only the existing unconditional DURA Core paths are supported')
            features=self.encoder(value)
            value=features[-1]
            for block,skip in zip(self.decode,features[-2::-1]):
                if skip.shape[-1]!=2*value.shape[-1]:raise ValueError('Original twofold decoder grid required')
                value=block(torch.cat((linear2x(value),skip),1))
            return self.out(value)

    cores=[model.frozen.coarse.spectral,model.frozen.coarse.temporal,*model.prior]
    if len(cores)!=4 or any(type(core) is not Core or core.conditional for core in cores):
        raise ValueError('Only original full two-stage DURA with four unconditional Cores supported')

    def switch(mode):
        if mode not in ('native_linear','slice_linear2x'):raise ValueError('Undeclared interpolation case')
        for core in cores:core.__class__=Core if mode=='native_linear' else SlicedCore
    return switch
