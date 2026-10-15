"""One t2 dropout change and a read-only same-trajectory tail average."""
import torch
from torch import nn
from experiments.cvs_feature_restitution.features import layout

MODES=('ELEMENT','NONE','LOCKED')
TAIL_EPOCHS=(17,18,19,20)


def install_dropout(model,mode):
    if mode not in MODES:raise ValueError('Fixed ELEMENT/NONE/LOCKED mode required')
    core,prefix,_=layout(model);original=core.t2.drop
    if type(original) is not nn.Dropout or original.p!=.1 or original.inplace:
        raise ValueError('Expected the original elementwise non-inplace t2 Dropout(0.10)')
    replacement=original if mode=='ELEMENT' else nn.Identity() if mode=='NONE' else nn.Dropout1d(original.p)
    replacement.train(original.training);core.t2.drop=replacement
    return dict(mode=mode,layer=prefix+'.t2.drop',probability=0. if mode=='NONE' else original.p,
        shared_over_time=mode=='LOCKED',signature_tap='unchanged t2 output after pool/drop',
        extra_parameters=0,additional_IQ_forwards=0,default_head_changed=False)


class TailAverage:
    """Average all model parameters in FP64, including the default head/gains.

    The caller uses one instance per scratch trajectory. Buffers must remain
    constant: the declared GroupNorm/LayerNorm model needs no BN recalibration.
    No state is loaded into the live student or its own EMA teacher.
    """
    def __init__(self,trajectory_id):
        if not isinstance(trajectory_id,str) or not trajectory_id:raise ValueError('One named trajectory required')
        self.trajectory_id=trajectory_id;self.epochs=[];self.sums={};self.buffers={};self.dtypes={}

    @torch.no_grad()
    def add(self,model,epoch,*,trajectory_id):
        if trajectory_id!=self.trajectory_id:raise ValueError('Cannot average across trajectories/seeds')
        if epoch not in TAIL_EPOCHS:return
        if len(self.epochs)>=len(TAIL_EPOCHS) or epoch!=TAIL_EPOCHS[len(self.epochs)]:
            raise ValueError('Exactly the four consecutive fixed tail epochs are required')
        if any(isinstance(module,nn.modules.batchnorm._BatchNorm) for module in model.modules()):
            raise ValueError('BN/running-statistics models are outside this fixed export')
        parameters=dict(model.named_parameters());state=model.state_dict()
        if not set(parameters)<=set(state):raise ValueError('All parameters must be in the exported state')
        if self.epochs and set(state)!=set(self.sums)|set(self.buffers):raise ValueError('Model state changed within the trajectory')
        for name,value in state.items():
            current=value.detach().cpu()
            if not torch.isfinite(current).all():raise ValueError('Finite tail state required')
            if name in parameters:
                if not (current.is_floating_point() or current.is_complex()):raise ValueError('Floating model parameters required')
                if not self.epochs:
                    self.dtypes[name]=current.dtype
                    self.sums[name]=current.to(torch.complex128 if current.is_complex() else torch.float64).clone()
                else:
                    if current.dtype!=self.dtypes[name] or current.shape!=self.sums[name].shape:raise ValueError('Parameter coordinate/dtype changed')
                    self.sums[name].add_(current.to(self.sums[name].dtype))
            elif not self.epochs:self.buffers[name]=current.clone()
            elif not torch.equal(current,self.buffers[name]):raise ValueError('A state buffer changed; no implicit statistics recalibration')
        self.epochs.append(epoch)

    @torch.no_grad()
    def state_dict(self):
        if tuple(self.epochs)!=TAIL_EPOCHS:raise ValueError('The full preregistered tail window is required')
        return dict({name:(value/len(TAIL_EPOCHS)).to(self.dtypes[name]) for name,value in self.sums.items()},
            **{name:value.clone() for name,value in self.buffers.items()})
