"""Parameter-only means. No forward, labels, losses, validation, or model choice."""
import copy
import torch
from . import design as d


class SourceParameterMeans:
    def __init__(self,model):
        self.model=model
        self.parameters=dict(model.named_parameters())
        if not self.parameters or any(p.dtype!=torch.float32 for p in self.parameters.values()):
            raise ValueError('Nonempty fixed FP32 source parameters required')
        first={id(p):n for n,p in self.parameters.items()}
        self.aliases={n:first[id(p)] for n,p in model.named_parameters(remove_duplicate=False)}
        self.buffers={n:b.detach().clone() for n,b in model.named_buffers(remove_duplicate=False)}
        self.last_update=self.last_epoch=self.ema_count=self.mean_count=0
        self.ema=self.mean=None

    def _check_model(self,model):
        if model is not self.model:raise ValueError('Only the same source trajectory may update or export means')
        current=dict(model.named_parameters())
        if list(current)!=list(self.parameters) or any(current[n] is not p for n,p in self.parameters.items()):
            raise ValueError('Source graph parameters changed during averaging')

    @torch.no_grad()
    def after_step(self,model,global_update):
        self._check_model(model)
        if type(global_update) is not int or global_update!=self.last_update+1 or not 1<=global_update<=d.UPDATES:
            raise ValueError('Every consecutive completed source update required')
        self.last_update=global_update
        if global_update>(d.EMA_START_EPOCH-1)*d.UPDATES_PER_EPOCH:
            if self.ema is None:self.ema={n:p.detach().clone() for n,p in self.parameters.items()}
            else:
                for n,p in self.parameters.items():self.ema[n].lerp_(p.detach(),1-d.EMA_DECAY)
            self.ema_count+=1

    @torch.no_grad()
    def after_epoch(self,model,epoch):
        self._check_model(model)
        if type(epoch) is not int or epoch!=self.last_epoch+1 or self.last_update!=epoch*d.UPDATES_PER_EPOCH or not 1<=epoch<=20:
            raise ValueError('Only consecutive full source epochs may take snapshots')
        self.last_epoch=epoch
        if epoch in d.MEAN_EPOCHS:
            if self.mean is None:self.mean={n:p.detach().clone() for n,p in self.parameters.items()}
            else:
                for n,p in self.parameters.items():self.mean[n].lerp_(p.detach(),1/(self.mean_count+1))
            self.mean_count+=1

    @torch.no_grad()
    def state_for(self,model,readout):
        self._check_model(model)
        if readout not in d.ARMS or (self.last_update,self.last_epoch,self.ema_count,self.mean_count)!=(4940,20,2470,5):
            raise ValueError('All three fixed readouts require the entire single source path')
        current=dict(model.named_buffers(remove_duplicate=False))
        if set(current)!=set(self.buffers) or any(not torch.equal(current[n],b) for n,b in self.buffers.items()):
            raise ValueError('Only fixed buffers;no hidden normalization or calibration updates allowed')
        values=self.parameters if readout=='final' else self.mean if readout=='late_mean' else self.ema
        if any(not torch.isfinite(p).all() for p in values.values()):raise ValueError('Finite frozen readout parameters required')
        state={n:values[first].detach().clone() for n,first in self.aliases.items()}
        state.update({n:b.detach().clone() for n,b in self.buffers.items()})
        if set(state)!=set(model.state_dict()):raise ValueError('Complete portable source state required')
        return state

    def metadata(self,readout):
        if readout not in d.ARMS:raise ValueError('Fixed readout required')
        return dict(readout=readout,rule=copy.deepcopy(d.configuration()['readouts'][readout]),
            shared_source_epochs=self.last_epoch,shared_source_updates=self.last_update,
            ema_updates=self.ema_count,late_epoch_snapshots=self.mean_count,
            unique_parameters_averaged=len(self.parameters),parameter_aliases=len(self.aliases),
            buffers_averaged=False,teacher_changed=False,extra_neural_forwards=0,
            new_trainable_parameters=0,TX_reads=0,F_reads=0,V_reads=0,target_reads=0)
