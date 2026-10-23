"""The same ordinary hard labels and view losses; guard their stronger parameter gradients."""
import torch
from .hard_pl_objective import HardPLObjective
from .hard_pl_no_as_mixstyle_objective import NoAsMixStyleObjective
from .u_gradprotect_objective import protected_loss


class Guarded:
    def __init__(self,c):
        super().__init__(c);self.pending_gradient_loss=None;self.guard_calls=0
        self.guard_conflicts={g:0 for g in ('t2','cls_head','other')};self.guard_caps=dict(self.guard_conflicts)

    def loss(self,student,device):
        if self.pending_gradient_loss is not None:raise ValueError('Previous U gradient loss was not consumed')
        self.pending_gradient_loss=super().loss(student,device)
        return self.pending_gradient_loss.detach()*0.

    def guide(self,labeled,student):
        if self.pending_gradient_loss is None:raise ValueError('Current hidden-label U loss required')
        unlabeled,self.pending_gradient_loss=self.pending_gradient_loss,None
        named=[(n,p) for n,p in student.named_parameters() if p.requires_grad]
        value,details,gradients=protected_loss(labeled,unlabeled,named);self.guard_calls+=1
        self.metrics['HPL_GP_calls']=self.guard_calls
        for group,items in details.items():
            self.guard_conflicts[group]+=int(items['conflict']);self.guard_caps[group]+=int(items['capped'])
            for name,number in items.items():self.metrics['HPL_GP_'+group+'_'+name]=float(number)
            self.metrics['HPL_GP_'+group+'_conflicts']=self.guard_conflicts[group]
            self.metrics['HPL_GP_'+group+'_caps']=self.guard_caps[group]
        if self.c.get('hard_pl_smoke'):
            self.smoke_guard_expected={n:g for (n,_),g in zip(named,gradients)};self.smoke_guard_details=details
        return value

    def state_dict(self):
        state=super().state_dict()
        state['U_gradient_guard']=dict(recipe=self.c['hard_pl_gradient_guard'],calls=self.guard_calls,
            conflicts=dict(self.guard_conflicts),caps=dict(self.guard_caps),pending_U_loss=self.pending_gradient_loss is not None)
        return state


class GuardedHardPL(Guarded,HardPLObjective):pass
class GuardedNoAsMixStyle(Guarded,NoAsMixStyleObjective):pass


def objective(c):return (GuardedHardPL if c['hard_pl']['method']=='HPL2' else GuardedNoAsMixStyle)(c)
