"""Complete nativeAs losses;only U is constrained by all three supervised directions."""
import torch
from .hard_pl_native_as_objective import GuardedNativeAs
from .branch_gradient_projection import branch_projected_gradients,GROUPS,ANCHORS
from .labeled_branch_capture import capture,components,labeled_context_active
from .u_gradprotect_objective import _ExplicitGradient


def protected_branch_loss(labeled,unlabeled,named,clean,physical,*,capture_anchors=None):
    parameters=tuple(p for _,p in named)
    gl=torch.autograd.grad(labeled,parameters,retain_graph=True,allow_unused=True)
    gu=torch.autograd.grad(unlabeled,parameters,retain_graph=True,allow_unused=True)
    gc=torch.autograd.grad(clean,parameters,retain_graph=True,allow_unused=True)
    ga=torch.autograd.grad(physical,parameters,retain_graph=True,allow_unused=True)
    if capture_anchors is not None:capture_anchors(named,gl,gc,ga)
    gradients,details=branch_projected_gradients(named,gl,gu,gc,ga,max_u_ratio=2.)
    value=_ExplicitGradient.apply((labeled+unlabeled).detach(),gradients,*parameters)
    return value,details,gradients


class BranchGuardedNativeAs(GuardedNativeAs):
    def __init__(self,c):
        super().__init__(c);self.branch_calls=0
        self.branch_conflicts={group:{anchor:0 for anchor in ANCHORS} for group in GROUPS}

    def labeled_context(self):return capture()

    def guide(self,labeled,student):
        clean,physical=components(labeled)
        if self.pending_gradient_loss is None:raise ValueError('Current hidden-label native U loss required')
        unlabeled,self.pending_gradient_loss=self.pending_gradient_loss,None
        named=[(n,p) for n,p in student.named_parameters() if p.requires_grad]
        value,details,gradients=protected_branch_loss(labeled,unlabeled,named,clean,physical,
            capture_anchors=getattr(self,'capture_step_anchors',None))
        self.branch_calls+=1;self.guard_calls+=1
        self.metrics.update(HPL_GP_calls=self.guard_calls,HPL_GP_U_to_L_norm_cap=2.,
            HPL_BG_calls=self.branch_calls,HPL_BG_gradient_calls=4*self.branch_calls,
            HPL_BG_extra_autograd_calls=2*self.branch_calls,HPL_BG_L_pair_captures=4*self.branch_calls)
        for group,items in details.items():
            self.guard_conflicts[group]+=int(items['conflict']);self.guard_caps[group]+=int(items['capped'])
            for name in ('L_norm','U_norm','applied_U_norm','dot_before','dot_after','conflict','capped','tiny_L'):
                self.metrics['HPL_GP_'+group+'_'+name]=float(items[name])
            self.metrics['HPL_GP_'+group+'_conflicts']=self.guard_conflicts[group]
            self.metrics['HPL_GP_'+group+'_caps']=self.guard_caps[group]
            for anchor in ANCHORS:
                before=items['anchor_cosines_before'][anchor];after=items['anchor_cosines_after'][anchor]
                self.branch_conflicts[group][anchor]+=int(before is not None and before<-1e-6)
                for phase,number in (('before',before),('after',after)):
                    self.metrics['HPL_BG_'+group+'_'+anchor+'_cosine_'+phase]=0. if number is None else number
                self.metrics['HPL_BG_'+group+'_'+anchor+'_conflicts']=self.branch_conflicts[group][anchor]
        if self.c.get('hard_pl_smoke'):
            self.smoke_guard_expected={n:g for (n,_),g in zip(named,gradients)};self.smoke_guard_details=details
        return value

    def state_dict(self):
        state=super().state_dict()
        if self.guard_calls!=self.branch_calls:raise ValueError('One current branch guard per complete native source step required')
        state['U_branch_guard']=dict(recipe=self.c['hard_pl_branch_guard'],calls=self.branch_calls,
            gradient_calls=4*self.branch_calls,extra_autograd_calls=2*self.branch_calls,L_pair_captures=4*self.branch_calls,
            pending_capture=labeled_context_active(),pending_U_loss=self.pending_gradient_loss is not None,
            anchor_conflicts={g:dict(values) for g,values in self.branch_conflicts.items()},
            extra_teacher_or_student_forward=0,new_head=False,U_truth_reads=0,target_reads=0)
        return state
