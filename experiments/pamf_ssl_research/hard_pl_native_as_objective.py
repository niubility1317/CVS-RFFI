"""Original mixed-half native As input; no extra teacher,loss,student pass or view gate."""
import torch
from .hard_pl_joint_objective import JointNoAsMixStyle
from .hard_pl_guard_objective import Guarded


class NativeAs(JointNoAsMixStyle):
    def __init__(self,c):
        super().__init__(c);self.nas_calls=self.nas_generated=self.nas_accepted=0

    def mixed_view(self,student,x,ordinary,physical,metadata,epoch,step,pseudo,accepted):
        if self.c.get('hard_pl_no_uas') or self.style_active or self.joint_active:
            raise ValueError('Native sourceAs generation must precede transient student contexts')
        selected={k:v[physical] if torch.is_tensor(v) and v.ndim and len(v)==len(x) else v for k,v in metadata.items()}
        simulated=student._cd_practical(x[physical],selected,epoch,step)
        if simulated.shape!=x[physical].shape or not bool(torch.isfinite(simulated).all()):
            raise ValueError('Finite aligned native sourceAs IQ required')
        mixed=ordinary.clone();mixed[physical]=simulated
        self.nas_calls+=1;self.nas_generated+=int(physical.sum());self.nas_accepted+=int(accepted[physical].sum())
        if self.c.get('hard_pl_smoke'):
            self.smoke_native_As_matches=bool(torch.equal(mixed[physical],simulated))
            self.smoke_native_nonphysical_ordinary_matches=bool(torch.equal(mixed[~physical],ordinary[~physical]))
        return mixed

    def loss(self,student,device):
        value=super().loss(student,device)
        self.metrics.update(HPL_NAS_calls=self.nas_calls,HPL_NAS_generated=self.nas_generated,
            HPL_NAS_raw_accepted_half=self.nas_accepted,HPL_NAS_accepted_student_views=2*self.nas_accepted,
            HPL_NAS_extra_As_teacher=0,HPL_NAS_raw_fallback=0,HPL_NAS_CE_normalizer=1.25)
        return value

    def state_dict(self):
        s=super().state_dict()
        if (self.nas_calls!=self.calls or self.nas_generated!=self.seen//2 or self.physical_seen!=self.nas_generated
            or self.physical_student_views!=2*self.nas_generated or self.raw_As_seen!=0
            or not 0<=self.nas_accepted<=min(self.accepted,self.nas_generated)):
            raise ValueError('Native As actual generation,student or original raw acceptance counts differ')
        s['U_native_As']=dict(recipe=self.c['hard_pl_native_as'],calls=self.nas_calls,generated=self.nas_generated,
            student_physical_views=self.physical_student_views,raw_accepted_half=self.nas_accepted,
            accepted_student_views=2*self.nas_accepted,raw_fallback=0,extra_As_teacher=0,
            all_raw_accepted_retained=True,extra_student_forwards=0,new_head=False,EMA_updates=0,U_truth_read=False,CE_dose_normalizer=1.25)
        return s


class GuardedNativeAs(Guarded,NativeAs):pass
