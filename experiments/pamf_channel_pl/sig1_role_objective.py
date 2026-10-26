"""Replace mixed ordinary slots with two restored views; keep CE dose and RX terms."""
from contextlib import contextmanager
from types import MethodType
import torch
from experiments.cric_dura_comparison.io import read
from .restored_source import SCENES,source_channel,snapshot,unchanged
from .sig1_repl_offset_objective import NoStyleNoRXE

METHODS=('REPL_G_U','REPL_G_L','REPL_G_LU')
ROLES=dict(REPL_G_U=(False,True),REPL_G_L=(True,False),REPL_G_LU=(True,True))


def view_plan(step,role,view):
    if role not in ('L','U') or view not in ('a','b') or type(step) is not int or step<=4440:raise ValueError('New E21+ paired source views only')
    epoch=(step-1)//222+1;scene=SCENES[(step-1+int(view=='b'))%3]
    return scene,'PAMF/SIG1/role_train/'+role+'/E'+str(epoch)+'/'+view


def mask(n,step,device):
    if n<=0 or n%2 or type(step) is not int or step<=0:raise ValueError('Even original batch and positive ordered step required')
    return (torch.arange(n,device=device)+step).remainder(2).bool()


def replace_ordinary(original,restored,physical):
    if (original.ndim!=3 or original.shape[1:]!=(2,256) or physical.shape!=(len(original),) or physical.dtype!=torch.bool
            or restored.shape!=(int((~physical).sum()),2,256) or not bool(torch.isfinite(restored).all())):
        raise ValueError('Only aligned ordinary half may be replaced')
    value=original.clone();value[~physical]=restored.detach()
    if not torch.equal(value[physical],original[physical]):raise ValueError('Physical As/raw half changed')
    return value


class RestoredRoles(NoStyleNoRXE):
    def __init__(self,c,g,engine):
        super().__init__(c);self.enable_L,self.enable_U=ROLES[c['paper_method']]
        self.g=g;self.engine=engine;self.contract=read(c['source_contract'])
        self.L_calls=self.L_replaced=self.U_replaced=self.G_calls=self.G_packets=0
        self.active_L=False;self.L_forward_index=0;self.current_raw_U=None
        self.last_views={};self.last_diversity={};self.g_before=snapshot(g)

    def validate_unlabeled_batch(self,batch):
        super().validate_unlabeled_batch(batch)
        if bool(batch[3]['tx_label_visible'].any()):raise ValueError('U true TX must remain hidden')
        self.current_raw_U=batch[0]

    def restored(self,raw,metadata,physical,step,role,view):
        scene,namespace=view_plan(step,role,view)
        n=len(raw);visible=metadata['tx_label_visible']
        if bool(visible.all())!=(role=='L') or (role=='U' and bool(visible.any())):raise ValueError('Role and TX visibility differ')
        sub={k:v[(~physical).to(v.device)] if torch.is_tensor(v) and v.ndim and len(v)==n else v for k,v in metadata.items()}
        with torch.no_grad():
            h=source_channel(raw[~physical].detach(),sub,scene,self.contract,namespace=namespace)
            out=self.engine(h.clone())[0].clone().detach()
        if out.shape!=(n//2,2,256) or not bool(torch.isfinite(out).all()):raise ValueError('Finite exact source restored half required')
        self.G_calls+=1;self.G_packets+=len(out)
        if role=='L':self.L_replaced+=len(out)
        else:self.U_replaced+=len(out)
        if self.c.get('candidate_smoke'):
            if view=='a':self.last_views[role]=out.detach().clone()
            else:
                delta=float((out-self.last_views[role]).norm()/self.last_views[role].norm().clamp_min(1e-12))
                if delta<=1e-7:raise ValueError('Different channel realizations must yield different actual G views')
                self.last_diversity[role]=delta
        return out

    @contextmanager
    def labeled(self,student,kwargs):
        if self.active_L or kwargs['labeled_count']!=128:raise ValueError('One ordered L128 hook only')
        self.active_L=True;self.L_forward_index=0;original=student.forward_identity_only
        raw=kwargs['labeled_inputs'].detach();metadata=kwargs['extra'][1];step=kwargs['optimizer_step']+self.c['step_offset']
        physical=mask(128,kwargs['optimizer_step'],raw.device)
        def forward(self_model,value,*args,**options):
            # Five original L forwards precede the unchanged six U forwards.
            if self.L_forward_index<5:
                index=self.L_forward_index;self.L_forward_index+=1
                if len(value)!=128:raise ValueError('Original L128 forward order differs')
                if self.enable_L and index in (1,2):
                    restored=self.restored(raw,metadata,physical,step,'L','a' if index==1 else 'b')
                    value=replace_ordinary(value,restored,physical)
            return original(value,*args,**options)
        student.forward_identity_only=MethodType(forward,student)
        try:
            yield
            if self.L_forward_index!=5:raise ValueError('Original ordinary/mixed/raw L forward count differs')
            self.L_calls+=1
        finally:
            student.forward_identity_only=original;self.active_L=False

    def unlabeled_student_forward(self,student,x,domain,pseudo,accepted,metadata):
        index=self._joint_forward_index+1
        if self.enable_U and index in (3,4):
            if self.current_raw_U is None or len(self.current_raw_U)!=len(x):raise ValueError('Aligned raw U required')
            raw=self.current_raw_U.to(x.device);step=student._feature_l_steps+self.c['step_offset'];physical=mask(len(x),step,x.device)
            restored=self.restored(raw,metadata,physical,step,'U','a' if index==3 else 'b')
            x=replace_ordinary(x,restored,physical)
        return super().unlabeled_student_forward(student,x,domain,pseudo,accepted,metadata)

    def guide(self,labeled,student):
        value=super().guide(labeled,student)
        self.metrics.update(RG_L_calls=self.L_calls,RG_L_replaced=self.L_replaced,RG_U_replaced=self.U_replaced,
            RG_G_calls=self.G_calls,RG_G_packets=self.G_packets,RG_extra_F_forwards=0,RG_G_updates=0)
        return value

    def state_dict(self):
        state=super().state_dict()
        if self.active_L or self.L_calls!=self.calls:raise ValueError('One full L/U update required')
        state['restored_roles']=dict(enable_L=self.enable_L,enable_U=self.enable_U,L_calls=self.L_calls,L_replaced=self.L_replaced,
            U_replaced=self.U_replaced,G_calls=self.G_calls,G_packets=self.G_packets,G_warmup_packets=2,
            extra_F_forwards=0,G_updates=0,As_and_raw_halves_preserved=True,source_G_unchanged=unchanged(self.g,self.g_before),
            class_CE_labels='all true L;raw95 pseudo accepted U',channel_realizations='a/b fixed per physical packet and cumulative epoch',
            scene_schedule='a:(step-1)%3;b:step%3;independenta/brealizations',postG_noise=False,lowconfidence_extra_loss=False)
        if not state['restored_roles']['source_G_unchanged']:raise ValueError('Frozen source G changed')
        return state
