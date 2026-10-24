"""Selected REPL arithmetic, with class-count portability and declared switches."""
from contextlib import contextmanager
import torch

def source_partners(pseudo,accepted,receivers,allowed,num_classes,dtype):
    from experiments.pamf_ssl_research.u_mixstyle_objective import protected_partners
    if num_classes not in (6,10) or bool(((pseudo<0)|(pseudo>=num_classes)).any()):raise ValueError('Registered class space required')
    p=torch.nn.functional.one_hot(pseudo,num_classes).to(dtype)
    return protected_partners(p,accepted.to(dtype),receivers,allowed,1.)

def portable_style_forward(self,student,x,domain,pseudo,accepted,metadata):
    from experiments.cric_dura_comparison.f_model import PREFIX
    from experiments.pamf_ssl_research.hard_pl_mixstyle_objective import style_hooks
    if self.style_active:raise ValueError('Nested source style context')
    n=len(x);recipe=self.c['hard_pl_mixstyle']
    if self.style_calls%6==0:
        rx=metadata['rx_i'].to(x.device).long()
        partner,valid=source_partners(pseudo,accepted,rx,self.c['source_rxs'],self.c['num_classes'],x.dtype)
        if bool((valid&(~accepted|~accepted[partner]|(pseudo!=pseudo[partner])|(rx==rx[partner]))).any()):raise ValueError('Cross-RX pseudo identity differs')
        alpha=torch.tensor(recipe['alpha'],device=x.device)
        lam=torch.distributions.Beta(alpha,alpha).sample((n,1,1))
        enabled={name:bool(torch.rand((),device=x.device)<recipe['probability_per_level']) for name in recipe['levels']}
        self.shared_plan=(partner,valid,lam,enabled);self.style_plan_calls+=1
    partner,valid,lam,enabled=self.shared_plan
    if len(partner)!=n:raise ValueError('Aligned six U views required')
    capture={} if self.c.get('hard_pl_smoke') else None;self.style_active=True
    try:
        with style_hooks(student,PREFIX,recipe,partner,valid,lam,force=bool(self.c.get('hard_pl_smoke')),
                         capture=capture,enabled_by_level=enabled,prepend=True) as applied:
            logits=student.forward_identity_only(x,domain_labels=domain)['tx_logits']
    finally:self.style_active=False
    self.style_calls+=1;self.style_seen+=n;self.style_eligible+=int(valid.sum())
    for name,v in applied.items():self.style_applied[name]+=v
    if capture is not None:self.style_smoke_maps=capture
    return logits

@contextmanager
def portable_scope():
    from experiments.pamf_ssl_research.hard_pl_no_as_mixstyle_objective import NoAsMixStyleObjective
    old=NoAsMixStyleObjective.unlabeled_student_forward
    NoAsMixStyleObjective.unlabeled_student_forward=portable_style_forward
    try:yield
    finally:NoAsMixStyleObjective.unlabeled_student_forward=old

def loader_config(c):
    # f_data.context historically chose 128 for matched PL. REPL and the matched
    # ordinary-PL E20 prefix require the original native 256-U loader, including
    # its partial last batches and independent same-seed iterator.
    return dict(c,regime='native_cric' if c['U_use'] else 'matched_sat_supervised')

@contextmanager
def loader_scope():
    from experiments.cric_dura_comparison import f_data,f_train
    original,alias=f_data.context,f_train.context
    def context(c):return original(loader_config(c))
    f_data.context=f_train.context=context
    try:yield
    finally:f_data.context,f_train.context=original,alias

def objective(c):
    from experiments.pamf_ssl_research.hard_pl_receiver_extrapolation_objective import ReceiverExtrapolation,GuardedReceiverExtrapolation
    from experiments.pamf_ssl_research.hard_pl_joint_objective import JointNoAsMixStyle,GuardedJointNoAs
    b=c['repl_row']['blocks']
    cls=(GuardedReceiverExtrapolation if b['guard'] else ReceiverExtrapolation) if b['extrapolation'] else (GuardedJointNoAs if b['guard'] else JointNoAsMixStyle)
    return cls(c)

@contextmanager
def installed(c,holder,repl_enabled):
    from experiments.cric_dura_comparison import t1_model,f_train
    from experiments.pamf_ssl_research.hard_pl_train import installed as hard_installed
    old_install,old_pair,old_args=t1_model.install,t1_model.pair,f_train.arguments
    def build(model,cfg):
        if cfg['repl_row']['group']=='strict_structure':
            from experiments.practical_full_2x2.pamf_model import install
            model=install(model,cfg)
        else:model=old_install(model,cfg)
        arm=cfg['repl_row']['structure']
        if arm!='full':
            from experiments.cric_dura_comparison.supervised_physics import install
            model=install(model,arm,cfg['model_seed'])
            model._feature_learning_report.update(source_U_IQ_used=cfg['U_use'],structure_training_recipe=cfg['paper_method'])
        return model
    def args(cfg,device):
        a=old_args(cfg,device);a.use_unlabeled=a.use_ema_teacher=False;a.lambda_u=a.lambda_ent=0.
        return a
    def pair(a,b,y,smoothing=.01,kl_weight=1.):return old_pair(a,b,y,smoothing,kl_weight=kl_weight if c['repl_row']['L_B'] else 0.)
    t1_model.install,t1_model.pair,f_train.arguments=build,pair,args
    try:
        with loader_scope():
            if repl_enabled:
                with portable_scope(),hard_installed(c,holder,objective_cls=objective):yield
            else:yield
    finally:t1_model.install,t1_model.pair,f_train.arguments=old_install,old_pair,old_args
