"""Audit actual labeled-only teacher exposure and ordinary joint CE updates."""
from collections import Counter
from contextlib import contextmanager
import gzip,json
import torch
from . import design as d
from experiments.cvs_urban_identity50.features import parameter_groups,boundary_parameters


class Observer:
    def __init__(self,c):d.validate(c);self.c=c;self.steps=0;self.coverage=Counter();self.forwards=Counter();self.history=[];self.updates=[]

    def initialize(self,model,c,out):
        self.model=model;self.out=out;self.inner=d.read(out/'inner_source_contract.json')
        self.allowed=set(self.inner['train_native_base_indices']);self.seen=set()
        self.groups=dict(parameter_groups(model),**{k:[p] for k,p in boundary_parameters(model).items()})
        if any(not p.requires_grad for p in model.parameters()):raise ValueError('Teacher default head and backbone must train jointly')
        self.previous=self.snapshot();self.trace=gzip.open(out/'teacher_batches.jsonl.gz','xt',encoding='utf-8',compresslevel=3)
        def count(module,args,output):self.forwards['train' if module.training else 'eval']+=len(args[0])
        self.handle=model.register_forward_hook(count)

    def snapshot(self):return {k:torch.cat([p.detach().flatten() for p in ps]).clone() for k,ps in self.groups.items()}

    @contextmanager
    def native_scope(self,n):
        if n._urban_practical_training_hook is not None or n._source_feature_auxiliary_hook is not None:raise ValueError('Another hook owns process')
        n._urban_practical_training_hook=self.observe
        try:yield
        finally:n._urban_practical_training_hook=None

    def observe(self,*,model,args,epoch,batch_idx,clean_iq,labels,clean_meta,clean_output,clean_ce,
                unlabeled_iq,unlabeled_meta,teacher_prob,weak_mask,teacher_model=None):
        step=(epoch-1)*222+batch_idx
        if step!=self.steps+1 or model is not self.model or len(labels)!=128 or bool((labels<0).any()):raise ValueError('Actual L-only teacher step differs')
        if (unlabeled_iq is not None or teacher_prob is not None or teacher_model is not None
                or args.use_unlabeled or args.use_ema_teacher or args.use_concat_sat_channel_aug or args.sat_training_mode!='disabled'):
            raise ValueError('No U iterator/EMA/augmentation or additional teacher in first pilot')
        base=clean_meta['base_index'].detach().cpu().tolist();rx=clean_meta['rx_i'].detach().cpu().tolist()
        day=clean_meta['day_i'].detach().cpu().tolist();tx=labels.detach().cpu().tolist()
        if not set(base)<=self.allowed or (self.c['arm']=='EXCLUDE_RX1' and 1 in rx):raise ValueError('Excluded/dev source reached teacher update')
        self.steps=step;self.seen.update(base);self.coverage.update(zip(tx,rx,day))
        self.trace.write(json.dumps(dict(step=step,epoch=epoch,base_index=base,RX=rx,day=day,TX=tx),allow_nan=False)+'\n')
        if step==1 or step%222==0:
            names=list(self.groups);params=[p for key in names for p in self.groups[key]]
            gradients=torch.autograd.grad(clean_ce,params,allow_unused=True,retain_graph=True);offset=0;values={}
            for key in names:
                count=len(self.groups[key]);vs=gradients[offset:offset+count];offset+=count
                values[key]=sum(float(v.detach().double().square().sum()) for v in vs if v is not None)**.5
            if values['default_head']<=0 or values['t2']<=0 or values['SIG_local']<=0:raise ValueError('Original default CE graph detached')
            self.history.append(dict(epoch=epoch,step=step,CE_grad_norms=values))
            d.write(self.out/'gradient_history.json',dict(records=self.history,head_fit_updates=0,extra_objectives=0))
        return clean_ce*0.,{'teacher/L_only_packets':128.,'teacher/U_packets':0.,'teacher/EMA':0.,'teacher/source_RX1_packets':float(rx.count(1))}

    def epoch(self,model,ctx,rows,out,device):
        current=self.snapshot();epoch=int(rows[-1]['epoch'])
        self.updates.append(dict(epoch=epoch,groups={k:dict(update_norm=float((v-self.previous[k]).norm()),parameter_norm=float(v.norm())) for k,v in current.items()}))
        self.previous=current;self.trace.flush()
        d.write(out/'parameter_update_history.json',dict(records=self.updates,original_CE_joint_updates=True))
        d.write(out/'teacher_coverage.json',dict(records=[dict(TX=k[0],RX=k[1],day=k[2],count=v) for k,v in sorted(self.coverage.items())],
            unique_train_samples=len(self.seen),forward_packets=dict(self.forwards),source_only=True,U_forward_packets=0,EMA_updates=0))

    def finish(self):
        self.handle.remove();self.trace.close()
        if self.steps!=11100 or len(self.updates)!=50 or self.seen!=self.allowed or self.forwards['train']!=11100*128:
            raise ValueError('Teacher E50 update or actual physical sample coverage incomplete')
        result=dict(status='VERIFIED',source_only=True,head_fit_updates=0,optimizer_updates=self.steps,
            training_epochs=50,steps_per_epoch=222,unique_train_samples=len(self.seen),forward_packets=dict(self.forwards),
            initial_checkpoint_smoke_packets=2,U_forward_packets=0,U_truth_read=False,EMA_updates=0,student_models_trained=0,
            train_receivers=self.inner['train_receivers'],all_TX=True,auxiliary_objectives=0)
        d.write(self.out/'training_mechanism_complete.json',result);return result
