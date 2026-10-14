import gzip
import json
from contextlib import contextmanager
import torch
from . import design as d
from .objective import feature_mask,selected_masks,ce
from experiments.cvs_local_order50.observer import derivative
from experiments.cvs_source_layer_order.probe import gradient_stats

class Observer:
    def __init__(self,*,design=None):self.design=design or d
    def initialize(self,model,c,out):
        d=self.design;d.validate(c)
        self.c,self.model,self.out=c,model,out
        self.steps=self.U_steps=self.U_packets=self.U_accepted=0
        self.extra_forwards=self.forward_packets=self.original_packets=0
        self.pending=None;self.saved=[];self.grad=[];self.records=[]
        named=dict(model.named_parameters());self.params=[named[n] for n in d.PARAMETERS]
        self.head=model.id_backbone.encoder.core.id_backbone.cls_head.weight
        if not all(p.requires_grad for p in self.params+[self.head]):raise ValueError('Default joint learning disabled')
        self.stream=gzip.open(out/'temporal_rsc_steps.jsonl.gz','xt',encoding='utf8',compresslevel=3)
    @contextmanager
    def native_scope(self,n):
        if any(getattr(n,key) is not None for key in ('_labeled_identity_forward_hook','_labeled_identity_loss_hook','_trusted_pseudo_loss_hook')):
            raise ValueError('Another L/U hook owner active')
        n._labeled_identity_forward_hook=self.forward
        n._labeled_identity_loss_hook=self.labeled;n._trusted_pseudo_loss_hook=self.unlabeled
        try:yield
        finally:
            n._labeled_identity_forward_hook=n._labeled_identity_loss_hook=n._trusted_pseudo_loss_hook=None
            self.stream.close()
    def forward(self,*,forward,model,args,epoch,batch_idx,inputs,labels,receivers,clean_count,role):
        step=(epoch-1)*222+batch_idx
        if role!='L_s' or model is not self.model or not model.training or self.pending is not None or step!=self.steps+1:
            raise ValueError('Actual RSC L forward context differs')
        if clean_count!=len(labels) or len(inputs)!=clean_count or inputs.shape[1:]!=(2,256) or args.label_smoothing!=.01:
            raise ValueError('RSC requires original single clean labelled view')
        mode=self.c['temporal_rsc']['mode']
        if mode not in ('SIG_TIME','MATCHED_RANDOM','TEMPORAL_RSC'):raise ValueError('Invalid training arm')
        if getattr(model,'_feature_extra_l_forward',False):raise ValueError('Nested L replay context')
        before_cpu=torch.get_rng_state().clone();device=inputs.device
        devices=[device.index if device.index is not None else torch.cuda.current_device()] if device.type=='cuda' else []
        before_cuda=torch.cuda.get_rng_state(devices[0]).clone() if devices else None
        local=dict(mask=None,feature=None)
        def capture(module,arguments,out):
            if out.shape!=(len(labels),96,32):raise ValueError('Complete actual t2 shape differs')
            if local['mask'] is None:local['feature']=out
            else:return out*local['mask'][:,None,:].to(out.dtype)
        layer=model.id_backbone.encoder.core.id_backbone.t2
        handle=layer.register_forward_hook(capture)
        def replay(mask,*,grad):
            local['mask']=mask
            model._feature_extra_l_forward=True
            try:
                with torch.random.fork_rng(devices=devices):
                    torch.set_rng_state(before_cpu)
                    if devices:torch.cuda.set_rng_state(before_cuda,devices[0])
                    with torch.set_grad_enabled(grad):result=forward()
            finally:
                model._feature_extra_l_forward=False;local['mask']=None
            self.extra_forwards+=1;self.forward_packets+=len(labels)
            return result
        try:
            original=forward();feature=local['feature'];logits=original['tx_logits']
            if feature is None or not feature.requires_grad or logits.shape!=(len(labels),6):raise ValueError('Actual t2/default-head graph missing')
            reference=ce(logits,labels)
            locating=torch.autograd.grad(logits.gather(1,labels[:,None]).sum(),feature,retain_graph=True)[0]
            mask=feature_mask(locating,role=role)
            spatial=replay(mask,grad=False)
            rsc,random,selected,drop=selected_masks(logits,spatial['tx_logits'],labels,mask,self.design.MASK_SEED+step,role=role)
            chosen=torch.ones_like(rsc) if mode=='SIG_TIME' else random if mode=='MATCHED_RANDOM' else rsc
            final=replay(chosen,grad=True)
            if mode=='SIG_TIME':
                if not torch.equal(final['tx_logits'].detach(),logits.detach()):raise ValueError('Unmasked train-mode dropout replay differs')
                returned=original
            else:returned=final
            # Probe forwards consume no additional ambient dropout random draws.
            selected_logits=returned['tx_logits']
            detail=dict(step=step,epoch=epoch,batch=batch_idx,mode=mode,source_packets=clean_count,
                original_CE=float(reference.detach()),selected_CE=float(ce(selected_logits,labels).detach()),
                selected_packets=int(selected.sum()),masked_positions=int((~chosen).sum()),
                candidate_masked_positions=int((~rsc).sum()),random_masked_positions=int((~random).sum()),
                spatial_true_probability_drop_mean=float(drop.mean()),extra_L_forwards=2,
                original_correct=int((logits.detach().argmax(1)==labels).sum()),
                selected_correct=int((selected_logits.detach().argmax(1)==labels).sum()))
            diag=None
            if epoch in self.design.DIAGNOSTIC_EPOCHS and batch_idx%16==0:
                diag={k:v.detach().cpu() for k,v in dict(temporal_score=locating.mean(1),base_mask=mask,
                    rsc_mask=rsc,random_mask=random,applied_mask=chosen,selected_packets=selected,
                    probability_drop=drop,labels=labels,original_logits=logits,
                    spatial_logits=spatial['tx_logits'],selected_logits=selected_logits).items()}
            self.original_packets+=clean_count;self.forward_packets+=clean_count
            self.pending=dict(logits=selected_logits,reference=reference,detail=detail,diagnostic=diag)
            return returned
        finally:handle.remove()
    def labeled(self,*,model,args,epoch,batch_idx,logits,labels,receivers,clean_count,original_loss,role):
        step=(epoch-1)*222+batch_idx;p=self.pending
        if model is not self.model or role!='L_s' or p is None or p['logits'] is not logits or p['detail']['step']!=step:
            raise ValueError('Native optimizer did not consume chosen RSC logits')
        if abs(float(original_loss.detach())-p['detail']['selected_CE'])>2e-6:raise ValueError('Native chosen CE differs')
        reference=original_loss if self.c['arm']=='SIG_TIME' else p['reference']
        if p['diagnostic'] is not None:
            delta=original_loss-reference;objects=self.params+[self.head]
            vectors={key:derivative(loss,objects) for key,loss in [('original',reference),('selected',original_loss),('delta',delta)]}
            self.grad.append(dict(step=step,epoch=epoch,vectors=vectors,mask=p['diagnostic']))
            self.records.append(dict(step=step,epoch=epoch,layers=[dict(parameter=name,
                selected=gradient_stats(vectors['original'][k],vectors['selected'][k]),
                delta=gradient_stats(vectors['original'][k],vectors['delta'][k])) for k,name in enumerate([*self.design.PARAMETERS,'existing_default_W'])]))
        self.steps=step;self.stream.write(json.dumps(p['detail'],allow_nan=False)+chr(10));self.pending=None
        return original_loss
    def unlabeled(self,*,model,args,epoch,batch_idx,loss_l,loss_u,loss_ent,active,logits,pseudo,mask):
        if model is not self.model or self.pending is not None or self.steps!=(epoch-1)*222+batch_idx or getattr(model,'_feature_extra_l_forward',False):
            raise ValueError('U hook not paired to cleared L step')
        if bool(active)!=(epoch>=34) or (args.lambda_u,args.lambda_ent,args.ema_decay)!=(.16,.01,.999):raise ValueError('Original U objective differs')
        if active:self.U_steps+=1;self.U_packets+=len(pseudo);self.U_accepted+=int(mask.sum())
        return loss_u
    def epoch(self,model,ctx,rows,out,device):
        epoch=int(rows[-1]['epoch']);self.stream.flush()
        if self.steps!=epoch*222 or self.extra_forwards!=self.steps*2 or getattr(model,'_feature_extra_l_steps',0)!=self.extra_forwards:
            raise ValueError('Native/additional L forward accounting differs')
        if epoch in self.design.DIAGNOSTIC_EPOCHS:
            if len(self.grad)!=13:raise ValueError('RSC derivative period incomplete')
            path=out/f'temporal_rsc_gradients_e{epoch}.pt';torch.save(self.grad,path)
            actual=torch.load(path,map_location='cpu',weights_only=False)
            ref=[r for r in self.records if r['epoch']==epoch]
            for a,b in zip(actual,ref):
                for k,row in enumerate(b['layers']):
                    for key in ('selected','delta'):
                        if gradient_stats(a['vectors']['original'][k],a['vectors'][key][k])!=row[key]:raise ValueError('Saved derivative recount differs')
            self.saved.append(epoch);self.grad=[]
            self.design.write(out/'temporal_rsc_gradient_records.json',dict(status='VERIFIED',epochs=self.saved,
                batch_records=self.records,saved_gradient_recount='VERIFIED',head_fit_updates=0,default_W_normal_joint_CE=True,U_truth_read=False))
    def finish(self):
        if self.steps!=11100 or self.extra_forwards!=22200 or self.U_steps!=3774 or self.U_packets!=963900 or self.pending is not None:
            raise ValueError('RSC native/U/additional-forward budget incomplete')
        if self.saved!=list(self.design.DIAGNOSTIC_EPOCHS) or len(self.records)!=52:raise ValueError('RSC derivative diagnostics incomplete')
        value=dict(status='VERIFIED',mode=self.c['arm'],steps=self.steps,U_steps=self.U_steps,U_packets=self.U_packets,
            U_accepted=self.U_accepted,gradient_batches=52,layer_records=208,saved_vectors=624,gradient_epochs=self.saved,
            extra_model_forwards=self.extra_forwards,actual_L_network_forwards=self.steps+self.extra_forwards,
            actual_original_L_packets=self.original_packets,actual_forward_L_packet_observations=self.forward_packets,
            native_L_contexts=self.model._feature_l_steps,additional_L_forward_counter=self.model._feature_extra_l_steps,
            new_parameters=0,new_heads=0,head_fit_updates=0,default_head_jointly_trained=True,U_loss_returned_unchanged=True,
            U_truth_read=False,saved_gradient_recount='VERIFIED')
        self.design.write(self.out/'temporal_rsc_training_observation.json',value);return value
