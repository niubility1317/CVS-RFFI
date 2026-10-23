"""Raw student U moments align same pseudo classes across original source receivers."""
from itertools import combinations
import torch
from .objective import Objective


def conditional_moments(h,target,weight,rx,source_rxs,recipe):
    if (h.ndim!=2 or target.ndim!=2 or len(target)!=len(h) or weight.shape!=(len(h),)
        or rx.shape!=(len(h),) or set(rx.tolist())-set(source_rxs)):
        raise ValueError('Current U descriptor/target and original source RX required')
    p=target.detach().argmax(1);w=weight.detach()
    eligible=w>=recipe['eligible_weight_min']
    w=w*eligible
    scale=h.detach().std(0,unbiased=False).clamp_min(recipe['descriptor_std_floor'])
    z=(h-h.detach().mean(0))/scale
    class_losses=[];class_mean=[];class_std=[];valid_cells=pairs=0
    for cls in range(target.shape[1]):
        moments=[]
        for receiver in source_rxs:
            mask=(p==cls)&(rx==receiver)&eligible
            cw=w[mask]
            if not len(cw) or not bool(cw.sum()>0):continue
            # The threshold uses the exact weighted ESS, without treating count as ESS.
            total=cw.double().sum();squared=cw.double().square().sum()
            if bool(total.square()<recipe['min_effective_samples']*squared):continue
            values=z[mask];mean=(values*cw[:,None]).sum(0)/cw.sum()
            variance=(cw[:,None]*(values-mean).square()).sum(0)/cw.sum()
            std=(variance+recipe['moment_eps']).sqrt()
            moments.append((mean,std));valid_cells+=1
        mean_values=[];std_values=[]
        for (a,sa),(b,sb) in combinations(moments,2):
            mean_values.append((a-b).square().mean());std_values.append((sa-sb).square().mean());pairs+=1
        if mean_values:
            mu=torch.stack(mean_values).mean();sigma=torch.stack(std_values).mean()
            class_losses.append(mu+sigma);class_mean.append(mu);class_std.append(sigma)
    zero=h.sum()*0.
    value=torch.stack(class_losses).mean() if class_losses else zero
    detail=dict(eligible=int(eligible.sum()),valid_cells=valid_cells,valid_classes=len(class_losses),receiver_pairs=pairs,
                mean_loss=torch.stack(class_mean).mean().detach() if class_mean else zero.detach(),
                std_loss=torch.stack(class_std).mean().detach() if class_std else zero.detach())
    return value,detail


class UnlabeledReceiverMomentsObjective(Objective):
    def __init__(self,student,config):
        super().__init__(student,config)
        self.u_cmr_calls=self.u_cmr_seen=self.u_cmr_eligible=0
        self.u_cmr_pairs={name:0 for name in config['u_cmr']['levels']}
        self._raw_context_active=False;self._cmr_metrics={}

    def loss(self,student,device):
        value=super().loss(student,device)
        self.metrics.update(self._cmr_metrics)
        return value

    def unlabeled_source_loss(self,student,x,domain,target,weight,rx):
        if self.current_epoch<31 or self._raw_context_active:
            raise ValueError('One current raw U moment forward at E31+ required')
        self._raw_context_active=True;maps={};handles=[]
        try:
            for name in self.c['u_cmr']['levels']:
                def capture(module,inputs,output,name=name):
                    if name in maps:raise ValueError('Repeated raw U layer capture')
                    maps[name]=output
                    if self.c.get('u_cmr_smoke'):output.retain_grad()
                handles.append(student.get_submodule('id_backbone.encoder.core.id_backbone.'+name).register_forward_hook(capture))
            student.forward_identity_only(x,domain_labels=domain)
        finally:
            for handle in handles:handle.remove()
            self._raw_context_active=False
        values=[];metrics={};recipe=self.c['u_cmr']
        for name in recipe['levels']:
            if name not in maps:raise ValueError('Raw U moment feature absent')
            h=maps[name];descriptor=torch.cat((h.mean(-1),h.std(-1,unbiased=False)),1)
            loss,detail=conditional_moments(descriptor,target,weight,rx,self.c['source_rxs'],recipe)
            values.append(loss);self.u_cmr_pairs[name]+=detail['receiver_pairs']
            metrics.update({'SSL_U_CMR_'+name+'_'+key:value for key,value in detail.items()})
            metrics['SSL_U_CMR_'+name+'_pairs_total']=self.u_cmr_pairs[name]
        value=recipe['weight']*torch.stack(values).mean()
        self.u_cmr_calls+=1;self.u_cmr_seen+=len(x)
        self.u_cmr_eligible+=int((weight.detach()>=recipe['eligible_weight_min']).sum())
        self._cmr_metrics=dict(metrics,SSL_U_CMR_loss=value.detach(),SSL_U_CMR_calls=self.u_cmr_calls,
            SSL_U_CMR_seen=self.u_cmr_seen,SSL_U_CMR_eligible=self.u_cmr_eligible)
        if self.c.get('u_cmr_smoke'):self._smoke_maps=maps
        return value

    def state_dict(self):
        state=super().state_dict()
        state.update(u_cmr=self.c['u_cmr'],u_cmr_calls=self.u_cmr_calls,u_cmr_seen=self.u_cmr_seen,
                     u_cmr_eligible=self.u_cmr_eligible,u_cmr_pairs=self.u_cmr_pairs)
        return state

    def load_state_dict(self,state):
        if state.get('u_cmr') and state['u_cmr']!=self.c['u_cmr']:
            raise ValueError('Moment state belongs to another recipe')
        super().load_state_dict(state)
        for key in ('u_cmr_calls','u_cmr_seen','u_cmr_eligible'):setattr(self,key,state.get(key,0))
        self.u_cmr_pairs=dict(state.get('u_cmr_pairs',self.u_cmr_pairs))
