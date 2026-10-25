"""Do not replace an own pseudo label; require both frozen source peers to agree."""
from contextlib import contextmanager
from functools import wraps
import torch
from experiments.pamf_ssl_research.hard_pl_objective import HardPLObjective
from .plan import validate_config


def consensus_mask(confidence,pseudo,peer_logits,threshold=.95):
    if confidence.ndim!=1 or pseudo.shape!=confidence.shape or pseudo.dtype!=torch.long or len(peer_logits)!=2:
        raise ValueError('Own labels and exactly two aligned peers required')
    accepted=confidence>=threshold
    for logits in peer_logits:
        if logits.ndim!=2 or len(logits)!=len(pseudo) or not bool(torch.isfinite(logits).all()):
            raise ValueError('Finite aligned source peer default-head logits required')
        conf,label=logits.softmax(1).max(1)
        accepted=accepted&(label==pseudo)&(conf>=threshold)
    return accepted


class PeerObjective(HardPLObjective):
    def __init__(self,c,peers):
        validate_config(c);super().__init__(c)
        if len(peers)!=2:raise ValueError('Both peers must be loaded before native training scope')
        self.peers=peers;self.peer_views=0;self.raw_accepted=0;self.owner=None
        for model in peers:
            if model.training or any(p.requires_grad for p in model.parameters()):raise ValueError('Frozen eval peers required')

    def validate_unlabeled_batch(self,batch):
        super().validate_unlabeled_batch(batch)
        meta=batch[3]
        if 'tx_label_visible' not in meta or bool(torch.as_tensor(meta['tx_label_visible']).any()):
            raise ValueError('Hidden source U TX required')
        if set(torch.as_tensor(meta['rx_i']).tolist())-set(self.c['source_rxs']):raise ValueError('Non-source U RX')

    def accept_pseudo(self,student,x,confidence,pseudo):
        logits=[]
        with torch.no_grad():
            for model in self.peers:
                if model.training or any(p.requires_grad for p in model.parameters()):raise ValueError('Peers cannot train or update')
                z=model(x)
                if z.shape!=(len(x),self.c['num_classes']):raise ValueError('Original peer class space required')
                logits.append(z)
        accepted=consensus_mask(confidence,pseudo,logits,self.settings['threshold'])
        self.raw_accepted+=int((confidence>=self.settings['threshold']).sum());self.peer_views+=2*len(x)
        return accepted

    def loss(self,student,device):
        if not student.training or self.owner is not None and self.owner is not student:
            raise ValueError('One native training student per peer objective required')
        self.owner=student
        value=super().loss(student,device)
        self.metrics.update(PEER_views=self.peer_views,PEER_raw_accepted=self.raw_accepted,
            PEER_consensus_accepted=self.accepted,PEER_updates=0,PEER_label_replacements=0)
        return value

    def state_dict(self):
        state=super().state_dict()
        if not 0<=self.accepted<=self.raw_accepted<=self.seen or self.peer_views!=2*self.seen:
            raise ValueError('Aligned peer acceptance/exposure required')
        return dict(state,recipe=self.c['peer_supervised'],weight=self.settings['weight'],raw_accepted=self.raw_accepted,
            peer_views=self.peer_views,own_eval_views=self.seen,peer_updates=0,peer_label_replacements=0,
            peers=[dict(model_seed=p['model_seed'],source=p['parent_source'],epoch=20) for p in self.c['peers']],
            source_V_training_reads=0,target_reads=0,G_training_reads=0,new_head=False,new_inference_parameters=0,
            As_confidence_gate=False,per_accepted_sample_CE_dose=self.settings['weight'],
            As_CE_fraction_of_physical_packet=.4 if self.settings['method']!='HPL2' else 0.)


_ACTIVE=False


@contextmanager
def installed(c,holder,objective):
    global _ACTIVE
    if _ACTIVE:raise ValueError('No nested peer source scope')
    from experiments.pamf_repl_completion.objectives import installed as baseline
    from experiments.cric_dura_comparison import t1_model
    original_plain,original_hook=t1_model.plain_pl,t1_model.hook
    holder['objective']=objective
    def plain(model,device):return objective.loss(model,device)
    @wraps(original_hook)
    def hook(self,**kwargs):
        value,metrics=original_hook(self,**kwargs)
        return value,dict(metrics,**objective.metrics)
    _ACTIVE=True
    try:
        with baseline(c,holder,False):
            t1_model.plain_pl,t1_model.hook=plain,hook
            try:yield
            finally:t1_model.plain_pl,t1_model.hook=original_plain,original_hook
    finally:_ACTIVE=False
