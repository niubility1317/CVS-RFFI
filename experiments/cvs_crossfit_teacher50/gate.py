"""Native old U gate functions replayed on source L_dev without its TX labels."""
import torch
from . import design as d
from experiments.cvs_phase1_stack.runtime import installed

FIELDS=('rx_i','day_i','eq_i','sig_i','base_index')


def strong_forward(model,x,domains):
    """Native train-mode Dropout/BN on ephemeral buffers; real state stays frozen."""
    if any(p.requires_grad or p.grad is not None for p in model.parameters()):raise ValueError('Frozen baseline required')
    modes={m:m.training for m in model.modules()};model.train()
    parameters={name:p.detach() for name,p in model.named_parameters()}
    buffers={name:b.detach().clone() for name,b in model.named_buffers()}
    try:
        with torch.no_grad():
            return torch.func.functional_call(model,(parameters,buffers),(x,),
                dict(y_tx=None,grl_lambda=1.,return_aux=True,domain_labels=domains))['tx_logits'].detach()
    finally:
        for module,mode in modes.items():module.training=mode


def batch_masks(native,args,weak_logits,strong_logits,meta):
    if (set(meta)!=set(FIELDS) or weak_logits.ndim!=2 or weak_logits.shape[1]!=6 or strong_logits.shape!=weak_logits.shape
            or any(len(v)!=len(weak_logits) for v in meta.values()) or args.pseudo_domain_gate
            or not args.pseudo_temporal_gate or args.pseudo_temporal_mode!='batch_neighbor' or not args.pseudo_strong_agreement
            or args.wisig_domain!='rx_day'):
        raise ValueError('Actual old native RX/day confidence, batch-neighbor and strong-agreement gate required; no TX metadata')
    conf,pseudo=weak_logits.softmax(1).max(1)
    # Only the equality partition matters for native within-domain quantiles.
    # This integer encoding is a relabeling of the native RX/day domain IDs.
    domains=meta['rx_i'].long()*10+meta['day_i'].long()
    confidence=native._threshold_mask(conf,domains,args)
    temporal=native._temporal_mask_tensor(pseudo,conf,(domains,meta),args,weak_logits.device)
    agreement=strong_logits.argmax(1)==pseudo
    return dict(confidence=confidence,temporal=temporal,strong=agreement,accepted=confidence&temporal&agreement,
        weak_pseudo=pseudo,weak_confidence=conf)


def replay(student,teacher,panel,metadata):
    if len(panel['ids'])!=720 or len(metadata)!=720:raise ValueError('Exact source L_dev720 required')
    table={r['physical_id']:r for r in metadata}
    if set(table)!=set(panel['ids']):raise ValueError('Actual native metadata physical IDs differ')
    args=d.parent.make_args(d.operational_parent(),'cpu')
    before={k:v.clone() for k,v in student.state_dict().items()};tb={k:v.clone() for k,v in teacher.state_dict().items()}
    if student.training or teacher.training:raise ValueError('Frozen eval baseline states required')
    order=torch.randperm(720,generator=torch.Generator().manual_seed(d.EVALUATION_SEED));outputs={};forwards=0
    with installed(d.operational_parent()) as native,torch.random.fork_rng(devices=[]):
        torch.manual_seed(d.EVALUATION_SEED)
        for start in range(0,720,256):
            selected=order[start:start+256];x=panel['x'][selected]
            meta={key:torch.tensor([table[panel['ids'][int(i)]][key] for i in selected]) for key in FIELDS}
            # Original domain map is sorted source RX/day; no classifier gate is active.
            groups={(rx,day):i for i,(rx,day) in enumerate(( (rx,day) for rx in (1,3,4,6,8) for day in (1,2,3) ))}
            domains=torch.tensor([groups[int(rx),int(day)] for rx,day in zip(meta['rx_i'],meta['day_i'])])
            with torch.no_grad():weak=teacher(x,y_tx=None,grl_lambda=1.,return_aux=True,domain_labels=domains)['tx_logits']
            strong=strong_forward(student,native._strong_augment(x,float(args.strong_noise_std)),domains)
            masks=batch_masks(native,args,weak,strong,meta);forwards+=2*len(selected)
            for j,index in enumerate(selected.tolist()):outputs[index]={k:float(v[j]) if v.dtype!=torch.bool else bool(v[j]) for k,v in masks.items()}
    if any(not torch.equal(v,student.state_dict()[k]) for k,v in before.items()) or any(not torch.equal(v,teacher.state_dict()[k]) for k,v in tb.items()):
        raise ValueError('Source gate replay changed real baseline/EMA state')
    if forwards!=1440 or set(outputs)!=set(range(720)):raise ValueError('Actual gate source coverage differs')
    return dict(records=[outputs[i] for i in range(720)],forward_packets=forwards,source_only=True,U_opened=False,query_read=False,
        actual_U_decisions=False,model_state_unchanged=True,EMA_state_unchanged=True,
        scope='Native threshold/temporal/strong gates on a fixed shuffled source L_dev720; train-mode strong Dropout/BN uses cloned ephemeral buffers. Last source batch208; this is not actual U batching or a historical U gate decision.',
        native_args={k:getattr(args,k) for k in ('pseudo_threshold_mode','tau_min','tau_max','pseudo_quantile','pseudo_temporal_window',
            'pseudo_temporal_min_conf','pseudo_strong_agreement','strong_noise_std','wisig_domain')})
