"""Label-blind acceptance for the extra urban loss; native old gates stay intact."""
import torch


def acceptance(prob,mode,*,threshold=.95,weak_prob=None):
    if prob.requires_grad or prob.ndim!=2 or prob.shape[1]!=6:
        raise ValueError('Detached six-class clean teacher probability required')
    if not torch.isfinite(prob).all():raise ValueError('Nonfinite clean teacher')
    confidence,prediction=prob.max(1);mask=confidence>=threshold
    if mode=='same_sample_weak_agreement':
        if weak_prob is None or weak_prob.requires_grad or weak_prob.shape!=prob.shape or not torch.isfinite(weak_prob).all():
            raise ValueError('Detached same-sample weak teacher required')
        wc,wp=weak_prob.max(1);mask=mask & (wc>=threshold) & (wp==prediction)
    elif mode!='clean_confidence':raise ValueError('Unknown extra urban acceptance')
    return mask


def weak_iq(x,seed,std=.01):
    noise=torch.randn(x.shape,generator=torch.Generator().manual_seed(seed),dtype=x.dtype)
    return x+std*noise.to(x.device)


def calibrate(teacher,loader,device,plan):
    """V only scores frozen teacher health; never updates model/EMA/normalization."""
    before={k:v.clone() for k,v in teacher.state_dict().items()}
    modes={m:m.training for m in teacher.modules()};teacher.eval()
    if any(p.requires_grad for p in teacher.parameters()):raise ValueError('Own EMA must be frozen')
    total=selected=correct=packets=0
    try:
        with torch.no_grad():
            for batch_idx,(x,y,domain,meta) in enumerate(loader):
                if bool((y<0).any()):raise ValueError('Teacher health requires legal V labels')
                x=x.to(device);y=y.to(device)
                logits=teacher(x,y_tx=None,grl_lambda=1.,return_aux=True,domain_labels=None)['tx_logits'];packets+=len(x)
                if batch_idx==0:
                    direct=teacher.forward_identity_only(x)['tx_logits'];packets+=len(x)
                    if not torch.equal(logits,direct):raise ValueError('Actual/default EMA head path differs')
                prob=logits.softmax(1);weak=None
                if plan['acceptance']=='same_sample_weak_agreement':
                    z=weak_iq(x,plan['weak_seed']+batch_idx,plan['weak_std'])
                    weak=teacher(z,y_tx=None,grl_lambda=1.,return_aux=True,domain_labels=None)['tx_logits'].softmax(1);packets+=len(x)
                mask=acceptance(prob,plan['acceptance'],threshold=plan['confidence_threshold'],weak_prob=weak)
                total+=len(y);selected+=int(mask.sum());correct+=int(((prob.argmax(1)==y)&mask).sum())
    finally:
        for module,mode in modes.items():module.training=mode
    if total!=27000 or any(not torch.equal(v,teacher.state_dict()[k]) for k,v in before.items()):
        raise ValueError('V count or frozen EMA state differs')
    precision=correct/selected if selected else None;coverage=selected/total
    return dict(source_only=True,source_role='V',count=total,selected=selected,precision=precision,
        coverage=coverage,passed=precision is not None and precision>=.95 and coverage>=.1,
        state_unchanged=True,forward_packets=packets,U_truth_read=False,query_read=False,head_fit_updates=0)
