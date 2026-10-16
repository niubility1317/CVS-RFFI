"""The same native augmentation up to its final PA intervention."""
import torch
from experiments.cvs_ordinary_fold20.streams import stream


def paired(augmentor,x,y,seed,call):
    if x.ndim!=3 or x.shape[1:]!=(2,256) or not bool(torch.isfinite(x).all()):raise ValueError('Finite IQ256 required')
    if augmentor.enable_class_signature or augmentor.p_dac!=0 or augmentor.p_pa<=0:raise ValueError('Fixed original PA-only intervention boundary required')
    original=augmentor._apply_general_channel;tape=[]
    def capture(value):
        out=original(value);tape.append(out.detach().clone());return out
    augmentor._apply_general_channel=capture
    try:
        with stream(seed,call,x.device):full,strength=augmentor(x.clone(),labels=y,no_pa=False,return_pa_strength=True)
        with stream(seed,call,x.device):no_pa,zero=augmentor(x.clone(),labels=y,no_pa=True,return_pa_strength=True)
    finally:augmentor._apply_general_channel=original
    if len(tape)!=2 or not torch.equal(tape[0],tape[1]):raise ValueError('Pre-PA upstream IQ differs')
    acted=strength>0;changed=(full!=no_pa).flatten(1).any(1)
    if bool(zero.any()) or bool((changed&~acted).any()) or not torch.equal(no_pa,tape[0]) or not torch.equal(full[~acted],no_pa[~acted]):
        raise ValueError('Only final PA may differ; no-PA output must equal upstream')
    if any(not bool(torch.isfinite(v).all()) for v in (full,no_pa,strength)):raise FloatingPointError('Nonfinite native PA view')
    return full,no_pa,strength


def waveform(x):
    power=x.square().sum(1).mean(1);peak=x.square().sum(1).max(1).values
    return dict(power=power.tolist(),max_abs=x.abs().flatten(1).max(1).values.tolist(),
        PAPR=(peak/power.clamp_min(1e-20)).tolist())


def decisions(model,x,y):
    logits=model.forward_identity_only(x)['tx_logits']
    if logits.shape!=(len(x),6) or not bool(torch.isfinite(logits).all()):raise FloatingPointError('Finite original default-head logits required')
    other=logits.masked_fill(torch.nn.functional.one_hot(y,6).bool(),-torch.inf).max(1).values
    margin=(logits.gather(1,y[:,None]).flatten()-other)/30.
    return logits.argmax(1).tolist(),margin.tolist()
