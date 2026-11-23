"""Two-stage WiSig restoration with a received-waveform forward action."""

from dataclasses import dataclass

import torch
from torch.nn import functional as F

from .wisig_anchored_window import AnchoredWindow, cnn, generic_crop, iq_fields, normalized_error
from .wisig_timing import NOMINAL, PERIOD


def shifted(x, offset):
    if offset>0:
        return F.pad(x[...,offset:],(0,offset))
    return F.pad(x[...,:offset],(-offset,0))


@dataclass
class Action:
    gain: torch.Tensor
    image: torch.Tensor
    left: torch.Tensor
    right: torch.Tensor

    def apply(self, x):
        return self.gain*x + self.image*x.conj() + self.left*shifted(x,-1) + self.right*shifted(x,3)

    def adjoint(self, y):
        return (self.gain.conj()*y + self.image*y.conj() + shifted(self.left.conj()*y,1)
                + shifted(self.right.conj()*y,-3))

    def diagonal(self, weight, penalty):
        main = (self.gain.abs().square()+self.image.abs().square())*weight + penalty
        main = main + shifted(self.left.abs().square()*weight,1) + shifted(self.right.abs().square()*weight,-3)
        image = 2.*weight*self.gain.conj()*self.image
        return main,image


def dot(a, b):
    return (a.conj()*b).real.sum(-1,keepdim=True)


def solve(action, observed, prior, weight, penalty, *, coupled):
    diagonal,image = action.diagonal(weight,penalty)
    determinant = (diagonal.square()-image.abs().square()).clamp_min(1e-6)

    def precondition(value):
        return (diagonal*value-image*value.conj())/determinant

    residual = action.adjoint(weight*(observed-action.apply(prior)))
    direction = precondition(residual)
    if not coupled:
        return prior+direction
    change = torch.zeros_like(prior)
    previous = dot(residual,direction)
    for index in range(2):
        response = action.adjoint(weight*action.apply(direction)) + penalty*direction
        step = previous / dot(direction,response).clamp_min(1e-12)
        change = change+step*direction
        if index==0:
            residual = residual-step*response
            next_direction = precondition(residual)
            following = dot(residual,next_direction)
            direction = next_direction+(following/previous.clamp_min(1e-12))*direction
            previous = following
    return prior+change


class ObservedSolve(AnchoredWindow):
    def __init__(self, coupled):
        super().__init__(known_waveform=True)
        self.coupled = coupled
        self.estimator = cnn(12,24,9)
        self.coarse = cnn(15,32,2)
        self.refine = cnn(21,32,2)
        self.register_buffer('long_energy',self.long_basis.abs().square().sum(-1),persistent=False)

    @staticmethod
    def predict(net, fields):
        x = torch.stack(fields,dim=1)
        mixed = net.training and x.is_cuda and torch.cuda.is_bf16_supported()
        with torch.autocast('cuda',dtype=torch.bfloat16,enabled=mixed):
            return net(x).float()

    def gate_scores(self, analysis):
        raw,_,omega,_,_,_,late = analysis
        energy = raw.abs().square() @ self.long_view.to(raw.real.dtype).T
        coherence = (late.abs().square()*self.long_energy/energy.clamp_min(1e-6)).clamp(0.,1.)
        center = coherence[:,coherence.shape[-1]//2]
        shift_evidence = (coherence.max(-1).values-center).clamp_min(0.)
        stf = raw[:,40:160]*torch.polar(torch.ones_like(raw[:,40:160].real),-omega[:,None]*self.time[40:160])
        repeats = stf.reshape(-1,6,PERIOD)
        a,b = repeats[:,:-1],repeats[:,1:]
        mismatch = (a-b).abs().square().sum(-1)
        power = a.abs().square().sum(-1)+b.abs().square().sum(-1)
        noise_evidence = (mismatch/power.clamp_min(1e-6)).median(1).values
        return shift_evidence,noise_evidence

    def stages(self, analysis, *, use_action=True, use_noise=True, coupled=None):
        raw,level,omega,noise,logits,short_gain,long_gain = analysis
        y,support,guide,shift,margin = self.select(raw,omega,logits,short_gain,long_gain)
        at = (self.time.to(y.real.dtype)/255.).expand_as(support)
        noise_at = noise[:,None].expand_as(support)
        carrier = omega[:,None].expand_as(support)
        canonical = self.canonical.expand_as(y)
        fields = (*iq_fields(y,guide,y-guide,canonical),support,noise_at.log1p(),at,carrier)
        log_gain,phase,ir,ii,lr,li,rr,ri,extra = self.predict(self.estimator,fields).tanh().unbind(1)
        gain = torch.polar((.65*log_gain).exp(),1.50*phase)
        image = .23*torch.complex(ir,ii)
        left,right = (.38*torch.complex(r,i) for r,i in ((lr,li),(rr,ri)))
        estimate = Action(gain,image,left,right)
        variance = (.008 + .75*noise_at) * (2.0*extra).exp()
        if use_action:
            action = estimate
        else:
            action = Action(torch.ones_like(gain),*(torch.zeros_like(gain) for _ in range(3)))
        weight = support / (.025 + variance if use_noise else torch.full_like(variance,.16))
        fields = (*fields,*iq_fields(action.gain-1.),variance.log().clamp(-7.,3.)/4.)
        prior_1 = y + 1.2*self.correction(self.coarse,fields)
        coupling = self.coupled if coupled is None else coupled
        map_1 = solve(action,y,prior_1,weight,2.0,coupled=coupling)
        innovation = action.adjoint(support*(y-action.apply(map_1)))
        fields = (*fields,*iq_fields(prior_1,map_1,innovation))
        prior_2 = map_1 + .85*self.correction(self.refine,fields)
        map_2 = solve(action,y,prior_2,weight,2.5,coupled=coupling)
        return (y,prior_1,map_1,prior_2,map_2),estimate,variance,support,level,shift,margin,noise

    @classmethod
    def correction(cls, net, fields):
        real,imag = cls.predict(net,fields).tanh().unbind(1)
        return torch.complex(real,imag)

    def restore(self, analysis):
        stages,_,_,_,level,shift,margin,noise = self.stages(analysis)
        return tuple(value*level for value in stages),shift,margin,noise


def gate_features(model, analysis):
    return model.gate_scores(analysis)


def generic_observation(source, rng, model):
    """Add target-independent, smoothly changing gain, phase and I/Q mixing."""
    shape = (len(source),1)
    def draw():
        return torch.rand(shape,device=source.device,generator=rng)
    time = model.time/255.
    freq = .25+2.*draw()
    path = (2.*torch.pi*(freq*time+draw())).sin()
    strength = draw()*(draw()>.35)
    phase = 1.45*strength*path + .7*strength*(2.*draw()-1.)
    gain = (.48*strength*path).exp()
    image = .16*strength*torch.polar(torch.ones_like(phase),3.*path)
    signal = source * torch.polar(gain,phase)+image*source.conj()
    return generic_crop(signal,rng,model)


def objective(model, source, rng, *, identity):
    views = torch.stack((generic_observation(source,rng,model),generic_observation(source,rng,model)),dim=1)
    if identity:
        views[-3:] = source[-3:,None]
    refs = source[:,None].expand_as(views).flatten(0,1)
    stages,action,variance,support,level,*_ = model.stages(model.analyze(views.flatten(0,1)))
    timed,prior_1,map_1,prior_2,map_2 = (value*level for value in stages)
    action_error = (stages[0]-action.apply(refs/level)).abs().square()
    measured = support.sum().clamp_min(1.)
    forward_fit = (action_error*support).sum()/measured
    variance_fit = (support*(action_error.detach()/variance+variance.log())).sum()/measured
    pair = map_2.unflatten(0,views.shape[:2])
    consistency = ((pair[:,0]-pair[:,1]).abs().square().mean(-1)/source.abs().square().mean(-1).clamp_min(.01)).mean()
    weak = torch.fft.fft(map_2-refs,norm='ortho').abs().square()
    weak = (weak/(.025+torch.fft.fft(refs,norm='ortho').abs().square())).log1p().mean()
    edit_scale = (action.image.abs().square()+action.left.abs().square()+action.right.abs().square()).mean()
    values = torch.stack((normalized_error(timed,refs),normalized_error(prior_1,refs),normalized_error(map_1,refs),
                          normalized_error(prior_2,refs),normalized_error(map_2,refs),forward_fit,variance_fit,
                          consistency,weak,edit_scale))
    return (values*values.new_tensor((.17,.16,.35,.12,1.,.70,.04,.12,.12,.05))).sum(),values
