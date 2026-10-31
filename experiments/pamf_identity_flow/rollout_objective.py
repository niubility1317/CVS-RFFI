"""Source loss through the actual full deployment-time flow trajectories."""
from .model import frozen_identity_consistency


def rollout_objective(model,teacher,clean,received,noise):
    """Supervise both four-step paths and frozen F0, with no clean-fed states."""
    a=model.integrate(received,noise,noise_scale=.02,steps=4)
    b=model.integrate(received,-noise,noise_scale=.02,steps=4)
    energy=clean.detach().square().mean((1,2)).clamp_min(1e-4)
    iq=.5*((a-clean.detach()).square().mean((1,2))+(b-clean.detach()).square().mean((1,2)))/energy
    anchor,agreement=frozen_identity_consistency(teacher,clean,a,b)
    identity=anchor+.25*agreement
    return iq+identity,dict(iq=iq,identity=identity,anchor=anchor,agreement=agreement,path_a=a,path_b=b)
