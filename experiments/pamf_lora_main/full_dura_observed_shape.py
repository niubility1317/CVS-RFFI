"""Train DURA to recover non-carrier LoRa structure in the observed coordinates."""
import torch
from torch import nn

from .dura_repair import PacketRestorer, no_tone


CAP=.15
TEACHER_WEIGHT=.25


class ObservedShape(nn.Module):
    def __init__(self):
        super().__init__()
        self.tracker=PacketRestorer('consensus').eval()

    @torch.no_grad()
    def coordinates(self,iq):
        observed,rotation,*_=self.tracker.track(iq[:,None])
        observed,rotation=observed[:,0],rotation[:,0]
        rms=observed.abs().square().mean(-1,keepdim=True).clamp_min(1e-8).sqrt()
        return observed/rms,rotation,rms

    def parts(self,output,coordinates):
        observed,rotation,rms=coordinates
        raw=torch.complex(output[:,0],output[:,1])*rotation.conj()/rms
        raw_shape=no_tone(raw)
        direct=no_tone(observed)
        delta=raw_shape-direct
        shrink=(1.+delta.abs().square().mean(-1,keepdim=True)/CAP**2).rsqrt()
        return direct+shrink*delta,raw_shape,raw.mean(-1)-observed.mean(-1)

    def restore(self,received,output):
        coordinates=self.coordinates(received)
        observed,rotation,rms=coordinates
        shape,*_=self.parts(output,coordinates)
        delta=(shape-no_tone(observed))*rotation*rms
        return received+torch.stack((delta.real,delta.imag),dim=1)

    def forward(self,received,clean,outputs,teacher=None):
        coordinates=self.coordinates(received)
        direct=no_tone(coordinates[0])
        truth=no_tone(self.coordinates(clean)[0])
        scale=truth.abs().square().mean(-1)+.002

        def distance(a,b):
            return (a-b).abs().square().mean(-1)/scale

        parts=[self.parts(output,coordinates) for output in outputs]
        losses=[distance(shape,truth)+.10*distance(raw,truth)+.50*tone.abs().square()
                 for shape,raw,tone in parts]
        # The last unroll remains primary. Intermediate losses protect against
        # relearning a trajectory with a harmful final step.
        loss=.5*losses[-1]+.5*torch.stack(losses).mean(0)
        if teacher is not None:
            with torch.no_grad():
                base_error=distance(direct,truth)
                guides=[self.parts(output,coordinates)[0] for output in teacher]
                useful=[distance(guide,truth)<base_error for guide in guides]
            transfer=[distance(shape,guide)*keep for (shape,_,_),guide,keep
                      in zip(parts,guides,useful,strict=True)]
            loss=loss+TEACHER_WEIGHT*torch.stack(transfer).mean(0)
        return loss
