"""Source-paired conditional residual diffusion with a joint time/frequency Transformer.

EDM equations are applied to clean-received residuals. No classifier, labels,
channel truth, clean reference or query statistics enter the deployed sampler.
"""
import math
import torch
from torch import nn


def fft_iq(value):
    pairs=value.reshape(len(value),-1,2,256)
    z=torch.fft.fft(torch.complex(pairs[:,:,0],pairs[:,:,1]),norm='ortho')
    return torch.stack((z.real,z.imag),2).flatten(1,2)


def ifft_iq(value):
    pairs=value.reshape(len(value),-1,2,256)
    z=torch.fft.ifft(torch.complex(pairs[:,:,0],pairs[:,:,1]),norm='ortho')
    return torch.stack((z.real,z.imag),2).flatten(1,2)


class NoiseBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm1=nn.LayerNorm(128,elementwise_affine=False)
        self.norm2=nn.LayerNorm(128,elementwise_affine=False)
        self.attention=nn.MultiheadAttention(128,4,dropout=0.,batch_first=True)
        self.mlp=nn.Sequential(nn.Linear(128,512),nn.GELU(),nn.Linear(512,128))
        self.condition=nn.Linear(128,512)
        nn.init.zeros_(self.condition.weight);nn.init.zeros_(self.condition.bias)

    def forward(self,tokens,noise):
        scale1,shift1,scale2,shift2=self.condition(noise)[:,None].chunk(4,-1)
        q=self.norm1(tokens)*(1+scale1)+shift1
        tokens=tokens+self.attention(q,q,q,need_weights=False)[0]
        return tokens+self.mlp(self.norm2(tokens)*(1+scale2)+shift2)


class TimeFrequencyDenoiser(nn.Module):
    def __init__(self):
        super().__init__()
        self.time_in=nn.Conv1d(4,128,4,stride=4)
        self.freq_in=nn.Conv1d(4,128,4,stride=4)
        self.noise_mlp=nn.Sequential(nn.Linear(64,128),nn.SiLU(),nn.Linear(128,128))
        self.domain=nn.Parameter(torch.randn(2,1,128)*.02)
        self.blocks=nn.ModuleList(NoiseBlock() for _ in range(6))
        self.norm=nn.LayerNorm(128)
        self.time_out=nn.Linear(128,8);self.freq_out=nn.Linear(128,8)
        for layer in [self.time_out,self.freq_out]:
            nn.init.zeros_(layer.weight);nn.init.zeros_(layer.bias)
        frequency=torch.exp(-math.log(10000)*torch.arange(32,dtype=torch.float32)/31)
        self.register_buffer('noise_frequency',frequency)
        p=torch.arange(64,dtype=torch.float32)[:,None]
        f=torch.exp(-math.log(10000)*torch.arange(64,dtype=torch.float32)/64)
        self.register_buffer('position',torch.cat(((p*f).sin(),(p*f).cos()),1)[None])

    @staticmethod
    def unpack(value):
        return value.reshape(len(value),64,2,4).permute(0,2,1,3).reshape(len(value),2,256)

    def forward(self,state,received,log_sigma):
        pair=torch.cat((state,received),1)
        time=self.time_in(pair).transpose(1,2)+self.position+self.domain[0]
        freq=self.freq_in(fft_iq(pair)).transpose(1,2)+self.position+self.domain[1]
        phase=log_sigma[:,None]*self.noise_frequency[None]
        noise=self.noise_mlp(torch.cat((phase.sin(),phase.cos()),1))
        tokens=torch.cat((time,freq),1)
        for block in self.blocks:tokens=block(tokens,noise)
        time,freq=self.norm(tokens).split(64,1)
        return (self.unpack(self.time_out(time))+ifft_iq(self.unpack(self.freq_out(freq))))/math.sqrt(2)


class ResidualDiffusionRestorer(nn.Module):
    sigma_data=.5

    def __init__(self):
        super().__init__();self.core=TimeFrequencyDenoiser()
        generator=torch.Generator(device='cpu').manual_seed(2026110204)
        self.register_buffer('fixed_latent',torch.randn(1,2,256,generator=generator))
        r=torch.linspace(0,1,8,dtype=torch.float64)
        schedule=(80**(1/7)+r*(.002**(1/7)-80**(1/7)))**7
        self.register_buffer('sigma_steps',torch.cat((schedule,torch.zeros(1))).float())

    def denoise(self,noisy_residual,received,sigma):
        sigma=sigma.reshape(-1,1,1)
        denominator=sigma.square()+self.sigma_data**2
        skip=self.sigma_data**2/denominator
        out=sigma*self.sigma_data/denominator.sqrt()
        prediction=self.core(noisy_residual/denominator.sqrt(),received,sigma.flatten().log()/4)
        return skip*noisy_residual+out*prediction

    def forward(self,received):
        if received.ndim!=3 or received.shape[1:]!=(2,256) or received.dtype!=torch.float32:
            raise ValueError('Current per-record FP32 IQ2x256 required')
        # One fixed latent for every record; no RNG, ID or batch-derived selection.
        state=self.fixed_latent.expand_as(received)*self.sigma_steps[0]
        for index in range(8):
            sigma,next_sigma=self.sigma_steps[index:index+2]
            denoised=self.denoise(state,received,sigma.expand(len(received)))
            derivative=(state-denoised)/sigma
            proposal=state+(next_sigma-sigma)*derivative
            if index<7:
                next_denoised=self.denoise(proposal,received,next_sigma.expand(len(received)))
                next_derivative=(proposal-next_denoised)/next_sigma
                state=state+(next_sigma-sigma)*(derivative+next_derivative)/2
            else:state=proposal
        return received+state


def diffusion_loss(model,received,clean,sigma,noise):
    """One call per source pair; clean constructs the independently noised training residual.

    Neither clean nor the sampled noise is a separate denoiser input, and the
    deployed sampler starts from its saved latent without a clean reference.
    """
    residual=clean.detach()-received
    noisy=residual+sigma[:,None,None]*noise
    restored=received+model.denoise(noisy,received,sigma)
    weight=(sigma.square()+model.sigma_data**2)/(sigma*model.sigma_data).square()
    loss=weight*(restored-clean.detach()).square().mean((1,2))
    return restored,dict(diffusion_mse=loss,diffusion_sigma=sigma)


def scratch_residual_diffusion(seed):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed);model=ResidualDiffusionRestorer()
    model.initialization_report=dict(mode='scratch_joint_single30',model_seed=seed,ancestor_checkpoints=[],
        training_stages=1,teacher_features_are_loss_only=True,pretrained_diffusion=False)
    return model
