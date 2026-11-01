"""Scratch TF action estimator and full-covariance Gaussian IQ posterior."""
import math
import torch
from torch import nn
from .covariance_inverse import dense_action,gaussian_posterior


class CovarianceMAP(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer('prior_mean',torch.zeros(512))
        self.register_buffer('prior_covariance',torch.eye(512))
        self.register_buffer('coordinates',torch.linspace(-1,1,256)[None,None])
        self.register_buffer('impulse',torch.nn.functional.one_hot(torch.tensor(0),64).float())
        self.time_patch=nn.Conv1d(5,128,4,stride=4)
        self.frequency_patch=nn.Conv1d(5,128,4,stride=4)
        self.position=nn.Parameter(torch.empty(1,128,128));nn.init.normal_(self.position,std=.02)
        layer=nn.TransformerEncoderLayer(128,4,512,dropout=0.,activation='gelu',batch_first=True,norm_first=True)
        self.encoder=nn.TransformerEncoder(layer,6,enable_nested_tensor=False)
        self.norm=nn.LayerNorm(128)
        self.kernel_head=nn.Linear(128,128)
        self.coefficient_head=nn.ConvTranspose1d(128,4,4,stride=4)
        self.noise_head=nn.Linear(128,1)
        for head in (self.kernel_head,self.coefficient_head,self.noise_head):
            nn.init.zeros_(head.weight);nn.init.zeros_(head.bias)

    def bind_prior(self,mean,covariance):
        if mean.shape!=(512,) or covariance.shape!=(512,512) or not torch.isfinite(mean).all() or not torch.isfinite(covariance).all():
            raise ValueError('Matched original-source IQ mean and full covariance required')
        covariance=covariance.to(self.prior_covariance)
        torch.linalg.cholesky(covariance.double())
        self.prior_mean.copy_(mean.to(self.prior_mean));self.prior_covariance.copy_(covariance)

    def estimate(self,received):
        if received.ndim!=3 or received.shape[1:]!=(2,256) or received.dtype!=torch.float32:
            raise ValueError('Current per-record FP32 received IQ2x256 required')
        n=len(received);mean=self.prior_mean.reshape(1,2,256).expand(n,-1,-1)
        coordinates=self.coordinates.expand(n,-1,-1)
        time=self.time_patch(torch.cat((received,mean,coordinates),1)).transpose(1,2)
        fft=lambda value:torch.view_as_real(torch.fft.fft(torch.complex(value[:,0],value[:,1]),norm='ortho')).transpose(1,2)
        frequency=self.frequency_patch(torch.cat((fft(received),fft(mean),coordinates),1)).transpose(1,2)
        tokens=self.norm(self.encoder(torch.cat((time,frequency),1)+self.position));pooled=tokens.mean(1)
        h=self.kernel_head(pooled);kernel=torch.complex(self.impulse+h[:,:64],h[:,64:])
        kernel=kernel/kernel.abs().square().sum(1,keepdim=True).sqrt().clamp_min(1e-6)
        c=self.coefficient_head(tokens[:,:64].transpose(1,2))
        main=1+torch.complex(c[:,0],c[:,1]);image=torch.complex(c[:,2],c[:,3])
        log_variance=(math.log(.0005)+self.noise_head(pooled).squeeze(1)).clamp(math.log(1e-7),0.)
        variance=log_variance.exp()
        action=dense_action(kernel.to(torch.complex128),main.to(torch.complex128),image.to(torch.complex128))
        return dict(action_matrix=action,variance=variance)

    def forward(self,received,*,return_aux=False):
        aux=self.estimate(received);n=len(received)
        restored,residual=gaussian_posterior(aux['action_matrix'],received.flatten(1).double(),self.prior_mean.double(),self.prior_covariance.double(),aux['variance'].double())
        restored=restored.float().reshape(n,2,256)
        return (restored,dict(**aux,relative_solve_residual=residual)) if return_aux else restored


def source_action_loss(aux,kernel,coefficients,variance):
    target=dense_action(torch.complex(kernel[:,0],kernel[:,1]).to(torch.complex128),
        torch.complex(coefficients[:,0],coefficients[:,1]).to(torch.complex128),
        torch.complex(coefficients[:,2],coefficients[:,3]).to(torch.complex128))
    error=(aux['action_matrix']-target).square().sum((1,2))/target.square().sum((1,2)).clamp_min(1e-8)
    noise_error=(aux['variance'].log()-variance.clamp_min(1e-7).log()).square()
    return .1*error.float()+.001*noise_error,dict(action_matrix_relative_mse=error.float(),noise_log_mse=noise_error)


def scratch_covariance(seed):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed);model=CovarianceMAP()
    model.initialization_report=dict(mode='scratch_single30_with_source_statistic_buffers',model_seed=seed,
        ancestor_checkpoints=[],training_stages=1,prior_is_fitted_source_IQ_statistics=True)
    return model
