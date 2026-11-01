import torch
from phase1_axial_spectral_mask import AxialSpectralMask

def main():
    torch.manual_seed(91);torch.set_num_threads(2);model=AxialSpectralMask();x=torch.randn(2,2,256)
    y=model(x);torch.testing.assert_close(y,x,rtol=2e-5,atol=2e-5)
    with torch.no_grad():model.output.weight.normal_(0,.01);model.temporal_output.weight.normal_(0,.01)
    z=model(x);(z.square().mean()).backward()
    for grad in (model.output.weight.grad,model.temporal_output.weight.grad,model.time_attention.in_proj_weight.grad,model.freq_attention.in_proj_weight.grad,model.temporal_input.weight.grad):
        assert grad is not None and torch.isfinite(grad).all() and grad.abs().sum()>0
    assert torch.isfinite(z).all()
    print('VERIFIED direct IQ network, identity initialization and temporal/spectral gradients')
if __name__=='__main__':main()
