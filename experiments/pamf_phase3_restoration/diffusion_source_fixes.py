"""Two fixed source-motivated hypotheses: early readout and matched-scale training."""
import torch


def early_denoised4(model,received):
    state=model.fixed_latent.expand_as(received)*model.sigma_steps[0]
    for index in range(4):
        sigma,next_sigma=model.sigma_steps[index:index+2]
        denoised=model.denoise(state,received,sigma.expand(len(received)))
        if index==3:return received+denoised
        derivative=(state-denoised)/sigma;proposal=state+(next_sigma-sigma)*derivative
        correction=model.denoise(proposal,received,next_sigma.expand(len(received)))
        next_derivative=(proposal-correction)/next_sigma
        state=state+(next_sigma-sigma)*(derivative+next_derivative)/2


def mix_sigmas(original,steps,update):
    # A per-update private generator preserves the parent's global sigma/noise RNG.
    # Its state is reproducible from the saved update, so no extra recovery state.
    generator=torch.Generator(device=original.device).manual_seed(2026110205+update)
    use=torch.rand(len(original),device=original.device,generator=generator)<.5
    index=torch.randint(8,(len(original),),device=original.device,generator=generator)
    return torch.where(use,steps[index],original),torch.where(use,index+1,torch.zeros_like(index))
