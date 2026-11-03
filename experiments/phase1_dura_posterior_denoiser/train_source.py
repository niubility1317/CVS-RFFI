"""Train deterministic antithetic IQ denoising trajectories on paired source."""
from experiments.phase1_dynamic_volterra import train_source as base
from experiments.phase1_dura_posterior_denoiser.model import PosteriorDenoiser, denoising_loss
from experiments.phase1_dura_safe_memory.objective import protected_source_objective
from experiments.phase1_dura_volterra.pipeline import load_components, input_noise, predict_batch

_current=[None]


def make_model():
    _current[0]=PosteriorDenoiser()
    return _current[0]


def denoiser_training_objective(reader,output,anchor,clean,clean_features,groups,received,log_variance):
    if _current[0] is None: raise RuntimeError('Current denoiser model required')
    protected,terms=protected_source_objective(reader,output,anchor,clean,clean_features,groups)
    denoising=denoising_loss(_current[0],received,anchor,clean,log_variance,groups)
    return protected+.25*denoising,dict(terms,denoising=denoising)


def main():
    names=('DynamicVolterra','SCHEMA','LOG_PREFIX','training_source_objective','USES_DURA',
           'load_components','input_noise','predict_batch','extra_frozen_components')
    old=tuple(getattr(base,name) for name in names)
    new=(make_model,'phase1_dura_posterior_denoiser_v1','DURA_POSTERIOR_DENOISER',denoiser_training_objective,True,
         load_components,input_noise,predict_batch,lambda dura,eye:(dura,))
    for name,value in zip(names,new):setattr(base,name,value)
    try:base.main()
    finally:
        for name,value in zip(names,old):setattr(base,name,value)
        _current[0]=None


if __name__=='__main__':main()
