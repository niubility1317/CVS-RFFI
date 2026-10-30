"""Scratch E30: double the IQ-error phase component in the strongest FFT quartile."""
import math
import torch
from . import no_variance
from experiments.pamf_dura_efficiency.model import direct_objective as source_objective
from experiments.pamf_restoration.models import complex_iq
base=no_variance.base
RUN='20261031-phase1-DURA-strongPhase-E30-s2026092701-r01'
MODULE='experiments.pamf_dura_direct_train.strong_phase'


def learning_rate(update):
    if not 0<=update<7410:raise ValueError('Fixed E30 update budget required')
    if update<247:return .001*(update+1)/247
    return .00002+(.001-.00002)*.5*(1+math.cos(math.pi*(update-247)/(7410-247-1)))


def strong_phase_component(restored,clean):
    reference=torch.fft.fft(complex_iq(clean.detach()),norm='ortho')
    predicted=torch.fft.fft(complex_iq(restored),norm='ortho')
    power=reference.abs().square()
    order=torch.argsort(power,dim=1,descending=True,stable=True)[:,:64]
    cross=predicted*reference.conj()
    component=2*(cross.abs()-cross.real).clamp_min(0)
    # This is exactly the phase part of complex IQ squared error.
    # Match the original per-packet IQ normalization, including its floor.
    scale=512*clean.detach().square().mean((1,2)).clamp_min(1e-4)
    value=torch.gather(component,1,order).sum(1)/scale
    share=torch.gather(power,1,order).sum(1)/power.sum(1).clamp_min(1e-4)
    return value,share


def objective(student,received,clean,action,variance,probes):
    _,terms,restored=source_objective(student,received,clean,action,variance,probes,return_restored=True)
    phase,share=strong_phase_component(restored,clean)
    loss=terms['iq_nmse']+.1*terms['prior_nmse']+.05*terms['spectral_error']+.05*terms['action_probe_nmse']+phase
    return loss,dict(terms,strong_phase_nmse=phase,strong_clean_energy_share=share)


def configure():
    no_variance.configure();previous_configuration=base.configuration;previous_command=base.command
    def configuration():
        value=previous_configuration()
        value.update(run=RUN,recipe='scratch_E30_strong_quartile_phase_component',epochs=30,updates=7410,
            new_cumulative_epochs=30,L_exposures=189000,U_exposures=1701000,corruption_epochs=20,
            source_freeze='fixedfinalE30/7410,noearlystop/source-target-ranking',
            objective='noVar4base + strongest64cleanFFTbins phasecomponent of IQMSE;phaseweight1to2;no phase quotient',
            strong_phase_extra_weight=1.,strong_frequency_bin_count=64,frequency_partition='PerpacketsourcecleanFFTpowerdescending/stableties;traininglossonly',
            extra_loss_metrics=['strong_phase_nmse','strong_clean_energy_share'],
            source_rationale='SourcephaseQ1oracle noVarcached01/20 improvescleanFchoice478to488/496to502;onesourcemidscene loses1;proxy notTXaccuracy/guarantee',
            teacher_model=None,training_F_access=False,F_guided_training=False,source_F_informed_method_research=True,
            training_stages=1,initialization_checkpoint_loads=0,parameters=266951,
            additional_training_FFTs_per_student_forward=2,additional_inference_FFTs=0,
            allowed_GPUs=[4,6,7],max_total_visible_GPU_processes=3,host_numpy_conversion='DLPack_owned_FP32',
            optimization_control='Samebudget E30physicalreadout;physicalGnoFgradient butwholeoldprocessF-guided/costdisclosed/not independentF-free')
        return value
    def command(args,mode,arm=None,role=None):
        return [MODULE if v==no_variance.MODULE else v for v in previous_command(args,mode,arm,role)]
    base.RUN=RUN;base.configuration=configuration;base.command=command;base.learning_rate=learning_rate
    base.direct_objective=objective


if __name__=='__main__':configure();base.main()
