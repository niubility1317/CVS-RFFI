"""Matched LoRa long fits with conditional-uncertainty and fixed readouts."""

from functools import partial

import numpy as np
import torch

from .channel_blind import corruption_score
from .channel_blind_source import FitRecipe, clean_threshold
from .data import SOURCE
from .dura_repair import complex_iq, real_iq
from .joint_spectrum_source import evaluate, smoke
from .joint_unroll_source import coordinates
from .posterior_transport import (PosteriorTransport, condition_waveforms, initialize, objective,
                                  transport_strength)
from .spectral_block_source import main
from .variance_transport import physical_norm


RUN = '20261124-phase1-lora-posterior-transport-source-m4-r01'
MODES = ('direct','reference','fixed_coarse','fixed_final','adaptive_coarse','adaptive_final',
         'same_weight_fixed','same_weight_mean','same_weight_half','unit_action','fixed_noise','no_feedback')
COMPARISONS = (('adaptive_final','fixed_final'), ('adaptive_coarse','fixed_coarse'),
               ('adaptive_final','same_weight_fixed'), ('adaptive_final','same_weight_mean'),
               ('adaptive_final','same_weight_half'), ('adaptive_final','adaptive_coarse'),
               ('adaptive_final','unit_action'), ('adaptive_final','fixed_noise'),
               ('adaptive_final','no_feedback'),
               *((name, base) for name in MODES[2:] for base in ('direct','reference')))
FIELDS = ('active','measured_common','measured_private','held_mismatch','estimated_action',
          'mean_variance_common','mean_variance_private','strength_common','strength_private',
          'coarse_rms','final_rms','fixed_rms','mean_rms')


@torch.no_grad()
def calibrate(model,tracker,calibration):
    threshold = float(clean_threshold(tracker,calibration))
    rng = torch.Generator(device=calibration.device).manual_seed(2026112463)
    summary = []
    rows = calibration.unflatten(0,(len(SOURCE),-1))
    samples = rows[:,:min(rows.shape[1],100)].flatten(0,1)
    for batch in samples.split(24):
        source = complex_iq(batch)
        wave = condition_waveforms(source,rng)
        received, phase, _, rms = coordinates(tracker,real_iq(wave))
        context = model.observe(received)
        last,variance = model.estimate(received,rms,context)[1]
        truth = model.spectral(source/(phase*rms))
        mask = model.frequency!=0.
        standardized = ((context[0]+last-truth).abs().square()/variance)[...,mask]
        summary.append(torch.stack((standardized[:,0].mean(-1),standardized[:,1:].mean((1,2))),-1))
    values = torch.cat(summary)
    if not torch.isfinite(values).all():
        raise FloatingPointError('Nonfinite source-only uncertainty readback')
    return threshold,dict(variance_diagnostic_packets=len(samples),
                          standardized_squared_error_median=values.median(0).values.tolist())


RECIPES = {
    name: FitRecipe(name,40_000,partial(PosteriorTransport,adaptive=name=='adaptive'),
                    condition_waveforms,None,coarse_first=True,batch=24,fit_per_rx=4000,
                    validation_steps=(1800,6000,12000,20000,30000),
                    initialize=initialize,finalize_calibration=calibrate)
    for name in ('fixed','adaptive')
}


def recover(iq,tracker,models,threshold,*,compare=False):
    y, phase, _, rms = coordinates(tracker,iq)
    active = corruption_score(y)>threshold
    count = int(active.sum())
    out = iq.expand(2*len(models)+(6 if compare else 0),*iq.shape).clone()
    evidence = iq.new_zeros((len(iq),len(FIELDS)))
    if not count:
        return out,count,evidence
    y, phase, rms = y[active],phase[active],rms[active]
    edits = []
    for model in models:
        context = model.observe(y)
        first,last = model.estimate(y,rms,context)
        noise = context[2]
        edits.extend(model.readout(y,rms,stage,noise) for stage in (first,last))
    if compare:
        fixed = model.readout(y,rms,last,noise,adaptive=False)
        mean = model.waveform(last[0])
        edits.extend((fixed,mean,.5*mean))
        for override in (dict(use_action=False),dict(use_noise=False)):
            swapped=model.observe(y,**override)
            other=model.estimate(y,rms,swapped)[1]
            edits.append(model.readout(y,rms,other,swapped[2]))
        disconnected = model.estimate(y,rms,context,use_feedback=False)[1]
        edits.append(model.readout(y,rms,disconnected,noise))
        strength = transport_strength(last[1],noise)
        size = physical_norm(y,rms).flatten().clamp_min(1e-8)
        measured = [physical_norm(e,rms).flatten()/size for e in (edits[2],edits[3],fixed,mean)]
        evidence[active]=torch.stack((torch.ones_like(size),noise[:,0,0],noise[:,1,0],
                         context[4].mean((1,2)),(context[3]-1).abs().mean((1,2)),
                         last[1][:,0].mean(-1),last[1][:,1:].mean((1,2)),
                         strength[:,0].mean(-1),strength[:,1:].mean((1,2)),*measured),1)
    wave = torch.stack(edits)*phase*rms
    out[:,active] = iq[active]+torch.stack((wave.real,wave.imag),dim=-2)
    return out,count,evidence


def render(iq,_unused,baseline,tracker,models,*,threshold,measurements=None,switch=None):
    reference,_ = baseline(iq)
    predictions,count,evidence = recover(iq,tracker,models,threshold,compare=True)
    if measurements is not None:
        measurements.append(np.asarray(evidence.tolist()))
    return torch.cat((torch.stack((iq,reference)),predictions)),count


if __name__=='__main__':
    main(run=RUN,runner=__spec__.name,recipes=RECIPES,modes=MODES,comparisons=COMPARISONS,calculate=objective,
         run_smoke=partial(smoke,render_fn=render,recover_fn=recover),
         run_evaluate=partial(evaluate,fields=FIELDS,render_fn=render))
