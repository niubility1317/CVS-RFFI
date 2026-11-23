"""Run the matched LoRa local-phase window study and engineering controls."""

from functools import partial
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from experiments.cric_dura_comparison.io import write

from .channel_blind import corruption_score
from .channel_blind_source import FitRecipe, panel, receive_only, setup
from .complex_vector_source import load_reader, score
from .data import SOURCE, VIEWS
from .dura_repair import complex_iq, real_iq
from .joint_unroll_source import coordinates, source_control
from .local_phase import LocalPhase, generic_phase_observation, objective, rotation
from .local_wiener_source import freeze
from .spectral_block_source import load_models, main


RUN = '20261124-phase1-lora-local-phase-source-m4-r01'
RECIPES = {name:FitRecipe(name,40_000,partial(LocalPhase,edges),generic_phase_observation,None,
                          batch=24,fit_per_rx=4000,validation_steps=(1800,6000,12000,20000,30000))
           for name, edges in (('middle',False),('edges',True))}
MODES = ('direct','reference','sync','sync_action','middle_coarse','middle_full','middle_half',
         'edges_coarse','edges_full','edges_half','edges_without_action','edges_without_noise','edges_learned')
COMPARISONS = (('sync','direct'),('sync_action','sync'),('edges_coarse','middle_coarse'),
               ('edges_full','middle_full'),('edges_half','middle_half'),('edges_full','edges_coarse'),
               ('edges_full','edges_without_action'),('edges_full','edges_without_noise'),
               ('edges_full','edges_learned'),*((name,'reference') for name in MODES[2:]))
DIAGNOSTICS = ('active','outer_variance','repeat_disagreement','relative_fit_confidence',
               'middle_rms_dose','edge_rms_dose','largest_amplitude_error')


def recover(iq, tracker, models, threshold, *, controls=False):
    wave = complex_iq(iq)
    y, frame, quality, _ = coordinates(tracker,iq)
    enabled = corruption_score(y)>threshold
    energy = wave.abs().square()
    output = [iq.clone() for _ in range(3*len(models)+(5 if controls else 0))]
    summary = iq.new_zeros((len(iq),len(DIAGNOSTICS))) if controls else None
    active = int(enabled.sum())
    if not active:
        return output, active, summary
    z, q, power = y[enabled],quality[enabled],energy[enabled]
    received = wave[enabled]
    updates = []
    for model in models:
        coarse, fine, estimates = model.phases(z,q,power)
        updates.extend((received*rotation(coarse),received*rotation(fine),received*rotation(.5*fine)))
    if controls:
        protocol = tracker.reference.chirp
        measured = frame[enabled]*protocol.conj()
        correction = measured[..., :1]*measured.conj()
        synced = received*correction
        _, action, noise, disagreement, confidence = estimates
        regular_inverse = 1.05*action.conj()/(action.abs().square()+.05)
        equalized = synced*(1.+confidence*(regular_inverse-1.))
        edge = models[-1]
        ablations = [received*rotation(edge.phases(z,q,power,use_action=a,use_noise=n)[1])
                     for a,n in ((False,True),(True,False),(False,False))]
        magnitude_change = (updates[4].abs()-received.abs()).abs().amax((1,2))
        rms = power.sum((1,2)).clamp_min(1e-8)
        dose = [(v-received).abs().square().sum((1,2)).div(rms).sqrt() for v in (updates[1],updates[4])]
        summary[enabled] = torch.stack((torch.ones_like(rms),noise.mean((1,2)),disagreement.mean((1,2)),
                                        confidence.mean((1,2)),*dose,magnitude_change),dim=1)
        updates = [synced,equalized,*updates,*ablations]
    for out, value in zip(output,updates,strict=True):
        out[enabled] = real_iq(value)
    return output, active, summary


def render(iq, _unused, reference, tracker, models, *, threshold, diagnostics=None):
    baseline, _ = reference(iq)
    waves, active, metrics = recover(iq,tracker,models,threshold,controls=True)
    if diagnostics is not None:
        diagnostics.append(np.asarray(metrics.tolist(),dtype=np.float32))
    return torch.stack((iq,baseline,*waves)), active


@torch.no_grad()
def smoke(rt, recipes, modes):
    device = 'cuda:0'
    root = Path(rt['run_root'])
    (_, source, _), contract = setup(rt,device,smoke=True)
    models, threshold = load_models(root,contract,device,recipes)
    _, _, expected, baseline, tracker = source_control(rt,'full',device)
    if contract != expected:
        raise ValueError('LoRa no-query smoke must use the matching source capture')
    reader = load_reader(root,contract,device)
    rng = torch.Generator(device=device).manual_seed(2026112329)
    candidate = real_iq(generic_phase_observation(complex_iq(source),rng))
    scores = corruption_score(coordinates(tracker,candidate)[0])
    disturbed = candidate[scores.argmax().view(1)]
    received = torch.cat((source[:1],disturbed))
    waves, active = render(received,None,baseline,tracker,models,threshold=threshold)
    logits = reader(waves.flatten(0,1))[0]
    phase_only = waves[4:]
    amplitude_error = (phase_only.square().sum(-2)-received.square().sum(-2)).abs().max()
    if logits.shape != (2*len(modes),10) or not torch.isfinite(logits).all() or not active or amplitude_error>1e-4:
        raise ValueError('LoRa real no-query smoke has invalid predictions or phase amplitude')
    times = {}
    for label, iq in (('source_clean',source[:1]),('generic_disturbed',disturbed)):
        records = []
        for i in range(28):
            torch.cuda.synchronize()
            begun = perf_counter()
            _, enabled, _ = recover(iq,tracker,(models[-1],),threshold)
            torch.cuda.synchronize()
            if i>=8:
                records.append(1000*(perf_counter()-begun))
        times[label] = dict(active=bool(enabled),median_ms=float(np.median(records)))
    write(root/'smoke.json',dict(status='PASS',target_reads=0,V_reads=0,max_power_error=float(amplitude_error),
          parameters={name:sum(p.numel() for p in model.parameters()) for name,model in zip(recipes,models,strict=True)},
          b1=times, device=torch.cuda.get_device_name()))


def summarize_diagnostics(root, route, chosen, diagnostics):
    by_scene = np.concatenate(diagnostics).reshape(len(VIEWS),len(chosen),len(DIAGNOSTICS))
    result = {}
    for key, rows in zip(VIEWS,by_scene,strict=True):
        quantiles = {}
        for receiver, entries in zip(SOURCE,np.split(rows,len(SOURCE)),strict=True):
            use = entries[:,0] > 0
            stat = {'active':int(use.sum())}
            if use.any():
                bounds = np.quantile(entries[use,1:],(.1,.5,.9),axis=0).T.tolist()
                stat.update(zip(DIAGNOSTICS[1:],bounds,strict=True))
            quantiles[receiver] = stat
        result[key] = quantiles
    write(root/f'{route}-estimates.json',dict(quantiles=[.1,.5,.9],by_scene=result))


def evaluate(rt, route, recipes, modes):
    device = 'cuda:0'
    root = Path(rt['run_root'])
    binding = source_control(rt,route,device)
    models, threshold = load_models(root,binding[2],device,recipes)
    diagnostics = []

    def finalize(output, route, source, chosen, ids, predictions, metrics, modes):
        summarize_diagnostics(output,route,chosen,diagnostics)
        freeze(output,route,source,chosen,ids,predictions,metrics,modes)

    score(rt,route,panel=panel,modes=modes,render=partial(render,threshold=threshold,diagnostics=diagnostics),
          binding=binding,frozen_models=models,require_clean_bypass=False,replay_fn=receive_only,finalize=finalize)


if __name__=='__main__':
    main(run=RUN,runner=__spec__.name,recipes=RECIPES,modes=MODES,comparisons=COMPARISONS,
         calculate=objective,run_smoke=smoke,run_evaluate=evaluate)
