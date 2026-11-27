"""Matched source study of differentiable LoRa recovery and PAMF W24."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
from functools import partial
import importlib
import json
import multiprocessing
import os
from pathlib import Path, PurePosixPath
import re
import subprocess
import sys
from time import monotonic, sleep

import numpy as np
import torch
from torch.nn import functional as F

# Spawned CPU replay workers must import from the same archived W24 CWD
# before the parent's activated historical paths.
if __name__ == '__mp_main__':
    sys.path.insert(0, str(Path.cwd()))

if not __package__:
    from experiments import pamf_lora_main
    pamf_lora_main.__path__.append(str(Path(__file__).resolve().parent))

from experiments.cric_dura_comparison.io import read, tensor, write
from experiments.pamf_lora_main.dura_repair import real_iq
from experiments.pamf_lora_main.full_packet_views import native, source_contract, source_role, target_contract
from experiments.pamf_lora_main.packet_runner import schedule
from experiments.pamf_lora_main.runner import activate, available
from experiments.pamf_lora_main.w24_edit_cross import check_controls
from experiments.pamf_lora_main.w24_factor_audit import AUG_RUN, load_classifier, panel, score_source
from experiments.pamf_lora_main.w24_formal import FULL, RUN_OUTER, append_blind_prediction, receipt, wide_delta
from experiments.pamf_lora_main.w24_native_full import (
    DATA_RUN, LIMIT, SOURCE, VIEWS, W24_COMMIT, W24_RUN, channel_job, classify_aperture,
)
from experiments.pamf_lora_main.w24_observed_dose import FULL_CASES, prior_target, source_receive
from experiments.pamf_lora_main.w24_local_covariance import CovarianceRefiner
from experiments.pamf_lora_main import w24_native_training as raw_train
from experiments.pamf_lora_main.w24_second_stage import SecondStage
from experiments.pamf_lora_main.w24_strong_source import (
    COLOR_CUTOFF_HZ, STRONG_FRACTION, STRONG_SNR_DB, PassbandMatchedNoise, StrongSourceMix,
)
from experiments.pamf_lora_main.w24_task_gain import TaskGain
from experiments.pamf_lora_main.w24_third_prior import UNET_FIDELITY_ARMS, UNET_FIDELITY_RUN, load_candidate
from experiments.pamf_lora_main.w24_wide_transfer import TRAIN_RUN, load_restorer, received_aperture, training_runtime
from experiments.pamf_lora_main.w24_outer_objective_source import RUN as DURA_RUN


RUN = '20261126-phase1-lora-w24-joint-pamf-s2026112991-r01'
RUN_PL = '20261126-phase1-lora-w24-joint-pamf-clean-pl-s2026112991-r01'
RUN_FIXED_PL = '20261126-phase1-lora-w24-fixed-e104-clean-pl-s2026112991-r01'
RUN_FEATURE = '20261126-phase1-lora-w24-clean-feature-recovery-s2026112993-r01'
RUN_REFINE = '20261126-phase1-lora-w24-second-stage-s2026112995-r01'
RUN_STRONG = '20261127-phase1-lora-w24-strong-source-s2026112997-r01'
RUN_COLOR = '20261127-phase1-lora-w24-colored-source-s2026112999-r01'
RUN_GAIN = '20261127-phase1-lora-w24-task-gain-s2026113003-r01'
RUN_COV = '20261127-phase1-lora-w24-local-covariance-s2026113005-r01'
RUN_VIEW = '20261127-phase1-lora-w24-paired-view-risk-s2026113007-r01'
ARMS = ('pamf_only', 'frozen_pamf_ce', 'joint')
PL = 'joint_clean_pl'
FIXED_PL = 'fixed_e104_clean_pl'
CONTINUED_PL, FEATURE = 'continued_clean_pl', 'clean_feature'
CONTINUATION_ARMS = (CONTINUED_PL, FEATURE)
RAW_REFINE, REFINE = 'raw_second_stage', 'conditioned_second_stage'
REFINE_ARMS = (RAW_REFINE, REFINE)
STRONG_CONTROL, STRONG = 'generic_source', 'strong_source'
STRONG_ARMS = (STRONG_CONTROL, STRONG)
WHITE, COLOR = 'white_source', 'colored_source'
COLOR_ARMS = (WHITE, COLOR)
CONSTANT_GAIN, ADAPT_GAIN = 'constant_gain', 'adaptive_gain'
GAIN_ARMS = (CONSTANT_GAIN, ADAPT_GAIN)
DIAGONAL_COV, CROSS_COV = 'diagonal_covariance', 'cross_chirp_covariance'
COV_ARMS = (DIAGONAL_COV, CROSS_COV)
MEAN_VIEW, HARD_VIEW = 'mean_view_risk', 'hard_view_risk'
VIEW_ARMS = (MEAN_VIEW, HARD_VIEW)
AUGMENTED_ARMS = (*STRONG_ARMS, *COLOR_ARMS, *GAIN_ARMS, *COV_ARMS, *VIEW_ARMS)
PARENT_TRIALS = {RUN_FEATURE: CONTINUATION_ARMS, RUN_REFINE: REFINE_ARMS, RUN_STRONG: STRONG_ARMS,
                 RUN_COLOR: COLOR_ARMS, RUN_GAIN: GAIN_ARMS, RUN_COV: COV_ARMS, RUN_VIEW: VIEW_ARMS}
PARENT_ARMS = tuple(arm for arms in PARENT_TRIALS.values() for arm in arms)
PARENT_CONTROLS = {candidate: control for control, candidate in PARENT_TRIALS.values()}
PSEUDO_ARMS = {RUN_PL: PL, RUN_FIXED_PL: FIXED_PL}
FIXED_READER_ARMS = ('frozen_pamf_ce', FIXED_PL, *PARENT_ARMS)
USE_PSEUDO = (*PSEUDO_ARMS.values(), *PARENT_ARMS)
PL_THRESHOLD = .95
SEED, STEPS, BATCH, PRED_BATCH = 2026112991, 6000, 16, 32
FEATURE_SEED, FEATURE_WEIGHT = 2026112993, 16.
REFINE_SEED, REFINE_STEPS = 2026112995, 9000
STRONG_SEED, STRONG_STEPS = 2026112997, 10000
COLOR_SEED = 2026112999
GAIN_SEED = 2026113003
COV_SEED, COV_STEPS = 2026113005, 9000
VIEW_SEED, VIEW_STEPS, VIEW_BATCH = 2026113007, 12000, 8
VIEW_TILT, VIEW_TEMPERATURE = .75, .25
FIXED_PL_COMMIT = '37215dd615d0f651ffebe87d6aaba6af0ab7a847'
STRONG_COMMIT = '76385bf9174440cc4e437a47bfc2fd7aff1b1fcb'
CENTER, CANDIDATE = 384, 'frozen_pamf_ce'
FORMAL_ARMS = {RUN: CANDIDATE, RUN_FIXED_PL: FIXED_PL,
               **{run: candidate for run, (_, candidate) in PARENT_TRIALS.items()},
               **{run: arms[-1] for run, arms in raw_train.RUNS.items()}}
MODES = ('direct', 'outer_half', 'unet_parent', *ARMS, 'joint_g_fixed_e104', 'joint_f_parent_g')
CONTRASTS = (('joint','pamf_only'), ('joint','frozen_pamf_ce'), ('joint','unet_parent'),
             ('joint','joint_g_fixed_e104'), ('joint','joint_f_parent_g'),
             ('frozen_pamf_ce','unet_parent'), ('frozen_pamf_ce','outer_half'))
FIT_FIELDS = ('project','frozen_sources','run_root','source_root','classifier_root','training_release_root','unet_parent')


def allowed_fit_runtime(rt, root):
    return {**{k: rt[k] for k in FIT_FIELDS}, 'run_root': str(root)}


def training_labels(source, contract):
    # Only L_s may supply truth during fit.
    with np.load(source/'L_s_metadata.npz', allow_pickle=False) as meta:
        labels = meta['y']
        if meta['ids'].tolist() != contract['role_ids']['L_s'] or labels.shape != (contract['counts']['L_s'],):
            raise ValueError('Invalid registered source L_s target mapping')
    if labels.dtype.kind not in 'iu' or (labels.min(), labels.max()) != (0, 9):
        raise ValueError('Expected the ten source classifier labels')
    return labels.astype(np.int64, copy=False)


def normalize_aperture(iq):
    scale = iq[..., :CENTER].abs().square().mean(dim=(1,2), keepdim=True).clamp_min(1e-12).sqrt()
    return real_iq(iq/scale), scale


def recovered(g, tracker, observation):
    update = .5 * wide_delta(g, tracker, observation)
    return torch.cat((observation[..., :CENTER]+update, observation[..., CENTER:]), dim=-1)


def boot(rt, device):
    activate(rt)
    project = Path(rt['project'])
    if (Path(rt['unet_parent']) != project/'runs'/UNET_FIDELITY_RUN
            or Path(rt['classifier_root']) != project/'runs'/AUG_RUN):
        raise ValueError('Expected exact lawful source parent and registered E104')
    contract = source_contract(Path(rt['source_root']))
    _, tracker, _, _ = load_restorer(*training_runtime(rt), contract, device)
    classifier = load_classifier(rt, contract, device)
    g, _ = load_candidate(Path(rt['unet_parent']), contract, device, UNET_FIDELITY_ARMS[0])
    return contract, classifier, g, tracker


def reduce_view_risk(loss, tilt):
    """Reduce two classification losses from the same physical source packet."""
    mean = loss.mean(0)
    if not tilt:
        return mean
    # Centering preserves equal-view loss and avoids magnifying large common losses.
    deviation = (loss - mean) / VIEW_TEMPERATURE
    soft_max = VIEW_TEMPERATURE * (deviation.logsumexp(0) - .69314718056)
    return mean + tilt * soft_max


def clean_pseudo_loss(f, anchor, clean_u, restored_u, *, preserve_feature=False, view_tilt=None):
    with torch.no_grad():
        teacher = anchor(clean_u)
        confidence, pseudo = teacher[0].softmax(-1).max(-1)
        active = confidence >= PL_THRESHOLD
    accepted = active.sum()
    if preserve_feature:
        logits, features = f(restored_u)
        ce = F.cross_entropy(logits,pseudo,reduction='none').masked_fill(~active,0).mean()
        feature = (1-F.cosine_similarity(features,teacher[1].detach(),dim=-1,eps=1e-6)).mean()
        return ce, accepted, feature
    if not active.any():
        return confidence.new_zeros(()), accepted
    if view_tilt is not None:
        views = restored_u.unflatten(0, (2, len(clean_u)))[:, active].flatten(0, 1)
        loss = F.cross_entropy(f(views)[0], pseudo[active].repeat(2), reduction='none').reshape(2, -1)
        return reduce_view_risk(loss, view_tilt).sum() / len(clean_u), accepted
    loss = F.cross_entropy(f(restored_u[active])[0],pseudo[active],reduction='sum') / len(clean_u)
    return loss, accepted


def loss_parts(g, f, anchor, tracker, generic, raw, label, rng, *, pseudo=False, preserve_feature=False, view_tilt=None):
    labeled = len(label)
    if labeled == 0 or len(raw) != 2 * labeled:
        raise ValueError('Expected equal labeled and unlabeled IQ groups, with labels only for the first group')
    raw = raw / raw.abs().square().mean(-1,keepdim=True).clamp_min(1e-12).sqrt()
    clean = tracker.aperture(raw)
    views = 1 if view_tilt is None else 2
    noisy = [tracker.aperture(generic(raw.unflatten(-1, (8,1024)), rng).flatten(-2)) for _ in range(views)]
    base, scale = normalize_aperture(torch.cat(noisy) if views == 2 else noisy[0])
    restore = recovered(g, tracker, base)
    restored = restore.unflatten(0, (views, len(raw)))
    clean_input, _ = normalize_aperture(clean[:labeled])
    clean_out = recovered(g, tracker, clean_input)
    f_restore, f_clean, f_clean_out = f(torch.cat((restored[:, :labeled].flatten(0, 1),
                                                  clean_input, clean_out)))[0].split((views*labeled,labeled,labeled))
    with torch.no_grad():
        teacher = anchor(clean_input)[0].softmax(-1)
    if views == 2:
        ce_views = F.cross_entropy(f_restore, label.repeat(2), reduction='none').reshape(2, labeled)
        restore_ce = reduce_view_risk(ce_views, view_tilt).mean()
    else:
        restore_ce = F.cross_entropy(f_restore, label)
    clean_ce = .5 * (F.cross_entropy(f_clean, label) + F.cross_entropy(f_clean_out, label))
    anchor_kl = .5 * (F.kl_div(f_clean.log_softmax(-1),teacher,reduction='batchmean')
                      + F.kl_div(f_clean_out.log_softmax(-1),teacher,reduction='batchmean'))
    paired_clean = torch.cat((clean, clean)) if views == 2 else clean
    target = real_iq(paired_clean/scale)[..., :CENTER]
    change = (restore[..., :CENTER]-target).permute(0,1,3,2).contiguous()
    ref = target.permute(0,1,3,2).contiguous()
    error = torch.fft.fft(torch.view_as_complex(change),norm='ortho')[...,1:].abs().square()
    power = torch.fft.fft(torch.view_as_complex(ref),norm='ortho')[...,1:].abs().square()
    spectral = (error/(.001+power+.03*power.mean(-1,keepdim=True))).log1p().mean()
    amplitude = change.square().mean()
    identity = (clean_out-clean_input).square().mean()
    parts = (restore_ce,clean_ce,anchor_kl,spectral,amplitude,identity)
    if pseudo:
        clean_u, _ = normalize_aperture(clean[labeled:])
        parts += clean_pseudo_loss(f,anchor,clean_u,restored[:, labeled:].flatten(0, 1),
                                   preserve_feature=preserve_feature,view_tilt=view_tilt)
    if views == 2:
        parts += ((ce_views[0] - ce_views[1]).abs().mean().detach(),)
    return torch.stack(parts)


def norm_gradient(values):
    return sum(float(x.detach().float().square().sum()) for x in values if x is not None)**.5


def source_parent(run):
    return RUN_STRONG if run in (RUN_COLOR,RUN_GAIN,RUN_COV,RUN_VIEW,*raw_train.RUNS) else RUN_FIXED_PL


def parent_modes(parent):
    modes = (*MODES,FIXED_PL)
    return (*modes,*STRONG_ARMS) if parent == RUN_STRONG else modes


def continuation_parent(rt, contract, f0, g0, device, parent=RUN_FIXED_PL):
    runs = Path(rt['project'])/'runs'
    g = load_row(runs/RUN_FIXED_PL,FIXED_PL,contract,f0,g0,device)[0]
    if parent == RUN_STRONG:
        g = load_row(runs/RUN_STRONG,STRONG,contract,f0,g,device)[0]
    return g


def training_recipe(arm):
    if arm in REFINE_ARMS:
        return REFINE_SEED, REFINE_STEPS, 8e-5, RUN_FIXED_PL
    if arm in CONTINUATION_ARMS:
        return FEATURE_SEED, STEPS, 4e-5, RUN_FIXED_PL
    if arm in STRONG_ARMS:
        return STRONG_SEED, STRONG_STEPS, 4e-5, RUN_FIXED_PL
    if arm in COLOR_ARMS:
        return COLOR_SEED, STRONG_STEPS, 4e-5, RUN_STRONG
    if arm in VIEW_ARMS:
        return VIEW_SEED, VIEW_STEPS, 4e-5, RUN_STRONG
    if arm in COV_ARMS:
        return COV_SEED, COV_STEPS, 8e-5, RUN_STRONG
    if arm in GAIN_ARMS:
        return GAIN_SEED, STEPS, 4e-4, RUN_STRONG
    return SEED, STEPS, 8e-5, UNET_FIDELITY_RUN


def initialize_recovery(arm, g, device):
    if arm in COV_ARMS:
        return CovarianceRefiner(g,cross_chirp=arm==CROSS_COV,seed=COV_SEED).to(device)
    if arm in REFINE_ARMS:
        return SecondStage(g,conditioned=arm==REFINE,seed=REFINE_SEED).to(device)
    if arm in GAIN_ARMS:
        return TaskGain(g,adaptive=arm==ADAPT_GAIN,seed=GAIN_SEED).to(device)
    return g


def fit(config, arm, *, smoke, device='cuda:0'):
    rt = read(config)
    if set(rt) != set(FIT_FIELDS):
        raise ValueError('Unexpected non-source input in joint training')
    out = Path(rt['run_root'])
    source = Path(rt['source_root'])
    contract, f, g, tracker = boot(rt, device)
    continuation = arm in CONTINUATION_ARMS
    refinement = arm in REFINE_ARMS
    gain_fit = arm in GAIN_ARMS
    covariance_fit = arm in COV_ARMS
    fixed_core = refinement or gain_fit or covariance_fit
    seed, budget, lr, checkpoint_parent = training_recipe(arm)
    if arm in PARENT_ARMS:
        g = continuation_parent(rt,contract,f,g,device,checkpoint_parent)
    g = initialize_recovery(arm,g,device)
    generic = importlib.import_module('lora_wide_dura_archive.dura_w24_model').condition_waveforms
    labeled, _, _, _ = source_role(source, contract, 'L_s')
    unlabeled, _, _, _ = source_role(source, contract, 'U_s')
    labels = training_labels(source,contract)
    anchor = deepcopy(f)
    learn_g, learn_f = arm != 'pamf_only', arm not in FIXED_READER_ARMS
    g.requires_grad_(learn_g).train(learn_g)
    if fixed_core:
        g.parent.eval().requires_grad_(False)
    g_parameters = tuple(p for p in g.parameters() if p.requires_grad)
    # Train the selected W24 weights against its deployed normalization and dropout behavior.
    f.requires_grad_(learn_f).eval()
    groups = []
    if learn_g:
        groups.append(dict(params=g_parameters, lr=lr))
    if learn_f:
        groups.append(dict(params=f.parameters(), lr=3e-5))
    optimizer = torch.optim.AdamW(groups,weight_decay=1e-4)
    steps = 1 if smoke else budget
    view_fit = arm in VIEW_ARMS
    view_tilt = (VIEW_TILT if arm == HARD_VIEW else 0.) if view_fit else None
    batch = 2 if smoke else VIEW_BATCH if view_fit else BATCH
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,steps,eta_min=2e-6)
    order = np.random.default_rng(seed)
    rng = torch.Generator(device=device).manual_seed(seed)
    if arm in AUGMENTED_ARMS:
        separate_rng = torch.Generator(device=device).manual_seed(seed+1)
        noise_shape = PassbandMatchedNoise(tracker,torch.Generator(device=device).manual_seed(seed+2)) if arm==COLOR else None
        generic = StrongSourceMix(generic,separate_rng,device,enabled=arm!=STRONG_CONTROL,noise_shape=noise_shape)
    pseudo = arm in USE_PSEUDO
    preserve_feature = arm == FEATURE
    weights = torch.tensor((1.,.65,.3,.30,.25,1.5) + ((1.,0.) if pseudo else ())
                           + ((FEATURE_WEIGHT if preserve_feature else 0.,) if continuation else ())
                           + ((0.,) if view_fit else ()),device=device)
    start = monotonic()
    ce_norm = None
    pseudo_ce_norm = None
    feature_norm = None
    accepted = torch.zeros((),device=device)
    for step in range(1,steps+1):
        l = order.choice(len(labeled), batch, replace=False)
        u = order.choice(len(unlabeled),len(l),replace=False)
        raw = np.concatenate((labeled[l],unlabeled[u]))
        raw = tensor(raw.view('float32'),device).view(torch.complex64)
        target = tensor(labels[l],device).long()
        if smoke and fixed_core:
            with torch.no_grad():
                aperture, _ = normalize_aperture(tracker.aperture(raw))
                torch.testing.assert_close(recovered(g,tracker,aperture),recovered(g.parent,tracker,aperture),
                                           rtol=0.,atol=0.)
        optimizer.zero_grad(set_to_none=True)
        parts = loss_parts(g,f,anchor,tracker,generic,raw,target,rng,pseudo=pseudo,preserve_feature=continuation,view_tilt=view_tilt)
        if pseudo:
            accepted += parts[7].detach()
        if not torch.isfinite(parts).all():
            raise FloatingPointError('Nonfinite source objective')
        if step == 1 and learn_g:
            ce_norm = norm_gradient(torch.autograd.grad(parts[0],g_parameters,allow_unused=True,retain_graph=True))
            if not (0 < ce_norm < float('inf')):
                raise ValueError('No supervised CE gradient reaches the restorer')
        if smoke and (arm == FIXED_PL or view_fit):
            if parts[7] == 0:
                raise ValueError('Real-checkpoint smoke needs accepted clean U_s pseudo labels')
            pseudo_ce_norm = norm_gradient(torch.autograd.grad(
                parts[6],g_parameters,allow_unused=True,retain_graph=True))
            if not (0 < pseudo_ce_norm < float('inf')):
                raise ValueError('No pseudo-label CE gradient reaches the restorer')
        if smoke and preserve_feature:
            feature_norm = norm_gradient(torch.autograd.grad(
                parts[8],g_parameters,allow_unused=True,retain_graph=True))
            if not (0 < feature_norm < float('inf')):
                raise ValueError('No paired clean-feature gradient reaches the restorer')
        (parts @ weights).backward()
        gradients = {name: norm_gradient(p.grad for p in model.parameters())
                     for name,model in (('G',g),('F',f))} if step in (1,steps) else None
        if gradients is not None and any(not ((0 < gradients[k] < float('inf')) if active else gradients[k] == 0)
                                         for k,active in (('G',learn_g),('F',learn_f))):
            raise ValueError('Unexpected source model gradient activity')
        torch.nn.utils.clip_grad_norm_([p for model in (g,f) for p in model.parameters() if p.requires_grad],
                                       1.,error_if_nonfinite=True)
        optimizer.step()
        scheduler.step()
        if step == 1 or step % 50 == 0:
            print('W24_JOINT_FIT', arm, step, *parts.detach().tolist(),flush=True)
    if device.startswith('cuda'):
        torch.cuda.synchronize()
    proof = dict(arm=arm, steps=steps, seed=seed, g_parent=checkpoint_parent, f_parent=AUG_RUN,
                 L_s=len(labeled), U_s=len(unlabeled), labeled_exposures=steps*batch,
                 unlabeled_exposures=steps*batch, U_TX_reads=0, V_reads=0, target_reads=0,
                 practical_reads=0, trace_reads=0, G_trainable=learn_g, F_trainable=learn_f)
    if pseudo:
        proof.update(U_PL_teacher=AUG_RUN,U_PL_threshold=PL_THRESHOLD,U_PL_accepted=int(accepted))
    if continuation:
        proof['feature_weight'] = FEATURE_WEIGHT if preserve_feature else 0.
    if refinement:
        proof['conditioned_on_first'] = arm == REFINE
    if gain_fit:
        proof['adaptive_gain'] = arm == ADAPT_GAIN
    if covariance_fit:
        proof['cross_chirp_covariance'] = arm == CROSS_COV
    if view_fit:
        proof.update(views_per_packet=2,view_tilt=view_tilt,view_temperature=VIEW_TEMPERATURE)
    if arm in AUGMENTED_ARMS:
        proof.update(strong_fraction=STRONG_FRACTION if generic.enabled else 0.,
                     strong_snr_db=list(STRONG_SNR_DB),strong_exposures=int(generic.selected))
    if arm in COLOR_ARMS:
        proof['color_cutoff_hz'] = list(COLOR_CUTOFF_HZ) if arm==COLOR else None
    summary = dict(**proof, status='PASS' if smoke else 'SOURCE_FROZEN', CE_to_G_norm=ce_norm,
                   gradients=gradients, worker_seconds=monotonic()-start,
                   peak_cuda_bytes=torch.cuda.max_memory_allocated() if device.startswith('cuda') else 0,
                   G_parameters=sum(x.numel() for x in g.parameters()),F_parameters=sum(x.numel() for x in f.parameters()))
    if smoke and (arm == FIXED_PL or view_fit):
        summary['pseudo_CE_to_G_norm'] = pseudo_ce_norm
    if smoke and preserve_feature:
        summary['feature_to_G_norm'] = feature_norm
    if fixed_core:
        summary['G_trainable_parameters'] = sum(p.numel() for p in g_parameters)
        summary['G1_gradient_norm'] = norm_gradient(p.grad for p in g.parent.parameters())
        if summary['G1_gradient_norm'] != 0:
            raise ValueError('Frozen first-stage network received gradients')
    if not smoke:
        with (out/'model.pt').open('xb') as stream:
            torch.save(dict(contract=contract, proof=proof,G=g.state_dict(),F=f.state_dict()),stream)
    write(out/('smoke.json' if smoke else 'complete.json'),summary)


def load_row(root, arm, contract, f0, g0, device):
    folder = root / arm
    summary = read(folder/'complete.json')
    state = torch.load(folder/'model.pt',map_location=device,weights_only=True)
    proof = state['proof']
    continuation = arm in CONTINUATION_ARMS
    refinement = arm in REFINE_ARMS
    gain_fit = arm in GAIN_ARMS
    covariance_fit = arm in COV_ARMS
    fixed_core = refinement or gain_fit or covariance_fit
    seed, budget, _, parent = training_recipe(arm)
    if (summary['status'] != 'SOURCE_FROZEN' or state['contract'] != contract
            or (proof['arm'],proof['steps'],proof['seed'],proof['g_parent'],proof['f_parent']) != (
                arm,budget,seed,parent,AUG_RUN)
            or any(proof[k] for k in ('U_TX_reads','V_reads','target_reads','practical_reads','trace_reads'))
            or (proof['G_trainable'],proof['F_trainable']) != (arm!='pamf_only',arm not in FIXED_READER_ARMS)
            or any(summary[k] != v for k,v in proof.items()) or read(folder/'smoke.json')['status'] != 'PASS'):
        raise ValueError(f'Invalid source freeze for {arm}')
    if arm in USE_PSEUDO and (proof.get('U_PL_teacher') != AUG_RUN
            or proof.get('U_PL_threshold') != PL_THRESHOLD
            or not 0 <= proof['U_PL_accepted'] <= proof['unlabeled_exposures']):
        raise ValueError('Unexpected clean U_s pseudo-label policy')
    if continuation and proof.get('feature_weight') != (FEATURE_WEIGHT if arm == FEATURE else 0.):
        raise ValueError('Unexpected paired feature weight')
    if refinement and proof.get('conditioned_on_first') is not (arm == REFINE):
        raise ValueError('Unexpected second-stage input')
    if gain_fit and proof.get('adaptive_gain') is not (arm == ADAPT_GAIN):
        raise ValueError('Unexpected amplitude calibration')
    if covariance_fit and proof.get('cross_chirp_covariance') is not (arm == CROSS_COV):
        raise ValueError('Unexpected spectral covariance mode')
    if arm in VIEW_ARMS and (proof.get('views_per_packet'),proof.get('view_tilt'),proof.get('view_temperature'),
                             proof['labeled_exposures'],proof['unlabeled_exposures']) != (
                                 2,VIEW_TILT if arm == HARD_VIEW else 0.,VIEW_TEMPERATURE,
                                 budget*VIEW_BATCH,budget*VIEW_BATCH):
        raise ValueError('Unexpected same-packet view objective')
    if arm in AUGMENTED_ARMS:
        strong_count = proof['strong_exposures']
        enabled = arm != STRONG_CONTROL
        total = (proof['labeled_exposures'] + proof['unlabeled_exposures']) * proof.get('views_per_packet',1)
        exposure_ok = 0 < strong_count <= total if enabled else strong_count == 0
        if (not exposure_ok or proof['strong_fraction'] != (STRONG_FRACTION if enabled else 0.)
                or proof['strong_snr_db'] != list(STRONG_SNR_DB)):
            raise ValueError('Unexpected strong source mixture')
    if arm in COLOR_ARMS and proof['color_cutoff_hz'] != (list(COLOR_CUTOFF_HZ) if arm==COLOR else None):
        raise ValueError('Unexpected receive-band noise shape')
    g0 = initialize_recovery(arm,deepcopy(g0),device) if fixed_core else g0
    g, f = deepcopy(g0), deepcopy(f0)
    for model, key in ((g,'G'),(f,'F')):
        model.load_state_dict(state[key],strict=True)
        prior = (g0 if key=='G' else f0).state_dict()
        if fixed_core and key == 'G' and any(
                not torch.equal(v,prior[k]) for k,v in model.state_dict().items() if k.startswith('parent.')):
            raise ValueError('First-stage weights differ from the frozen source parent')
        changed = any(not torch.equal(v,prior[k]) for k,v in model.state_dict().items())
        if changed != proof[f'{key}_trainable']:
            raise ValueError(f'Unexpected checkpoint update of {key} in {arm}')
    return g.eval().requires_grad_(False), f.eval().requires_grad_(False)


def source_predictions(rt, source, contract, render):
    indices, rx, ids = panel(source,contract)
    raw = native(source,'V',contract['counts']['V'])[indices]
    jobs = [(int(i),SOURCE[int(r)],x) for i,r,x in zip(indices,rx,raw,strict=True)]
    matrix = {}
    with ProcessPoolExecutor(max_workers=rt['cpu_workers'],mp_context=multiprocessing.get_context('spawn')) as pool:
        for view in VIEWS:
            stream = pool.map(partial(source_receive,view),jobs,chunksize=8)
            groups = []
            for offset in range(0,len(jobs),PRED_BATCH):
                raw = np.stack([next(stream) for _ in range(min(PRED_BATCH,len(jobs)-offset))])
                groups.append(render(raw))
            matrix[view] = np.concatenate(groups,axis=1)
            print('W24_JOINT_SOURCE',view,len(jobs),flush=True)
    return ids, matrix


def write_source_predictions(root, ids, matrix, modes, *, steps=STEPS, **waveform):
    out = root/FULL
    out.mkdir()
    np.savez(out/'predictions.npz',**matrix)
    write(out/'source-predictions.json',dict(status='PREDICTIONS_FROZEN',modes=list(modes),views=list(VIEWS),
          physical_ids=ids,target_reads=0,source_truth_reads=0,waveform=dict(route=FULL,warm_steps=steps,**waveform)))


@torch.no_grad()
def predict_source(rt):
    root, source, device = Path(rt['run_root']), Path(rt['source_root']), 'cuda:0'
    contract, f0, g0, tracker = boot(rt,device)
    rows = [load_row(root,arm,contract,f0,g0,device) for arm in ARMS]
    (g_pamf,f_pamf),(g_fixed,_),(g_joint,f_joint) = rows

    def render(raw):
        iq = received_aperture(raw,device)
        base = recovered(g0,tracker,iq)
        updated = recovered(g_joint,tracker,iq)
        pairs = ((f0,iq),(f0,base),(f_pamf,recovered(g_pamf,tracker,iq)),
                 (f0,recovered(g_fixed,tracker,iq)),(f_joint,updated),(f0,updated),(f_joint,base))
        return np.stack([classify_aperture(f,x) for f,x in pairs])

    u = native(source,'U_s',contract['counts']['U_s'])[0]
    sample = np.stack([source_receive(v,(0,SOURCE[0],u)) for v in ('clean', VIEWS[-1])])
    cpu_frontend = received_aperture(np.stack((u,u)),device)
    source_frontend, _ = normalize_aperture(tracker.aperture(
        tensor(np.stack((u,u)).view('float32'),device).view(torch.complex64)))
    torch.testing.assert_close(source_frontend,cpu_frontend,rtol=1e-5,atol=2e-6)
    if render(sample).shape != (len(MODES)-1,2):
        raise ValueError('Expected a complete real-checkpoint source smoke')

    ids, matrix = source_predictions(rt,source,contract,render)
    check_controls(Path(rt['unet_parent'])/FULL,('direct','unet_standard_long_half'),[0,1],ids,matrix)
    outer = Path(rt['project'])/'runs'/DURA_RUN/FULL
    mark = read(outer/'source-predictions.json')
    if (mark['status'], mark['physical_ids'], mark['views'], mark['target_reads'], mark['source_truth_reads']) != (
            'PREDICTIONS_FROZEN', ids, list(VIEWS), 0, 0):
        raise ValueError('Archived DURA source predictions differ')
    col = mark['modes'].index('outer_objective_half')
    with np.load(outer/'predictions.npz',allow_pickle=False) as old:
        matrix = {v: np.concatenate((data[:1],old[v][[col]],data[1:])) for v,data in matrix.items()}
    write_source_predictions(root,ids,matrix,MODES,frozen_E104_formal_candidate=CANDIDATE,joint_source_only=True)


@torch.no_grad()
def predict_pseudo(rt):
    root, source, device = Path(rt['run_root']), Path(rt['source_root']), 'cuda:0'
    contract, f0, g0, tracker = boot(rt,device)
    run = rt['run_id']
    if run in PARENT_TRIALS:
        arms, parent_run = PARENT_TRIALS[run], source_parent(run)
        baseline_modes = parent_modes(parent_run)
        g0 = continuation_parent(rt,contract,f0,g0,device,parent_run)
    else:
        arms, parent_run, baseline_modes = (PSEUDO_ARMS[run],), RUN, MODES
    rows = [load_row(root,arm,contract,f0,g0,device) for arm in arms]

    def render(raw):
        observed = received_aperture(raw,device)
        return np.stack([classify_aperture(f0,observed), *(
            classify_aperture(f,recovered(g,tracker,observed)) for g,f in rows)])

    u = native(source,'U_s',contract['counts']['U_s'])[0]
    smoke = np.stack([source_receive(v,(0,SOURCE[0],u)) for v in ('clean',VIEWS[-1])])
    if render(smoke).shape != (1+len(arms),2):
        raise ValueError('Expected real frozen clean-PL predictions')
    ids, matrix = source_predictions(rt,source,contract,render)
    comparison = root.parent/parent_run/FULL
    reference = read(comparison/'source-predictions.json')
    if (reference['modes'] != list(baseline_modes) or reference['views'] != list(VIEWS)
            or reference['source_truth_reads'] or reference['target_reads']):
        raise ValueError('Expected the preregistered true-label comparison')
    check_controls(comparison,('direct',),[0],ids,matrix)
    with np.load(comparison/'predictions.npz',allow_pickle=False) as archived:
        predictions = {view: np.concatenate((archived[view],matrix[view][1:])) for view in VIEWS}
    write_source_predictions(root,ids,predictions,(*baseline_modes,*arms),steps=training_recipe(arms[0])[1],joint_source_only=run == RUN_PL,
                             matched_run=parent_run,U_PL_teacher=AUG_RUN,U_PL_threshold=PL_THRESHOLD)


@torch.no_grad()
def predict_raw_source(rt):
    root, source, device = Path(rt['run_root']), Path(rt['source_root']), 'cuda:0'
    joint = sys.modules[__name__]
    f,model,tracker,contract = raw_train.load_readers(joint,rt,device)
    run = rt['run_id']
    stages = raw_train.RUNS[run]
    baseline_modes = parent_modes(RUN_STRONG)

    def render(raw):
        return np.stack([classify_aperture(f,received_aperture(raw,device)),
                         *(raw_train.render(joint,f,model,stage,raw,device) for stage in stages)])

    u = native(source,'U_s',contract['counts']['U_s'])[0]
    sample = np.stack([source_receive(v,(0,SOURCE[0],u)) for v in ('clean',VIEWS[-1])])
    corrected = raw_train.restored_raw(model,sample,stages[-1],device)
    expected = received_aperture(corrected,device)
    differentiable,_ = normalize_aperture(tracker.aperture(tensor(corrected.view(np.float32),device).view(torch.complex64)))
    torch.testing.assert_close(differentiable,expected,rtol=1e-5,atol=3e-6)
    if render(sample).shape != (1+len(stages),2):
        raise ValueError('Expected frozen native IQ source smoke')
    ids,matrix = source_predictions(rt,source,contract,render)
    comparison = root.parent/RUN_STRONG/FULL
    reference = read(comparison/'source-predictions.json')
    if (reference['modes'] != list(baseline_modes) or reference['views'] != list(VIEWS)
            or reference['source_truth_reads'] or reference['target_reads']):
        raise ValueError('Expected exact native test comparison')
    check_controls(comparison,('direct',),[0],ids,matrix)
    with np.load(comparison/'predictions.npz',allow_pickle=False) as archived:
        predictions = {view: np.concatenate((archived[view],matrix[view][1:])) for view in VIEWS}
    write_source_predictions(root,ids,predictions,(*baseline_modes,*stages),
                             steps=sum(raw_train.STEPS[a] for a in stages),matched_run=RUN_STRONG,
                             model_input_length=8192,preprocess_after_restoration=True)


def qualifies(scores, candidate=CANDIDATE):
    if candidate not in FORMAL_ARMS.values():
        raise ValueError('Expected a frozen-E104 candidate')
    test = scores[candidate]
    controls = [scores[key] for key in ('direct','outer_half','unet_parent')]
    if candidate == FIXED_PL:
        control = scores[CANDIDATE]
        if test['low_urban_correct'] <= control['low_urban_correct']:
            return False
        controls.append(control)
    if candidate in PARENT_CONTROLS:
        control = scores[PARENT_CONTROLS[candidate]]
        parent = scores[STRONG if candidate in (COLOR,ADAPT_GAIN,CROSS_COV,HARD_VIEW) else FIXED_PL]
        if (test['low_urban_correct'] <= control['low_urban_correct']
                or test['by_scene']['clean'] < parent['by_scene']['clean']):
            return False
        controls.extend((control,parent))
    if candidate in (raw_train.STAGES[-1],raw_train.TASK):
        controls.append(scores[STRONG])
        if test['by_scene']['clean'] < scores[STRONG]['by_scene']['clean']:
            return False
        if candidate == raw_train.STAGES[-1]:
            controls.extend(scores[stage] for stage in raw_train.STAGES[:-1])
    return (test['six_correct'] > max(x['six_correct'] for x in controls)
            and test['low_urban_correct'] >= max(x['low_urban_correct'] for x in controls)
            and test['by_scene']['clean'] >= scores['direct']['by_scene']['clean'])


def formal_receipt(case, run=RUN):
    if case not in FULL_CASES:
        raise ValueError('Expected Full-only shard')
    if run in PARENT_TRIALS or run in raw_train.RUNS:
        parent = source_parent(run)
        old = formal_receipt(case,parent)
    else:
        old, parent = receipt(case,True,True), RUN_OUTER
    return dict(old,modes=[*old['modes'],FORMAL_ARMS[run]],candidate_run=run,reference_run=parent)


def formal_reference(rt, case):
    if rt['run_id'] not in PARENT_TRIALS and rt['run_id'] not in raw_train.RUNS:
        return prior_target(rt,case)
    project = Path(rt['project'])
    run = source_parent(rt['run_id'])
    parent = project/'runs'/run
    release = read(project/'releases'/f'{run}-release'/'runtime.json')
    state = read(parent/'state.json')
    expected_commit = STRONG_COMMIT if run == RUN_STRONG else FIXED_PL_COMMIT
    if (release['code_commit'] != expected_commit or Path(release['run_root']) != parent
            or any(release[k] != rt[k] for k in ('source_root','native_target_root','w24_release_root'))
            or state != dict(status='PREDICTIONS_FROZEN',cases=list(FULL_CASES),target_truth_read=False)):
        raise ValueError('Expected the frozen E104 clean-PL formal comparison')
    shard = parent/'predictions'/case
    if read(shard/'complete.json') != formal_receipt(case,run):
        raise ValueError('Mismatched frozen E104 clean-PL formal comparison')
    return shard


@torch.no_grad()
def predict_formal(rt,case):
    root, device = Path(rt['run_root']), 'cuda:0'
    candidate = FORMAL_ARMS[rt['run_id']]
    proof = read(root/FULL/'source-scores.json')
    if (proof['status'] != 'ANALYZED' or not proof['source_truth_last'] or proof['target_reads']
            or proof['n_packets'] != 300 or proof['n_practical'] != 1800 or not qualifies(proof['scores'],candidate)):
        raise ValueError('Frozen E104 source qualification required')
    expected = formal_receipt(case,rt['run_id'])
    baseline = formal_reference(rt,case)
    activity = read(baseline/'activity.json')
    native_run = rt['run_id'] in raw_train.RUNS
    if native_run:
        f0,model,_,_ = raw_train.load_readers(sys.modules[__name__],rt,device)
    else:
        contract,f0,g0,tracker = boot(rt,device)
        if candidate in PARENT_ARMS:
            g0 = continuation_parent(rt,contract,f0,g0,device,source_parent(rt['run_id']))
        model,_ = load_row(root,candidate,contract,f0,g0,device)
    target = Path(rt['native_target_root'])
    target_contract(target)
    out = root/'predictions'/case
    out.mkdir(parents=True)
    counts = {}
    with ProcessPoolExecutor(max_workers=rt['cpu_workers'],mp_context=multiprocessing.get_context('spawn')) as pool:
        for rx in expected['receivers']:
            raw = native(target,rx,LIMIT)
            decisions = {}
            with np.load(baseline/f'{rx}.npz',allow_pickle=False) as old:
                for view in VIEWS:
                    stream = iter(raw) if view=='clean' else pool.map(partial(channel_job,view,rx),enumerate(raw),chunksize=16)
                    pieces = []
                    batch = 8 if native_run else PRED_BATCH
                    for i in range(0,LIMIT,batch):
                        packets = np.stack([next(stream) for _ in range(min(batch,LIMIT-i))])
                        observed = received_aperture(packets,device)
                        decision = (raw_train.render(sys.modules[__name__],f0,model,candidate,packets,device)
                                    if native_run else classify_aperture(f0,recovered(model,tracker,observed)))
                        pieces.append(np.stack((classify_aperture(f0,observed),decision)))
                    key = f'{rx}/{view}'
                    decisions[view],counts[key] = append_blind_prediction(
                        old[view],np.concatenate(pieces,axis=1)[None],activity[key],LIMIT)
                    print('W24_JOINT_FIXED_E104_FORMAL',rx,view,flush=True)
            np.savez(out/f'{rx}.npz',**decisions)
    write(out/'activity.json',counts)
    write(out/'complete.json',expected)


def dispatch(runtime):
    rt = read(runtime)
    root, reference = Path(rt['run_root']), Path(rt['w24_release_root'])
    root.mkdir()
    state = root/'state.json'
    run = rt['run_id']
    pseudo_arm = PSEUDO_ARMS.get(run)
    arms = raw_train.RUNS.get(run) or PARENT_TRIALS.get(run,(pseudo_arm,) if pseudo_arm else ARMS)

    def status(**values):
        write(state,values,exclusive=False)

    def launch(action, label, gpu, config, *details):
        env = dict(os.environ,CUDA_VISIBLE_DEVICES='' if gpu is None else str(gpu),
                   PYTHONPATH=str(reference),OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1')
        cmd=[rt['python'],'-u',str(Path(__file__).resolve()),action,str(config),*details]
        with (root/f'{label}.log').open('x',encoding='utf-8') as log:
            job=subprocess.Popen(cmd,cwd=reference,env=env,stdout=log,stderr=subprocess.STDOUT)
        print('W24_JOINT_LAUNCHED',label,job.pid,gpu,flush=True)
        return job

    try:
        if read(reference/'submit.json')['code_commit'] != W24_COMMIT:
            raise ValueError('Expected archived W24 implementation')
        if run in PARENT_TRIALS or run in raw_train.RUNS:
            # Check the already frozen parent's source selection outside fit.
            ancestor = source_parent(run)
            score = read(root.parent/ancestor/FULL/'source-scores.json')
            if ((score['status'],score['source_truth_last'],score['target_reads'],score['n_packets'])
                    != ('ANALYZED',True,0,300) or not qualifies(score['scores'],FORMAL_ARMS[ancestor])):
                raise ValueError('Expected source-selected clean-PL parent')
        for arm in arms:
            out = root/arm
            out.mkdir()
            write(out/'fit-inputs.json',allowed_fit_runtime(rt,out))
        cards = available(rt['allowed_gpus'],rt['max_compute_processes'])
        if not cards:
            raise RuntimeError('No sufficient N607 training memory')
        status(status='SOURCE_TRAINING',steps=({a:raw_train.STEPS[a] for a in arms} if run in raw_train.RUNS
                                              else training_recipe(arms[0])[1]),arms=list(arms))
        schedule(rt,[('fit-row',arm) for arm in arms],lambda _,action,arm,gpu:
                 launch(action,f'{arm}-fit',gpu,root/arm/'fit-inputs.json',arm))
        if pseudo_arm or run in PARENT_TRIALS:
            comparison = root.parent/(source_parent(run) if run in PARENT_TRIALS else RUN)
            deadline = monotonic()+12*60*60
            status(status='WAITING_FOR_MATCHED_SOURCE_PREDICTIONS',comparison=comparison.name)
            while not (comparison/FULL/'source-predictions.json').is_file():
                if read(comparison/'state.json')['status']=='FAILED' or monotonic()>=deadline:
                    raise RuntimeError('Matched true-label source predictions are unavailable')
                sleep(30)
        deadline=monotonic()+600
        while not (cards := available(rt['allowed_gpus'],rt['max_compute_processes'])):
            if monotonic()>=deadline:
                raise RuntimeError('No GPU free for frozen source predictions')
            sleep(15)
        for action,gpu in (('predict-source',cards[0]),('score-source',None)):
            job=launch(action,action,gpu,runtime)
            status(status='RUNNING',phase=action,pid=job.pid,gpu=gpu)
            if job.wait():
                raise RuntimeError(f'{action} failed')
        promote = run in FORMAL_ARMS and qualifies(read(root/FULL/'source-scores.json')['scores'],FORMAL_ARMS[run])
        status(status='SOURCE_SCORED',promote=promote)
        if promote:
            status(status='FORMAL_PREDICTING',route=FULL)
            formal_runtime = dict(rt,max_workers=2) if run in raw_train.RUNS else rt
            schedule(formal_runtime,[('predict-formal',case) for case in FULL_CASES],lambda _,action,case,gpu:
                     launch(action,case,gpu,runtime,case))
            if any(read(root/'predictions'/c/'complete.json') != formal_receipt(c,run) for c in FULL_CASES):
                raise RuntimeError('Incomplete formal Full predictions')
            status(status='PREDICTIONS_FROZEN',cases=list(FULL_CASES),target_truth_read=False)
    except Exception as error:
        status(status='FAILED',error=str(error))
        raise


def collect(args):
    from experiments.pamf_lora_main.w24_formal_score import full_target_acceptance, load_frozen, score_matrix

    root = PurePosixPath(read(args.submit)['run_root'])
    if root.name not in FORMAL_ARMS or not re.fullmatch(r'/[A-Za-z0-9_./-]+',str(root)):
        raise ValueError('Expected an immutable fixed-E104 run root')
    conn=['-F',str(args.ssh_config),'-o','BatchMode=yes','-o','ConnectTimeout=10']
    deadline,last=monotonic()+12*60*60,None
    while monotonic()<deadline:
        state=None
        try:
            remote=subprocess.run(['ssh',*conn,'-T','N607','cat '+str(root/'state.json')],timeout=25,capture_output=True)
            if remote.returncode==0:
                state=json.loads(remote.stdout)
        except (subprocess.TimeoutExpired,json.JSONDecodeError):
            pass
        if state:
            if state['status']!=last:
                print(state,flush=True)
                last=state['status']
            if last=='FAILED':
                raise RuntimeError(state['error'])
            if last=='SOURCE_SCORED' and not state['promote']:
                return
            if last=='PREDICTIONS_FROZEN':
                break
        sleep(40)
    else:
        raise TimeoutError('Formal E104 collector exceeded its observation window')
    args.output.mkdir(parents=True)
    subprocess.run(['scp',*conn,'-r','N607:'+str(root/'predictions'),str(args.output)],check=True,timeout=300)
    plans={case:formal_receipt(case,root.name) for case in FULL_CASES}
    matrices,activity=load_frozen(args.output/'predictions',ungated=True,observed_plans=plans)
    spec=plans[FULL_CASES[0]]
    score_matrix(matrices,activity,{FULL:spec['modes']},args.dataset,args.output/'scores',tuple(spec['seeds']),
                 lambda result:full_target_acceptance(result[FULL][FORMAL_ARMS[root.name]]['pooled']))


def main():
    from experiments.pamf_lora_main.publish import release, release_commit

    root = argparse.ArgumentParser(description=__doc__)
    actions = root.add_subparsers(dest='action',required=True)
    for name in ('dispatch','fit-row','fit-smoke','fit','predict-source','score-source','predict-formal'):
        cmd=actions.add_parser(name)
        cmd.add_argument('runtime',type=Path)
        if name.startswith('fit'):
            cmd.add_argument('arm',choices=(*ARMS,*USE_PSEUDO,*raw_train.STEPS))
        if name=='predict-formal':
            cmd.add_argument('case',choices=FULL_CASES)
    pub=actions.add_parser('publish')
    pseudo_options = pub.add_mutually_exclusive_group()
    pseudo_options.add_argument('--clean-pl',dest='run_id',action='store_const',const=RUN_PL)
    pseudo_options.add_argument('--fixed-clean-pl',dest='run_id',action='store_const',const=RUN_FIXED_PL)
    pseudo_options.add_argument('--clean-feature',dest='run_id',action='store_const',const=RUN_FEATURE)
    pseudo_options.add_argument('--second-stage',dest='run_id',action='store_const',const=RUN_REFINE)
    pseudo_options.add_argument('--strong-source',dest='run_id',action='store_const',const=RUN_STRONG)
    pseudo_options.add_argument('--colored-source',dest='run_id',action='store_const',const=RUN_COLOR)
    pseudo_options.add_argument('--task-gain',dest='run_id',action='store_const',const=RUN_GAIN)
    pseudo_options.add_argument('--local-covariance',dest='run_id',action='store_const',const=RUN_COV)
    pseudo_options.add_argument('--paired-view-risk',dest='run_id',action='store_const',const=RUN_VIEW)
    pseudo_options.add_argument('--raw-dura',dest='run_id',action='store_const',const=raw_train.DURA_RUN)
    pseudo_options.add_argument('--raw-e104',dest='run_id',action='store_const',const=raw_train.TASK_RUN)
    col=actions.add_parser('collect')
    for cmd,fields in ((pub,('repo','context','ssh-config','output')),(col,('submit','dataset','ssh-config','output'))):
        for field in fields:
            cmd.add_argument('--'+field,type=Path,required=True)
    args=root.parse_args()
    if args.action=='dispatch':
        dispatch(args.runtime)
    elif args.action=='fit-row':
        # Separate processes exercise the true path before starting the fixed fit.
        entry=str(Path(__file__).resolve())
        for name in ('fit-smoke','fit'):
            subprocess.run([sys.executable,'-u',entry,name,str(args.runtime),args.arm],check=True)
    elif args.action in ('fit-smoke','fit'):
        if args.arm in raw_train.STEPS:
            raw_train.fit(sys.modules[__name__],args.runtime,args.arm,smoke=args.action=='fit-smoke')
        else:
            fit(args.runtime,args.arm,smoke=args.action=='fit-smoke')
    elif args.action=='predict-source':
        runtime=read(args.runtime)
        if runtime['run_id'] in raw_train.RUNS:
            predict_raw_source(runtime)
        else:
            (predict_source if runtime['run_id']==RUN else predict_pseudo)(runtime)
    elif args.action=='score-source':
        runtime=read(args.runtime)
        arm = PSEUDO_ARMS.get(runtime['run_id'])
        modes, contrasts = MODES, CONTRASTS
        if arm:
            control = CANDIDATE if arm == FIXED_PL else 'joint'
            modes, contrasts = (*MODES,arm), ((arm,control),(arm,'outer_half'),(arm,'direct'))
        if runtime['run_id'] in PARENT_TRIALS:
            control, candidate = PARENT_TRIALS[runtime['run_id']]
            ancestor = source_parent(runtime['run_id'])
            parent = FORMAL_ARMS[ancestor]
            modes = (*parent_modes(ancestor),control,candidate)
            contrasts = ((candidate,control),(candidate,parent),(candidate,'outer_half'),(candidate,'direct'),
                         (control,parent))
        if runtime['run_id'] in raw_train.RUNS:
            stages = raw_train.RUNS[runtime['run_id']]
            modes = (*parent_modes(RUN_STRONG),*stages)
            contrasts = tuple((stage,prior) for i,stage in enumerate(stages)
                              for prior in ('direct','outer_half',STRONG,*stages[:i]))
        score_source(args.runtime,modes=modes,output=Path(runtime['run_root'])/FULL,contrasts=contrasts)
    elif args.action=='predict-formal':
        predict_formal(read(args.runtime),args.case)
    elif args.action=='collect':
        collect(args)
    else:
        ctx=read(args.context)
        rt={k:ctx[k] for k in ('project','python','frozen_sources')}
        project=PurePosixPath(rt['project'])
        releases,runs=project/'releases',project/'runs'
        run = args.run_id or RUN
        gpu_preferences = {RUN: [3,0,7,2,6,5], RUN_PL: [2,6,5], RUN_FIXED_PL: [7,2,3,0,6,5],
                           RUN_FEATURE: [0,3,7,2,6,5], RUN_REFINE: [7,3,0,2,6,5],
                           RUN_STRONG: [7,0,3,2,6,5], RUN_COLOR: [0,3,7,2,6,5], RUN_GAIN: [7,2,0,3,6,5],
                           RUN_COV: [7,6,0,3,2,5], RUN_VIEW: [7,0,2,6,3,5],
                           raw_train.DURA_RUN: [6,5], raw_train.TASK_RUN: [2,3]}
        rt.update(run_id=run,runner=__spec__.name,code_commit=release_commit(args.repo),
                  release_root=str(releases/f'{run}-release'),run_root=str(runs/run),
                  source_root=str(releases/f'{DATA_RUN}-release'/'data/source'),
                  w24_release_root=str(releases/f'{W24_RUN}-release'),training_release_root=str(releases/f'{TRAIN_RUN}-release'),
                  classifier_root=str(runs/AUG_RUN),unet_parent=str(runs/UNET_FIDELITY_RUN),
                  allowed_gpus=gpu_preferences[run],
                  max_workers=1 if run in (RUN_PL,*raw_train.RUNS) else 3,max_compute_processes=8,cpu_workers=6)
        rt['external_releases']=[rt[k] for k in ('source_root','w24_release_root',
                                                'training_release_root','classifier_root','unet_parent')]
        rt['external_releases'] += [str(runs/r) for r in (DURA_RUN,TRAIN_RUN)]
        if run in PSEUDO_ARMS:
            rt['external_releases'].append(str(runs/RUN))
        if run in PARENT_TRIALS or run in raw_train.RUNS:
            rt['external_releases'] += [str(runs/RUN_FIXED_PL),str(releases/f'{RUN_FIXED_PL}-release')]
        if run in raw_train.RUNS or (run in PARENT_TRIALS and source_parent(run) == RUN_STRONG):
            rt['external_releases'] += [str(runs/RUN_STRONG),str(releases/f'{RUN_STRONG}-release')]
        if run in FORMAL_ARMS:
            rt['native_target_root']=str(releases/f'{DATA_RUN}-release'/'data/target')
            rt['external_releases'] += [rt['native_target_root'],str(runs/RUN_OUTER)]
        release(args.repo,args.output,rt,args.ssh_config)


if __name__=='__main__':
    main()
