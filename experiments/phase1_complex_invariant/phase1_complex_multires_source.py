"""Phase1 source-only, TX/F-free multi-resolution complex-mask training."""
import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from phase1_complex_multires_mask import ComplexMultiResolutionMask
from phase1_complex_invariant_loss import complex_gain_invariant_error
from experiments.pamf_phase1_flow import experiment as core
from experiments.pamf_phase1_flow import anchor_source_smoke as prior
from experiments.pamf_phase1_flow import feature_conditioned_experiment as label_reader

VIEWS = ('clean', 'practical_low_suburban', 'practical_mid_urban', 'practical_low_urban')
EPOCHS = (2, 5, 10)
BLENDS = (.25, .5, 1.)


def align_to_reference(wave, reference, width=64):
    """Fit one complex gain and a bounded delay on the central samples."""
    pad = np.pad(wave, ((0, 0), (width, width)), mode='reflect')
    ref = reference[:, width:-width]
    best = np.full(len(wave), np.inf)
    chosen = np.zeros_like(wave)
    for shift in range(-width, width + 1):
        full = pad[:, width + shift:width + shift + 256]
        part = full[:, width:-width]
        gain = np.sum(np.conj(part) * ref, axis=1) / np.maximum(np.sum(abs(part)**2, axis=1), 1e-20)
        error = np.mean(abs(gain[:, None] * part - ref)**2, axis=1)
        use = error < best
        best[use] = error[use]
        chosen[use] = gain[use, None] * full[use]
    return chosen, best


def iq(z):
    return np.stack((z.real, z.imag), axis=1).astype(np.float32)


def source_inputs(root, smoke=False):
    """Build immutable source views without loading F or transmitter labels."""
    from experiments.cric_dura_comparison.bootstrap import activate
    runtime=core.read(root/'releases'/(core.F_RUN+'-release')/'runtime.json')
    if runtime['code_commit']!='3ff8f4b7e85b5608f5b45350aa18081499a7ef1d':raise ValueError('Exact Phase1 runtime required')
    activate(runtime)
    from leo_practical.channel import Config, receiver_for_session, stable_seed, normalize_input
    from leo_practical.residual import ResidualChannel
    complete = core.read(root/'runs'/core.F_RUN/'F'/core.F_ROW/'source/complete.json')
    contract = core.read(complete['config']['source_contract'])
    cache, clean, roles = core.data(root, contract)
    ids = np.load(cache/'train/physical_ids.npy', allow_pickle=False)
    rx = np.load(cache/'train/rx.npy', allow_pickle=False)
    day = np.load(cache/'train/day.npy', allow_pickle=False)
    rng = np.random.default_rng(2026110551)
    groups = [np.sort(rng.choice(np.flatnonzero(roles == role), size=take, replace=False))
              for role, take in (('L_s', 1024), ('U_s', 3072))]
    fixed = np.concatenate(groups)
    assert len(fixed) == len(np.unique(fixed)) == 4096
    source_rx, source_day = rx[fixed], day[fixed]
    split = np.full(len(fixed), -1, dtype=np.int64)
    split[np.isin(source_rx, [1, 3, 4]) & np.isin(source_day, [1, 2])] = 0
    split[np.isin(source_rx, [4, 6]) & (source_day == 3)] = 1
    split[(source_rx == 8) & (source_day == 3)] = 2
    keep = split >= 0
    fixed, source_rx, source_day, split = fixed[keep], source_rx[keep], source_day[keep], split[keep]
    if smoke:
        selected = np.concatenate((np.flatnonzero(split == 0)[:12], np.flatnonzero((split == 1) & (source_rx == 4))[:6], np.flatnonzero((split == 1) & (source_rx == 6))[:6], np.flatnonzero(split == 2)[:12]))
        fixed, source_rx, source_day, split = fixed[selected], source_rx[selected], source_day[selected], split[selected]
    assert len(np.unique(ids[fixed])) == len(fixed)
    assert not np.intersect1d(ids[fixed[split == 0]], ids[fixed[split > 0]]).size
    n = len(fixed)
    clean_iq = np.asarray(clean[fixed], dtype=np.float32)
    clean_complex = clean_iq[:, 0].astype(np.float64) + 1j*clean_iq[:, 1]
    dura = prior.frozen_dura(root, 'cpu')
    data = {}
    for view in VIEWS:
        if view == 'clean':
            received = clean_iq
        else:
            cfg = Config(fs_hz=25e6, fc_hz=2462e6, scenario=view, processing_route='residual', mode='post_sync', equalization_enabled=False, reference_snr_db=30., boundary='reflect')
            received = []
            for idx in fixed:
                pid = str(ids[idx]); session = 'rx%d:day%d:eq1' % (int(rx[idx]), int(day[idx]))
                hardware = receiver_for_session(cfg, 2026110553, session)
                value = np.asarray(clean[idx], dtype=np.float64)
                signal, _ = normalize_input(value[0] + 1j*value[1])
                value, meta = ResidualChannel(cfg, stable_seed(2026110552, 'source-complex-multires-2026110552', view, pid), hardware).process(signal)
                if not np.isfinite(value).all() or meta['channel_equalization_applied']:
                    raise ValueError('Invalid source channel output')
                received.append(iq(value[None])[0])
            received = np.stack(received)
        anchors = []
        with torch.no_grad():
            for start in range(0, n, 32):
                anchors.append(dura(core.tensor(received[start:start+32], 'cpu')).cpu())
        anchor = np.from_dlpack(torch.cat(anchors).contiguous())
        z = anchor[:, 0].astype(np.float64) + 1j*anchor[:, 1]
        raw = received[:, 0].astype(np.float64) + 1j*received[:, 1]
        aligned_raw, _ = align_to_reference(raw, z)
        _, residual = align_to_reference(z, raw)
        _, aligned_clean_error = align_to_reference(z, clean_complex)
        target, _ = align_to_reference(clean_complex, z)
        power = np.mean(abs(raw[:, 64:-64])**2, axis=1).clip(min=1e-20)
        fraction = np.clip(residual/power, 1e-8, 1.)
        proxy = 10*np.log10(np.maximum(1-fraction, 1e-8)/fraction)
        ratio = np.sqrt(np.sum((received-anchor)**2, axis=(1, 2))/np.maximum(np.sum(received**2, axis=(1, 2)), 1e-12))
        gate = (ratio > .2) & (proxy < 10.)
        data[view] = dict(anchor=anchor.copy(), observed=received.copy(), received=iq(aligned_raw), target=iq(target), gate=gate,
                          clean_error=aligned_clean_error)
        print('SOURCE_VIEW_READY', view, n, flush=True)
    return cache, clean, roles, contract, fixed, source_rx, split, clean_iq, clean_complex, data


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--commit', required=True)
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--smoke', action='store_true')
    args = ap.parse_args()
    if args.out.exists():
        raise FileExistsError('Immutable output root already exists')
    args.out.mkdir(parents=True)
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(2026110554)
    rng = np.random.default_rng(2026110555)
    cache, clean, roles, contract, fixed, source_rx, split, clean_iq, clean_complex, data = source_inputs(args.root, args.smoke)
    n = len(fixed)
    model = ComplexMultiResolutionMask().to(args.device).train()
    opt = torch.optim.AdamW(model.parameters(), lr=.0003, weight_decay=.001)
    train_indices = [(view, int(i)) for view in VIEWS[1:] for i in np.flatnonzero((split == 0) & data[view]['gate'])]
    if len(train_indices) < (3 if args.smoke else 100):
        raise ValueError('Insufficient gated source training records')
    states, losses = {}, []
    for epoch in range(1, 11):
        order = rng.permutation(len(train_indices)); total = 0.
        for start in range(0, len(order), 32):
            batch = [train_indices[k] for k in order[start:start+32]]
            a = core.tensor(np.stack([data[v]['anchor'][i] for v, i in batch]), args.device)
            r = core.tensor(np.stack([data[v]['received'][i] for v, i in batch]), args.device)
            target = core.tensor(np.stack([data[v]['target'][i] for v, i in batch]), args.device)
            gate = torch.ones(len(batch), device=args.device)
            output = model(a, r, gate)
            norm = target.square().mean((1, 2)).clamp_min(1e-8)
            iq_loss = complex_gain_invariant_error(output, target)
            deviation = (output-a).square().mean((1, 2))/norm
            loss = (iq_loss + .02*deviation).mean()
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite source IQ objective')
            opt.zero_grad(set_to_none=True); loss.backward()
            if model.mask_net[-1].weight.grad is None or not torch.isfinite(model.mask_net[-1].weight.grad).all() or not model.mask_net[-1].weight.grad.abs().sum()>0:
                raise FloatingPointError('Invalid mask gradient')
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            opt.step(); total += float(loss)*len(batch)
        losses.append(total/len(train_indices))
        if epoch in EPOCHS:
            states[epoch] = {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
            print('SOURCE_MASK_EPOCH', epoch, losses[-1], flush=True)
    # Freeze by paired source IQ only. No F parameters, outputs, or TX labels
    # are loaded or consulted for training or candidate selection.
    model.eval(); eval_ids=np.flatnonzero(split>0); candidates={}; validation=[]
    for epoch in EPOCHS:
        model.load_state_dict(states[epoch]); model.eval()
        with torch.no_grad():
            for view in VIEWS:
                outputs=[]
                for start in range(0,len(eval_ids),32):
                    idx=eval_ids[start:start+32]
                    a=core.tensor(data[view]['anchor'][idx],args.device)
                    r=core.tensor(data[view]['received'][idx],args.device)
                    g=core.tensor(data[view]['gate'][idx].astype(np.float32),args.device)
                    outputs.append(model(a,r,g).cpu())
                full=np.from_dlpack(torch.cat(outputs).contiguous())
                anchor=data[view]['anchor'][eval_ids]
                for alpha in BLENDS:
                    candidates[epoch,alpha,view]=anchor+alpha*(full-anchor)
        print('SOURCE_MASK_CANDIDATE_WAVEFORMS_FIXED',epoch,flush=True)
    selected=dict(kind='dura',validation_improvement=0.)
    val_mask=split[eval_ids]==1
    val_ids=eval_ids[val_mask]
    val_truth=clean_complex[val_ids]
    baseline_errors={}
    for view in VIEWS:
        anchor=data[view]['anchor'][val_ids]
        _,baseline_errors[view]=align_to_reference(anchor[:,0].astype(np.float64)+1j*anchor[:,1],val_truth)
    for epoch in EPOCHS:
        for alpha in BLENDS:
            per={};groups={};improvement=0.
            for view in VIEWS:
                wave=candidates[epoch,alpha,view][val_mask]
                _,new=align_to_reference(wave[:,0].astype(np.float64)+1j*wave[:,1],val_truth)
                old=baseline_errors[view]
                per[view]=dict(baseline=float(old.mean()),candidate=float(new.mean()))
                if view!='clean':improvement+=per[view]['baseline']-per[view]['candidate']
                for group in (4,6):
                    gm=source_rx[val_ids]==group
                    if not gm.any():raise ValueError('Both validation receivers are required')
                    groups.setdefault(str(group),{})[view]=dict(baseline=float(old[gm].mean()),candidate=float(new[gm].mean()))
            guard=all(item['candidate']<=item['baseline']+1e-8 for g in groups.values() for item in g.values())
            validation.append(dict(epoch=epoch,alpha=alpha,improvement=improvement,per_view=per,groups=groups,guards=guard))
            if guard and improvement>selected['validation_improvement']+1e-8:
                selected=dict(kind='complex_multires_mask',epoch=epoch,alpha=alpha,validation_improvement=improvement)
    # Selected checkpoint is persisted before F and labels are loaded.
    if selected['kind']=='complex_multires_mask':
        with (args.out/'selected.pt').open('xb') as f:
            torch.save(dict(commit=args.commit,selected=selected,model=states[selected['epoch']]),f)
    with (args.out/'source_selection.json').open('x',encoding='utf8') as f:
        json.dump(dict(selected=selected,validation=validation,F_loaded=False,TX_labels_read=0),f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
    print('SOURCE_IQ_ONLY_SELECTION_FIXED',json.dumps(selected),flush=True)
    # Frozen reader is used solely for post-selection source evaluation.
    teacher,teacher_contract=core.teacher(args.root,'cpu')
    if teacher_contract['role_ids']!=contract['role_ids']:raise ValueError('Frozen ground reader source contract differs')
    clean_prob=[]; basep={}; preds={}; chosen={}
    with torch.no_grad():
        for start in range(0,n,64):
            x=core.tensor(clean_iq[start:start+64],'cpu')
            clean_prob.append(F.softmax(teacher.classify_features(teacher.features(x)),1).cpu())
        clean_prob=np.from_dlpack(torch.cat(clean_prob).contiguous())
        for view in VIEWS:
            anchor=data[view]['anchor'][eval_ids]
            wave=(candidates[selected['epoch'],selected['alpha'],view]
                  if selected['kind']=='complex_multires_mask' else anchor)
            chosen[view]=wave; before=[]; after=[]
            for start in range(0,len(eval_ids),64):
                a=core.tensor(anchor[start:start+64],'cpu');x=core.tensor(wave[start:start+64],'cpu')
                before.append(F.softmax(teacher.classify_features(teacher.features(a)),1).cpu())
                after.append(F.softmax(teacher.classify_features(teacher.features(x)),1).cpu())
            basep[view]=np.from_dlpack(torch.cat(before).contiguous())
            preds[view]=np.from_dlpack(torch.cat(after).contiguous())
    print('SELECTED_SOURCE_PREDICTIONS_FIXED_BEFORE_SOURCE_TX_LABELS',flush=True)
    labels=label_reader.source_labels(args.root,cache,clean,roles,contract)
    labeled=roles[fixed]=='L_s';assert np.all(labels[fixed[~labeled]]==-1)
    target=np.where(labeled,labels[fixed],clean_prob.argmax(1));valid=labeled|(clean_prob.max(1)>=.95)
    evaluation={};quality={}
    for part in (1,2):
        mask=(split[eval_ids]==part)&valid[eval_ids];iqmask=split[eval_ids]==part;per={};q={}
        for view in VIEWS:
            before=basep[view].argmax(1)==target[eval_ids];after=preds[view].argmax(1)==target[eval_ids];lm=mask&labeled[eval_ids]
            per[view]=dict(valid_count=int(mask.sum()),L_s_count=int(lm.sum()),baseline=int((mask&before).sum()),student=int((mask&after).sum()),repairs=int((mask&~before&after).sum()),damages=int((mask&before&~after).sum()),L_s_baseline=int((lm&before).sum()),L_s_student=int((lm&after).sum()))
            anchor=data[view]['anchor'][eval_ids][iqmask];wave=chosen[view][iqmask];truth=clean_complex[eval_ids][iqmask]
            _,old=align_to_reference(anchor[:,0].astype(np.float64)+1j*anchor[:,1],truth)
            _,new=align_to_reference(wave[:,0].astype(np.float64)+1j*wave[:,1],truth)
            q[view]=dict(count=int(iqmask.sum()),aligned_mse_before=float(old.mean()),aligned_mse_after=float(new.mean()),improved=int((new<old).sum()),worsened=int((new>old).sum()))
        evaluation[str(part)]=per;quality[str(part)]=q
    result=dict(status='SOURCE_ONLY_COMPLEX_INVARIANT_MASK',commit=args.commit,source_ids=n,train_ids=int((split==0).sum()),validation_ids=int((split==1).sum()),heldout_ids=int((split==2).sum()),training_gated_views=len(train_indices),training_loss=losses,selected=selected,evaluation=evaluation,quality=quality,F_loaded_only_after_selection=True,training_TX_labels_read=0,U_s_labels_read=0,V_reads=0,test_IQ_reads=0,predictions_before_source_labels=True)
    with (args.out/'result.json').open('x',encoding='utf8') as f:json.dump(result,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
    print('SOURCE_COMPLEX_MULTIRES_COMPLETE',json.dumps(selected),flush=True)


if __name__=='__main__':main()
