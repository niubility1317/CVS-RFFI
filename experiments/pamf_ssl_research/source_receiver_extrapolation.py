"""Frozen source-only test of full-feature same-pseudo cross-RX extrapolation."""
import math
import torch
from torch.nn import functional as F
from .matched_report import SEEDS

ALPHAS=(0.,.25,.5,1.)


def extrapolate(h,pseudo,accepted,rx,partner,valid,alpha):
    n=len(h)
    if (h.shape!=(n,160) or not bool(torch.isfinite(h).all()) or alpha not in ALPHAS
        or any(v.shape!=(n,) for v in (pseudo,accepted,rx,partner,valid))
        or any(v.dtype!=torch.long for v in (pseudo,rx,partner)) or accepted.dtype!=torch.bool or valid.dtype!=torch.bool
        or (n and (int(partner.min())<0 or int(partner.max())>=n))):
        raise ValueError('Finite fulljoint160 and fixed aligned source pseudo/RX partner masks required')
    if bool((valid&(~accepted|~accepted[partner]|(pseudo!=pseudo[partner])|(rx==rx[partner]))).any()):
        raise ValueError('Accepted same-pseudo different-sourceRX endpoints required')
    out=h.clone();out[valid]=h[valid]+alpha*(h[valid]-h[partner[valid]])
    return out


def select_scale(rows):
    if (len(rows)!=6 or {(r['model_seed'],r['role']) for r in rows}!={(s,r) for s in SEEDS for r in ('U_s','V')}
        or any([v['alpha'] for v in r['cases']]!=list(ALPHAS) for r in rows)):
        raise ValueError('All three fixed seeds and both complete source roles required')
    decisions=[]
    for alpha in ALPHAS[1:]:
        checks=[]
        for r in rows:
            c=next(v for v in r['cases'] if v['alpha']==alpha);base=r['cases'][0]
            n=c['selected'];correct=c['true_correct'] if r['role']=='V' else c['pseudo_agreements']
            if (type(n) is not int or n<=0 or type(correct) is not int or not 0<=correct<=n
                or any(not math.isfinite(v['pseudo_hard_CE']) or v['pseudo_hard_CE']<0 for v in (c,base))):
                raise ValueError('Finite complete source counts and CE required')
            ratio=None if base['pseudo_hard_CE']<=1e-12 else c['pseudo_hard_CE']/base['pseudo_hard_CE']
            enough=n>=(800 if r['role']=='U_s' else 600);fidelity=100*correct/n
            good=enough and fidelity>=99. and (r['role']=='V' or ratio is not None and ratio>=1.25)
            checks.append(dict(model_seed=r['model_seed'],role=r['role'],selected=n,
                fidelity_percent=fidelity,fidelity_is_actual_U_accuracy=False,hard_CE_ratio=ratio,pass_rule=good))
        decisions.append(dict(alpha=alpha,passes=all(v['pass_rule'] for v in checks),checks=checks))
    passed=[v['alpha'] for v in decisions if v['passes']]
    return dict(status='SOURCE_CANDIDATE_SUPPORTED' if passed else 'NO_SOURCE_CANDIDATE',
        selected_alpha=max(passed) if passed else None,decisions=decisions,
        rule='Largest fixed nonzero alpha;each seed V trueaccuracy>=99%/eligible>=600,U fixed-pseudoagreement>=99%/eligible>=800 and hardCE>=1.25x raw.No state update or target selection.',
        U_actual_accuracy='UNKNOWN;original raw pseudo labels only',new_training_launched=False)


def collect(rt,plan,analysis_commit):
    import gc,time
    from pathlib import Path
    from experiments.cric_dura_comparison.bootstrap import activate,numerical_policy
    from experiments.cric_dura_comparison.io import read
    from experiments.cric_dura_comparison.f_data import SourceDataset
    from experiments.cric_dura_comparison.frozen_inference import reader_from_payload
    from .hard_pl_early_train import source_parents
    from .hard_pl_joint import validate
    from .source_geometry import fixed_indices,capture_deployed
    from .u_mixstyle_objective import protected_partners
    activate(rt);numerical_policy();torch.set_num_threads(2)
    if plan['alphas']!=list(ALPHAS) or plan['model_seeds']!=list(SEEDS) or plan['source_epoch']!=20:
        raise ValueError('Fixed originalE20 source-only scale grid required')
    rows=[];start=time.monotonic()
    for seed in SEEDS:
        row=next(v for v in rt['rows'] if v['model_seed']==seed);c=read(row['train_config'])
        parents,contract=source_parents(c,validate_config=validate);done=parents[-1];cfg=done['config']
        payload=torch.load(Path(done['folder'])/'final_ssdg.pth',map_location='cpu',weights_only=False)
        if (done['epoch']!=20 or done['updates']!=4440 or payload['run_id']!=cfg['run_id'] or payload['candidate_id']!=cfg['row_id']
            or payload['epoch']!=20 or payload['a1_ema_successful_updates']!=4440 or payload['args']['teacher_ckpt']
            or payload['args']['use_ema_teacher'] or payload['args']['use_unlabeled']
            or payload['args']['baseline_ckpt']!=('' if cfg.get('parent_source') is None else cfg['parent_source']+'/final_ssdg.pth')):
            raise ValueError('Actual fixed same-contract originalE20 student/defaulthead required')
        reader=reader_from_payload(dict(cfg,parent_source=None),payload,'cpu')
        norm={n:b.detach().clone() for n,b in reader.named_buffers() if n.endswith(('running_mean','running_var','num_batches_tracked'))}
        data=SourceDataset(c['source_cache'],'L_s',contract);_,smoke=capture_deployed(reader,data.x[:2]);del data
        if smoke.shape!=(2,6):raise ValueError('Real originalE20 no-query source smoke failed')
        for role in ('U_s','V'):
            data=SourceDataset(c['source_cache'],role,contract);ix=fixed_indices(data.metadata,role);h=[];logits=[]
            for begin in range(0,len(ix),256):
                features,z=capture_deployed(reader,data.x[ix[begin:begin+256]]);h.append(features['joint160']);logits.append(z)
            h=torch.cat(h);logits=torch.cat(logits);q=logits.softmax(1);pseudo=q.argmax(1);accepted=q.max(1).values>=.95
            rx=torch.tensor(data.metadata['rx_i'][ix].tolist(),dtype=torch.long);torch.manual_seed(seed+2026102601)
            partner,valid=protected_partners(q,accepted.float(),rx,c['source_rxs'],1.)
            truth=torch.tensor(data.metadata['y'][ix].tolist(),dtype=torch.long) if role=='V' else None
            cases=[]
            with torch.inference_mode():
                for alpha in ALPHAS:
                    value=extrapolate(h,pseudo,accepted,rx,partner,valid,alpha)
                    z=logits if alpha==0 else reader.classify_features(value)
                    if not bool(torch.isfinite(z).all()) or z.shape!=(len(ix),6):raise ValueError('Finite existing-head predictions required')
                    n=int(valid.sum());pred=z.argmax(1);target=pseudo[valid]
                    cases.append(dict(alpha=alpha,selected=n,pseudo_agreements=int((pred[valid]==target).sum()),
                        true_correct=int((pred[valid]==truth[valid]).sum()) if truth is not None else None,
                        pseudo_hard_CE=float(F.cross_entropy(z[valid],target)),pseudo_LS01_CE=float(F.cross_entropy(z[valid],target,label_smoothing=.01)),
                        mean_relative_full_feature_displacement=float(((value[valid]-h[valid]).norm(dim=1)/h[valid].norm(dim=1).clamp_min(1e-12)).mean())))
            rows.append(dict(model_seed=seed,role=role,source_run_id=cfg['run_id'],source_row_id=cfg['row_id'],source_epoch=20,samples=len(ix),
                raw_accepted=int(accepted.sum()),partner_eligible=int(valid.sum()),
                true_TX_pair_agreement=int((truth[valid]==truth[partner[valid]]).sum()) if truth is not None else None,
                U_TX_truth_read=False,partners_chosen_without_V_truth=True,cases=cases))
            del data,h,logits;gc.collect()
        if reader.training or any(p.requires_grad for p in reader.parameters()) or any(not torch.equal(dict(reader.named_buffers())[n],v) for n,v in norm.items()):
            raise ValueError('Frozen source reader and unchanged normalization required')
        del reader,payload;gc.collect()
    return dict(status='VERIFIED_FIXED_SOURCE_EXTRAPOLATION',analysis_code_commit=analysis_commit,seconds=time.monotonic()-start,
        model_seeds=list(SEEDS),source_epoch=20,rows=rows,selection=select_scale(rows),GPU_workers=0,CPU_threads=2,
        original_source_backbone_packets=5040,real_checkpoint_L_smoke_packets=6,extra_existing_head_calls=18,extra_existing_head_packets=15120,
        model_updates=0,normalization_updates=0,EMA_updates=0,prototype_updates=0,head_fit_updates=0,V_gradients=0,target_reads=0,U_truth_reads=0,
        live_jobs_changed=False,interpretation='Fixed originalE20 source calibration only,fulljoint preserved;no branch deletion or fitted/replacement head.U fidelity means agreement with own raw pseudo label,not actualUaccuracy.Nonzeroalpha expands full same-pseudo crossRX differences,not a verified physical channel or target transfer guarantee.')
