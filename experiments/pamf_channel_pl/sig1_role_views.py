"""Frozen L/U study of same-packet restored views and noise placement."""
import argparse
from pathlib import Path
import os
import time
from experiments.cric_dura_comparison.io import read,write
from .restored_source import SCENES,snapshot,unchanged,source_channel

RUN='20261026-diagnostic-PAMF-SIG1-LU-restored-views-m1-r01'
RELEASE='20261026-diagnostic-PAMF-SIG1-LU-restored-views-release-r01'
SEED=2026092701
NAMES=('raw','raw_noise40')+tuple(s+'::'+v for s in SCENES for v in ('G_a','G_b','pre40','post40'))


def validate_runtime(rt):
    if (set(rt)!= {'run_id','project','python','release_root','run_root','context','runtime_file','code_commit','frozen_sources','allowed_gpus','launch_owner'}
            or rt['run_id']!=RUN or rt['run_root']!=rt['project']+'/runs/'+RUN
            or rt['release_root']!=rt['project']+'/releases/'+RELEASE or rt['context']!=rt['release_root']+'/context.json'
            or rt['runtime_file']!=rt['release_root']+'/runtime.json' or rt['allowed_gpus']!=list(range(8))
            or rt['launch_owner']!='codex-pamf-sig1-role-views-20261026'):
        raise ValueError('One immutable source-only role/view diagnostic required')


def select_U(metadata,receivers):
    import numpy as np
    if 'y' in metadata:raise ValueError('U true TX forbidden')
    selected=[]
    for rx in receivers:
        candidates=np.flatnonzero(metadata['rx_i']==rx)
        if len(candidates)<1260:raise ValueError('Each source RX needs 1260 U records')
        selected.extend(np.random.default_rng(20261026311+rx).permutation(candidates)[:1260].tolist())
    selected=np.asarray(sorted(selected),np.int64)
    if selected.shape!=(6300,) or len(set(selected.tolist()))!=6300:raise ValueError('Fixed source U6300 subset required')
    return selected


def hidden(batch,role):
    import torch
    if role not in ('L_s','U_s'):raise ValueError('Only legal source L/U')
    visible=batch[3]['tx_label_visible']
    if role=='U_s':
        if not bool((batch[1]==-1).all()) or bool(visible.any()):raise ValueError('U TX exposed')
    elif not bool(visible.all()) or bool(((batch[1]<0)|(batch[1]>=6)).any()):raise ValueError('Visible true L required')


def noise40(x,meta,stage):
    import numpy as np
    import torch
    if x.ndim!=3 or x.shape[1:]!=(2,256) or not bool(torch.isfinite(x).all()):raise ValueError('Finite IQ required')
    from leo_practical.channel import stable_seed
    values={k:meta[k].detach().cpu().tolist() for k in ('eq_i','rx_i','day_i','sig_i','base_index')}
    power=x.detach().square().sum(1).mean(1).cpu().tolist();noise=[]
    for i,power_i in enumerate(power):
        if power_i<=0:raise ValueError('Nonzero IQ needed for relative SNR')
        key=tuple(int(values[k][i]) for k in values)
        rng=np.random.default_rng(stable_seed(20261026317,'PAMF/SIG1/role_views/noise40',stage,key))
        value=rng.normal(size=(2,256));value*=np.sqrt(power_i/10000/np.mean((value*value).sum(0)));noise.append(value)
    a=np.ascontiguousarray(noise,np.float32)
    return x+torch.frombuffer(bytearray(a.tobytes()),dtype=torch.float32).reshape(a.shape).to(x.device)


def cross_rx_pairs(pred,accepted,rx,true=None):
    import numpy as np
    pred=np.asarray(pred);accepted=np.asarray(accepted);rx=np.asarray(rx)
    if pred.ndim!=1 or accepted.dtype!=bool or accepted.shape!=pred.shape or rx.shape!=pred.shape:raise ValueError('Aligned class/RX/mask required')
    total=correct=0
    for cls in range(6):
        mask=accepted&(pred==cls);counts=[int((mask&(rx==r)).sum()) for r in np.unique(rx)]
        total+=sum(counts)**2-sum(n*n for n in counts)
        if true is not None:
            for label in range(6):
                counts=[int((mask&(rx==r)&(true==label)).sum()) for r in np.unique(rx)]
                correct+=sum(counts)**2-sum(n*n for n in counts)
    result=dict(ordered_same_pseudo_cross_RX_pairs=total,true_pair_quality_known=true is not None)
    if true is not None:result.update(true_same_TX_pairs=correct,false_positive_pairs=total-correct,positive_pair_precision=correct/total if total else None)
    return result


def stats(value):
    import numpy as np
    a=np.asarray(value,np.float64)
    if not a.size:return None
    if not np.isfinite(a).all():raise ValueError('Finite complete diagnostic values required')
    return dict(n=int(a.size),mean=float(a.mean()),median=float(np.median(a)),p10=float(np.quantile(a,.1)),p90=float(np.quantile(a,.9)))


def summarize(logits,features,geometry,metadata,role,gradient):
    import numpy as np
    if logits.shape!=(14,6300,6) or features.shape!=(14,6300,160) or not np.isfinite(logits).all() or not np.isfinite(features).all():raise ValueError('Complete source role/view arrays required')
    if ('y' in metadata)!=(role=='L_s'):raise ValueError('No U truth may be supplied')
    value=logits-logits.max(-1,keepdims=True);prob=np.exp(value);prob/=prob.sum(-1,keepdims=True)
    pred=prob.argmax(-1);confidence=prob.max(-1);accepted=confidence[0]>=.95;rx=metadata['rx_i'];y=metadata.get('y')
    masks=[('ALL',np.ones(6300,bool)),('raw_accepted',accepted),('raw_rejected',~accepted)]
    masks += [('RX'+str(r),rx==r) for r in np.unique(rx)]
    rows=[]
    for i,name in enumerate(NAMES):
        for group,mask in masks:
            if not mask.any():continue
            row=dict(view=name,group=group,n=int(mask.sum()),raw_prediction_flip=float((pred[i,mask]!=pred[0,mask]).mean()),mean_confidence=float(confidence[i,mask].mean()))
            if y is not None:row.update(accuracy=float((pred[i,mask]==y[mask]).mean()),true_CE=float(-np.log(prob[i,mask,y[mask]].clip(1e-30)).mean()))
            rows.append(row)
    pairs=[]
    for scene in SCENES:
        a,b=[NAMES.index(scene+'::'+k) for k in ('G_a','G_b')]
        agreement=pred[a]==pred[b]
        for name,mask in masks[:3]:
            if not mask.any():continue
            row=dict(scene=scene,group=name,n=int(mask.sum()),same_packet_G_argmax_agreement=float(agreement[mask].mean()),
                normalized_feature_distance=stats(np.square(features[a,mask]-features[b,mask]).sum(1)))
            if y is not None:row.update(agree_but_wrong=float((agreement[mask]&(pred[a,mask]!=y[mask])).mean()),
                raw_correct_both_G_wrong=float(((pred[0,mask]==y[mask])&(pred[a,mask]!=y[mask])&(pred[b,mask]!=y[mask])).mean()))
            pairs.append(row)
    counts=dict(raw95_accepted=int(accepted.sum()),raw95_coverage=float(accepted.mean()),
        cross_RX_pseudo_pairs=cross_rx_pairs(pred[0],accepted,rx,y),same_packet_positive_pairs_without_TX_truth=3*6300)
    if y is not None:
        counts.update(raw95_pseudo_precision=float((pred[0,accepted]==y[accepted]).mean()) if accepted.any() else None,
            clean_rejected_correct=int(((~accepted)&(pred[0]==y)).sum()),clean_rejected_wrong=int(((~accepted)&(pred[0]!=y)).sum()),
            cross_RX_true_label_pairs=cross_rx_pairs(y,np.ones(6300,bool),rx,y))
    return dict(role=role,records=6300,results=rows,same_packet=pairs,counts=counts,waveform_geometry=geometry,
        frozen_existing_head_feature_gradient=gradient,U_true_precision='UNKNOWN' if role=='U_s' else None,
        scope='L is already-seen training L;U is fixed RX-balanced subset,not full U;no target generalization conclusion')


def feature_gradients(model,h,raw_prob,pseudo,y):
    import torch
    from torch.nn import functional as F
    with torch.enable_grad():
        z=h.detach().clone().requires_grad_(True);q=model.classify_features(z)
        gt=torch.autograd.grad(F.cross_entropy(q,y,reduction='sum',label_smoothing=.01),z,retain_graph=True)[0]
        gp=torch.autograd.grad(F.cross_entropy(q,pseudo,reduction='sum',label_smoothing=.01),z,retain_graph=True)[0]
        gs=torch.autograd.grad(-(raw_prob.detach()*q.log_softmax(1)).sum(),z)[0]
        a=F.cosine_similarity(gt,gp,dim=1);norm=gs.norm(dim=1)
        b=torch.where(norm>1e-7,F.cosine_similarity(gt,gs,dim=1),torch.zeros_like(norm))
    return a.detach(),b.detach(),norm.detach()


def run(rt):
    import numpy as np
    import torch
    from torch.nn import functional as F
    from .sig1_repl_rx_source import load_reader
    from experiments.cric_dura_comparison.f_data import SourceDataset
    from experiments.cric_dura_comparison.external_dura import fp32_copy
    from experiments.pamf_stage2_inference.engine import InferenceEngine
    validate_runtime(rt);root=Path(rt['run_root']);start=time.perf_counter()
    if not root.is_dir() or (root/'smoke.json').exists() or (root/'complete.json').exists():raise ValueError('Fresh own dispatch root required')
    model,g,c,contract=load_reader(rt);engine=InferenceEngine(g,reuse=True,graph=True,native=False)
    labeled=SourceDataset(c['source_cache'],'L_s',contract);unlabeled=SourceDataset(c['source_cache'],'U_s',contract)
    if len(labeled)!=6300 or len(unlabeled)!=56700:raise ValueError('Original source roles only')
    ui=select_U(unlabeled.metadata,c['source_rxs'])
    if set(labeled.metadata['ids'])&set(unlabeled.metadata['ids'][ui]):raise ValueError('Source roles must be physically disjoint')
    with torch.no_grad():
        x=labeled.x[:2].to('cuda:0');restored=engine(x.clone())[0].clone();q=model(restored)
        if q.shape!=(2,6) or not bool(torch.isfinite(q).all()):raise ValueError('Actual single-view E20/defaulthead/G source smoke failed')
    write(root/'smoke.json',dict(status='PASS',source_L_packets=2,signature_views=1,signature_parameters=18976,
        real_E20_defaulthead_and_sameG=True,optimizer_updates=0,U_truth_reads=0,target_reads=0))
    before=snapshot(model);gbefore=snapshot(g);role_results=[]
    for role,data,indices in [('L_s',labeled,np.arange(6300)),('U_s',unlabeled,ui)]:
        folder=root/role;folder.mkdir();metadata={k:v[indices] for k,v in data.metadata.items() if k in ('ids','rx_i','day_i','y')}
        if ('y' in metadata)!=(role=='L_s'):raise ValueError('TX visibility differs')
        logits=np.empty((14,6300,6),np.float32);features=np.empty((14,6300,160),np.float32);offset=0;wave={};grad={}
        loader=torch.utils.data.DataLoader(torch.utils.data.Subset(data,indices.tolist()),batch_size=128,shuffle=False,num_workers=0)
        for batch in loader:
            hidden(batch,role);x=batch[0].to('cuda:0');n=len(x);views=[x,noise40(x,batch[3],'raw')]
            with torch.no_grad():
                for scene in SCENES:
                    ha=source_channel(x,batch[3],scene,contract,namespace='PAMF/SIG1/role_views/a')
                    hb=source_channel(x,batch[3],scene,contract,namespace='PAMF/SIG1/role_views/b')
                    ga=engine(ha.clone())[0].clone();gb=engine(hb.clone())[0].clone()
                    pre=engine(noise40(ha,batch[3],scene+'::pre'))[0].clone();post=noise40(ga,batch[3],scene+'::post')
                    views.extend((ga,gb,pre,post))
                    base=ga.flatten(1).norm(dim=1).clamp_min(1e-12)
                    for name,value in [('G_b_vs_a',(gb-ga).flatten(1).norm(dim=1)/base),('pre40_vs_a',(pre-ga).flatten(1).norm(dim=1)/base),('post40_vs_a',(post-ga).flatten(1).norm(dim=1)/base)]:
                        wave.setdefault(scene+'::'+name,[]).extend(value.cpu().tolist())
                raw_h=model.features(x);raw_q=model.classify_features(raw_h);raw_prob=raw_q.softmax(1);raw_conf,pseudo=raw_prob.max(1)
                for j,view in enumerate(views):
                    h=raw_h if j==0 else model.features(view);q=raw_q if j==0 else model.classify_features(h)
                    if h.shape!=(n,160) or not bool(torch.isfinite(h).all()) or bool((h.norm(dim=1)<=1e-8).any()) or not bool(torch.isfinite(q).all()):raise ValueError('Finite actual full features/defaulthead required')
                    logits[j,offset:offset+n]=fp32_copy(q);features[j,offset:offset+n]=fp32_copy(F.normalize(h,dim=1))
                    if role=='L_s':
                        a,b,gn=feature_gradients(model,h,raw_prob,pseudo,batch[1].to(x.device))
                        for key,value in [('pseudo_vs_true',a),('raw_soft_vs_true_zero_if_tiny',b),('raw_soft_gradient_norm',gn),('raw_soft_nontrivial_fraction',(gn>1e-7).float())]:
                            grad.setdefault(NAMES[j]+'::'+key,[]).extend(value.cpu().tolist())
            offset+=n;write(root/'progress.json',dict(status='FROZEN_ROLE_VIEWS',role=role,seen=offset,total=6300,new_model_optimizer_updates=0,U_truth_reads=0,target_reads=0),exclusive=False)
            if offset%1024==0 or offset==6300:print('FROZEN_ROLE_VIEWS',role,offset,6300,flush=True)
        if offset!=6300 or not unchanged(model,before) or not unchanged(g,gbefore):raise ValueError('Coverage or frozen F/G changed')
        np.savez(folder/'source_predictions_features.npz',logits=logits,features=features,**metadata)
        geometry={k:stats(v) for k,v in wave.items()};gradient={}
        accepted=np.max(np.exp(logits[0]-logits[0].max(1,keepdims=True))/np.exp(logits[0]-logits[0].max(1,keepdims=True)).sum(1,keepdims=True),axis=1)>=.95
        for key,values in grad.items():
            values=np.asarray(values);gradient[key]={name:stats(values[mask]) for name,mask in [('ALL',np.ones(6300,bool)),('raw_accepted',accepted),('raw_rejected',~accepted)]}
        result=summarize(logits,features,geometry,metadata,role,gradient);role_results.append(result);write(folder/'complete.json',dict(status='ROLE_VIEWS_ANALYZED',**result))
    write(root/'complete.json',dict(status='SIG1_LU_RESTORED_VIEWS_ANALYZED',run_id=RUN,code_commit=rt['code_commit'],model_seed=SEED,teacher_epoch=20,results=role_results,
        F_backbone_logical_packets=176402,G_logical_packets=113402,feature_only_gradient_is_not_parameter_or_optimizer_gradient=True,
        parameters_and_all_registered_buffers_unchanged=True,optimizer_updates=0,U_truth_reads=0,V_reads=0,target_reads=0,new_head=False,
        noise_SNR_dB=40.,noise_power='relative to signal at injection point;no IQ re-normalization',seconds=time.perf_counter()-start,
        candidate_selection='none;diagnostic only;does not change active REPL source gates or launch new students'))


def dispatch(rt):
    # A single worker is admitted against all owners; no foreign task adoption.
    from experiments.practical_full_2x2.cache import atomic
    from experiments.pamf_repl_completion.runner import capacity,processes
    from experiments.pamf_peer_supervised.runner import launch
    validate_runtime(rt);root=Path(rt['run_root']);root.mkdir(parents=True,exist_ok=False)
    job=dict(id='LUViews',kind='views',config=rt['runtime_file'],entrypoint='experiments.pamf_channel_pl.sig1_role_views',execution_release=rt['release_root'],execution_python=rt['python'])
    write(root/'jobs.json',[job]);state=dict(status='QUEUED',pid=os.getpid(),code_commit=rt['code_commit'],jobs={})
    while True:
        pending=processes();available=[]
        for gpu in range(8):
            pids,free=capacity(gpu);load=len(pids|{pid for pid,g in pending.items() if g==gpu})
            if load<2 and free>=10000:available.append((load,-free,gpu))
        if available:break
        atomic(root/'pipeline_state.json',state);time.sleep(12)
    child,item=launch(rt,job,min(available)[2]);state.update(status='RUNNING',jobs={'LUViews':item});atomic(root/'pipeline_state.json',state)
    code=child.wait();path=root/'complete.json';good=code==0 and path.is_file() and read(path)['status']=='SIG1_LU_RESTORED_VIEWS_ANALYZED'
    state.update(status='ANALYZED' if good else 'STOPPED_FOR_TECHNICAL_FAILURE',jobs={'LUViews':dict(item,status='COMPLETE' if good else 'FAILED',exit_code=code)});atomic(root/'pipeline_state.json',state)


def worker(rt):
    from experiments.pamf_repl_completion.runner import activate
    activate(rt)
    import torch
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    run(rt)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['dispatch','worker']);p.add_argument('--kind');p.add_argument('--config',required=True)
    a=p.parse_args();c=read(a.config);dispatch(c) if a.action=='dispatch' else worker(c)
