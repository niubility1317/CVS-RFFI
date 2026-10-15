"""Frozen default-head errors, matched reference geometry and true paired gradients."""
import copy,gzip,json,time,os,sys
from collections import defaultdict
from pathlib import Path
import torch
from torch.nn import functional as F
from . import design as d
from experiments.cvs_urban_semantic50 import objective as semantic
from experiments.cvs_urban_semantic50.model import own_model
from experiments.cvs_urban_branch50 import branches as b,gradient as g,objective as positive
from experiments.cvs_urban_branch50.channel import PairedChannel
from experiments.cvs_urban_branch50.mechanism import developer_panel,read_features,contributions
from experiments.cvs_urban_sigtime50.panel import metadata,apply_fixed_scene
from experiments.cvs_urban_sigtime50.mechanism import quality_bin,gates
from experiments.cvs_urban_identity50.mechanism import identity_distances,variance
from experiments.cvs_urban_identity50.features import boundary_parameters

KEYS=(*b.BRANCHES,'J')


def reference_gap(anchors,keys,y,rx,day):
    """Mean positive-minus-negative cosine over exactly matched RX/day triplets."""
    n=len(y)
    if anchors.shape!=(n,160) or keys.shape!=anchors.shape or bool(((y<0)|(y>=6)).any()):
        raise ValueError('Legal L labels and actual joint160 required')
    if rx.shape!=(n,) or day.shape!=(n,) or not set(rx.tolist())<={1,3,4,6,8} or not set(day.tolist())<={1,2,3}:
        raise ValueError('Only registered source RX/day references allowed')
    a,k=(F.normalize(v,dim=1,eps=1e-4) for v in (anchors,keys))
    total=a.new_zeros(n);count=a.new_zeros(n)
    for receiver,date in sorted(set(zip(rx.tolist(),day.tolist()))):
        selected=(rx==receiver)&(day==date);sim=a@k[selected].T
        pos=y[:,None]==y[selected][None,:];neg=~pos
        np_=pos.sum(1);nn=neg.sum(1);pairs=np_*nn*(rx!=receiver)
        # Exact triplet mean without allocating a dense N**3 tensor.
        gap=(sim*pos).sum(1)/np_.clamp_min(1)-(sim*neg).sum(1)/nn.clamp_min(1)
        total+=gap*pairs;count+=pairs
    return dict(gap=total/count.clamp_min(1),triplets=count)


def auxiliary(model,arm,clean,urban,target,y,rx,day,q,recipe):
    if arm in ('D0','D1'):return urban['J'].sum()*0.,{}
    if arm=='D2':
        reliability=positive.labeled_reliability(target['logits'],y)
        loss=positive.losses(urban,target['features'],reliability)['J']
        return recipe['positive']['coefficients']['J']['coefficient']*loss,dict(teacher_eligible=int(reliability.sum()))
    loss,stats=semantic.semantic(clean['J'],urban['J'],y,rx,day,q,bidirectional=arm=='D4')
    return recipe['semantic']['coefficients'][arm]['coefficient']*loss,{k:v for k,v in stats.items() if not torch.is_tensor(v)}


def gradient_record(model,output,values,target,y,rx,day,q,arm,recipe,scene,batch):
    n=len(y);clean={k:v[:n] for k,v in values.items()};urban={k:v[n:] for k,v in values.items()}
    theta=b.feature_parameters(model,output['id_feat_joint'])
    if sum(p.numel() for _,p in theta)!=195041:raise ValueError('Actual original feature-gradient scope changed')
    cc=.5*F.cross_entropy(output['tx_logits'][:n],y,label_smoothing=.01)
    sc=.5*F.cross_entropy(output['tx_logits'][n:],y,label_smoothing=.01)
    extra,stats=auxiliary(model,arm,clean,urban,target,y,rx,day,q,recipe)
    gc,gs,ga=(g.vectors(loss,theta) for loss in (cc,sc,extra))
    boundary=boundary_parameters(model)
    grads=torch.autograd.grad(extra,list(boundary.values()),allow_unused=True,retain_graph=True)
    if any(v is not None and float(v.abs().max())>1e-8 for v in grads):raise ValueError('Extra gradient leaked to W/gains')
    return dict(scene=scene,batch=batch,auxiliary=stats,global_vs_clean=g.compare(gc,ga),global_vs_satellite=g.compare(gs,ga),
        clean_vs_satellite=g.compare(gc,gs),layer_vs_aux={name:g.compare({name:gs[name]},{name:ga[name]}) for name,_ in theta},
        auxiliary_boundary_norms={key:0. if v is None else float(v.norm()) for key,v in zip(boundary,grads)},
        optimizer_updates=0,scope='Eval paired source graph, not stochastic/clipped AdamW training update')


def summarize(items):
    n=len(items);fixed=[v for v in items if v['fixed_A0_clean_correct']]
    return dict(count=n,accuracy=sum(v['correct'] for v in items)/n,
        fixed_A0_clean_correct_count=len(fixed),fixed_A0_retention=sum(v['correct'] for v in fixed)/len(fixed) if fixed else None,
        A0_errors_repaired=sum(v['correct'] and not v['A0_satellite_correct'] for v in fixed),
        A0_correct_new_errors=sum(not v['correct'] and v['A0_satellite_correct'] for v in fixed),
        margin=sum(v['margin'] for v in items)/n,
        urban_to_clean_reference_gap=sum(v['urban_to_clean_reference_gap'] for v in items)/n,
        clean_to_urban_reference_gap=sum(v['clean_to_urban_reference_gap'] for v in items)/n,
        identity={key:{name:sum(v['identity'][key][name] for v in items)/n for name in items[0]['identity'][key]} for key in KEYS})


def run():
    rows,inner=d.eligible_rows()
    if d.BASE.exists():raise FileExistsError('Existing source analysis must be reconciled, never overwritten')
    d.BASE.mkdir();torch.set_num_threads(2);started=time.perf_counter()
    reference=own_model(rows[0][1],'cpu')
    with torch.no_grad():smoke=reference.forward_identity_only(torch.zeros(2,2,256))['tx_logits']
    if smoke.shape!=(2,6) or not torch.isfinite(smoke).all():raise ValueError('Real default checkpoint smoke failed')
    d.write(d.BASE/'source_smoke.json',dict(status='PASS',query_read=False,source_only=True,forward_packets=2))
    print('SMOKE PASS real matched A0/default head; query_read=false; source-only continues',flush=True)
    panel=developer_panel(inner);y=panel['y'];forwards=2
    with torch.no_grad():fixed=read_features(reference,panel['x'].clone())['logits'].argmax(1)==y
    forwards+=720;del reference
    channel=PairedChannel(d.training.parent.config(d.training.parent.rows()[0]));views={}
    d.write(d.BASE/'resolved_config.json',dict(run_id=d.RUN,launch_owner=d.OWNER,pid=os.getpid(),cwd=os.getcwd(),
        python=sys.executable,torch_version=torch.__version__,threads=torch.get_num_threads(),
        CUDA_VISIBLE_DEVICES=os.environ.get('CUDA_VISIBLE_DEVICES'),source_role='L_dev720',U_opened=False,V_forward_packets=0,
        checkpoints={row['arm']:c['output_root']+'/final_ssdg.pth' for row,c in rows},
        channel_module_file=channel.ch.__file__,residual_module_file=channel.residual.__file__,
        source_scene_stream=2026101402,forward_budget=d.FORWARD_PACKETS,formal_test_changed=False,
        commit=(d.ROOT/'release_commit.txt').read_text().strip()))
    for scene in d.training.SCENES:
        arrays=[];records=[]
        for start in range(0,720,90):
            x,meta=apply_fixed_scene(channel,panel['x'][start:start+90],metadata(panel,start,start+90),scene)
            arrays.append(x);records.extend(meta)
        views[scene]=(torch.cat(arrays),records)
    recipe=dict(positive=d.training.parent.recipe(),semantic=d.training.recipe())
    results={};baseline={}
    with gzip.open(d.BASE/'sample_identity.jsonl.gz','xt',encoding='utf-8',compresslevel=3) as stream:
        for row,c in rows:
            arm=row['arm'];model=own_model(c,'cpu');cp=torch.load(Path(c['output_root'])/'final_ssdg.pth',map_location='cpu',weights_only=False)
            teacher=copy.deepcopy(model);teacher.load_state_dict(cp['ema_model'],strict=True);teacher.eval().requires_grad_(False)
            before={k:v.clone() for k,v in model.state_dict().items()};tb={k:v.clone() for k,v in teacher.state_dict().items()}
            with torch.no_grad():clean=read_features(model,panel['x'].clone());target=read_features(teacher,panel['x'].clone())
            forwards+=1440;clean_correct=clean['logits'].argmax(1)==y;scenes={};gradients=[]
            model.requires_grad_(True).eval()
            try:
                for scene,(satellite,details) in views.items():
                    features=defaultdict(list);logits=[];paired=scene in ('practical_mid_urban','practical_low_urban')
                    for start in range(0,720,90):
                        stop=start+90;x=satellite[start:stop]
                        with torch.set_grad_enabled(paired):
                            actual=torch.cat((panel['x'][start:stop],x)) if paired else x.clone()
                            output,values=b.forward(model,actual);forwards+=len(actual)
                            if paired:
                                tgt=dict(logits=target['logits'][start:stop],features={k:v[start:stop] for k,v in target['features'].items()})
                                q=x.new_tensor([v['q'] for v in details[start:stop]])
                                gradients.append(gradient_record(model,output,values,tgt,y[start:stop],panel['rx'][start:stop],
                                    panel['day'][start:stop],q,arm,recipe,scene,start//90))
                        logits.append(output['tx_logits'][-90:].detach())
                        for key in KEYS:features[key].append(values[key][-90:].detach())
                    logits=torch.cat(logits);features={k:torch.cat(v) for k,v in features.items()};correct=logits.argmax(1)==y
                    if arm=='D1':baseline[scene]=correct.clone()
                    distance={k:identity_distances(features[k],clean['features'][k],target['features'][k],y) for k in KEYS}
                    other=logits.masked_fill(F.one_hot(y,6).bool(),-torch.inf).max(1).values
                    margin=(logits.gather(1,y[:,None]).squeeze(1)-other)/30
                    gap=reference_gap(features['J'],clean['features']['J'],y,panel['rx'],panel['day'])
                    reverse=reference_gap(clean['features']['J'],features['J'],y,panel['rx'],panel['day'])
                    items=[];groups=defaultdict(list)
                    for i,meta in enumerate(details):
                        item=dict(physical_id=panel['ids'][i],correct=bool(correct[i]),clean_correct=bool(clean_correct[i]),
                            fixed_A0_clean_correct=bool(fixed[i]),A0_satellite_correct=bool(baseline[scene][i]),margin=float(margin[i]),
                            urban_to_clean_reference_gap=float(gap['gap'][i]),clean_to_urban_reference_gap=float(reverse['gap'][i]),
                            matched_reference_triplets=int(gap['triplets'][i]),
                            identity={k:{name:float(value[i]) for name,value in distance[k].items()} for k in KEYS})
                        key=(int(panel['rx'][i]),meta['state_end'],meta['receiver_lock_status'],quality_bin(meta['signal_to_added_noise_db']))
                        groups[key].append(item);items.append(item)
                        stream.write(json.dumps(dict(arm=arm,scene=scene,RX=key[0],state=key[1],lock=key[2],quality=key[3],
                            receiver_quality_proxy_db=meta['quality_snr_db'],signal_to_added_noise_db=meta['signal_to_added_noise_db'],
                            residual_CFO_hz=meta['output_frequency_hz'],residual_kernel_length=meta['residual_kernel_length'],**item),allow_nan=False)+'\n')
                    scenes[scene]=dict(overall=summarize(items),strata=[dict(RX=k[0],state=k[1],lock=k[2],quality=k[3],**summarize(v)) for k,v in sorted(groups.items())],
                        branch_variance=variance(features),gain_contribution=contributions(model,features))
            finally:model.requires_grad_(False)
            if any(not torch.equal(v,model.state_dict()[k]) for k,v in before.items()) or any(not torch.equal(v,teacher.state_dict()[k]) for k,v in tb.items()):
                raise ValueError('Read-only analysis changed a model or own EMA state')
            if any(p.grad is not None or p.requires_grad for p in model.parameters()) or any(p.grad is not None or p.requires_grad for p in teacher.parameters()):
                raise ValueError('Source diagnostic accumulated gradients or teacher fit')
            results[arm]=dict(clean_accuracy=float(clean_correct.float().mean()),clean_branch_variance=variance(clean['features']),
                clean_gain_contribution=contributions(model,clean['features']),gates=gates(model),scenes=scenes,gradients=gradients,
                model_state_unchanged=True,EMA_state_unchanged=True,source_config=c)
            d.write(d.BASE/'progress.json',dict(completed=list(results),forward_packets=forwards,source_only=True,query_read=False))
    if forwards!=d.FORWARD_PACKETS:raise ValueError('Declared five-row source forward budget differs')
    result=dict(status='ANALYZED',source_only=True,query_read=False,U_truth_read=False,source_role='L_dev720',
        V_forward_packets=0,optimizer_updates=0,EMA_updates=0,head_fit_updates=0,feature_modules_added=0,
        fixed_A0_clean_correct_count=int(fixed.sum()),reference='D1/A0 final50 fixed clean-correct set; not a selected teacher or trainable upper bound',
        rows=results,forward_packets=forwards,seconds=time.perf_counter()-started,source_scene_stream=2026101402,
        gradients='Eval paired graph with common .5 clean/.5 satellite CE probes, including hypothetical satellite CE for clean-only D0; actual training updates come only from native logs',
        relative_geometry='Same-TX positives/different-TX negatives from the same reference RX/day, cross-RX anchors; global nearest-wrong cosine also separately reported',
        formal_test_changed=False,commit=(d.ROOT/'release_commit.txt').read_text().strip())
    d.write(d.BASE/'results.json',result)
    d.write(d.BASE/'complete.json',{k:result[k] for k in ('status','source_only','query_read','forward_packets','seconds','commit')})
    print(json.dumps({k:result[k] for k in ('status','source_only','forward_packets','seconds','fixed_A0_clean_correct_count')}),flush=True)
    return result


if __name__=='__main__':run()
