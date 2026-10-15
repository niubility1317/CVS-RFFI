"""Frozen source-only identity geometry, errors and branch variance; no fitted head."""
import copy,gzip,json,time
from collections import defaultdict
import torch
from torch.nn import functional as F
from . import design as d,features as feature
from experiments.cvs_urban_sigtime50.panel import load,metadata,apply_fixed_scene
from experiments.cvs_urban_sigtime50.channel import PairedChannel
from experiments.cvs_urban_sigtime50.mechanism import quality_bin,gates
EPS=1e-4


def identity_distances(satellite,clean,teacher,labels):
    if satellite.shape!=clean.shape or satellite.ndim!=2 or satellite.shape[1]!=160 or teacher.shape!=clean.shape:
        raise ValueError('Actual corresponding joint160 arrays required')
    if labels.shape!=(len(clean),) or bool(((labels<0)|(labels>5)).any()):
        raise ValueError('Legal L_s labels required; hidden U is forbidden')
    zs,zc,zt=(F.normalize(z,dim=1,eps=EPS) for z in (satellite,clean,teacher))
    same=(zs*zc).sum(1);teacher_same=(zs*zt).sum(1)
    wrong=labels[:,None]!=labels[None,:]
    if not bool(wrong.any(1).all()):raise ValueError('Need legal source wrong-TX comparators')
    closest=(zs@zc.T).masked_fill(~wrong,-torch.inf).max(1).values
    return dict(clean_pair_cosine=same,EMA_pair_cosine=teacher_same,closest_wrong_TX_cosine=closest,
        correct_minus_wrong_cosine=same-closest,unit_pair_distance=(zs-zc).norm(dim=1))


def variance(values):
    return {k:dict(mean_variance=float(v.var(0,unbiased=False).mean()),
        mean_norm=float(v.norm(dim=1).mean()),minimum_coordinate_variance=float(v.var(0,unbiased=False).min()))
        for k,v in values.items()}


def forward(model,x,batch=90):
    outputs=[];branches=defaultdict(list)
    for start in range(0,len(x),batch):
        with feature.capture(model) as values:out=model.forward_identity_only(x[start:start+batch])
        # Same algebra and deployment path as training; read-only under no_grad.
        z=feature.auxiliary_joint(model,out,values)
        outputs.append(dict(joint=z,logits=out['tx_logits']))
        for key,value in values.items():branches[key].append(value)
    return dict(joint=torch.cat([v['joint'] for v in outputs]),logits=torch.cat([v['logits'] for v in outputs]),
        branches={k:torch.cat(v) for k,v in branches.items()})


def actual_rows():
    from .model import own_model as new_model
    from experiments.cvs_urban_sigtime50.model import own_model as old_model
    result=[]
    for arm in ('C3','B0'):
        c=d.reference.config(next(r for r in d.reference.rows() if r['arm']==arm))
        result.append((arm,c,old_model))
    result.extend((r['arm'],d.config(r),new_model) for r in d.rows())
    for _,c,_ in result:
        src=__import__('pathlib').Path(c['output_root'])
        done=d.read(src/'completion.json');frozen=d.read(src/'source_frozen.json')
        if done['epoch']!=50 or done['optimizer_steps']!=11100 or done['target_access'] or done['head_fit_updates']:
            raise ValueError('Need source-only E50/11100 default-head checkpoints')
        if frozen['epoch']!=50 or frozen['selection']!='final_only' or frozen['target_access']:
            raise ValueError('Need final50 source freeze')
    return result


def summarize(items):
    n=len(items);fixed=[v for v in items if v['fixed_A_clean_correct']]
    return dict(count=n,accuracy=sum(v['correct'] for v in items)/n,
        fixed_A_clean_correct_count=len(fixed),
        fixed_A_retention=sum(v['correct'] for v in fixed)/len(fixed) if fixed else None,
        A_errors_repaired=sum(v['correct'] and not v['A_satellite_correct'] for v in fixed),
        A_correct_new_errors=sum(not v['correct'] and v['A_satellite_correct'] for v in fixed),
        margin=sum(v['margin'] for v in items)/n,
        identity={key:sum(v[key] for v in items)/n for key in
            ('clean_pair_cosine','EMA_pair_cosine','closest_wrong_TX_cosine','correct_minus_wrong_cosine','unit_pair_distance')})


def run():
    eligible=actual_rows();out=d.BASE/'source_identity_mechanism';out.mkdir(exist_ok=False)
    torch.set_num_threads(2);started=time.perf_counter();panel=load();y=panel['y']
    channel=PairedChannel(d.reference.config(d.reference.rows()[0]))
    results={};A_clean_correct=None;A_sat_correct={};forwards=0
    with gzip.open(out/'sample_identity.jsonl.gz','xt',encoding='utf-8',compresslevel=3) as stream:
        for arm,c,build in eligible:
            model=build(c,'cpu');checkpoint=torch.load(c['output_root']+'/final_ssdg.pth',map_location='cpu',weights_only=False)
            teacher=copy.deepcopy(model);teacher.load_state_dict(checkpoint['ema_model'],strict=True)
            teacher.eval().requires_grad_(False)
            before={k:v.clone() for k,v in model.state_dict().items()};tb={k:v.clone() for k,v in teacher.state_dict().items()}
            with torch.no_grad():
                clean=forward(model,panel['x']);target=forward(teacher,panel['x']);forwards+=1440
                if arm=='C3':print('SMOKE PASS actual source C3/own EMA checkpoint; query_read=false; updates=0',flush=True)
                clean_correct=clean['logits'].argmax(1)==y
                if A_clean_correct is None:A_clean_correct=clean_correct.clone()
                scenes={}
                for scene in d.SCENES:
                    batches=[];details=[]
                    for start in range(0,720,90):
                        x,meta=apply_fixed_scene(channel,panel['x'][start:start+90],metadata(panel,start,start+90),scene)
                        batches.append(x);details.extend(meta)
                    satellite=forward(model,torch.cat(batches));forwards+=720
                    correct=satellite['logits'].argmax(1)==y
                    if arm=='C3':A_sat_correct[scene]=correct.clone()
                    distances=identity_distances(satellite['joint'],clean['joint'],target['joint'],y)
                    logits=satellite['logits'];other=logits.masked_fill(F.one_hot(y,6).bool(),-torch.inf).max(1).values
                    margin=(logits.gather(1,y[:,None]).squeeze(1)-other)/30
                    groups=defaultdict(list);items=[]
                    for i,m in enumerate(details):
                        item=dict(physical_id=panel['ids'][i],correct=bool(correct[i]),clean_correct=bool(clean_correct[i]),
                            fixed_A_clean_correct=bool(A_clean_correct[i]),A_satellite_correct=bool(A_sat_correct[scene][i]),
                            margin=float(margin[i]),**{k:float(v[i]) for k,v in distances.items()})
                        state=m['state_end'];lock=m['receiver_lock_status'];quality=quality_bin(m['signal_to_added_noise_db'])
                        groups[(state,lock,quality)].append(item);items.append(item)
                        stream.write(json.dumps(dict(arm=arm,scene=scene,state=state,lock=lock,quality=quality,**item),allow_nan=False)+'\n')
                    scenes[scene]=dict(overall=summarize(items),strata=[dict(state=k[0],lock=k[1],quality=k[2],**summarize(v)) for k,v in sorted(groups.items())],
                        branch_variance=variance(satellite['branches']),joint_variance=variance({'joint':satellite['joint']})['joint'])
            if any(not torch.equal(v,model.state_dict()[k]) for k,v in before.items()) or any(not torch.equal(v,teacher.state_dict()[k]) for k,v in tb.items()):
                raise ValueError('Frozen source identity analysis mutated model/EMA')
            if any(p.grad is not None or p.requires_grad for p in model.parameters()) or any(p.grad is not None or p.requires_grad for p in teacher.parameters()):
                raise ValueError('Read-only source identity analysis has training gradients')
            results[arm]=dict(clean_accuracy=float(clean_correct.float().mean()),clean_branch_variance=variance(clean['branches']),
                clean_joint_variance=variance({'joint':clean['joint']})['joint'],gates=gates(model),scenes=scenes,
                model_state_unchanged=True,EMA_state_unchanged=True,checkpoint=c['output_root']+'/final_ssdg.pth')
            d.write(out/'progress.json',dict(completed=list(results),source_only=True,query_read=False))
    result=dict(status='ANALYZED',source_only=True,query_read=False,U_truth_read=False,source_role='L_s',source_count=720,
        reference='Fixed C3 E50 A clean-correct set; diagnostic reference only, no training inheritance',
        fixed_A_clean_correct_count=int(A_clean_correct.sum()),rows=results,
        no_classifier_fitted=True,head_fit_updates=0,optimizer_updates=0,forward_packets=forwards,
        seconds=time.perf_counter()-started,dropout='eval for all read-only diagnostic paths',
        wrong_TX_comparison='closest cosine among all 720 source clean vectors with different true L_s TX; no fitted decision rule',
        causal_limit='Same full channel draws compare invariance/error changes, not a controlled damage-removal causal claim',
        commit=(d.ROOT/'release_commit.txt').read_text().strip())
    if forwards!=23040:raise ValueError('Unexpected diagnostic forward budget')
    d.write(out/'results.json',result)
    print(json.dumps({k:result[k] for k in ('status','source_only','forward_packets','seconds','fixed_A_clean_correct_count')}),flush=True)
    return result


if __name__=='__main__':run()
