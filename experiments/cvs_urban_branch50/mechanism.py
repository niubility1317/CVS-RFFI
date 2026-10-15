"""Frozen internal-L development geometry and shared-path gradients; no fitting."""
import copy,gzip,json,time
from collections import defaultdict
from pathlib import Path
import torch
from torch.nn import functional as F
from . import design as d,branches as b,gradient as g,objective
from .model import own_model
from .channel import PairedChannel
from experiments.cvs_urban_sigtime50.panel import metadata,apply_fixed_scene
from experiments.cvs_urban_sigtime50.mechanism import quality_bin,gates
from experiments.cvs_urban_identity50.mechanism import identity_distances,variance

KEYS=(*b.BRANCHES,'J')
OUTPUT='source_branch_mechanism'


def eligible_rows():
    result=[];reference=None
    for row in d.rows():
        c=d.config(row);source=Path(c['output_root'])
        done=d.read(source/'completion.json');frozen=d.read(source/'source_frozen.json')
        inner=d.read(source/'inner_source_contract.json')
        if (done['config']!=c or done['epoch']!=50 or done['optimizer_steps']!=11100
                or done['target_access'] or done['head_fit_updates'] or frozen['epoch']!=50
                or frozen['selection']!='final_only' or frozen['target_access']):
            raise ValueError('All eight need fixed source-only final50/default-head models')
        if reference is None:reference=inner
        if inner!=reference:raise ValueError('Eight frozen rows must have the same actual internal physical fold')
        result.append((row,c))
    return result,reference


def developer_panel(inner):
    from experiments.cvs_nocfo_foundation.source import load_source
    from experiments.cvs_nocfo_foundation import design as source_design
    L,U,V,contract=load_source(source_design.config(source_design.rows()[0]))
    if U is not None:raise ValueError('Source developer analysis must not open U')
    panel=select_developer(L,inner,contract)
    # Loading the existing source cache materializes V; no V IQ is retained,
    # forwarded, used as a comparator, or used to change a parameter here.
    del V,L
    return panel


def select_developer(L,inner,contract):
    keep=inner['role_ids']['L_dev'];train=set(inner['role_ids']['L_train'])
    if (len(keep)!=720 or len(set(keep))!=720 or len(train)!=5580 or train&set(keep)
            or train|set(keep)!=set(contract['role_ids']['L_s']) or inner['L_dev_updates']):
        raise ValueError('Actual legal internal L developer physical IDs differ')
    lookup={pid:i for i,pid in enumerate(L['ids'])};indices=torch.tensor([lookup[pid] for pid in keep])
    panel={key:L[key][indices] for key in ('x','y','rx','day')};panel['ids']=list(keep)
    return panel


def read_features(model,x):
    logits=[];features=defaultdict(list)
    for start in range(0,len(x),90):
        out,values=b.forward(model,x[start:start+90]);logits.append(out['tx_logits'])
        for key in KEYS:features[key].append(values[key])
    return dict(logits=torch.cat(logits),features={key:torch.cat(v) for key,v in features.items()})


def weighted_loss(losses,arm,recipe):
    if arm=='A0':return sum(losses.values())*0.
    position=arm[2:];coefficients=recipe['coefficients']
    if position=='ALL':
        return coefficients['ALL']['coefficient']*objective.mixture(losses,
            {key:coefficients[key]['coefficient'] for key in b.BRANCHES},recipe['ALL_weights'])
    return coefficients[position]['coefficient']*losses[position]


def gradient_record(model,output,values,target_logits,target,y,arm,recipe,scene,batch):
    theta=b.feature_parameters(model,output['id_feat_joint'])
    if sum(p.numel() for _,p in theta)!=195041:raise ValueError('Actual feature extractor gradient scope changed')
    reliability=objective.labeled_reliability(target_logits,y)
    losses=objective.losses(values,target,reliability)
    ce=.5*F.cross_entropy(output['tx_logits'],y,label_smoothing=.01)
    cg=g.vectors(ce,theta);extra=weighted_loss(losses,arm,recipe);ag=g.vectors(extra,theta)
    tg,sg=(g.vectors(losses[key],theta) for key in ('T','S'))
    shared=[name for name,_ in theta if float(tg[name].norm())>1e-12 and float(sg[name].norm())>1e-12]
    from experiments.cvs_urban_identity50.features import boundary_parameters
    boundary=boundary_parameters(model)
    bg=torch.autograd.grad(extra,list(boundary.values()),allow_unused=True,retain_graph=True)
    if any(v is not None and float(v.abs().max())>1e-8 for v in bg):
        raise ValueError('Frozen auxiliary diagnostics leaked gradient to W or fusion gains')
    return dict(scene=scene,batch=batch,teacher_eligible=int(reliability.sum()),global_gradient=g.compare(cg,ag),
        layer_gradients={name:g.compare({name:cg[name]},{name:ag[name]}) for name,_ in theta},
        T_S_full_feature_cosine=g.compare(tg,sg)['cosine'],
        T_S_shared_only=g.compare({k:tg[k] for k in shared},{k:sg[k] for k in shared}) if shared else None,
        T_S_shared_parameter_names=shared,
        shared_scope='same parameters with both current source T and S gradient norms >1e-12; excludes independent projections',
        auxiliary_boundary_norms={key:0. if value is None else float(value.norm()) for key,value in zip(boundary,bg)},
        optimizer_updates=0,interpretation='Eval source diagnostic, not actual stochastic/clipped AdamW training update')


@torch.no_grad()
def contributions(model,values):
    encoder,core=b.layout(model);joint_norm=values['J'].norm(dim=1).clamp_min(1e-8)
    gains=dict(P=core.cls_head.gain.tanh(),R=encoder.response_gain.tanh(),S=.2*encoder.sequence_pool.gain_raw.tanh())
    return {key:dict(gain_norm=float(gain.detach().norm()),mean_absolute_gain=float(gain.detach().abs().mean()),
        mean_contribution_norm=float((gain*values[key]).norm(dim=1).mean()),
        mean_contribution_to_joint_norm_ratio=float(((gain*values[key]).norm(dim=1)/joint_norm).mean()))
        for key,gain in gains.items()}


def summarize(items):
    n=len(items);fixed=[v for v in items if v['fixed_A0_clean_correct']]
    return dict(count=n,accuracy=sum(v['correct'] for v in items)/n,
        fixed_A0_clean_correct_count=len(fixed),fixed_A0_retention=sum(v['correct'] for v in fixed)/len(fixed) if fixed else None,
        A0_errors_repaired=sum(v['correct'] and not v['A0_satellite_correct'] for v in fixed),
        A0_correct_new_errors=sum(not v['correct'] and v['A0_satellite_correct'] for v in fixed),
        margin=sum(v['margin'] for v in items)/n,
        identity={key:{name:sum(v['identity'][key][name] for v in items)/n for name in items[0]['identity'][key]} for key in KEYS})


def run():
    rows,inner=eligible_rows();out=d.BASE/OUTPUT
    if out.exists():raise FileExistsError('Reconcile an existing source analysis; do not overwrite or resubmit')
    out.mkdir();torch.set_num_threads(2);started=time.perf_counter();panel=developer_panel(inner)
    recipe=d.recipe();y=panel['y'];channel=PairedChannel(d.config(d.rows()[0]));results={};forwards=0
    A0=own_model(d.config(d.rows()[0]),'cpu')
    with torch.no_grad():fixed=read_features(A0,panel['x'])['logits'].argmax(1)==y
    forwards+=720;del A0
    print('SMOKE PASS actual frozen A0/internal L_dev/default head; query_read=false; updates=0',flush=True)
    baseline={}
    with gzip.open(out/'sample_branch_identity.jsonl.gz','xt',encoding='utf-8',compresslevel=3) as stream:
        for row,c in rows:
            arm=row['arm'];model=own_model(c,'cpu')
            checkpoint=torch.load(c['output_root']+'/final_ssdg.pth',map_location='cpu',weights_only=False)
            teacher=copy.deepcopy(model);teacher.load_state_dict(checkpoint['ema_model'],strict=True);teacher.eval().requires_grad_(False)
            before={k:v.clone() for k,v in model.state_dict().items()};tb={k:v.clone() for k,v in teacher.state_dict().items()}
            with torch.no_grad():clean=read_features(model,panel['x']);target=read_features(teacher,panel['x'])
            forwards+=1440;clean_correct=clean['logits'].argmax(1)==y;scenes={};gradients=[]
            model.requires_grad_(True).eval()
            try:
                for scene in d.SCENES:
                    details=[];features=defaultdict(list);logits=[]
                    for start in range(0,720,90):
                        stop=start+90;satellite,meta=apply_fixed_scene(channel,panel['x'][start:stop],metadata(panel,start,stop),scene)
                        details.extend(meta)
                        with torch.set_grad_enabled(scene in ('practical_mid_urban','practical_low_urban')):
                            output,values=b.forward(model,satellite);forwards+=len(satellite)
                            if scene in ('practical_mid_urban','practical_low_urban'):
                                gradients.append(gradient_record(model,output,values,target['logits'][start:stop],
                                    {key:target['features'][key][start:stop] for key in KEYS},y[start:stop],arm,recipe,scene,start//90))
                        logits.append(output['tx_logits'].detach())
                        for key in KEYS:features[key].append(values[key].detach())
                    logits=torch.cat(logits);features={key:torch.cat(v) for key,v in features.items()}
                    correct=logits.argmax(1)==y
                    if arm=='A0':baseline[scene]=correct.clone()
                    distance={key:identity_distances(features[key],clean['features'][key],target['features'][key],y) for key in KEYS}
                    other=logits.masked_fill(F.one_hot(y,6).bool(),-torch.inf).max(1).values
                    margin=(logits.gather(1,y[:,None]).squeeze(1)-other)/30
                    groups=defaultdict(list);items=[]
                    for i,meta in enumerate(details):
                        item=dict(physical_id=panel['ids'][i],correct=bool(correct[i]),clean_correct=bool(clean_correct[i]),
                            fixed_A0_clean_correct=bool(fixed[i]),A0_satellite_correct=bool(baseline[scene][i]),margin=float(margin[i]),
                            identity={key:{name:float(value[i]) for name,value in distance[key].items()} for key in KEYS})
                        key=(int(panel['rx'][i]),meta['state_end'],meta['receiver_lock_status'],quality_bin(meta['signal_to_added_noise_db']))
                        groups[key].append(item);items.append(item)
                        stream.write(json.dumps(dict(arm=arm,scene=scene,RX=key[0],state=key[1],lock=key[2],quality=key[3],
                            receiver_quality_proxy_db=meta['quality_snr_db'],signal_to_added_noise_db=meta['signal_to_added_noise_db'],
                            residual_CFO_hz=meta['output_frequency_hz'],residual_kernel_length=meta['residual_kernel_length'],**item),allow_nan=False)+'\n')
                    scenes[scene]=dict(overall=summarize(items),strata=[dict(RX=k[0],state=k[1],lock=k[2],quality=k[3],**summarize(v)) for k,v in sorted(groups.items())],
                        actual_branch_variance=variance(features),gain_contribution=contributions(model,features))
            finally:model.requires_grad_(False)
            if any(not torch.equal(v,model.state_dict()[k]) for k,v in before.items()) or any(not torch.equal(v,teacher.state_dict()[k]) for k,v in tb.items()):
                raise ValueError('Source branch mechanism mutated frozen model or EMA')
            if any(p.grad is not None or p.requires_grad for p in model.parameters()) or any(p.grad is not None or p.requires_grad for p in teacher.parameters()):
                raise ValueError('Read-only source branch analysis accumulated parameter/teacher gradients')
            results[arm]=dict(clean_accuracy=float(clean_correct.float().mean()),clean_actual_branch_variance=variance(clean['features']),
                clean_gain_contribution=contributions(model,clean['features']),gates=gates(model),scenes=scenes,gradients=gradients,
                model_state_unchanged=True,EMA_state_unchanged=True,checkpoint=c['output_root']+'/final_ssdg.pth')
            d.write(out/'progress.json',dict(completed=list(results),source_only=True,query_read=False))
    if forwards!=46800:raise ValueError('Unexpected eight-row source branch forward budget')
    result=dict(status='ANALYZED',source_only=True,query_read=False,U_truth_read=False,source_role='L_dev',source_count=720,
        V_forward_packets=0,optimizer_updates=0,head_fit_updates=0,feature_modules_added=0,
        fixed_A0_clean_correct_count=int(fixed.sum()),reference='Fixed matched A0 final50 clean-correct L_dev set; no weight inheritance',
        branches=list(KEYS),rows=results,forward_packets=forwards,seconds=time.perf_counter()-started,
        dropout='eval for all zero-update source diagnostic paths',
        wrong_TX_comparison='nearest cosine among all 720 different-TX legal source clean vectors; no fitted classifier',
        interpretation='Identity direction, variance/gain and error evidence; not a trainable upper bound or unique causal damage attribution',
        commit=(d.ROOT/'release_commit.txt').read_text().strip())
    d.write(out/'results.json',result)
    print(json.dumps({k:result[k] for k in ('status','source_only','forward_packets','seconds','fixed_A0_clean_correct_count')}),flush=True)
    return result


if __name__=='__main__':run()
