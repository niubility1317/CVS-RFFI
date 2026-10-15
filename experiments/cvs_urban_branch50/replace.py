"""Same-model clean branch replacement on legal source IQ; no optimization."""
import gzip,inspect,json,time
from collections import defaultdict
from pathlib import Path
import torch
from torch.nn import functional as F
from . import branches as b
from experiments.cvs_urban_sigtime50 import design as reference
from experiments.cvs_urban_sigtime50.model import own_model
from experiments.cvs_urban_sigtime50.panel import load,metadata,apply_fixed_scene
from experiments.cvs_urban_sigtime50.channel import PairedChannel,audit
from experiments.cvs_urban_sigtime50.mechanism import quality_bin

RUN='20261015-phase1-sigtime-branch-replace-source-manysig-s2026092701-r01'
BASE=Path(reference.PROJECT)/'runs'/RUN


def metrics(items):
    if not items:raise ValueError('Empty replacement cohort')
    correct=sum(r['correct'] for r in items);base=sum(r['baseline_correct'] for r in items)
    clean=[r for r in items if r['clean_correct']]
    return dict(count=len(items),accuracy=correct/len(items),baseline_accuracy=base/len(items),
        difference=(correct-base)/len(items),repaired=sum(r['correct'] and not r['baseline_correct'] for r in items),
        new_errors=sum(not r['correct'] and r['baseline_correct'] for r in items),
        mean_margin=sum(r['margin'] for r in items)/len(items),clean_correct_count=len(clean),
        clean_correct_retention=sum(r['correct'] for r in clean)/len(clean) if clean else None)


def run():
    c=reference.config(next(r for r in reference.rows() if r['arm']=='B0'))
    model=own_model(c,'cpu');panel=load();torch.set_num_threads(2)
    if BASE.exists():raise FileExistsError('Reconcile existing branch replacement output')
    BASE.mkdir();root=Path(__file__).resolve().parents[2];started=time.perf_counter()
    spec=reference.read(root/'experiments/cvs_urban_branch50/registration_replace.json')
    spec['code']['commit']=(root/'release_commit.txt').read_text().strip();reference.write(BASE/'experiment.json',spec)
    before={k:v.clone() for k,v in model.state_dict().items()};channel=PairedChannel(c)
    groups=defaultdict(list);forwards=0
    with torch.no_grad(),gzip.open(BASE/'sample_replacements.jsonl.gz','xt',encoding='utf-8',compresslevel=3) as stream:
        for start in range(0,720,90):
            stop=start+90;x=panel['x'][start:stop];y=panel['y'][start:stop]
            clean,cv=b.forward(model,x);forwards+=len(x)
            clean_correct=clean['tx_logits'].argmax(1)==y
            if start==0:
                print('SMOKE PASS frozen source B0 checkpoint; actual T/F/P/R/S/J; query_read=false; updates=0',flush=True)
            for scene in reference.SCENES:
                satellite,details=apply_fixed_scene(channel,x,metadata(panel,start,stop),scene)
                original,urban=b.forward(model,satellite);forwards+=len(x)
                baseline_correct=original['tx_logits'].argmax(1)==y
                interventions={'full':original}
                for key in b.BRANCHES:
                    with b.replace_one(model,cv,key):changed,values=b.forward(model,satellite)
                    forwards+=len(x)
                    if not torch.equal(values[key],cv[key]):raise ValueError('Selected branch was not replaced exactly')
                    for other in (*b.BRANCHES,'_rho'):
                        if other!=key and not torch.equal(values[other],urban[other]):
                            raise ValueError('Replacement changed another branch: '+key+'/'+other)
                    interventions[key]=changed
                for key,output in interventions.items():
                    logits=output['tx_logits'];correct=logits.argmax(1)==y
                    other=logits.masked_fill(F.one_hot(y,6).bool(),-torch.inf).max(1).values
                    margins=(logits.gather(1,y[:,None]).squeeze(1)-other)/30
                    for i,m in enumerate(details):
                        item=dict(physical_id=panel['ids'][start+i],correct=bool(correct[i]),baseline_correct=bool(baseline_correct[i]),
                            clean_correct=bool(clean_correct[i]),margin=float(margins[i]))
                        group=(scene,key,m['state_end'],m['receiver_lock_status'],quality_bin(m['signal_to_added_noise_db']))
                        groups[group].append(item)
                        stream.write(json.dumps(dict(scene=group[0],condition=group[1],state=group[2],lock=group[3],quality=group[4],**item),allow_nan=False)+'\n')
            reference.write(BASE/'progress.json',dict(completed_source_samples=stop,source_only=True,query_read=False,forward_packets=forwards))
    if any(not torch.equal(v,model.state_dict()[k]) for k,v in before.items()):raise ValueError('Replacement mutated checkpoint state')
    if any(p.grad is not None or p.requires_grad for p in model.parameters()):raise ValueError('Read-only replacement has training gradients')
    strata=[dict(scene=k[0],condition=k[1],state=k[2],lock=k[3],quality=k[4],**metrics(v)) for k,v in sorted(groups.items())]
    combined=defaultdict(list)
    for key,items in groups.items():combined[key[:2]].extend(items)
    overall=[dict(scene=k[0],condition=k[1],**metrics(v)) for k,v in sorted(combined.items())]
    if forwards!=26640:raise ValueError('Unexpected replacement diagnostic budget')
    result=dict(status='SOURCE_BRANCH_REPLACEMENT_COMPLETE',source_only=True,query_read=False,U_truth_read=False,
        source_role='L_s',source_count=720,source_checkpoint=c['output_root']+'/final_ssdg.pth',
        model_state_unchanged=True,optimizer_updates=0,head_fit_updates=0,feature_modules_added=0,
        branches=list(b.BRANCHES),core_module_file=inspect.getfile(type(b.layout(model)[1])),channel=audit(),
        overall=overall,strata=strata,forward_packets=forwards,seconds=time.perf_counter()-started,
        interpretation='Operational same-model clean replacement on previously trained source L samples; not a trainable upper bound, deployable clean input or branch ranking proof',
        formal_branch_matrix_filtering=False,training_initialization_used=False,
        commit=(root/'release_commit.txt').read_text().strip())
    reference.write(BASE/'results.json',result)
    print(json.dumps({k:result[k] for k in ('status','source_only','forward_packets','seconds','model_state_unchanged')}),flush=True)
    return result


if __name__=='__main__':run()
