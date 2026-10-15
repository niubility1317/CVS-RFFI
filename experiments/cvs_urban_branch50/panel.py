"""Read-only internal developer metrics, without V role changes or target scores."""
from collections import defaultdict
import torch
from torch.nn import functional as F
from . import design as d
from .channel import PairedChannel
from experiments.cvs_urban_sigtime50.panel import metadata,apply_fixed_scene

EVAL_EPOCHS=(1,5,6,10,15,20,25,30,33,34,35,40,45,50)


def evaluate(model,panel):
    modes={m:m.training for m in model.modules()};model.eval();device=next(model.parameters()).device
    channel=PairedChannel(d.config(d.rows()[0]));groups=defaultdict(list);clean_ids=[];hits=0
    try:
        with torch.no_grad():
            for start in range(0,720,90):
                stop=start+90;x=panel['x'][start:stop].to(device);y=panel['y'][start:stop].to(device)
                clean=model.forward_identity_only(x)['tx_logits'];correct=clean.argmax(1)==y;hits+=int(correct.sum())
                clean_ids.extend(pid for pid,ok in zip(panel['ids'][start:stop],correct.cpu().tolist()) if ok)
                for scene in d.SCENES:
                    sat,details=apply_fixed_scene(channel,x,metadata(panel,start,stop),scene)
                    output=model.forward_identity_only(sat)['tx_logits']
                    other=output.masked_fill(F.one_hot(y,6).bool(),-torch.inf).max(1).values
                    margin=(output.gather(1,y[:,None]).squeeze(1)-other)/30
                    for i,info in enumerate(details):
                        groups[(scene,int(panel['rx'][start+i]),info['state_end'],info['receiver_lock_status'])].append(
                            dict(correct=bool(output[i].argmax()==y[i]),margin=float(margin[i]),clean_correct=bool(correct[i])))
    finally:
        for module,mode in modes.items():module.training=mode
    strata=[];scenes=defaultdict(list);rx=defaultdict(list)
    for key,items in sorted(groups.items()):
        scenes[key[0]].extend(items);rx[(key[0],key[1])].extend(items)
        retained=[i for i in items if i['clean_correct']]
        strata.append(dict(scene=key[0],RX=key[1],state=key[2],lock=key[3],count=len(items),
            accuracy=sum(i['correct'] for i in items)/len(items),margin=sum(i['margin'] for i in items)/len(items),
            own_clean_correct_retention=sum(i['correct'] for i in retained)/len(retained) if retained else None))
    accuracy={k:sum(i['correct'] for i in v)/len(v) for k,v in scenes.items()}
    rx_accuracy={str(k):sum(i['correct'] for i in v)/len(v) for k,v in rx.items()}
    return dict(clean_accuracy=hits/720,clean_correct_ids=clean_ids,scene_accuracy=accuracy,
        urban_mean=sum(accuracy[k] for k in d.URBAN)/3,
        worst_urban_RX=min(sum(rx_accuracy[str((scene,r))] for scene in d.URBAN)/3 for r in (1,3,4,6,8)),
        strata=strata,source_count=720,source_role='L_dev',source_only=True,query_read=False,
        training_updates=0,head_fit_updates=0,forward_packets=5040,
        selection_uses_final_epoch_only=True)
