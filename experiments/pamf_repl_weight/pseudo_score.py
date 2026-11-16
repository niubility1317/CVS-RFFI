"""Independent truth-side aggregation of already-frozen U observations."""
import gzip
import json


def score_observations(path,*,truth,allowed_ids,num_classes,steps_per_epoch=222):
    """Truth remains outside training; exposures and unique U IDs are distinct."""
    allowed=set(allowed_ids)
    if len(allowed)!=len(allowed_ids) or not allowed or set(truth)!=allowed or num_classes<2:
        raise ValueError('Exact unique U role truth mapping required')
    if any(not isinstance(y,int) or not 0<=y<num_classes for y in truth.values()):
        raise ValueError('Truth labels must match the original class mapping')
    overall=dict(seen=0,accepted=0,accepted_correct=0)
    epochs={};true_classes={};predicted_classes={};unique_seen=set();unique_accepted=set()
    def add(group,accepted,correct):
        group['seen']+=1;group['accepted']+=int(accepted)
        group['accepted_correct']+=int(accepted and correct)
    def bucket(mapping,key):
        return mapping.setdefault(str(key),dict(seen=0,accepted=0,accepted_correct=0))
    last_step=0
    with gzip.open(path,'rt',encoding='utf-8') as stream:
        for line in stream:
            row=json.loads(line);step=row['step']
            fields=[row[key] for key in ('physical_ids','pseudo','confidence','accepted')]
            if not isinstance(step,int) or step<=last_step or not all(len(v)==len(fields[0]) for v in fields):
                raise ValueError('Observation steps and aligned batch fields required')
            last_step=step;epoch=(step-1)//steps_per_epoch+1
            for sid,pred,confidence,selected in zip(*fields):
                if (sid not in allowed or not isinstance(pred,int) or not 0<=pred<num_classes
                        or not isinstance(selected,bool) or not isinstance(confidence,(int,float))
                        or not 0<=confidence<=1):
                    raise ValueError('Invalid source observation')
                actual=truth[sid];correct=pred==actual
                for group in (overall,bucket(epochs,epoch),bucket(true_classes,actual),bucket(predicted_classes,pred)):
                    add(group,selected,correct)
                unique_seen.add(sid)
                if selected:unique_accepted.add(sid)
    if not overall['seen']:raise ValueError('Empty observations')
    return dict(overall=overall,per_epoch=epochs,per_true_class=true_classes,per_pseudo_class=predicted_classes,
                unique_U_total=len(allowed),unique_seen=len(unique_seen),unique_accepted=len(unique_accepted),
                role='U_s',truth_used_only_for_offline_diagnostic=True,training_or_selection_use=False,
                interpretation='Accuracy is accepted_correct/accepted over repeated exposures; '
                'unique_accepted counts distinct physical IDs accepted at least once.')
