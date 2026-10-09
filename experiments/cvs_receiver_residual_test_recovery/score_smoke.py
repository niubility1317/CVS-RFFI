"""Regression: truth contains physical RX labels, not dataset index strings."""
import numpy as np
from experiments.cvs_receiver_residual_test_recovery import evaluate as ev


def main():
    truth={};ids=[]
    for label in range(6):
        for rx in ev.HELDOUT_RX.values():
            for day in ev.TARGET_DAYS:
                for i in range(1000):
                    sid=f'{label}:{rx}:{day}:{i}';ids.append(sid)
                    truth[sid]=dict(label=label,receiver=rx,day=day)
    ids=np.asarray(ids);y,rx,days=ev.truth_arrays(truth,ids)
    assert len(y)==168000 and set(rx)==set(ev.HELDOUT_RX.values())
    key=ids[0];original=truth[key].copy()
    for field,value in [('receiver','0'),('receiver','1-19'),('label',6),('day','2021_01_01')]:
        truth[key]=dict(original,**{field:value})
        try:ev.truth_arrays(truth,ids)
        except ValueError:pass
        else:raise AssertionError('Invalid metadata accepted')
    truth[key]=original
    truth.pop(key)
    try:ev.truth_arrays(truth,ids)
    except ValueError:pass
    else:raise AssertionError('Missing physical ID accepted')
    print('PASS: physical RX/day coverage and five invalid metadata cases')


if __name__=='__main__':main()
