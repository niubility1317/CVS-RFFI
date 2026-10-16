"""Fit-L progress only; no network, checkpoint, validation or target access."""
import math
import numpy as np
FIT=(1,3,4,8)
def eligible(scene,m):
    value=m.get('signal_to_added_noise_db')
    return bool(scene in ('practical_mid_urban','practical_low_urban') and m.get('receiver_lock_status') in ('locked','degraded') and value is not None and math.isfinite(value) and 3.<=value<=15.)
def aggregate(cells,gradient):
    expected={(t,r,d) for t in range(6) for r in FIT for d in (1,2,3)};seen=set();count=correct=0;margin=ce=0.;acc=[];margins=[]
    for c in cells:
        key=(c['TX'],c['RX'],c['day'])
        if key not in expected or key in seen or c['count']<=0 or not 0<=c['correct']<=c['count'] or not math.isfinite(c['margin_sum']) or not math.isfinite(c['CE_sum']):raise ValueError('Only finite legal labeled fitL cell statistics')
        seen.add(key);n=c['count'];count+=n;correct+=c['correct'];margin+=c['margin_sum'];ce+=c['CE_sum'];acc.append(c['correct']/n);margins.append(c['margin_sum']/n)
    if count!=28416:raise ValueError('Exactly222 L128 primary statistics per epoch')
    return dict(L_packets=count,source_cells=len(seen),all_72_fit_cells=seen==expected,accuracy=correct/count,mean_margin=margin/count,mean_CE=ce/count,
        p10_cell_accuracy=float(np.quantile(acc,.1)),p10_cell_margin=float(np.quantile(margins,.1)),gradient=float(gradient))
class Policy:
    def __init__(self):self.epoch=1;self.eta=0.;self.history=[]
    def advance(self,epoch,metrics):
        if epoch!=self.epoch or epoch not in range(1,21):raise ValueError('One ordered update per fixedE20 epoch')
        required=('accuracy','p10_cell_accuracy','p10_cell_margin','mean_CE','gradient')
        finite=all(math.isfinite(metrics[k]) for k in required)
        absolute=finite and metrics['all_72_fit_cells'] and metrics['accuracy']>=.7 and metrics['p10_cell_accuracy']>=.6 and metrics['p10_cell_margin']>=.02 and metrics['mean_CE']<=1. and 0<=metrics['gradient']<=5.
        previous=[r['metrics'] for r in self.history[-3:]]
        stable=not previous or (metrics['mean_CE']<=1.05*float(np.median([r['mean_CE'] for r in previous])) and metrics['p10_cell_margin']>=float(np.median([r['p10_cell_margin'] for r in previous]))-.01 and metrics['gradient']<=2*max(1e-12,float(np.median([r['gradient'] for r in previous]))))
        before=self.eta
        if epoch>=5 and absolute and stable:self.eta=min(.5,round(self.eta+.1,10))
        self.history.append(dict(epoch=epoch,eta_used=before,eta_next=self.eta,absolute_ready=bool(absolute),stable=bool(stable),metrics=metrics));self.epoch+=1
        return self.history[-1]
    def summary(self):
        return dict(epochs=len(self.history),eta_final=self.eta,max_eta=max((r['eta_used'] for r in self.history),default=0.),history=self.history,
            progress_source='Existing first ordinary L default-head true-label logits and native gradient only',held_V_used=False,U_truth_used=False,target_access=False,extra_network_forwards=0)
