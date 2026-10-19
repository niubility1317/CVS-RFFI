import math
import torch
from . import design as d


class ObservedFocus:
    """Train-only scalar aggregation; no labels, fitted network, or inference state."""
    def __init__(self,arm,device='cpu'):
        d.configuration(arm);self.arm=arm;self.logq=torch.zeros(len(d.RX),device=device)
        self.updates=0;self.seen=0;self.selected=0
        self.rx_seen={str(rx):0 for rx in d.RX};self.rx_selected={str(rx):0 for rx in d.RX}

    def __call__(self,loss,rx):
        if (loss.ndim!=1 or not len(loss) or rx.shape!=loss.shape or rx.dtype!=torch.long or
            rx.device!=loss.device or self.logq.device!=loss.device or not torch.isfinite(loss).all() or
            not all(int(v) in d.RX for v in rx.unique().tolist())):
            raise ValueError('Finite observed-only source losses and matched legal RX required')
        n=len(loss);selected=torch.arange(n,device=loss.device)
        if self.arm==d.ARMS[0]:value=loss.mean()
        elif self.arm==d.ARMS[1]:
            selected=torch.topk(loss,max(1,math.ceil(.25*n)),sorted=False).indices
            value=loss[selected].mean()
        else:
            present=[i for i,domain in enumerate(d.RX) if bool((rx==domain).any())]
            means=torch.stack([loss[rx==d.RX[i]].mean() for i in present])
            with torch.no_grad():
                self.logq[present]+=.01*means.detach()
                self.logq-=self.logq.max()
                self.updates+=1
            # Missing groups receive no batch gradient; renormalize only present groups.
            weights=torch.softmax(self.logq[present],0).detach()
            value=(weights*means).sum()
        self.seen+=n;self.selected+=len(selected)
        for domain in d.RX:
            self.rx_seen[str(domain)]+=int((rx==domain).sum())
            self.rx_selected[str(domain)]+=int((rx[selected]==domain).sum())
        return value

    def state(self):
        return dict(arm=self.arm,source_RX=list(d.RX),updates=self.updates,observed_exposures=self.seen,
            selected_exposures=self.selected,selected_fraction=self.selected/self.seen if self.seen else 0.,
            RX_exposures=dict(self.rx_seen),selected_RX_exposures=dict(self.rx_selected),
            global_RX_weights=torch.softmax(self.logq,0).tolist(),log_RX_weights=self.logq.tolist(),
            TX_labels_read=0,target_reads=0,head_fit_updates=0)
