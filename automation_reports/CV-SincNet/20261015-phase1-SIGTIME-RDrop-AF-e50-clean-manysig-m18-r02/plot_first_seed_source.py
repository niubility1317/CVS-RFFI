"""Complete native source curves from immutable first-seed evidence."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
p=Path(__file__).resolve().parent;data=json.loads((p/'first_seed_source_audit.json').read_text(encoding='utf8'));fig,axes=plt.subplots(1,3,figsize=(14,4.3));colors=dict(A='#6b7280',B='#2563eb',C='#92400e',D='#9333ea',E='#ea580c',F='#059669')
for row in data['rows']:
 values=row['source_epoch_curve'];x=[r['epoch'] for r in values];arm=row['arm']
 for a,key in zip(axes,('train_loss_tx_labeled','val_tx_acc','train_pseudo_selected')):a.plot(x,[r.get(key,0) for r in values],label=arm,color=colors[arm],ls='--' if arm=='A' else '-',lw=1.4)
for a,title in zip(axes,('Native first-forward labeled CE','Source V accuracy (%)','Accepted hidden U per epoch')):a.set_title(title);a.set_xlabel('Epoch');a.set_xlim(1,50);a.grid(alpha=.2);a.axvline(33.5,color='#999999',lw=.8,ls='--')
axes[0].legend(ncol=2);fig.suptitle('SIG_TIME + RDrop A-F: complete50 source curves, seed2026092701',fontsize=13);fig.tight_layout(rect=(0,0,1,.93));fig.savefig(p/'first_seed_source_curves.png',dpi=160);plt.close(fig)
