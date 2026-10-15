"""Plot the full fixed50 source trajectories without fitting any model."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
p=Path(__file__).resolve().parent;data=json.loads((p/'source_curves_three_seed.json').read_text(encoding='utf8'));fig,axes=plt.subplots(3,3,figsize=(14,10),sharex=True);colors=dict(OLD_GATE_COS50='#6b7280',NORM_CONTROL='#2563eb',RX_CAGRAD='#dc2626')
for row,seed in enumerate((2026092701,2026092702,2026092703)):
 for v in data['rows']:
  if not v['row_id'].endswith(str(seed)):continue
  r=v['source_epoch_curve'];x=[a['epoch'] for a in r];name=v['arm'];color=colors[name]
  for col,key in enumerate(('train_loss_tx_labeled','val_tx_acc','train_pseudo_selected')):
   axes[row,col].plot(x,[a.get(key,0) for a in r],label=name,color=color,lw=1.5)
 for col in range(3):
  axes[row,col].axvline(33.5,color='#999999',lw=.8,ls='--');axes[row,col].grid(alpha=.2);axes[row,col].set_xlim(1,50)
 axes[row,0].set_ylabel(f'Seed {seed}\nL CE')
for a,title in zip(axes[0],('L identity CE','Source V accuracy (%)','Accepted hidden U per epoch')):a.set_title(title)
for a in axes[-1]:a.set_xlabel('Epoch')
axes[0,0].legend(fontsize=8);fig.suptitle('RX gradient coordination: complete fixed50 source curves, three seeds',fontsize=14);fig.tight_layout(rect=(0,0,1,.96));fig.savefig(p/'source_curves_three_seed.png',dpi=160);plt.close(fig)
