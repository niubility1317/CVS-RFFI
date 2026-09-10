"""Read complete extended curves to distinguish scheduled inactivity from collapse."""
import json,subprocess
from pathlib import Path
REMOTE=r'''
import json
from pathlib import Path
p=Path('/home/szu2070436088/2510044040/CV-SincNet/runs/a1_extended_s392005_20260910_r1')
out={}
for name in ['X2_E400','X2_E600','R3_CLEAN_RX_E400']:
 rows=[json.loads(l) for l in (p/name/'metrics_epoch.jsonl').read_text().splitlines() if l.strip()]
 keys=[k for k in rows[-1] if any(s in k for s in ['grad_norm','grad_backbone','loss_tx','ecrs_cross','source_val_sat','rc4_g_L','skipped_nonfinite']) or k in ['epoch','val_tx_acc','train_loss','lr']]
 out[name]=[{k:x.get(k) for k in keys} for x in rows]
print(json.dumps(out))
'''
r=subprocess.run(['ssh','-F','E:/type10-7/tools/n607_ssh_config','-o','BatchMode=yes','-o','ConnectTimeout=10','N607','python3 -'],input=REMOTE,text=True,encoding='utf-8',capture_output=True,timeout=30,check=True)
d=json.loads(r.stdout)
out=Path(__file__).parent/'mechanism_activation_audit_20260910'/'extended_signal_curves.json'
out.write_text(json.dumps(d,indent=2)+'\n',encoding='utf-8')
for n,rows in d.items():
    zeros=[x['epoch'] for x in rows if x.get('train_ecrs_cross_rx_identity_grad_norm')==0]
    chance=[x['epoch'] for x in rows if abs(x['val_tx_acc']-100/6)<1e-8]
    print(n,'epochs',len(rows),'zero_probe',zeros,'chance_clean',chance)
    for x in rows:
        if x['epoch'] in [65,68,70,71,72,73,74,75,80,100,150,180,len(rows)]:
            print(x['epoch'],{k:x.get(k) for k in ['val_tx_acc','stage_source_val_sat_mean_tx','train_ecrs_cross_rx_identity_grad_norm','train_ecrs_cross_rx_raw_loss','train_loss','train_loss_tx_labeled','train_grad_backbone','lr']})
