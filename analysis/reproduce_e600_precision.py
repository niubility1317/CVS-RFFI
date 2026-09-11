"""Compare FP16 loss scaling with FP32 in the unchanged E600 source entry."""
import json,sys
from pathlib import Path
from unittest.mock import patch
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'code'),str(ROOT/'code/scripts')]
from SSDG import train_ssdg as train
from run_a1_extended_budgets import extended_matrix
from check_a1_ecrs_cross_rx import run_check

out=ROOT/'analysis/e600_repair_20260911/precision_reproduction'
out.mkdir(parents=True,exist_ok=False)
results=[]
original=train.GradScaler
for amp in (True,False):
    changes=[]
    def factory(*a,**kw):
        scaler=original(*a,**kw)
        update=scaler.update
        def observed(*args,**kwargs):
            before=scaler.get_scale();ret=update(*args,**kwargs);after=scaler.get_scale()
            changes.append({'before':before,'after':after,'overflow_skip':after<before})
            return ret
        scaler.update=observed
        return scaler
    matrix=extended_matrix()
    row=next(r for r in matrix['rows'] if r['id']=='X2_E600')
    row['options']['--amp']=str(amp).lower()
    matrix['rows']=[row]
    with patch.object(train,'GradScaler',factory):
        check=run_check(torch.device('cuda:0'),out/('fp16' if amp else 'fp32'),matrix_factory=lambda:matrix)
    packet=out/('fp16' if amp else 'fp32')/'entry/X2_E600/first_rc4_anomaly.pt'
    anomaly=None
    if packet.exists():
        p=torch.load(packet,map_location='cpu',weights_only=False)
        anomaly={k:p[k] for k in ['epoch','batch_index','loss_finite','first_nonfinite_gradient']}
    result={'amp':amp,'successful_steps':check['actual_train_entry_successful_steps'],
            'scaler_updates':changes,'first_anomaly':anomaly,'source_only_no_checkpoint':True}
    results.append(result)
    print(json.dumps(result),flush=True)
(out/'result.json').write_text(json.dumps(results,indent=2)+'\n',encoding='utf-8')
assert not any(x['overflow_skip'] for x in results[1]['scaler_updates'])
assert results[1]['first_anomaly'] is None
