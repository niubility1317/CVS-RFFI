import json
from pathlib import Path
import subprocess
import sys

import numpy as np


def test_complete_source_role_holdouts(tmp_path):
    features=tmp_path/'features';features.mkdir()
    y=np.tile(np.repeat(np.arange(6),22),3)
    scene=np.repeat(np.arange(3),6*22)
    role=np.tile(np.array(['L_s']*20+['V']*2),18)
    x=np.zeros((len(y),288),dtype=np.float32);x[np.arange(len(y)),y]=1
    logits=np.eye(6)[y]*10
    np.savez(features/'features.npz',features=x,labels=y,logits=logits,scenes=scene,roles=role,receivers=np.ones(len(y),dtype=int))
    (features/'complete.json').write_text(json.dumps(dict(status='SOURCE_FEATURES_COMPLETE',target_access=False,classes=list('abcdef'))),encoding='utf-8')
    out=tmp_path/'output'
    spec=dict(execution=dict(remote_run_root=str(out)),development=dict(features=str(features),support_seed=1,
        validation_per_class=2,k=[1],k1_fft_grid=[0.0,1.0],expected_rows=120))
    path=tmp_path/'spec.json';path.write_text(json.dumps(spec),encoding='utf-8')
    root=Path(__file__).resolve().parents[1]
    subprocess.run([sys.executable,str(root/'tools/evaluate_d92_registration_proxy.py'),'--spec',str(path),'--commit','test'],check=True,capture_output=True)
    result=json.loads((out/'complete.json').read_text())
    assert result['rows']==120 and result['target_access'] is False
    assert result['novel_tx_performance_verified'] is False
    assert result['selected_k1_fft_weight']==0
    assert all(r['old_accuracy']==r['new_proxy_accuracy']==r['harmonic_mean']==1 for r in result['summary'])
    assert json.loads((out/'startup.json').read_text())['commit']=='test'
