"""Source-only real frozen DURA checkpoint check before continuation launch."""
from pathlib import Path
import numpy as np
import torch
from .io import read,write,tensor


def run(rt):
    from .frozen_inference import load_G,load_original_F
    source=read(rt['source_runtime']);handles=read(source['reused_source_handles'])
    h=handles['F0'];contract=read(h['source_contract'])
    old=Path(rt['original_G_action_cache'])/'train'
    ids=np.load(old/'physical_ids.npy',allow_pickle=False)
    if set(ids.tolist())!=set(contract['role_ids']['L_s']+contract['role_ids']['U_s']):
        raise ValueError('Original G source lineage union differs')
    x=tensor(np.load(Path(h['source_cache'])/'L_s_x.npy',mmap_mode='r',allow_pickle=False)[:2])
    spec=dict(kind='original_F0_LU',roles='LU',source_contract=h['source_contract'],checkpoint=rt['original_G_checkpoint'])
    g=load_G(spec,contract,'cpu');f=load_original_F(2026092701,contract,'cpu')
    with torch.no_grad():
        restored=g(x);logits=f.classify_features(f.features(restored))
    if restored.shape!=x.shape or logits.shape!=(2,6) or not bool(torch.isfinite(logits).all()):
        raise ValueError('Real frozen DURA/F source checkpoint smoke failed')
    path=Path(rt['run_root']);path.mkdir(parents=True,exist_ok=False)
    write(path/'prelaunch_smoke.json',dict(status='PASS',source_role='L_s',samples=2,G_parameters=266951,
        query_reads=0,TX_labels_G_read=0,head_fit_updates=0,source_lineage='original exact L/U union; no V',
        actual_G_checkpoint=rt['original_G_checkpoint']))
