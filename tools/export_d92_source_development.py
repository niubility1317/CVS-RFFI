"""Frozen source L/V feature export for D92 development; no target loader/truth."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace

from cvs_native_artifacts import read, verify_source


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--spec', type=Path, required=True)
    a = p.parse_args(); spec = read(a.spec); cfg = spec['development']
    sys.path.insert(0, cfg['native_code'])
    import numpy as np
    import torch
    from SSDG import train_ssdg as native
    from cvsrffi.checkpoint_loading import build_exact_ssdg_model_from_checkpoint
    from cvsrffi.identity_only_forward import identity_only_feature_forward
    from cvsrffi.xuc_fusion.native import role_ids_from_native
    from cvsrffi.game_tracking.data import opaque_id
    from cvsrffi.practical_adapter import practical_config
    from leo_practical import apply_leo_practical_channel_batch
    from cvsrffi.stage2_diag_cosine_exploration import registered_feature

    torch.set_num_threads(2)
    source = Path(cfg['source_root']); out = Path(cfg['feature_output'])
    if out.exists():
        raise FileExistsError(out)
    contract, resolved = verify_source(source, read(cfg['source_contract']), cfg['model_seed'])
    payload = torch.load(source / 'final_ssdg.pth', map_location='cpu', weights_only=False)
    if (payload['epoch'] != 200 or payload['args']['seed'] != cfg['model_seed']
            or not payload['args']['from_scratch'] or payload['args']['baseline_ckpt']
            or payload['args']['teacher_ckpt']):
        raise ValueError('Checkpoint provenance mismatch')
    device = torch.device(cfg['device'])
    model, audit = build_exact_ssdg_model_from_checkpoint(payload, input_len=256, device=device)
    model.eval()
    with torch.inference_mode():
        z, logits = identity_only_feature_forward(model, torch.zeros(2, 2, 256, device=device), 'z_id')
        if z.shape != (2, 160) or logits.shape != (2, 6) or not torch.isfinite(logits).all():
            raise ValueError('Real checkpoint no-query smoke failed')
    args = SimpleNamespace(**resolved)
    args.device = str(device); args.num_workers = 0; args.eval_batch_size = 256
    ctx = native._build_ssdg_wisig_data(args, device)
    if ctx['named_test_loaders'] or role_ids_from_native(ctx) != contract['role_ids']:
        raise ValueError('Source-only data contract mismatch')
    out.mkdir(parents=True, exist_ok=False)
    def write(name, value):
        (out / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    write('startup.json', dict(pid=os.getpid(), argv=sys.argv, cwd=os.getcwd(), python=sys.executable,
        model_seed=cfg['model_seed'], checkpoint=str(source / 'final_ssdg.pth'), loader=audit,
        checkpoint_provenance='MATCHED_SOURCE_ONLY_SCRATCH_FINAL200', encoder_updates=0,
        query_access=False, source_contract=cfg['source_contract'], config=cfg))
    features=[]; all_logits=[]; labels=[]; receivers=[]; scenes=[]; ids=[]; roles=[]
    started = time.perf_counter()
    for role, key in [('L_s', 'probe_train_loader'), ('V', 'val_loader')]:
        loader=ctx[key]; index=loader.dataset.index
        identifiers=[opaque_id(it) for it in index]
        if sorted(identifiers) != contract['role_ids'][role]:
            raise ValueError('Role physical-ID mismatch')
        offset=0
        for batch in loader:
            x,y,d,meta=batch
            batch_ids=identifiers[offset:offset+len(x)]; offset+=len(x)
            rx=np.asarray(meta['rx_i'].tolist(), dtype='int64')
            assigned=np.asarray([int(hashlib.sha256((str(cfg['scene_seed'])+'|'+s).encode()).hexdigest()[:8],16)%3 for s in batch_ids])
            original=np.asarray(x.tolist(), dtype='float64')
            received=np.empty_like(original)
            for scene_index,scene in enumerate(cfg['scenarios']):
                use=np.flatnonzero(assigned==scene_index)
                if not len(use):continue
                iq,_,_=apply_leo_practical_channel_batch(original[use],practical_config(scene,args),
                    seed=cfg['augmentation_seed'],sample_ids=[batch_ids[i] for i in use],
                    session_ids=['source_rx'+str(rx[i]) for i in use],
                    realization_namespace='d92_source_development_20260928',receiver_seed=2027)
                received[use]=iq
            with torch.inference_mode():
                z,score=identity_only_feature_forward(model,torch.tensor(received.tolist(),dtype=torch.float32,device=device),'z_id')
            block=registered_feature(received.astype('float32'),np.asarray(z.cpu().tolist(),dtype='float32'))
            features.extend(block);all_logits.extend(score.cpu().tolist());labels.extend(y.tolist())
            receivers.extend(rx.tolist());scenes.extend(assigned.tolist());ids.extend(batch_ids);roles.extend([role]*len(x))
            print(json.dumps(dict(role=role,processed=offset,total=len(index),elapsed_seconds=time.perf_counter()-started)),flush=True)
        if offset!=len(index):raise ValueError('Incomplete source export')
    if len(ids)!=len(set(ids)):raise ValueError('Physical roles overlap')
    np.savez_compressed(out/'features.npz',features=np.asarray(features,dtype='float32'),
        logits=np.asarray(all_logits,dtype='float32'),labels=np.asarray(labels,dtype='int64'),
        receivers=np.asarray(receivers),scenes=np.asarray(scenes),ids=np.asarray(ids),roles=np.asarray(roles))
    write('complete.json',dict(status='SOURCE_FEATURES_COMPLETE',count=len(ids),roles={r:roles.count(r) for r in set(roles)},
        checkpoint_sha256=hashlib.sha256((source/'final_ssdg.pth').read_bytes()).hexdigest(),
        classes=contract['classes'],encoder_updates=0,target_access=False,source_only=True,
        elapsed_seconds=time.perf_counter()-started))


if __name__=='__main__':main()
