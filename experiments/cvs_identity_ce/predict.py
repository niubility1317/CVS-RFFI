"""Frozen Phase1 predictions; no truth input and no source data loader."""
import argparse
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT/'code')]
import numpy as np
import torch
from experiments.cvs_identity_ce.model import IdentityOnlyCVS
from experiments.cvs_identity_ce.source import write


def read(path): return json.loads(Path(path).read_text(encoding='utf-8'))


def predict(c):
    if any(c.get(k) for k in ('truth', 'p1_truth', 'target_truth')):
        raise ValueError('Predictor rejects truth inputs')
    out, source = Path(c['output_root']), Path(c['source_output'])
    out.mkdir(parents=True, exist_ok=False)
    device = torch.device(c.get('device', 'cuda:0')); torch.set_num_threads(2)
    done, initial = read(source/'completion.json'), read(source/'initialization.json')
    contract, expected = read(source/'source_contract.json'), read(c['source_contract'])
    if (done['status'] != 'SOURCE_TRAINED' or done['epoch'] != 200 or
        initial['status'] != 'SCRATCH' or not initial['scratch_only'] or
        initial['checkpoint_sources'] or initial['ancestors'] or initial['target_access'] or
        initial['target_contact'] or initial['model_seed'] != c['model_seed']):
        raise ValueError('Checkpoint provenance invalid')
    for key in ('role_ids', 'source_rxs', 'source_days', 'ratios', 'split_seed'):
        if contract[key] != expected[key]: raise ValueError('CHECKPOINT_DATA_CONTRACT_MISMATCH: '+key)
    if contract['equalized'] != 1 or contract['out_len'] != 256 or not contract['normalize']:
        raise ValueError('Checkpoint input contract mismatch')
    payload = torch.load(source/'last.pt', map_location=device, weights_only=False)
    if (payload['epoch'] != 200 or payload['source_contract'] != contract or
        payload['initialization'] != initial or payload['selection'] != 'fixed_last_epoch' or
        payload['method'] != 'cvs_identity_ce'):
        raise ValueError('Checkpoint content mismatch')
    model = IdentityOnlyCVS().to(device)
    model.load_state_dict(payload['model'], strict=True); model.eval()
    with torch.no_grad():
        scores = model(torch.zeros(2,2,256,device=device))
    if scores.shape != (2,6) or not torch.isfinite(scores).all(): raise ValueError('Frozen checkpoint smoke failed')
    write(out/'provenance.json', dict(status='VERIFIED', source_roles='EXACT_MATCH',
        scratch_ancestors=[], checkpoint=str(source/'last.pt'), selection='epoch200', query_fit=False))
    capsule = Path(c['p1_capsule']); manifest = read(capsule/'manifest.json')
    if (manifest['status'] != 'VALIDATED_ONCE' or manifest['classes'] != contract['classes'] or
        manifest['channel'] != 'residual/post_sync/noeq'):
        raise ValueError('Phase1 class/channel capsule mismatch')
    with np.load(capsule/'index.npz', allow_pickle=False) as ix:
        ids = ix['ids'].copy()
    if len(ids) != len(set(ids.tolist())): raise ValueError('Duplicate physical IDs')
    write(out/'resolved_config.json', dict(c, truth_input=False, query_fit=False))
    predictions = {}
    with torch.no_grad():
        for view in ('clean', 'satellite'):
            array = np.load(capsule/(view+'.npy'), mmap_mode='r', allow_pickle=False)
            if array.shape != (len(ids),2,256): raise ValueError('Phase1 input shape mismatch')
            pred = []
            for start in range(0,len(ids),256):
                # Compatible with N607 Torch2.1/NumPy2 without their C ABI bridge.
                x = torch.tensor(np.asarray(array[start:start+256]).tolist(), device=device, dtype=torch.float32)
                logits = model(x)
                if logits.shape != (len(x),6) or not torch.isfinite(logits).all():
                    raise ValueError('Invalid frozen classifier scores')
                pred.extend(logits.argmax(1).cpu().tolist())
            predictions[view] = np.asarray(pred, dtype=np.int64)
            print('PREDICT '+json.dumps(dict(view=view,count=len(pred),truth_read=False)), flush=True)
    np.savez(out/'phase1_predictions.npz', ids=ids, **predictions)
    write(out/'phase1_complete.json', dict(status='PREDICTIONS_COMPLETE', count=len(ids),truth_read=False))


if __name__ == '__main__':
    p=argparse.ArgumentParser(); p.add_argument('--config',required=True); a=p.parse_args()
    predict(read(a.config))
