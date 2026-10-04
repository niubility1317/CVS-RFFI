"""All source choices fixed before query; own scratch final checkpoint only."""
import argparse
from pathlib import Path
from types import SimpleNamespace
import sys
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT),str(ROOT/'code')]
import numpy as np
import torch
from experiments.cvs_phase1_stack.design import *
from experiments.cvs_phase1_stack.runtime import installed
from experiments.cvs_phase1_stack.dispatch import validate_freeze
from experiments.cvs_equivariant_identity.precision import numerical_context


def predict(p):
    rid=p['row_id'];row=next(r for r in rows() if r['row_id']==rid)
    expected=dict(row_id=rid,source_config=BASE+'/configs/source-'+rid+'.json',source_output=BASE+'/'+rid+'/source',output_root=BASE+'/'+rid+'/prediction',p1_capsule=CAPSULE)
    if p!=expected:raise ValueError('Unregistered prediction input')
    whole=read(Path(BASE)/'all_sources_frozen.json')
    if whole!=dict(status='ALL_SOURCE_FROZEN',run_id=RUN,rows=[r['row_id'] for r in rows()],stages=list(STAGES),target_access=False):raise ValueError('Source selection incomplete')
    for stage in STAGES:validate_freeze(read(Path(BASE)/(stage+'_source_frozen.json')),stage)
    c=read(p['source_config']);validate(c);source=Path(p['source_output']);out=Path(p['output_root'])
    f=read(Path(BASE)/(c['stage']+'_source_frozen.json'))
    if next(r['config'] for r in f['rows'] if r['row_id']==rid)!=c:raise ValueError('Frozen config mismatch')
    init=read(source/'initialization.json');done=read(source/'completion.json');contract=read(source/'source_contract.json');expected_contract=read(SOURCE)
    if not init['scratch_only'] or init['checkpoint_sources'] or init['ancestors'] or init['target_contact'] or init['source_roles']!='EXACT_MATCH' or done['config']!=c or done['epoch']!=200 or done['target_access'] or done['target_evaluated']:raise ValueError('Checkpoint provenance invalid')
    for k in ['role_ids','classes','source_rxs','source_days','ratios','split_seed','equalized','out_len','normalize']:
        if contract[k]!=expected_contract[k]:raise ValueError('CHECKPOINT_DATA_CONTRACT_MISMATCH '+k)
    torch.set_num_threads(2);device=torch.device('cuda:0')
    with numerical_context(FULL_FP32_POLICY),installed(c) as n:
        ck=torch.load(source/'final_ssdg.pth',map_location=device,weights_only=False)
        if ck['epoch']!=200 or ck['args']['baseline_ckpt'] or ck['args']['teacher_ckpt'] or not ck['args']['from_scratch'] or ck['candidate_id']!=rid or ck['run_id']!=RUN or ck['checkpoint_selection']!='final_only':raise ValueError('Checkpoint contents disagree')
        from scripts.train_daot_rc4_baseline import resolved_config
        from experiments.cvs_phase1_stack.source import clean
        if clean(resolved_config(SimpleNamespace(**ck['args'])))!=read(source/'resolved_native_args.json'):raise ValueError('Native args provenance mismatch')
        m=n.build_baseline_model(SimpleNamespace(**ck['baseline_args']),device)
        m.load_state_dict(ck['model'],strict=True);m.eval()
        with torch.no_grad():scores=m(torch.zeros(2,2,256,device=device))
        if scores.shape!=(2,6) or not torch.isfinite(scores).all():raise ValueError('Checkpoint no-query smoke failed')
        manifest=read(Path(CAPSULE)/'manifest.json')
        if manifest['status']!='VALIDATED_ONCE' or manifest['classes']!=CLASSES or manifest['scenes']!=SCENES or manifest['channel']!='residual/post_sync/noeq':raise ValueError('Capsule mismatch')
        with np.load(Path(CAPSULE)/'index.npz',allow_pickle=False) as ix:ids=ix['ids'].copy()
        if len(ids)!=168000 or len(set(ids.tolist()))!=168000:raise ValueError('Physical IDs mismatch')
        out.mkdir(parents=True,exist_ok=False);write(out/'resolved_config.json',dict(p,truth_read=False,query_fit=False))
        predictions={}
        with torch.no_grad():
            for view in ['clean','satellite']:
                array=np.load(Path(CAPSULE)/(view+'.npy'),mmap_mode='r',allow_pickle=False)
                if array.shape!=(168000,2,256):raise ValueError('Query shape mismatch')
                pred=[]
                for i in range(0,len(ids),256):
                    logits=m(torch.tensor(np.asarray(array[i:i+256]).tolist(),device=device,dtype=torch.float32))
                    if not torch.isfinite(logits).all():raise FloatingPointError('Invalid logits')
                    pred.extend(logits.argmax(1).cpu().tolist())
                predictions[view]=np.asarray(pred,dtype=np.int64)
        np.savez(out/'phase1_predictions.npz',ids=ids,**predictions)
        write(out/'phase1_complete.json',dict(status='PREDICTIONS_COMPLETE',count=len(ids),truth_read=False))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);a=p.parse_args();predict(read(a.config))
