"""Frozen native CVS final evaluation / source-L and received-IQ feature export.

This process imports the exact native release, independently of the D92 package.
Neither action accepts query truth. Source-L preparation precedes target inference.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def verify_source(source, contract, seed):
    actual=read(source/'source_contract.json')
    initial=read(source/'initialization.json')
    done=read(source/'completion.json')
    cfg=read(source/'resolved_config.json')
    if initial.get('scratch_only') is not True or initial.get('checkpoint_sources') != [] or initial.get('target_contact') is not False:
        raise ValueError('CHECKPOINT_PROVENANCE_UNVERIFIED')
    if done.get('target_evaluated') is not False or done.get('epochs') != 200 or done.get('status') != 'TRAINING_COMPLETE':
        raise ValueError('CHECKPOINT_TARGET_CONTAMINATED_OR_INCOMPLETE')
    if actual.get('native_role_comparison') != 'EXACT_MATCH':
        raise ValueError('CHECKPOINT_DATA_CONTRACT_MISMATCH')
    for key in ('role_ids','source_rxs','source_days','ratios','split_seed','num_classes'):
        if actual.get(key) != contract.get(key):
            raise ValueError('CHECKPOINT_DATA_CONTRACT_MISMATCH: '+key)
    if cfg['seed'] != seed or cfg['checkpoint_selection'] != 'final_only' or cfg['a1_periodic_target_start'] != 0 or cfg['a1_final_weak_reference']:
        raise ValueError('Source version or selection mismatch')
    return actual,cfg


def main():
    p=argparse.ArgumentParser()
    p.add_argument('action',choices=['final-predict','source-features','received-features'])
    for name in ('source','contract','native-code','output'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--seed',type=int,required=True)
    p.add_argument('--capsule',type=Path)
    p.add_argument('--device',default='cpu')
    a=p.parse_args()
    sys.path.insert(0,str(a.native_code.resolve()))
    import numpy as np
    import torch
    from cvsrffi.checkpoint_loading import build_exact_ssdg_model_from_checkpoint
    from cvsrffi.identity_only_forward import identity_only_feature_forward
    torch.set_num_threads(2)
    contract,cfg=verify_source(a.source,read(a.contract),a.seed)
    if a.output.exists():
        raise FileExistsError(a.output)
    checkpoint=a.source/'final_ssdg.pth'
    payload=torch.load(checkpoint,map_location='cpu',weights_only=False)
    if payload['epoch'] != 200 or payload['args']['seed'] != a.seed or payload['args']['baseline_ckpt'] or payload['args']['teacher_ckpt'] or not payload['args']['from_scratch']:
        raise ValueError('Wrong final checkpoint or inheritance')
    device=torch.device(a.device)
    model,audit=build_exact_ssdg_model_from_checkpoint(payload,input_len=256,device=device)
    model.eval()
    def infer(x):
        result=identity_only_feature_forward(model,x,'z_id')
        if result is None:
            raise ValueError('Native identity-only forward unavailable')
        z,logits=result
        if z.ndim!=2 or z.shape[1]!=160 or logits.shape[1]!=6 or not torch.isfinite(z).all() or not torch.isfinite(logits).all():
            raise ValueError('Invalid native output')
        return torch.nn.functional.normalize(z,dim=1),logits
    with torch.inference_mode():
        infer(torch.zeros(2,2,256,device=device))
    a.output.mkdir(parents=True)
    provenance=dict(verdict='MATCHED_SOURCE_ONLY_SCRATCH',checkpoint=str(checkpoint),checkpoint_epoch=200,
                    initialization='scratch',checkpoint_inheritance=[],target_access_before_freeze=False,
                    source_role_comparison='EXACT_MATCH',classes=contract['classes'],model_seed=a.seed,
                    selection='fixed_final_epoch200',loader=audit,query_read_for_smoke=False)
    write(a.output/'startup.json',dict(argv=sys.argv,cwd=os.getcwd(),pid=os.getpid(),python=sys.executable,
          torch=torch.__version__,native_code=str(a.native_code),source_arguments=cfg,timestamp=time.time()))
    if a.action=='source-features':
        from types import SimpleNamespace
        from SSDG import train_ssdg as native
        from cvsrffi.xuc_fusion.native import role_ids_from_native
        from cvsrffi.game_tracking.data import opaque_id
        args=SimpleNamespace(**cfg)
        args.device=str(device)
        ctx=native._build_ssdg_wisig_data(args,device)
        if ctx['named_test_loaders'] or role_ids_from_native(ctx)!=contract['role_ids']:
            raise ValueError('Source-L data contract mismatch')
        loader=ctx['probe_train_loader']
        identifiers=[opaque_id(item) for item in loader.dataset.index]
        features=[];labels=[];domains=[]
        with torch.inference_mode():
            for x,y,d,*_ in loader:
                z,_=infer(x.to(device))
                features.extend(z.cpu().tolist());labels.extend(y.tolist())
                domains.extend([ctx['domain_label_map'][int(v)] for v in d.tolist()])
        if sorted(identifiers)!=contract['role_ids']['L_s'] or len(features)!=len(identifiers):
            raise ValueError('Source-L feature coverage mismatch')
        np.savez(a.output/'source_l_features.npz',identity160=np.asarray(features,dtype='float32'),
                 labels=np.asarray(labels,dtype='int64'),domains=np.asarray(domains,dtype='int64'),ids=np.asarray(identifiers))
        provenance.update(feature_role='L_s',checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest())
        write(a.output/'provenance.json',provenance)
        write(a.output/'class_registry.json',contract['classes'])
    else:
        if a.capsule is None:
            raise ValueError('Capsule required')
        manifest=read(a.capsule/'manifest.json')
        if a.action=='final-predict':
            if manifest['status']!='VALIDATED_ONCE' or manifest['classes']!=contract['classes']:
                raise ValueError('Capsule/class mismatch')
            index=np.load(a.capsule/'index.npz',allow_pickle=False)
            result={}
            with torch.inference_mode():
                for view in ('clean','satellite'):
                    iq=np.load(a.capsule/(view+'.npy'),mmap_mode='r',allow_pickle=False)
                    preds=[]
                    for start in range(0,len(iq),256):
                        _,scores=infer(torch.tensor(iq[start:start+256].tolist(),dtype=torch.float32,device=device))
                        preds.extend(scores.argmax(1).cpu().tolist())
                    result[view]=np.asarray(preds,dtype='int64')
                    print('PREDICTED '+view+' '+str(len(preds)),flush=True)
            np.savez(a.output/'predictions.npz',ids=index['ids'],**result)
            write(a.output/'predictions_complete.json',dict(status='PREDICTIONS_COMPLETE',count=len(index['ids']),truth_read=False))
        else:
            if manifest['protocol_schema']!='p2_min_v1' or manifest['phase2_data_status']!='VALIDATED_ONCE':
                raise ValueError('Unvalidated Phase2 capsule')
            data=np.load(a.capsule/'received.npz',allow_pickle=False)
            iq=data['iq'];features=[];logits=[]
            with torch.inference_mode():
                for start in range(0,len(iq),256):
                    z,s=infer(torch.tensor(iq[start:start+256].tolist(),dtype=torch.float32,device=device))
                    features.extend(z.cpu().tolist());logits.extend(s.cpu().tolist())
            np.savez(a.output/'received_features.npz',identity160=np.asarray(features,dtype='float32'),
                     logits=np.asarray(logits,dtype='float32'),ids=data['ids'])
            write(a.output/'features_complete.json',dict(status='FROZEN_FEATURES_COMPLETE',capsule_id=manifest['capsule_id'],count=len(iq),query_used_for_fitting=False))
        write(a.output/'checkpoint_provenance.json',provenance)
    print('COMPLETE '+a.action,flush=True)


if __name__=='__main__':
    main()
