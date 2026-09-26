"""Separate final clean/residual target evaluation; CPU queue and truth-last scoring."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'code')]


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value):
    Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')


def build_data(spec):
    import numpy as np
    from dataset_wisig import load_wisig_compact_pkl, WiSigCompactDataset
    from baselines.common.practical_source import physical_id
    from leo_practical.channel import Config, stable_seed
    from leo_practical.batch import apply_leo_practical_channel_batch
    out=Path(spec['execution']['remote_run_root'])/'data'
    out.mkdir(exist_ok=False)
    capsule=out/'capsule'
    capsule.mkdir()
    raw=load_wisig_compact_pkl(spec['data']['dataset'])
    contract=read(spec['data']['contract_ref'])
    targets=spec['data']['target_receivers']
    if set(targets)&set(contract['source_rxs']):
        raise ValueError('Target/source receiver overlap')
    ds=WiSigCompactDataset(raw,out_len=256,crop_mode='center',normalize=True,equalized=1,
        rx_keep=targets,day_keep=spec['data']['target_days'],domain='rx_day')
    source_ids=set().union(*(set(v) for v in contract['role_ids'].values()))
    scenes=['practical_high','practical_mid','practical_low_urban']
    ids=[physical_id(item) for item in ds.index]
    if len(set(ids))!=len(ids) or source_ids.intersection(ids):
        raise ValueError('Physical ID leakage/duplication')
    # Shuffle storage order: numerical position must not encode truth blocks.
    order=np.random.default_rng(spec['evaluation_seed']).permutation(len(ds))
    clean=np.lib.format.open_memmap(capsule/'clean.npy',mode='w+',dtype='float32',shape=(len(ds),2,256))
    sat=np.lib.format.open_memmap(capsule/'satellite.npy',mode='w+',dtype='float32',shape=(len(ds),2,256))
    names=[]
    scene_indices=[]
    truth={}
    for start in range(0,len(ds),256):
        batch=order[start:start+256]
        records=[ds[int(i)] for i in batch]
        names_batch=[ids[int(i)] for i in batch]
        x=np.asarray([r[0].tolist() for r in records],dtype='float32')
        assign=[stable_seed(spec['evaluation_seed'],'scene',sid)%3 for sid in names_batch]
        y=np.empty_like(x)
        for j,scene in enumerate(scenes):
            selected=[i for i,v in enumerate(assign) if v==j]
            if not selected:
                continue
            channel=Config(fs_hz=25000000,scenario=scene,processing_route='residual',mode='post_sync',equalization_enabled=False,
                           input_processing_state='WiSig_equalized1_center256_unit_rms')
            received,_,_=apply_leo_practical_channel_batch(x[selected],channel,seed=spec['augmentation_seed'],
                sample_ids=[names_batch[i] for i in selected],
                session_ids=[f"rx:{records[i][3]['rx']}/day:{records[i][3]['day']}" for i in selected],
                realization_namespace='phase1/final-target/v1',receiver_seed=2027,return_meta=False)
            y[selected]=received
        if not np.isfinite(x).all() or not np.isfinite(y).all():
            raise ValueError('Nonfinite evaluation IQ')
        clean[start:start+len(batch)],sat[start:start+len(batch)]=x,y
        names.extend(names_batch)
        scene_indices.extend(assign)
        for sid, record in zip(names_batch,records):
            truth[sid]=dict(label=record[1],receiver=record[3]['rx'],day=record[3]['day'])
        if start%8192==0:
            print('[BUILD] '+str(start)+'/'+str(len(ds)),flush=True)
    clean.flush()
    sat.flush()
    np.savez(capsule/'index.npz',ids=np.asarray(names),scenes=np.asarray(scene_indices))
    write(out/'truth.json',truth)
    write(capsule/'manifest.json',dict(status='VALIDATED_ONCE',count=len(ds),classes=[str(c) for c in raw['tx_list']],
        scenes=scenes,source_target_disjoint=True,unique_ids=True,one_satellite_observation_per_id=True,
        clean_role='Phase1 paired diagnostic only; never exported to Phase2',channel='residual/post_sync/noeq',fs_hz=25000000))


def predict(spec,row):
    import numpy as np
    import torch
    from tools.run_practical_phase2_baseline import source_provenance, factory, logits, tensor
    from baselines.common.run_logging import startup_record
    torch.set_num_threads(2)
    source=Path(row['source_root'])
    expected=read(spec['data']['contract_ref'])
    contract=source_provenance(source,expected,row['epochs'])
    cfg=read(source/'resolved_config.json')
    if cfg['model_seed']!=row['seeds']['model'] or cfg['method']!=row['model_method']:
        raise ValueError('Source model/seed mismatch')
    out=Path(row['output_root'])
    out.mkdir(exist_ok=False)
    startup_record(dict(spec=spec['run_id'],row=row,source_arguments=cfg),out)
    model=factory(cfg['method'],len(contract['classes']),len(contract['source_rxs'])).eval()
    checkpoint=torch.load(source/'last.pt',map_location='cpu',weights_only=False)
    if checkpoint['epoch']!=row['epochs']:
        raise ValueError('Wrong final epoch')
    model.load_state_dict(checkpoint['model'],strict=True)
    with torch.no_grad():
        smoke=logits(model,cfg['method'],torch.zeros(2,2,256))
    if not torch.isfinite(smoke).all():
        raise ValueError('Real-checkpoint no-query smoke failed')
    write(out/'checkpoint_provenance.json',dict(verdict='MATCHED_SOURCE_ONLY_SCRATCH',checkpoint=str(source/'last.pt'),
        epoch=checkpoint['epoch'],selection='fixed_final_epoch',smoke_input='synthetic_zero_no_query'))
    capsule=Path(spec['execution']['remote_run_root'])/'data/capsule'
    manifest=read(capsule/'manifest.json')
    if manifest['classes']!=contract['classes'] or manifest['status']!='VALIDATED_ONCE':
        raise ValueError('Class map/capsule mismatch')
    index=np.load(capsule/'index.npz',allow_pickle=False)
    result={}
    for view in ('clean','satellite'):
        iq=np.load(capsule/(view+'.npy'),mmap_mode='r',allow_pickle=False)
        predictions=[]
        with torch.no_grad():
            for start in range(0,len(iq),256):
                scores=logits(model,cfg['method'],tensor(iq[start:start+256],'cpu'))
                if scores.shape[1]!=len(contract['classes']) or not torch.isfinite(scores).all():
                    raise ValueError('Invalid query scores')
                predictions.extend(scores.argmax(1).tolist())
        result[view]=np.asarray(predictions,dtype='int64')
        print('[PREDICTED] '+view+' '+str(len(predictions)),flush=True)
    np.savez(out/'predictions.npz',ids=index['ids'],**result)
    write(out/'predictions_complete.json',dict(status='PREDICTIONS_COMPLETE',count=manifest['count'],truth_read=False))


def metrics(y,p,classes):
    import numpy as np
    cm=np.zeros((classes,classes),dtype='int64')
    np.add.at(cm,(y,p),1)
    tp=cm.diagonal()
    denominator=cm.sum(0)+cm.sum(1)
    f1=np.divide(2*tp,denominator,out=np.zeros(classes,dtype=float),where=denominator!=0)
    return dict(count=len(y),accuracy=float(tp.sum()/len(y)),macro_f1=float(f1.mean()),confusion=cm.tolist())


def score(spec):
    import numpy as np
    root=Path(spec['execution']['remote_run_root'])
    if any(not (Path(r['output_root'])/'predictions_complete.json').is_file() for r in spec['rows']):
        raise ValueError('All predictions must finish before truth')
    truth=read(root/'data/truth.json')
    index=np.load(root/'data/capsule/index.npz',allow_pickle=False)
    manifest=read(root/'data/capsule/manifest.json')
    results=[]
    for row in spec['rows']:
        pred=np.load(Path(row['output_root'])/'predictions.npz',allow_pickle=False)
        if not np.array_equal(pred['ids'],index['ids']):
            raise ValueError('Prediction identity mismatch')
        ids=pred['ids'].tolist()
        y=np.asarray([truth[s]['label'] for s in ids])
        receivers=np.asarray([str(truth[s]['receiver']) for s in ids])
        for view in ['clean']+manifest['scenes']:
            mask=np.ones(len(ids),dtype=bool) if view=='clean' else index['scenes']==manifest['scenes'].index(view)
            p=pred['clean' if view=='clean' else 'satellite']
            results.append(dict(row_id=row['row_id'],view=view,receiver='ALL',**metrics(y[mask],p[mask],len(manifest['classes']))))
            for receiver in sorted(set(receivers)):
                selected=mask&(receivers==receiver)
                results.append(dict(row_id=row['row_id'],view=view,receiver=receiver,**metrics(y[selected],p[selected],len(manifest['classes']))))
    with (root/'results.json').open('x',encoding='utf-8') as f:
        json.dump(dict(status='SCORED',target_feedback_forbidden=True,results=results),f)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['launch','dispatch','build','predict','score'])
    p.add_argument('--spec',type=Path,required=True)
    p.add_argument('--row')
    p.add_argument('--commit',default='not_provided')
    a=p.parse_args()
    spec=read(a.spec)
    root=Path(spec['execution']['remote_run_root'])
    if a.action=='launch':
        root.mkdir(parents=True,exist_ok=False)
        env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',COMPARISON_RELEASE_COMMIT=a.commit)
        with (root/'dispatcher.log').open('x',encoding='utf-8') as log:
            proc=subprocess.Popen([sys.executable,__file__,'dispatch','--spec',str(a.spec.resolve()),'--commit',a.commit],
                cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        write(root/'launch.json',dict(pid=proc.pid,cwd=str(ROOT),commit=a.commit,owner=spec['execution']['launch_owner']))
        print(json.dumps(dict(pid=proc.pid,root=str(root))))
    elif a.action=='dispatch':
        def child(action,row=None):
            command=[sys.executable,__file__,action,'--spec',str(a.spec.resolve())]
            if row:
                command+=['--row',row['row_id']]
            logpath=root/(action+'.log' if row is None else row['row_id']+'.log')
            with logpath.open('x',encoding='utf-8') as log:
                subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
        write(root/'state.json',dict(status='BUILDING_DATA',completed=[]))
        child('build')
        completed=[]
        for row in spec['rows']:
            write(root/'state.json',dict(status='WAITING_SOURCE',row=row['row_id'],completed=completed))
            while not (Path(row['source_root'])/'completion.json').exists():
                upstream=read(spec['source_state'])
                if upstream[row['row_id']]['status']=='TECHNICAL_FAILURE':
                    raise RuntimeError('Source dependency failed; no automatic retry')
                time.sleep(30)
            write(root/'state.json',dict(status='PREDICTING',row=row['row_id'],completed=completed))
            child('predict',row)
            completed.append(row['row_id'])
        write(root/'state.json',dict(status='PREDICTIONS_COMPLETE',completed=completed))
        child('score')
        write(root/'state.json',dict(status='SCORED',completed=completed))
    elif a.action=='build':
        build_data(spec)
    elif a.action=='predict':
        predict(spec,next(row for row in spec['rows'] if row['row_id']==a.row))
    else:
        score(spec)


if __name__=='__main__':
    main()
