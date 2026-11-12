"""One clean-only SCALR task; no queue, truth access or score execution."""
import argparse
from pathlib import Path
import time

import numpy as np
import torch

from .episodic_io import load_task
from .episodic_experiment import split_rows
from .io import read, require, write
from . import r42_suite_worker as common


def source_smoke(rt,reader,engine,device):
    from .clean_source_smoke import source_samples
    source_iq=source_samples(rt,2)
    _,source_logits=common.baseline_features(reader,source_iq,device,2,rt.get('fast_dlpack_transfer',False))
    require(source_logits.shape==(2,6) and np.isfinite(source_logits).all(),'PAMF source checkpoint smoke failed')
    restored=common._restore(engine,source_iq,device,2)
    require(restored.shape==source_iq.shape and np.isfinite(restored).all(),'DURA source checkpoint smoke failed')


def run(rt, job):
    require(rt.get('clean_root') and rt.get('data_root'), 'Clean and parent data roots required')
    require(rt.get('arm') == 'R42' and rt.get('kernel_method') == 'adaptive_residual_ensemble', 'SCALR R42 required')
    require(rt['model_seeds'] == [job['seed']], 'Wrong source seed')
    require(not job.get('reuse_file'), 'Do not reuse satellite predictions for clean inputs')
    points = sorted(set(tuple(map(int, p)) for p in job['points']))
    require(points and all(k in (1,5,10,20) and n in (2,5,10,20) for k,n in points), 'Invalid workpoints')
    cap=Path(rt['data_root'])/job['capsule']; manifest=read(cap/'manifest.json')
    specs={(s['k'],s['new_count']):s for s in (read(p) for p in (cap/'splits').glob('*.json'))}
    require(all((k,0) in specs and (k,n) in specs for k,n in points), 'Missing split')
    old_names=specs[points[0][0],0]['registered_classes']
    require(len(old_names)==6 and all(specs[k,n]['registered_classes'][:6]==old_names for k,n in points), 'Wrong old classes')
    columns=common.head_columns(read(rt['source_contract'])['classes'],old_names)
    source_root=Path(rt.get('scalr_source_root',rt['run_root']))
    rule=read(source_root/'source_selection.json')['K1_rule']
    root=Path(rt['run_root'])/'predictions'/manifest['receiver']
    root.mkdir(parents=True,exist_ok=False)
    device=rt.get('device','cuda:0'); batch=int(rt.get('batch_size',128)); decision_batch=int(rt.get('decision_batch',512))
    require(batch>0 and decision_batch>0, 'Invalid batch size')
    reader,engine=common.initialize(rt)
    before={k:v.detach().clone() for k,v in reader.state_dict().items()}
    g_before={k:v.detach().clone() for k,v in engine.pipeline.model.state_dict().items()}
    source_smoke(rt,reader,engine,device)
    bridge=common.AdaptiveResidualBridge.load(source_root/'source_orbit_cca.npz')
    fast=rt.get('fast_dlpack_transfer',False); rows=[]; started=time.perf_counter()

    def task_at(spec,stage):
        result=load_task(cap,spec['split_id'],stage=stage,clean_root=rt['clean_root'],task_root=rt['data_root'],
                         restored_root=rt.get('restored_root'))
        require(result['input_condition']=='clean_ground_pre_satellite_normalized','Wrong input')
        return result

    def save(stage,spec,path,query,pred,restricted=None):
        name=f"SCALR-{stage}-K{spec['k']}-N{spec['new_count']}-{path}.npz"
        payload=dict(query_indices=query,predicted_indices=np.asarray(pred,np.int16))
        if restricted is not None: payload['old_restricted_indices']=np.asarray(restricted,np.int16)
        np.savez_compressed(root/name,**payload)
        rows.append(dict(arm='R42',stage=stage,k=spec['k'],new_count=spec['new_count'],input_path=path,
                         input_condition='clean_ground_pre_satellite_normalized',split_id=spec['split_id'],
                         registered_classes=spec['registered_classes'],query_count=len(query),prediction=name))

    def inputs(iq,path,step,cached=None):
        if path and cached is not None:return cached
        return common._restore(engine,iq,device,step,fast_transfer=True) if path and fast else (
               common._restore(engine,iq,device,step) if path else iq)

    for k in sorted({k for k,_ in points}):
        spec=specs[k,0]; task=task_at(spec,'A'); query=split_rows(cap,spec)
        require(len(query)==len(task['query_x']),'A0 query alignment mismatch')
        for qp in (0,1):
            iq=inputs(task['query_x'],qp,batch,task.get('restored_query_x'))
            _,logits=common.baseline_features(reader,iq,device,batch,fast)
            pred,restricted=common._decisions(logits[:,columns],spec['registered_classes'])
            save('A0',spec,f'Q{qp}',query,pred,restricted)

    work=[('B',specs[k,0]) for k in sorted({k for k,_ in points})]
    work += [('C',specs[k,n]) for k,n in points]
    for stage,spec in work:
        task=task_at(spec,stage);query=split_rows(cap,spec)
        require(len(query)==len(task['query_x']),'Query alignment mismatch')
        for path in (0,1):
            support=inputs(task['support_x'],path,spec['k'],task.get('restored_support_x'))
            features,_=common.orbit_features(reader,support,device,spec['k'],fast)
            mapped=bridge.transform(features,restored=bool(path),old_columns=columns)
            state=common.fit_r42(mapped,task['support_y'],task['support_ids'],fixed=rule)
            iq=inputs(task['query_x'],path,batch,task.get('restored_query_x'))
            qfeatures,_=common.orbit_features(reader,iq,device,batch,fast)
            qfeatures=bridge.transform(qfeatures,restored=bool(path),old_columns=columns)
            scores=[]
            for start in range(0,len(iq),decision_batch):
                value,_=state.evaluate(common.subset(qfeatures,np.arange(start,min(start+decision_batch,len(iq)))))
                scores.append(value)
            pred,restricted=common._decisions(np.concatenate(scores),spec['registered_classes'])
            save(stage,spec,f'S{path}Q{path}',query,pred,restricted if spec['new_count'] else None)
    require(all(torch.equal(before[k],v) for k,v in reader.state_dict().items()), 'Frozen F changed')
    require(all(torch.equal(g_before[k],v) for k,v in engine.pipeline.model.state_dict().items()), 'Frozen G changed')
    require(len(rows)==2*len({k for k,_ in points})+2*len(work),'Wrong row count')
    write(root/'predictions.json',dict(capsule=job['capsule'],capsule_id=manifest['capsule_id'],model_seed=job['seed'],rows=rows))
    write(root/'complete.json',dict(status='PREDICTIONS_COMPLETE',capsule_id=manifest['capsule_id'],model_seed=job['seed'],
          rows=len(rows),truth_read=False,query_fit=False,F_updates=0,G_updates=0,paths=['S0Q0','S1Q1'],
          seconds=time.perf_counter()-started))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--runtime',required=True);p.add_argument('--job',required=True)
    args=p.parse_args();run(read(args.runtime),read(args.job))


if __name__=='__main__':main()
