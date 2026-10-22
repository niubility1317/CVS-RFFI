"""Frozen external default heads on shared, same-contract DURA received IQ."""
from pathlib import Path
import sys
import time
from .io import read, write, tensor
VIEWS=('clean','practical_high','practical_mid','practical_low_suburban',
       'practical_high_urban','practical_mid_urban','practical_low_urban')


def reject_truth(value):
    if isinstance(value,dict):
        if set(value)&{'truth','target_truth','labels','label'}:raise ValueError('Predictor rejects truth')
        for v in value.values():reject_truth(v)
    elif isinstance(value,list):
        for v in value:reject_truth(v)


def tasks(c):
    return ['main','source_unseen_date'] if c['protocol_id']=='chrono' else ['main']


def handle(root, classes):
    from .evaluation import handle as existing_handle
    return existing_handle(root,classes)

CONTROL = '20261023-phase1-external-DURA-control-r01'
SUP = '20261022-phase1-DURA-external-Lonly-readouts-m24-r01'
OTHER = '20261023-phase1-external-DURA-readouts-m492-r01'
CACHE = '20261023-phase1-external-DURA-IQ-cache-m34-r01'
ANALYSIS = '20261023-phase1-external-DURA-analysis-m516-r01'
RECOVERY = '20261023-phase1-PAMF-core-source-diagnostic-recovery-m2-r01'
CORE_RECOVERY = '20261023-phase1-PAMF-core-factorial-recovery-r01'
FAMILIES = ('riei', 'drift', 'ibrff', 'sdg')
READOUTS = ('raw', 'DURA-L', 'DURA-LU')
SEEDS = (2026092701, 2026092702, 2026092703)


def validate_rows(rows):
    """The already-trained scope; no new experiments or target-based selection."""
    from collections import Counter
    counts = Counter((r['protocol_id'], r['training_regime'], r['family'], r['variant']) for r in rows)
    expected = {}
    for pid in ('F0','F1','F2','chrono','ManyRx','label01','label05'):
        for regime in (('supervised','pl') if pid=='F0' else ('supervised',)):
            for family in (('ibrff',) if pid.startswith('label') else FAMILIES):
                for variant in ('ce','full'):expected[pid,regime,family,variant] = 3
    for m in (1,2,3,4,6):
        for budget in ('natural','fixed'):
            for regime in (('supervised','pl') if budget=='natural' else ('supervised',)):
                for family in FAMILIES:
                    for variant in ('ce','full'):expected['m%d-%s'%(m,budget),regime,family,variant] = 3
    if counts != expected or len(rows)!=516:
        raise ValueError('Expected exactly the516 previously trained four-method models')
    keys=set()
    seeds={}
    for r in rows:
        key=(r['run_id'],r['row_id'])
        if key in keys or r['model_seed'] not in SEEDS or r['U_s_use']!=(r['training_regime']=='pl'):
            raise ValueError('Duplicate model, missing seed, or widened source permissions')
        keys.add(key)
        group=(r['protocol_id'],r['training_regime'],r['family'],r['variant'])
        seeds.setdefault(group,set()).add(r['model_seed'])
    if any(s!=set(SEEDS) for s in seeds.values()):raise ValueError('Incomplete or duplicated seed within external method cell')
    return rows


def compatible_sources(external, restoration, classes):
    if any(external[k]!=restoration[k] for k in ('role_ids','counts','source_rxs','source_days')) or classes!=restoration['classes']:
        raise ValueError('CHECKPOINT_DATA_CONTRACT_MISMATCH')


def activate_external(c):
    """Use the original immutable external model/recipe, including author code."""
    import importlib
    package=importlib.import_module('experiments')
    for path in c['external_releases']:
        p=str(Path(path)/'experiments')
        if p not in package.__path__:package.__path__.append(p)
    code=str(Path(c['external_releases'][0])/'code')
    if code not in sys.path:sys.path.insert(0,code)
    vendor=str(Path(__file__).resolve().parent/'vendor')
    if vendor not in sys.path:sys.path.insert(0,vendor)
    # G's bootstrap may have already imported a different baselines package.
    baselines=importlib.import_module('baselines')
    path=str(Path(c['external_releases'][0])/'baselines')
    baselines.__path__=[path]+[p for p in baselines.__path__ if p!=path]


def load_external(c, device):
    activate_external(c)
    from experiments.phase1_external_dg_aug import source
    if c['source_suite']=='sourcecount360':
        from experiments.phase1_dg_sourcecount6.runtime import bind
        bind()
    else:
        from experiments.phase1_external_dg_aug.plan import make_plan
        source._PLAN=make_plan();source._ROWS={r['row_id']:r for r in source._PLAN['rows']}
    expected=read(c['external_source_contract'])
    compatible_sources(expected,read(c['source_contract']),c['classes'])
    model,actual=source.frozen_model(c['source_output'],expected,device)
    if (actual['row_id'],actual['model_seed'],actual['family'],actual['variant'],actual['training_regime']) != (
        c['source_row_id'],c['model_seed'],c['family'],c['variant'],c['training_regime']):
        raise ValueError('Frozen external source model differs')
    if actual['U_s_use']!=(c['training_regime']=='pl'):
        raise ValueError('Frozen external U permissions differ')
    return model.eval().requires_grad_(False),actual


def load_raw(c, task, ids):
    import numpy as np
    root=Path(c['raw_predictions_from']);done=read(root/'complete.json')
    if (done.get('status')!='ALL_PREDICTIONS_COMPLETE' or done.get('row_id')!=c['source_row_id']
            or done.get('truth_read') is not False or set(done['tests'])!=set(tasks(c))):
        raise ValueError('Frozen external raw output belongs to another row/task')
    with np.load(root/task/'predictions.npz',allow_pickle=False) as p:
        if not np.array_equal(p['ids'],ids) or set(p.files)!=set(VIEWS)|{'ids'}:
            raise ValueError('Frozen raw ID order/view coverage differs')
        values={v:p[v].copy() for v in VIEWS}
    for p in values.values():
        if p.shape!=ids.shape or p.dtype.kind not in 'iu' or p.min()<0 or p.max()>=len(c['classes']):
            raise ValueError('Invalid frozen raw class decisions')
    return values


def cache_preflight(c, name, ids, task):
    import numpy as np
    cc=read(c['cache_configs'][name]);root=Path(cc['output_root']);done=read(root/'complete.json')
    reject_truth(cc)
    if (cc['protocol_id']!=c['protocol_id'] or cc['readout']!=name or cc['G']['roles']!=name[5:]
            or cc['source_contract']!=c['source_contract'] or done.get('status')!='DURA_IQ_CACHE_COMPLETE'
            or done.get('row_id')!=cc['row_id'] or done.get('run_id')!=CACHE
            or done.get('truth_read') is not False or done.get('F_reads')!=0
            or done.get('tests')!=tasks(c) or done.get('views')!=list(VIEWS)):
        raise ValueError('Frozen restored-IQ role/contract/completion differs')
    with np.load(root/task/'index.npz',allow_pickle=False) as p:
        if not np.array_equal(p['ids'],ids):raise ValueError('Restored-IQ physical IDs differ')
    return root/task


def fp32_copy(value):
    """Exact dense FP32 CPU copy, also when Torch's NumPy bridge is unavailable."""
    import ctypes
    import numpy as np
    import torch
    if value.dtype!=torch.float32:raise ValueError('DURA cache must remain FP32')
    host=value.cpu().contiguous()
    buffer=ctypes.string_at(host.data_ptr(),host.numel()*host.element_size())
    return np.frombuffer(buffer,dtype=np.float32).reshape(tuple(host.shape))


def cache(c, device='cuda:0'):
    import numpy as np
    import torch
    from .frozen_inference import load_G
    from .f_data import SourceDataset
    from experiments.pamf_stage2_inference.engine import InferenceEngine
    reject_truth(c)
    if c['G']['roles']!=c['readout'][5:] or c['readout'] not in READOUTS[1:]:
        raise ValueError('Wrong DURA role')
    contract=read(c['source_contract']);model=load_G(c['G'],contract,device)
    call=InferenceEngine(model,reuse=True,graph=True,native=False)
    out=Path(c['output_root']);out.mkdir(parents=True,exist_ok=False)
    # Match the locked route to the eager teacher before any target IQ is read.
    source=SourceDataset(c['source_cache'],'L_s',contract).x[:2].to(device)
    with torch.inference_mode():
        reference=model(source);optimized=call(source)[0]
        if not torch.allclose(reference,optimized,atol=1e-5,rtol=1e-5):
            raise ValueError('Source graph/native DURA IQ differ')
    write(out/'source_execution_check.json',dict(status='PASS',source_samples=2,target_reads=0,atol=1e-5,rtol=1e-5))
    tic=time.monotonic();counts={}
    for task in tasks(c):
        manifest,ids,positions=handle(c['tests'][task]['views_root'],contract['classes'])
        folder=out/task;folder.mkdir();np.savez(folder/'index.npz',ids=ids)
        for view in VIEWS:
            src=manifest['clean_ref'] if view=='clean' else str(Path(manifest.get('observations_root',c['tests'][task]['views_root']))/(view+'.npy'))
            data=np.load(src,mmap_mode='r',allow_pickle=False)
            if data.shape[1:]!=(2,256) or (positions is None and len(data)!=len(ids)):
                raise ValueError('Received-IQ shape differs')
            if positions is not None and (positions.min()<0 or positions.max()>=len(data)):
                raise ValueError('Invalid received-IQ positions')
            result=np.lib.format.open_memmap(folder/(view+'.npy'),mode='w+',dtype=np.float32,shape=(len(ids),2,256))
            with torch.inference_mode():
                for start in range(0,len(ids),128):
                    ix=slice(start,start+128) if positions is None else positions[start:start+128]
                    x=tensor(np.ascontiguousarray(data[ix],dtype=np.float32),device)
                    restored=call(x)[0]
                    # Exact FP32 bytes without the unavailable Torch2.1/NumPy2 bridge.
                    result[start:start+len(x)]=fp32_copy(restored)
            result.flush();del result
            write(out/'progress.json',dict(task=task,view=view,count=len(ids),elapsed=time.monotonic()-tic),exclusive=False)
            print('EXTERNAL_DURA_CACHE',c['row_id'],task,view,len(ids),flush=True)
        counts[task]=len(ids)
    write(out/'complete.json',dict(status='DURA_IQ_CACHE_COMPLETE',run_id=CACHE,row_id=c['row_id'],
        protocol_id=c['protocol_id'],readout=c['readout'],G=c['G'],tests=tasks(c),views=list(VIEWS),counts=counts,
        dtype='float32',shape_per_packet=[2,256],extra_normalization=False,truth_read=False,F_reads=0,
        model_updates=0,seconds=time.monotonic()-tic))


def predict(c, device='cuda:0'):
    import numpy as np
    import torch
    reject_truth(c)
    if list(c['readouts'])!=list(READOUTS):raise ValueError('All raw/L/LU readouts required')
    net,actual=load_external(c,device)
    from experiments.phase1_external_dg_aug.models import identity
    out=Path(c['output_root']);out.mkdir(parents=True,exist_ok=False);completed={};tic=time.monotonic()
    for task in tasks(c):
        _,ids,_=handle(c['tests'][task]['views_root'],c['classes'])
        raw=load_raw(c,task,ids)
        cached={n:cache_preflight(c,n,ids,task) for n in READOUTS[1:]}
        values={'raw__'+v:raw[v] for v in VIEWS}
        for view in VIEWS:
            for name in READOUTS[1:]:
                data=np.load(cached[name]/(view+'.npy'),mmap_mode='r',allow_pickle=False)
                if data.dtype!=np.float32 or data.shape!=(len(ids),2,256):raise ValueError('Restored IQ schema differs')
                decisions=[]
                with torch.inference_mode():
                    for start in range(0,len(ids),128):
                        x=tensor(np.ascontiguousarray(data[start:start+128]),device)
                        logits=identity(net,actual,x)
                        if logits.shape!=(len(x),len(c['classes'])) or not bool(torch.isfinite(logits).all()):
                            raise FloatingPointError('Invalid original default-head logits')
                        decisions.extend(logits.argmax(1).cpu().tolist())
                values[name+'__'+view]=np.asarray(decisions,dtype=np.int64)
            print('EXTERNAL_DURA_PREDICT',c['row_id'],task,view,len(ids),flush=True)
        folder=out/task;folder.mkdir();np.savez(folder/'predictions.npz',ids=ids,**values)
        completed[task]=dict(count=len(ids),views=list(VIEWS),readouts=list(READOUTS))
    write(out/'complete.json',dict(status='ALL_PREDICTIONS_COMPLETE',run_id=c['run_id'],row_id=c['row_id'],
        tests=completed,views=list(VIEWS),readouts=list(READOUTS),truth_read=False,query_fit=False,
        new_F_training=False,new_G_training=False,head_fit_updates=0,raw_reused=True,seconds=time.monotonic()-tic))


def smoke(rt, output):
    """Actual checkpoints and only L IQ; original native logits must agree."""
    import numpy as np
    import torch
    from .bootstrap import activate,numerical_policy
    activate(rt);numerical_policy()
    from .frozen_inference import load_G
    from experiments.pamf_stage2_inference.engine import InferenceEngine
    m=rt['external_manifest'];records=[]
    for family in FAMILIES:
        item=next(r for r in m['rows'] if r['protocol_id']=='F0' and r['family']==family
                  and r['variant']=='full' and r['training_regime']=='supervised' and r['model_seed']==SEEDS[0])
        c=model_config(item,m['G_templates']['F0'],rt)
        net,actual=load_external(c,'cpu')
        from experiments.phase1_external_dg_aug.models import identity
        # The external loader retains its original input storage/selection. This
        # smoke tests adapter/native logits, not identity labels or trained quality.
        x=tensor(np.load(Path(item['source_cache'])/'L_s_x.npy',mmap_mode='r')[:3].copy(),'cpu')
        with torch.inference_mode():
            adapted=identity(net,actual,x)
            if family=='riei':native=net.ec(net.fed(x)[:,:net.emitter_feature_dim])
            elif family=='drift':native=net.tx_classifier(net.encoder(x)[:,:net.split_dim])
            else:native=net(x)[1]
            if not torch.equal(adapted,native) or not torch.isfinite(adapted).all():
                raise ValueError('External adapter changed native head logits')
        records.append(dict(family=family,checkpoint_loaded=True,native_logits_exact=True,source_samples=3))
    template=m['G_templates']['F0'];contract=read(template['source_contract'])
    from .f_data import SourceDataset
    x=SourceDataset(template['source_cache'],'L_s',contract).x[:2]
    checks=[]
    for name in READOUTS[1:]:
        g=load_G(template['readouts'][name],contract,'cpu')
        with torch.inference_mode():
            a=InferenceEngine(g,reuse=True,graph=False,native=False)(x)[0];b=g(x)
            if not torch.allclose(a,b,atol=1e-5,rtol=1e-5):raise ValueError('DURA source execution mismatch')
        checks.append(dict(readout=name,source_samples=2,source_route_agrees=True))
    # Reproduce the NumPy2/Torch2.1 failure's exact source selection/diagnostic path.
    from .frozen_inference import load_original_T1
    from .source_physics import run as physics
    reader=load_original_T1(SEEDS[0],contract,'cpu')
    physics(reader,template,Path(output).with_name('repaired_source_physics_smoke.json'),'cpu')
    write(output,dict(status='PASS',external_checkpoints=records,G_checkpoints=checks,
        source_physics_fixed=True,target_reads=0,U_IQ_reads=0,optimizer_updates=0,head_fit_updates=0))


def model_config(item, template, rt):
    return dict(protocol_id=item['protocol_id'],model_seed=item['model_seed'],source_suite=item['suite'],
        source_run_id=item['run_id'],source_row_id=item['row_id'],source_output=item['source_output'],
        external_source_contract=item['source_contract'],source_contract=template['source_contract'],
        classes=item['classes'],family=item['family'],variant=item['variant'],training_regime=item['training_regime'],
        source_days=template['source_days'],raw_predictions_from=item['prediction_output'],
        readouts=list(READOUTS),external_releases=rt['external_releases'],
        frozen_sources=rt['frozen_sources'],code_commit=rt['code_commit'])
