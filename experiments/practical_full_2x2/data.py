"""Reuse validated source roles/received IQ; build only the two new RX contracts."""
from copy import deepcopy
from pathlib import Path
import pickle
import numpy as np
from experiments.cric_dura_comparison.io import read,write
from experiments.cric_dura_comparison.source_iq import aligned_source_iq
from experiments.phase1_external_dg import data as base
from .plan import NEW_RX,TASKS

def metadata(records,source_rxs,days,role):
    keys=('rx_i','day_i','eq_i','sig_i','base_index','domain')
    out={k:[] for k in keys};out['ids']=[]
    if role!='U_s':out['y']=[]
    # base_index is a physical lookup coordinate, not an extra identity label.
    for serial,(tx,rx,day,eq,packet,sid) in enumerate(records):
        vals=(rx,day,eq,packet,serial,source_rxs.index(rx)*len(days)+days.index(day))
        for k,v in zip(keys,vals):out[k].append(v)
        out['ids'].append(sid)
        if role!='U_s':out['y'].append(tx)
    return {k:np.asarray(v,dtype=None if k=='ids' else np.int64) for k,v in out.items()}

def canonical_copy(handle,out):
    """Existing metadata ID order, with the repaired IQ binding preserved."""
    out=Path(out);out.mkdir(parents=True,exist_ok=False)
    c=deepcopy(read(handle['source_contract']))
    if 'classes' not in c:c['classes']=read(Path(handle['source_cache'])/'metadata.json')['classes']
    for role in ('L_s','U_s','V'):
        src=Path(handle['source_cache'])
        with np.load(src/(role+'_metadata.npz'),allow_pickle=False) as m:meta={k:m[k].copy() for k in m.files}
        if list(meta['ids'])!=c['role_ids'][role]:raise ValueError('Existing source role order differs')
        x,_=aligned_source_iq(src,role,meta['ids'])
        np.save(out/(role+'_x.npy'),x)
        np.savez(out/(role+'_metadata.npz'),**meta)
        ix=dict(ids=meta['ids'],rx=np.asarray([c['source_rxs'].index(int(v)) for v in meta['rx_i']],np.int64))
        if role!='U_s':ix['y']=meta['y']
        np.savez(out/(role+'_index.npz'),**ix)
    write(out/'source_contract.json',c)
    return dict(source_cache=str(out),source_contract=str(out/'source_contract.json'),classes=c['classes'],counts=c['counts'])

def new_source(data,p,out):
    out=Path(out);contract_path=base.build_source(data,p,out)
    records=base.selections(data,p);c=read(contract_path)
    for role,items in records.items():
        m=metadata(items,p['source_rxs'],p['source_days'],role)
        np.savez(out/(role+'_metadata.npz'),**m)
        if role=='U_s':
            x=np.lib.format.open_memmap(out/'U_s_x.npy',mode='w+',dtype=np.float32,shape=(len(items),2,256))
            for i,(tx,rx,day,eq,packet,_) in enumerate(items):x[i]=base.iq_record(data['data'][tx][rx][day][eq][packet])
            x.flush();del x
            np.savez(out/'U_s_index.npz',ids=m['ids'],rx=np.asarray([p['source_rxs'].index(int(v)) for v in m['rx_i']],np.int64))
    if c['counts']!=p['expected_counts']:raise ValueError('New RX role counts differ')
    return dict(source_cache=str(out),source_contract=contract_path,classes=c['classes'],counts=c['counts'])

def external_handles(h,root):
    result={};canonical=Path(h['source_cache']);c=read(h['source_contract'])
    for pl in (False,True):
        out=Path(root)/('pl' if pl else 'supervised');out.mkdir(parents=True,exist_ok=False)
        for role in (('L_s','U_s','V') if pl else ('L_s','V')):
            for suffix in ('_x.npy','_index.npz','_metadata.npz'):(out/(role+suffix)).symlink_to(canonical/(role+suffix))
        contract=dict(c,U_s_use=pl)
        if pl:contract['U_s_labels_hidden']=True
        write(out/'source_contract.json',contract)
        write(out/'metadata.json',dict(source_contract=contract,classes=c['classes'],counts=c['counts'],
            target_constructed=False,U_s_exported=pl,U_s_labels_exported=False))
        result['pl' if pl else 'supervised']=dict(h,source_cache=str(out),source_contract=str(out/'source_contract.json'))
    return result

def old_G_selector(rt,h,pid,role,out):
    c=read(h['source_contract']);allowed=set(c['role_ids']['L_s'])
    if role=='LU':allowed.update(c['role_ids']['U_s'])
    old=Path(rt['original_G_action_cache'])/'train';new=Path(rt['old_G_data'])/'ManySig'
    lookups=[{str(v):i for i,v in enumerate(np.load(root/'physical_ids.npy',allow_pickle=False))} for root in (old,new)]
    order=[]
    for sid in sorted(allowed):
        k=0 if sid in lookups[0] else 1
        if sid not in lookups[k]:raise ValueError('No existing source observation for new G contract')
        order.append((k,lookups[k][sid],sid))
    selector=Path(out)/(pid+'-'+role+'.npz');selector.parent.mkdir(parents=True,exist_ok=True)
    labeled=set(c['role_ids']['L_s'])
    np.savez(selector,ids=np.asarray([s for k,i,s in order]),kind=np.asarray([k for k,i,s in order],np.int64),
        position=np.asarray([i for k,i,s in order],np.int64),is_L=np.asarray([s in labeled for k,i,s in order],bool))
    return dict(selector=str(selector),old_action=rt['original_G_action_cache'],old_coarse=rt['original_G_coarse_cache'],
        new_cache=str(new),source_contract=h['source_contract'],roles=role)

def prepare(rt,plan):
    root=Path(rt['run_root'])/'data';root.mkdir(parents=True,exist_ok=False)
    original=read(rt['reused_source_handles']);handles={};datasets={}
    for dataset in ('ManySig','ManyRx'):
        with Path(rt['datasets'][dataset]).open('rb') as stream:datasets[dataset]=pickle.load(stream)
    for pid,p in plan['protocols'].items():
        h=new_source(datasets[p['dataset']],p,root/'source'/pid) if pid in NEW_RX else canonical_copy(original[pid],root/'source'/pid)
        c=read(h['source_contract'])
        if c['source_rxs']!=p['source_rxs'] or c['source_days']!=p['source_days'] or c['counts']!=p['expected_counts']:
            raise ValueError('Source contract differs from accepted plan: '+pid)
        handles[pid]=dict(h,external=external_handles(h,root/'external'/pid))
        print('SOURCE_READY',pid,c['counts'],flush=True)
    # The existing ManySig union already covers every physical RX/date.
    unions=rt['observation_unions'];records={d:base.inventory(datasets[d],d,list(range(12)) if d=='ManySig' else list(range(1,32,2))) for d in datasets}
    for d in datasets:
        with np.load(Path(unions[d])/'index.npz',allow_pickle=False) as ix:
            if not np.array_equal(ix['ids'],np.asarray([r[-1] for r in records[d]])):raise ValueError('Existing received-IQ union IDs differ')
    # ManyRx union historically has target odd RX only. Generate its source D0
    # diagnostic once with the same frozen observation function and seed.
    d='ManyRx';p=plan['protocols'][d]
    unseen=[r for r in base.inventory(datasets[d],d,p['source_rxs']) if r[2]==0]
    extra=root/'observations'/'ManyRx-source-D0';extra.mkdir(parents=True)
    np.savez(extra/'index.npz',ids=np.asarray([r[-1] for r in unseen]))
    x=np.lib.format.open_memmap(extra/'clean.npy',mode='w+',dtype=np.float32,shape=(len(unseen),2,256))
    for i,(tx,rx,day,eq,packet,_) in enumerate(unseen):x[i]=base.iq_record(datasets[d]['data'][tx][rx][day][eq][packet])
    x.flush();del x
    prior=read(Path(unions['ManySig'])/'manifest.json')
    for item in prior['evidence']:
        scene=item['scene'];dest=np.lib.format.open_memmap(extra/(scene+'.npy'),mode='w+',dtype=np.float32,shape=(len(unseen),2,256))
        from concurrent.futures import ProcessPoolExecutor
        tasks=[(str(extra),scene,np.arange(i,min(i+512,len(unseen))),item['config'],rt['frozen_sources']['G']) for i in range(0,len(unseen),512)]
        with ProcessPoolExecutor(max_workers=6) as pool:
            for pos,y,locks in pool.map(base.generate_chunk,tasks):dest[pos]=y
        dest.flush();del dest
        print('MANYRX_UNSEEN_DATE_VIEW',scene,len(unseen),flush=True)
    for pid,p in plan['protocols'].items():
        data=datasets[p['dataset']];union=unions[p['dataset']];rec=records[p['dataset']];tests={}
        for task,days in (('cross_rx',[1,2,3]),('cross_both',[0])):
            tests[task]=base.make_wrapper(data,dict(p,target_days=days),handles[pid]['source_contract'],union,rec,root/'views'/(pid+'-'+task))
        use_extra=p['dataset']=='ManyRx'
        tests['cross_date']=base.make_wrapper(data,p,handles[pid]['source_contract'],str(extra) if use_extra else union,
            unseen if use_extra else rec,root/'views'/(pid+'-cross_date'),diagnostic=True)
        handles[pid]['tests']={t:tests[t] for t in TASKS}
        print('TEST_HANDLES_READY',pid,flush=True)
    write(root/'handles.json',handles)
    return handles
