"""Construct new source subsets once; reuse every received-IQ observation."""
from copy import deepcopy
from pathlib import Path
import pickle
from experiments.phase1_external_dg import data as base
from .plan import make_plan, counts, TARGET, SOURCE_ORDER

read, write = base.read, base.write


def selections(data, p):
    natural = base.selections(data, dict(p, U_s_use=False, fixed_labels=None))
    if p['fixed_labels'] is None:return natural
    m=len(p['source_rxs'])
    if p['fixed_labels']!=1080 or p['fixed_validation']!=5400 or 60%m or 300%m:
        raise ValueError('Unregistered balanced fixed L/V budget')
    lq,vq=60//m,300//m
    labeled=[r for r in natural['L_s'] if r[4]<lq]
    hidden=sorted(natural['U_s']+[r for r in natural['L_s'] if r[4]>=lq])
    validation=[r for r in natural['V'] if r[4]<700+vq]
    result=dict(L_s=labeled,U_s=hidden,V=validation)
    if {k:len(v) for k,v in result.items()}!=counts(m,'fixed'):
        raise ValueError('Source cells do not match registered fixed budget')
    return result


def build_source(data,p,out):
    import numpy as np
    out=Path(out);out.mkdir(parents=True,exist_ok=False)
    records=selections(data,p);actual={k:len(v) for k,v in records.items()}
    if actual!=p['expected_counts']:raise ValueError('Physical source counts differ')
    classes=[str(v) for v in data['tx_list']]
    contract=dict(schema='phase1_sourcecount6_roles_v1',dataset='ManySig',
        source_rxs=p['source_rxs'],source_days=p['source_days'],classes=classes,num_classes=6,
        counts=actual,split_seed=392005,split_rule='per_tx_rx_day_eq_contiguous_signal_index',
        label_per_thousand=70,fixed_labels=p['fixed_labels'],fixed_validation=p['fixed_validation'],
        role_ids={k:[r[-1] for r in values] for k,values in records.items()},
        checkpoint_init='scratch_only',target_access_before_freeze=False,
        equalized=1,out_len=256,normalize=True,U_s_use=False)
    for role in ('L_s','V'):
        items=records[role]
        x=np.lib.format.open_memmap(out/(role+'_x.npy'),mode='w+',dtype=np.float32,shape=(len(items),2,256))
        for i,(tx,rx,day,eq,packet,sid) in enumerate(items):
            x[i]=base.iq_record(data['data'][tx][rx][day][eq][packet])
        x.flush();del x
        np.savez(out/(role+'_index.npz'),ids=np.asarray([r[-1] for r in items]),
            y=np.asarray([r[0] for r in items],dtype=np.int64),
            rx=np.asarray([p['source_rxs'].index(r[1]) for r in items],dtype=np.int64))
    write(out/'source_contract.json',contract)
    write(out/'metadata.json',dict(source_contract=contract,classes=classes,counts=actual,
        equalized=1,out_len=256,normalize=True,U_s_exported=False,target_constructed=False))
    return dict(source_cache=str(out),source_contract=str(out/'source_contract.json'),counts=actual)


def build_u(data,p,parent,out):
    import numpy as np
    out=Path(out);out.mkdir(parents=True,exist_ok=False)
    records=selections(data,p)['U_s'];contract=deepcopy(read(parent['source_contract']))
    if [r[-1] for r in records]!=contract['role_ids']['U_s']:
        raise ValueError('U IQ does not match the hidden source role')
    x=np.lib.format.open_memmap(out/'U_s_x.npy',mode='w+',dtype=np.float32,shape=(len(records),2,256))
    for i,(tx,rx,day,eq,packet,sid) in enumerate(records):
        x[i]=base.iq_record(data['data'][tx][rx][day][eq][packet])
    x.flush();del x
    np.savez(out/'U_s_index.npz',ids=np.asarray([r[-1] for r in records]),
        rx=np.asarray([p['source_rxs'].index(r[1]) for r in records],dtype=np.int64))
    contract.update(U_s_use=True,U_s_labels_hidden=True)
    meta=deepcopy(read(Path(parent['source_cache'])/'metadata.json'))
    meta.update(source_contract=contract,U_s_exported=True,U_s_labels_exported=False,
        role_cache_roots=dict(L_s=parent['source_cache'],V=parent['source_cache'],U_s=str(out)))
    write(out/'source_contract.json',contract);write(out/'metadata.json',meta)
    return dict(source_cache=str(out),source_contract=str(out/'source_contract.json'),counts=meta['counts'])


def prepare(rt):
    import numpy as np
    plan=make_plan();root=Path(rt['run_root']);out=root/'data';out.mkdir(exist_ok=False)
    old=read(Path(rt['parent_run_root'])/'data_handles.json')
    with Path(rt['datasets']['ManySig']).open('rb') as stream:data=pickle.load(stream)
    if (len(data['tx_list']),len(data['rx_list']))!=(6,12):raise ValueError('ManySig metadata differs')
    if [str(v) for v in data['capture_date_list']]!=['2021_03_01','2021_03_08','2021_03_15','2021_03_23']:
        raise ValueError('Date mapping differs')
    handles={}
    for pid,p in plan['protocols'].items():
        m=len(p['source_rxs'])
        if m==5 and p['fixed_labels'] is None:
            prior=old['F0_PL' if p['U_s_use'] else 'F0']
            meta=read(Path(prior['source_cache'])/'metadata.json')
            c=read(prior['source_contract'])
            if c['source_rxs']!=p['source_rxs'] or c['source_days']!=p['source_days'] or meta['counts']!=p['expected_counts']:
                raise ValueError('Five-source parent contract differs')
            h={k:deepcopy(prior[k]) for k in ('source_cache','source_contract','counts')}
        elif p['U_s_use']:
            parent=handles[pid.replace('-pl','-supervised')]
            h=build_u(data,p,parent,out/'source'/pid)
        else:h=build_source(data,p,out/'source'/pid)
        handles[pid]=h
        print('SOURCE_HANDLE',pid,h['counts'],flush=True)
    # The old suite stored all 12 receivers in one observations union. A new
    # fixed-target wrapper changes only positions; no IQ is resimulated.
    union=Path(rt['existing_observations_root'])
    records=base.inventory(data,'ManySig',list(range(12)))
    with np.load(union/'index.npz',allow_pickle=False) as ix:
        if not np.array_equal(ix['ids'],np.asarray([r[-1] for r in records])):
            raise ValueError('Existing observation positions differ')
    p=plan['protocols']['m6-natural-supervised']
    test=base.make_wrapper(data,p,handles[p['protocol_id']]['source_contract'],union,records,out/'fixed-target6')
    manifest=read(Path(test['views_root'])/'manifest.json')
    if manifest['count']!=144000 or manifest['target_rxs']!=list(TARGET):
        raise ValueError('Fixed target coverage differs')
    for h in handles.values():h['tests']=dict(main=test)
    write(root/'data_handles.json',handles)
    write(root/'data_status.json',dict(status='VALIDATED_ONCE',schema='phase1_sourcecount6_roles_v1',
        source_order=list(SOURCE_ORDER),target_rxs=list(TARGET),target_count=144000,
        observations_reused=True,U_labels_exported=False,protocol_count=len(handles)))
    return handles
