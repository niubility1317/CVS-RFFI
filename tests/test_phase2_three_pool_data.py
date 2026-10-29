import json
import numpy as np
import pytest
from experiments.pamf_phase2_practical.three_pool_data import assign_pools,prepare,bank,input_ready
from experiments.pamf_phase2_practical.episodic_io import load_task


def test_stratified_three_pools_are_exhaustive_disjoint_and_balanced():
    labels=[];days=[]
    for ci in range(3):
        for d in range(4):labels.extend([ci]*(69+ci+d));days.extend([d]*(69+ci+d))
    labels=np.asarray(labels);days=np.asarray(days)
    a=assign_pools(labels,days,classes=['a','b','c'],receiver='r',seed=123)
    assert np.array_equal(a,assign_pools(labels,days,classes=['a','b','c'],receiver='r',seed=123))
    b=assign_pools(labels,days,classes=['a','b','c'],receiver='r',seed=124)
    assert not np.array_equal(a,b)
    assert set(a)==set(range(3))
    for ci in range(3):
        counts=np.bincount(a[labels==ci],minlength=3);assert counts.max()-counts.min()<=1
        for d in range(4):
            counts=np.bincount(a[(labels==ci)&(days==d)],minlength=3);assert counts.max()-counts.min()<=1
    parts=[set(np.flatnonzero(a==p)) for p in range(3)]
    assert set.union(*parts)==set(range(len(labels)))
    assert all(not parts[a]&parts[b] for a in range(3) for b in range(a))


def test_partition_rejects_unknown_dates_and_wrong_pool_count():
    with pytest.raises(ValueError,match='Unknown TX/day'):
        assign_pools([0],[4],classes=['a'],receiver='r',seed=1)
    with pytest.raises(ValueError,match='partition input'):
        assign_pools([0],[0],classes=['a'],receiver='r',seed=1,pool_count=4)


def put(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value),encoding='utf-8')


def test_input_ready_waits_for_atomic_completion_not_partial_report(tmp_path):
    report=tmp_path/'worker_reports/E-r.json';report.parent.mkdir();report.write_text('{')
    assert not input_ready(tmp_path,'E','r')
    put(tmp_path/'progress/E-r.json',dict(status='GENERATING'))
    assert not input_ready(tmp_path,'E','r')
    # The producer closes its report, then publishes the atomic completion marker.
    report.write_text('{}')
    put(tmp_path/'progress/E-r.json',dict(status='VALIDATED_ONCE'))
    assert input_ready(tmp_path,'E','r')


def test_minimum_query_cannot_be_silently_relaxed(tmp_path):
    source=tmp_path/'source';classes=[f'tx{i}' for i in range(26)]
    labels=np.repeat(np.arange(26),500);days=np.tile(np.arange(4),len(labels)//4)
    folder=source/'builder_only/pool/r';folder.mkdir(parents=True)
    np.savez(folder/'records.npz',labels=labels,days=days,physical_ids=np.array([str(i) for i in range(len(labels))]))
    put(source/'builder_only/pool/manifest.json',dict(status='FULL_SHARED_RAW_POOL_READY',classes=classes,physical_records=len(labels)))
    rt=dict(input_data_root=str(source),run_root=str(tmp_path/'out'),old_classes=classes[:6],new_classes=classes[6:],
            source_receivers=['s'],target_receivers=['r'],expected_pool_records=len(labels),shots=[20],
            physical_pool_seed=42,physical_pools=['P01','P02','P03'],minimum_query_per_class=150)
    with pytest.raises(ValueError,match='fewer than minimum query'):
        prepare(rt)


def test_real_bank_repack_preserves_observations_and_only_opens_current_task(tmp_path):
    source=tmp_path/'source';out=tmp_path/'out';r='rx';classes=[f'tx{i}' for i in range(26)]
    labels=np.concatenate([np.full(4*(140+ci),ci,np.int16) for ci in range(26)])
    days=np.concatenate([np.repeat(np.arange(4,dtype=np.uint8),140+ci) for ci in range(26)])
    n=len(labels);physical=np.array([f'raw{i:06d}' for i in range(n)])
    private=source/'builder_only/pool'/r;private.mkdir(parents=True)
    np.savez(private/'records.npz',labels=labels,days=days,physical_ids=physical)
    put(source/'builder_only/pool/manifest.json',dict(status='FULL_SHARED_RAW_POOL_READY',classes=classes,physical_records=n))
    rt=dict(input_data_root=str(source),run_root=str(out),old_classes=classes[:6],new_classes=classes[6:],
            source_receivers=['src'],target_receivers=[r],expected_pool_records=n,shots=[1,5,10,20],new_counts=[0,2,5,10,20],
            physical_pool_seed=42,physical_pools=['P01','P02','P03'],minimum_query_per_class=150,code_commit='fixture',
            support_seed=44,query_order_seed=45,run_id='three-pool',input_run_id='input',dates=['d0','d1','d2','d3'])
    plans,counts,report=prepare(rt)
    assert sum(report['pool_totals'].values())==n
    for episode,seed in [('MIX01',111),('MIX02',222)]:
        perm=np.random.default_rng(seed).permutation(n);capsule=source/'capsules'/episode/r;capsule.mkdir(parents=True)
        received=np.arange(n,dtype=np.float32)[:,None,None]*np.ones((n,2,256),np.float32)+seed
        ids=np.array([f'{episode}-obs{i:06d}' for i in range(n)])
        np.save(capsule/'received.npy',received);np.save(capsule/'ids.npy',ids)
        put(capsule/'manifest.json',dict(protocol_schema='p2_min_v1',phase2_data_status='VALIDATED_ONCE',source_roles_unchanged=True,
            capsule_id='input:'+episode+':'+r,channel=dict(route='residual')))
        truth=source/'score_only'/episode;truth.mkdir(parents=True)
        np.savez(truth/(r+'.npz'),observation_ids=ids,physical_ids=physical[perm],labels=labels[perm],days=days[perm],scene_indices=np.zeros(n,np.uint8))
        put(source/'worker_reports'/(episode+'-'+r+'.json'),dict(status='VALIDATED_ONCE',episode=episode,receiver=r,observations=n,capsule=str(capsule.relative_to(source))))
        reports=bank(rt,episode,r,plans,counts)
        assert len(reports)==3 and sum(x['observations'] for x in reports)==n
        allids=[]
        for item in reports:
            dest=out/item['capsule'];destids=np.load(dest/'ids.npy');destiq=np.load(dest/'received.npy')
            lookup={str(x):i for i,x in enumerate(ids)};positions=np.array([lookup[str(x)] for x in destids])
            assert np.array_equal(destiq,received[positions]);allids.extend(destids.tolist())
            specs=[json.loads(p.read_text()) for p in (dest/'splits').glob('*.json')]
            bsplit=next(x for x in specs if x['new_count']==0 and x['k']==20)
            a=load_task(dest,bsplit['split_id'],stage='A');b=load_task(dest,bsplit['split_id'],stage='B')
            assert 'support_x' not in a and np.array_equal(a['query_ids'],b['query_ids'])
            cs=next(x for x in specs if x['new_count']==20 and x['k']==20)
            c=load_task(dest,cs['split_id'],stage='C')
            assert len(c['support_ids'])==520 and len(c['query_ids'])==len(destids)-520
            assert set(c['support_ids'])|set(c['query_ids'])==set(destids) and not set(c['support_ids'])&set(c['query_ids'])
        assert len(allids)==len(set(allids))==n and set(allids)==set(ids)
    # Pool physical membership is fixed across episodes; observations and support are episode-specific.
    for pid in rt['physical_pools']:
        p1=np.load(out/'score_only'/pid/'MIX01'/(r+'.npz'))
        p2=np.load(out/'score_only'/pid/'MIX02'/(r+'.npz'))
        assert set(p1['physical_ids'])==set(p2['physical_ids'])
        assert not set(p1['observation_ids'])&set(p2['observation_ids'])
        p1.close();p2.close()
