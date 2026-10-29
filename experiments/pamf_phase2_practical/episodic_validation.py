"""One data-side verification of the completed three-pool experiment matrix."""
from pathlib import Path
import time
import numpy as np
from .io import read,write,require
from .episodic_io import query_complement


def run(rt,output):
    root=Path(rt['data_root']);started=time.monotonic()
    while True:
        status=read(Path(rt['data_release_root'])/'preparation_status.json')
        if status['status']=='DATA_COMPLETE':break
        require(status['status']=='RUNNING','Input builder failed or stopped')
        require(time.monotonic()-started<rt.get('data_wait_seconds',86400),'Data preparation wait exceeded')
        time.sleep(15)
    complete=read(root/'builder_report.json');partition=read(root/'partition_report.json')
    require(complete['status']=='VALIDATED_ONCE' and complete['capsules']==483 and complete['splits']==9660,'Incomplete three-pool data')
    require(partition['pool_count']==3 and partition['min_query_at_max_K']>=150,'Wrong partition/query minimum')
    catalog=read(root/'catalog.json');require(len(catalog['capsules'])==483,'Wrong catalog')
    labels=read(root/'score_only/label_map.json');scenes=labels['scenes'];classes=rt['old_classes']+rt['new_classes']
    require(labels['classes']==classes,'Wrong class order')
    episodes={e['id']:e for e in rt['episodes']};base={};last_seed={};scene_totals={};split_count=0;observations=0;minimum=10**9
    for entry in sorted(catalog['capsules'],key=lambda x:(x['physical_pool'],x['receiver'],x['episode'])):
        pool,e,r=entry['physical_pool'],entry['episode'],entry['receiver'];cap=root/entry['capsule'];m=read(cap/'manifest.json')
        require(m['protocol_schema']=='p2_min_v1' and m['phase2_data_status']=='VALIDATED_ONCE','Invalid data handle')
        require(m['storage']=='episodic-pool-complement-v1' and m['source_roles_unchanged'],'Wrong stage reader layout')
        require(m['channel']==dict(route='residual',mode='post_sync',equalization_enabled=False,fs_hz=25e6),'Wrong practical route')
        ids=np.load(cap/'ids.npy');iq=np.load(cap/'received.npy',mmap_mode='r')
        require(iq.dtype==np.float32 and iq.shape==(len(ids),2,256),'Wrong received representation')
        with np.load(root/'score_only'/pool/e/(r+'.npz')) as z:
            truth={k:z[k] for k in ('physical_ids','observation_ids','labels','days','scene_indices','overlay_seeds','elevation_deg')}
        require(np.array_equal(ids,truth['observation_ids']) and len(set(ids.tolist()))==len(ids),'Wrong observation IDs')
        order=np.argsort(truth['physical_ids']);physical=truth['physical_ids'][order];seeds=truth['overlay_seeds'][order]
        key=pool,r
        if key in base:
            require(np.array_equal(base[key],physical),'Physical pool membership changed across repeats')
            require((last_seed[key]!=seeds).all(),'Channel seed reused across episodes')
        else:base[key]=physical.copy()
        last_seed[key]=seeds.copy()
        indices=truth['scene_indices'];angles=truth['elevation_deg'];require(np.isfinite(angles).all(),'Missing actual elevations')
        if 'scenario' in episodes[e]:
            require((indices==scenes.index(episodes[e]['scenario'])).all(),'Pure scenario contains another scene')
        for si,name in enumerate(scenes):
            selected=indices==si
            if not selected.any():continue
            low,high=(20,45) if 'scenario' in episodes[e] and name=='practical_mid_urban' else ((10,30) if 'low' in name else ((30,45) if 'mid' in name else (45,80)))
            require(((angles[selected]>=low-1e-8)&(angles[selected]<=high+1e-8)).all(),'Actual angle outside declared scenario')
            scene_totals[name]=scene_totals.get(name,0)+int(selected.sum())
        specs=[read(p) for p in (cap/'splits').glob('*.json')];require(len(specs)==20,'Missing K/N split')
        require({(s['k'],s['new_count']) for s in specs}=={(k,n) for k in rt['shots'] for n in rt['new_counts']},'Wrong K/N matrix')
        support_by={}
        for s in specs:
            k,n=s['k'],s['new_count'];support=np.asarray(s['support_indices']);registered=np.load(cap/'stage_rows'/f'N{n:02d}.npy');query=query_complement(registered,support)
            require(s['capsule_id']==m['capsule_id'] and s['registered_classes']==classes[:6+n],'Wrong split/classes')
            sy=np.load(cap/'support_labels'/(s['split_id']+'.npy'))
            require(len(support)==k*(6+n) and np.array_equal(sy,truth['labels'][support]),'Wrong support labels')
            require(np.array_equal(sy,np.repeat(np.arange(6+n),k)),'Wrong classwise K support')
            require(len(query)==s['query_count'] and set(registered.tolist())==set(np.flatnonzero(truth['labels']<6+n).tolist()),'Query not full registered complement')
            qcounts=np.bincount(truth['labels'][query],minlength=26)[:6+n];minimum=min(minimum,int(qcounts.min()));require((qcounts>=150).all(),'Query below150')
            support_by[k,n]=support.reshape(6+n,k)
        for k in rt['shots']:
            for n in rt['new_counts']:
                require(np.array_equal(support_by[k,n][:6],support_by[k,0]),'Old support changes B-to-C')
                require(np.array_equal(support_by[k,n],support_by[max(rt['shots']),20][:6+n,:k]),'K/N support not nested')
        split_count+=len(specs);observations+=len(ids)
        print('VALIDATED',pool,e,r,split_count,flush=True)
    physical_by_pool={}
    for (pool,r),values in base.items():physical_by_pool.setdefault(pool,set()).update(values.tolist())
    pools=list(physical_by_pool)
    require(len(pools)==3 and all(not physical_by_pool[a]&physical_by_pool[b] for i,a in enumerate(pools) for b in pools[:i]),'Three pools overlap')
    require(sum(map(len,physical_by_pool.values()))==554823 and observations==12760929 and split_count==9660,'Wrong final totals')
    write(output,dict(status='PASS',capsules=483,splits=split_count,physical_records=554823,observations=observations,
          physical_pools_disjoint=True,full_query_complement=True,nested_support=True,old_support_B_C_same=True,
          channel_seeds_distinct_per_repeat=True,actual_scene_and_elevation_checked=True,minimum_query=minimum,
          scene_counts=scene_totals,model_reads=0,predictions=0,seconds=time.monotonic()-started))
