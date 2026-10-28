"""Offline builder only: raw input, channel geometry and truth never enter fitting."""
from collections import Counter
from dataclasses import asdict
import hashlib
import math
from pathlib import Path
import numpy as np
from .io import read,write,require

def identity(value):
    return hashlib.sha256(repr(value).encode('utf-8')).hexdigest()[:24]

def allocation(available,groups,support_size,query_size,seed):
    """Date-balanced disjoint pools, with capacity checked before output creation."""
    require(len(available)==4 and min(support_size,query_size)>0,'Four dates and positive pool sizes required')
    required=[0]*4;plans=[]
    for group in range(groups):
        s=[support_size//4]*4;q=[query_size//4]*4
        for i in range(support_size%4):s[(group+i)%4]+=1
        for i in range(query_size%4):q[(group+support_size%4+i)%4]+=1
        required=[a+b+c for a,b,c in zip(required,s,q)];plans.append((s,q))
    require(all(need<=len(have) for need,have in zip(required,available)),f'Insufficient physical records: require {required}, have {[len(x) for x in available]}')
    rng=np.random.default_rng(seed);orders=[[day[int(i)] for i in rng.permutation(len(day))] for day in available]
    pos=[0]*4;out=[]
    for s,q in plans:
        roles=[]
        for size in (s,q):
            values=[]
            for di,n in enumerate(size):values.extend((di,index) for index in orders[di][pos[di]:pos[di]+n]);pos[di]+=n
            roles.append(values)
        out.append(roles)
    return out

def geometry(cfg,band,rng):
    from leo_practical.channel import Geometry,slant_range,GM,RE
    def psi(e):
        v=math.radians(e);return math.acos(RE/(RE+cfg.altitude_m)*math.cos(v))-v
    lo,hi=band;c=float(rng.uniform(math.cos(psi(lo)),math.cos(psi(hi))));p=math.acos(c)
    elevation=math.degrees(math.atan2(math.cos(p)-RE/(RE+cfg.altitude_m),math.sin(p)))
    elevation=min(hi,max(lo,elevation))
    speed=math.sqrt(GM/(RE+cfg.altitude_m));rate=float(rng.choice([-1,1])*speed*math.cos(math.radians(elevation)))
    return Geometry(cfg.altitude_m,elevation,float(slant_range(cfg.altitude_m,elevation)),rate,'coverage_uniform_cospsi_reference_not_ephemeris')

def observe(raw,scene_spec,*,sid,session,pool,mixture):
    from leo_practical.channel import Config,stable_seed,normalize_input,receiver_for_session
    from leo_practical.residual import ResidualChannel
    rng=np.random.default_rng(stable_seed(pool['scene_seed'],sid,'scene-and-geometry'))
    if 'scenario' in pool:
        spec=next(s for s in scene_spec if s['id']==pool['scenario']);g=None
    else:
        spec=scene_spec[int(rng.choice(len(scene_spec),p=[s['probability'] for s in scene_spec]))];g='mixed'
    cfg=Config(fs_hz=mixture['route']['fs_hz'],scenario=spec['id'],processing_route='residual',mode='post_sync',equalization_enabled=False,
               input_processing_state='WiSig_equalized1_center256_unit_rms')
    if g is not None:g=geometry(cfg,spec['mixed_conditional_elevation_range_deg'],rng)
    z=np.asarray(raw,np.float32)
    require(z.shape==(2,256),'Unexpected target IQ shape; resizing forbidden')
    z,rms=normalize_input(z[0].astype(np.float64)+1j*z[1].astype(np.float64))
    receiver=receiver_for_session(cfg,mixture['receiver_seed'],session)
    seed=stable_seed(pool['channel_seed'],'phase2-practical-20261028',spec['id'],sid)
    y,meta=ResidualChannel(cfg,seed,receiver,geometry=g).process(z)
    return np.stack((y.real,y.imag)).astype(np.float32),dict(scenario=spec['id'],overlay_seed=seed,input_rms=float(rms),geometry=asdict(g) if g else meta.get('geometry'))

def build(rt):
    import sys
    sys.path.insert(0,str(Path(rt['phase1_release'])/'code'))
    from dataset_wisig import load_wisig_compact_pkl
    from leo_practical.channel import stable_seed
    mix=read(rt['mixture_config']);raw=load_wisig_compact_pkl(rt['manytx'])
    tx=list(map(str,raw['tx_list']));rx=list(map(str,raw['rx_list']));dates=list(map(str,raw['capture_date_list']));eq=list(raw['equalized_list']).index(1)
    classes=rt['old_classes']+rt['new_classes'];pools=rt['pools'];sp=rt['support_pool_size'];qs=rt['query_size']
    require(not set(rt['target_receivers'])&set(rt['source_receivers']),'Source/target receiver overlap')
    require(len(set(classes))==26 and not set(rt['old_classes'])&set(rt['new_classes']),'Old/new TX mismatch')
    plans={}
    for r in rt['target_receivers']:
        for c in classes:
            arrays=raw['data'][tx.index(c)][rx.index(r)]
            available=[list(range(len(d[eq]))) if d[eq] is not None else [] for d in arrays]
            plans[(r,c)]=allocation(available,len(pools),sp,qs,stable_seed(mix['physical_allocation_seed'],r,c))
    root=Path(rt['data_root']);root.mkdir(parents=True,exist_ok=False);capsule=root/'capsule';capsule.mkdir();(capsule/'splits').mkdir();(root/'score_only').mkdir()
    arrays=[];ids=[];truth=[];groups={};seen=set();counts=Counter()
    for r in rt['target_receivers']:
        for c in classes:
            native=raw['data'][tx.index(c)][rx.index(r)]
            for pindex,pool in enumerate(pools):
                for role,records in zip(('support_pool','query'),plans[(r,c)][pindex]):
                    ix=[]
                    for di,si in records:
                        sid=identity(['WiSig/ManyTx',c,r,dates[di],1,si]);require(sid not in seen,'Repeated physical IQ');seen.add(sid)
                        value,meta=observe(np.asarray(native[di][eq][si],np.float32).T,mix['scenarios'],sid=sid,session=f'rx:{r}/day:{dates[di]}',pool=pool,mixture=mix)
                        require(np.isfinite(value).all(),'Nonfinite received IQ')
                        index=len(ids);ids.append(sid);arrays.append(value);ix.append(index);counts[(pool['id'],meta['scenario'])]+=1
                        truth.append(dict(physical_id=sid,index=index,transmitter=c,receiver=r,day=dates[di],signal_index=si,pool=pool['id'],role=role,**meta))
                    groups[(r,pool['id'],c,role)]=ix
        print('BUILD_RECEIVED',r,len(ids),flush=True)
    order=np.random.default_rng(mix['physical_allocation_seed']+1).permutation(len(ids));inverse=np.empty(len(ids),dtype=np.int64);inverse[order]=np.arange(len(ids))
    np.save(capsule/'received.npy',np.asarray(arrays,np.float32)[order],allow_pickle=False);ids=np.asarray(ids)[order];np.save(capsule/'ids.npy',ids,allow_pickle=False)
    groups={k:[int(inverse[i]) for i in v] for k,v in groups.items()}
    for entry in truth:entry['index']=int(inverse[entry['index']])
    capsule_id='practical-20261028-'+rt['run_id'].rsplit('-',1)[-1]+'-'+identity([rt['pools'],sp,qs,mix['physical_allocation_seed']])
    count=0
    for r in rt['target_receivers']:
        for pool in pools:
            pid=pool['id']
            for ss in rt['support_seeds']:
                support_order={c:np.asarray(groups[(r,pid,c,'support_pool')])[np.random.default_rng(stable_seed(ss,r,pid,c)).permutation(sp)].tolist() for c in classes}
                for k in rt['shots']:
                    for n in rt['new_counts']:
                        registered=rt['old_classes']+rt['new_classes'][:n]
                        support=[i for c in registered for i in support_order[c][:k]];labels=[j for j,c in enumerate(registered) for _ in range(k)]
                        query=[i for c in registered for i in groups[(r,pid,c,'query')]]
                        query=np.asarray(query)[np.random.default_rng(stable_seed(pool['evaluation_seed'],r,n,'query-order')).permutation(len(query))].tolist()
                        split=dict(protocol_schema='p2_min_v1',phase2_data_status='VALIDATED_ONCE',capsule_id=capsule_id,receiver=r,scenario=pid,k=k,
                                   registered_classes=registered,support_indices=support,support_labels=labels,query_indices=query,support_seed=ss)
                        split['split_id']=identity(split);write(capsule/'splits'/(split['split_id']+'.json'),split);count+=1
    manifest=dict(protocol_schema='p2_min_v1',phase2_data_status='VALIDATED_ONCE',capsule_id=capsule_id,received_count=len(ids),split_count=count,
                  signal_shape=[2,256],scenarios=[p['id'] for p in pools],channel=dict(route='residual',mode='post_sync',equalization_enabled=False,fs_hz=25e6),
                  support_pool_size=sp,query_size=qs,checks=dict(unique_physical_observation=True,scenes_physically_disjoint=True,support_query_physically_disjoint=True,query_labels_absent=True,source_paths_absent=True,finite_received=True))
    write(root/'score_only/truth.json',{r['physical_id']:r for r in truth});write(capsule/'manifest.json',manifest)
    write(root/'builder_report.json',dict(status='VALIDATED_ONCE',manifest=manifest,config=rt,realized_scene_counts=[dict(pool=p,scene=s,count=n) for (p,s),n in counts.items()],no_checkpoint_access=True))
    print('DATA_COMPLETE',len(ids),count,flush=True)
