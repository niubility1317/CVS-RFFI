"""Single-scene, physical-ID keyed FP32 L-input cache with asynchronous prefetch.

Only fixed input transformations are shared. No student predictions, labels from
U, validation, or target are available to the producer or a training consumer.
"""
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import json,os,time
from pathlib import Path
import numpy as np
from .plan import SCENES,SCENE_PROBABILITIES

def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def atomic(p,value):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_name(p.name+f'.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(value,allow_nan=False)+'\n',encoding='utf-8');os.replace(tmp,p)

def probability(progress):
    if not 0<=progress<=1:raise ValueError('Budget progress outside [0,1]')
    return .3 if progress<.2 else .6 if progress<.45 else .8

def key(epoch,stage,p):
    if epoch<1 or stage not in (1,2) or p not in (.3,.6,.8):raise ValueError('Unregistered cache coordinate')
    return f'stage{stage}-epoch{epoch:03d}-p{int(100*p):02d}'

def configuration(scene):
    from leo_practical.channel import Config
    return Config(fs_hz=25e6,fc_hz=2462e6,scenario=scene,processing_route='residual',
        mode='post_sync',equalization_enabled=False,reference_snr_db=30.,boundary='reflect')

def draw(sid,epoch,stage,p):
    from leo_practical.channel import stable_seed
    rng=np.random.default_rng(stable_seed(2026102301,'Full2x2',stage,epoch,str(sid)))
    active=float(rng.random())<p
    scene=int(rng.choice(3,p=SCENE_PROBABILITIES)) if active else -1
    return scene

def transform(x,sid,rx,day,epoch,stage,p):
    from leo_practical.channel import receiver_for_session,stable_seed
    from leo_practical.residual import ResidualChannel
    scene=draw(sid,epoch,stage,p)
    if scene<0:return np.asarray(x,np.float32).copy(),-1,'UNCHANGED'
    cfg=configuration(SCENES[scene]);session=f'rx{rx}:day{day}:eq1'
    receiver=receiver_for_session(cfg,2026102302,session)
    seed=stable_seed(2026102301,'Full2x2/residual',stage,epoch,str(sid),SCENES[scene])
    engine=ResidualChannel(cfg,seed=seed,receiver=receiver)
    z=np.asarray(x[0],np.float64)+1j*np.asarray(x[1],np.float64)
    value,meta=engine.process(z)
    y=np.stack((value.real,value.imag)).astype(np.float32)
    if y.shape!=x.shape or not np.isfinite(y).all():raise ValueError('Invalid residual IQ')
    if meta.get('channel_equalization_applied'):raise ValueError('Forbidden channel equalization')
    return y,scene,str(meta.get('receiver_lock_status','UNKNOWN'))

_SOURCE={}
def initialize(request):
    global _SOURCE
    folder=Path(request['source_cache'])
    with np.load(folder/'L_s_metadata.npz',allow_pickle=False) as m:meta={k:m[k].copy() for k in m.files}
    from experiments.cric_dura_comparison.source_iq import aligned_source_iq
    x,_=aligned_source_iq(folder,'L_s',meta['ids'])
    contract=read(request['source_contract'])
    if list(meta['ids'])!=contract['role_ids']['L_s']:raise ValueError('Cache L IDs differ')
    _SOURCE=dict(x=x,meta=meta)

def chunk(args):
    start,end,epoch,stage,p=args;m=_SOURCE['meta'];values=[];scenes=[];locks=[]
    for i in range(start,end):
        x,s,l=transform(_SOURCE['x'][i],str(m['ids'][i]),int(m['rx_i'][i]),int(m['day_i'][i]),epoch,stage,p)
        values.append(x);scenes.append(s);locks.append(l)
    return start,np.asarray(values,np.float32),np.asarray(scenes,np.int8),dict(Counter(locks))

def produce(request,workers=6):
    root=Path(request['cache_root'])/request['protocol_id']/key(request['epoch'],request['stage'],request['probability'])
    if (root/'complete.json').is_file():return
    root.mkdir(parents=True,exist_ok=False);started=time.monotonic()
    c=read(request['source_contract']);ids=c['role_ids']['L_s'];n=len(ids)
    np.save(root/'ids.npy',np.asarray(ids))
    x=np.lib.format.open_memmap(root/'x.npy',mode='w+',dtype=np.float32,shape=(n,2,256))
    scene=np.lib.format.open_memmap(root/'scene.npy',mode='w+',dtype=np.int8,shape=(n,));locks=Counter()
    args=[(i,min(i+128,n),request['epoch'],request['stage'],request['probability']) for i in range(0,n,128)]
    with ProcessPoolExecutor(max_workers=workers,initializer=initialize,initargs=(request,)) as pool:
        for i,a,s,l in pool.map(chunk,args,chunksize=1):x[i:i+len(a)]=a;scene[i:i+len(s)]=s;locks.update(l)
    x.flush();scene.flush()
    atomic(root/'complete.json',dict(status='L_AUGMENTATION_READY',request=request,count=n,
        scene_counts=dict(Counter(map(int,scene))),lock_counts=dict(locks),seconds=time.monotonic()-started,
        U_reads=0,V_reads=0,target_reads=0,labels_used_for_transform=0,dtype='float32',extra_normalization=False))

def service(c):
    queue=Path(c['augmentation_cache'])/'requests';queue.mkdir(parents=True,exist_ok=True)
    while not (queue.parent/'stop').exists():
        pending=sorted((p for p in queue.glob('*.json') if not p.with_suffix('.done').exists()),key=lambda p:p.stat().st_mtime)
        if not pending:time.sleep(1);continue
        request=pending[0];r=read(request)
        try:
            produce(r,c.get('augmentation_cpu_workers',6))
            request.with_suffix('.done').write_text('ready\n')
            print('AUG_CACHE_READY',r['protocol_id'],key(r['epoch'],r['stage'],r['probability']),flush=True)
        except Exception as e:
            atomic(request.with_suffix('.failed'),dict(error=f'{type(e).__name__}: {e}'));raise

class Reader:
    def __init__(self,c):
        self.c=c;self.arrays={};self.wait_seconds=0.;self.samples=0
        self.contract=read(c['source_contract']);self.ids=np.asarray(self.contract['role_ids']['L_s'])
    def request(self,epoch,stage,p):
        name=self.c['protocol_id']+'__'+key(epoch,stage,p)
        req=dict(protocol_id=self.c['protocol_id'],epoch=epoch,stage=stage,probability=p,
            source_cache=self.c['source_cache'],source_contract=self.c['source_contract'],cache_root=self.c['augmentation_cache'])
        path=Path(self.c['augmentation_cache'])/'requests'/(name+'.json')
        if not path.exists():atomic(path,req)
        return path,Path(req['cache_root'])/req['protocol_id']/key(epoch,stage,p)
    def get(self,indices,epoch,stage,progress,max_epochs):
        p=probability(progress);path,root=self.request(epoch,stage,p)
        if epoch<max_epochs:self.request(epoch+1,stage,probability(epoch/max_epochs))
        begin=time.monotonic()
        while not (root/'complete.json').exists():
            if path.with_suffix('.failed').exists():raise RuntimeError(read(path.with_suffix('.failed'))['error'])
            if time.monotonic()-begin>3600:raise TimeoutError('No augmentation cache progress within one hour')
            time.sleep(.5)
        self.wait_seconds+=time.monotonic()-begin
        if str(root) not in self.arrays:
            ids=np.load(root/'ids.npy',allow_pickle=False)
            if not np.array_equal(ids,self.ids):raise ValueError('Cache physical L order differs')
            self.arrays={str(root):(np.load(root/'x.npy',mmap_mode='r'),np.load(root/'scene.npy',mmap_mode='r'))}
        x,s=self.arrays[str(root)];idx=np.asarray(indices,dtype=np.int64)
        if idx.ndim!=1 or not len(idx) or idx.min()<0 or idx.max()>=len(self.ids):raise ValueError('Only row L indices allowed')
        self.samples+=len(idx);code=s[idx]
        return np.asarray(x[idx],np.float32),dict(applied=bool((code>=0).any()),applied_count=int((code>=0).sum()),
            scene_counts=dict(Counter(map(int,code))),scenario='practical_three_scene_mixture',probability=p)
