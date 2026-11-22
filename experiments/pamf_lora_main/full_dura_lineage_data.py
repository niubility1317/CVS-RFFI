"""Bind the original WiSig DURA curriculum to aligned LoRa source windows."""
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
from pathlib import Path

import numpy as np

from experiments.cric_dura_comparison.io import read, write
from .data import FC_HZ, FS_HZ, SOURCE, VIEWS
from .full_packet_data import COUNTS
from .full_restore_draw import RUN as DRAW_RUN, load_draw
from .full_supervised import handles


ROLES=('L_s','U_s')
DRAWS=3
LATENCY=8
L_COUNT=COUNTS['L_s']
COUNT=L_COUNT+COUNTS['U_s']


@lru_cache(maxsize=None)
def receiver(scene,rx):
    from leo_practical.channel import Config, receiver_for_session
    cfg=Config(fs_hz=FS_HZ,fc_hz=FC_HZ,scenario=scene,processing_route='residual',
               mode='post_sync',equalization_enabled=False,reference_snr_db=30.,
               boundary='reflect',fractional_delay_half_length=LATENCY)
    return cfg,receiver_for_session(cfg,2026112209,f'LoRa-source/{rx}')


def as_complex(a):
    return a[0].astype(np.float64)+1j*a[1].astype(np.float64)


def in_observed_phase(clean,received,coefficients=None):
    """Choose the observed phase while preserving the clean IQ amplitude and shape."""
    x=clean[:,0]+1j*clean[:,1]
    y=received[:,0]+1j*received[:,1]
    cross=(y*x.conj()).sum(-1,keepdims=True)
    move=(np.abs(cross[:,0])>1e-8)&np.any(clean!=received,axis=(1,2))
    phase=np.exp(1j*np.angle(cross[move]))
    ref=x[move]*phase
    target=clean.copy()
    target[move]=np.stack((ref.real,ref.imag),axis=1)
    if coefficients is None:
        return target
    # The rotated reference must produce the same physical observation.
    q=coefficients[move]
    main=(q[:,0]+1j*q[:,1])*phase.conj()
    image=(q[:,2]+1j*q[:,3])*phase
    output=coefficients.copy()
    output[move]=np.stack((main.real,main.imag,image.real,image.imag),axis=1)
    return target,output


def aligned_teacher(item):
    """Generate an exact known-action residual proxy at the zero-delay target origin."""
    from experiments.pamf_action_restore.data import teacher
    from leo_practical.channel import stable_seed
    from leo_practical.equalization import linear_filter
    clean,scene,rx,identity,draw=item
    cfg,hardware=receiver(scene,rx)
    seed=stable_seed(2026112209,'lora-aligned-action',identity,draw,scene)
    action,_=teacher(clean,cfg,seed,hardware)
    kernel=as_complex(action['kernel'])
    m=as_complex(action['coefficients'][:2])
    q=as_complex(action['coefficients'][2:])
    x=as_complex(clean)
    prior=linear_filter(x,kernel,'reflect')[0]
    noise=as_complex(action['received'])-m*prior-q*prior.conj()
    kept=kernel[LATENCY:]
    norm=np.linalg.norm(kept)
    energy=np.linalg.norm(kernel)
    if min(norm,energy)<=1e-12:
        raise ValueError('LoRa causal action teacher has no signal support')
    omitted=float(1-(norm/energy)**2)
    aligned_kernel=np.pad(kept/norm,(0,LATENCY))

    def after_delay(a):
        return np.pad(a,(0,LATENCY),mode='reflect')[LATENCY:]

    main=norm*after_delay(m)
    image=norm*after_delay(q)
    observed=linear_filter(x,aligned_kernel,'reflect')[0]
    observed=main*observed+image*observed.conj()+after_delay(noise)
    variance=float(.5*np.mean(np.abs(noise)**2))
    variance=np.clip(variance,1e-8,1.)
    arrays=(np.stack((observed.real,observed.imag)).astype(np.float32),
            np.stack((aligned_kernel.real,aligned_kernel.imag)).astype(np.float32),
            np.stack((main.real,main.imag,image.real,image.imag)).astype(np.float32),
            np.float32(variance))
    if any(not np.isfinite(a).all() for a in arrays):
        raise ValueError('Nonfinite exact LoRa training action components')
    return arrays,omitted


def identity_and_rx(source,contract):
    names=[]
    receivers=[]
    for role in ROLES:
        with np.load(source/f'{role}_metadata.npz',allow_pickle=False) as a:
            if a['ids'].tolist()!=contract['role_ids'][role] or ('y' in a.files)!=(role=='L_s'):
                raise ValueError('LoRa DURA source role changed')
            names.extend(a['ids'].tolist())
            receivers.extend(np.asarray(SOURCE)[a['rx_i']].tolist())
    if len(names)!=COUNT or len(set(names))!=COUNT:
        raise ValueError('LoRa source action identities must be unique physical packets')
    return names,receivers


def source_iq(cache):
    l=np.load(cache/'L_s.npy',allow_pickle=False,mmap_mode='r')
    u=np.load(cache/'U_s.npy',allow_pickle=False,mmap_mode='r')
    p=np.load(cache/'U_s_practical.npy',allow_pickle=False,mmap_mode='r')
    if (l.shape!=(7,L_COUNT,6,2,256) or u.shape!=(1,COUNTS['U_s'],6,2,256)
            or p.shape!=u.shape[1:] or any(a.dtype!=np.float32 for a in (l,u,p))):
        raise ValueError('Invalid matched LoRa source IQ')
    return l,u[0],p


def build(rt):
    root=Path(rt['run_root'])
    source,cache,c=handles(rt)
    l,u,_=source_iq(cache)
    ids,rx=identity_and_rx(source,c)
    out=root/'source_action'
    out.mkdir(exist_ok=False)
    sizes=dict(received=(2,256),kernel=(2,64),coefficients=(4,256),variance=())
    arrays={k:np.lib.format.open_memmap(out/f'{k}.npy',mode='w+',dtype=np.float32,shape=(DRAWS*COUNT,*v))
            for k,v in sizes.items()}
    omitted=np.empty(DRAWS*COUNT,dtype=np.float64)

    def items():
        for d in range(DRAWS):
            for i,(pid,group) in enumerate(zip(ids,rx,strict=True)):
                chirp=(i+2*d)%6
                clean=l[0,i,chirp] if i<L_COUNT else u[i-L_COUNT,chirp]
                yield clean,VIEWS[(i+d)%6+1],group,pid,d

    with ProcessPoolExecutor(12) as pool:
        for j,(row,discarded) in enumerate(pool.map(aligned_teacher,items(),chunksize=48)):
            for a,v in zip(arrays.values(),row,strict=True):
                a[j]=v
            omitted[j]=discarded
            if (j+1)%8000==0:
                print('LORA_LINEAGE_ACTION_READY',j+1,DRAWS*COUNT,flush=True)
    for a in arrays.values():
        a.flush()
    write(out/'complete.json',dict(status='SOURCE_ACTION_READY',source_contract=c,draws=DRAWS,n=DRAWS*COUNT,
          causal_zero_origin=True,proxy_train_only=True,omitted_kernel_energy=np.quantile(omitted,[.5,.95,1.]).tolist(),
          F_reads=0,TX_reads=0,target_reads=0))


class LoRaView:
    def __init__(self,rt,*,observed_phase=False,full_action=False,full_coarse=False):
        self.root=Path(rt['run_root'])
        self.observed_phase=observed_phase
        self.full_coarse=full_coarse
        src,cache,self.contract=handles(rt)
        self.l,self.u,self.u_first=source_iq(cache)
        alt=self.root.parent/DRAW_RUN
        self.l_second,self.u_second=(load_draw(alt,role) for role in ROLES)
        self.names,_=identity_and_rx(src,self.contract)
        p=self.root/'source_action'
        meta=read(p/'complete.json')
        if (meta['status']!='SOURCE_ACTION_READY' or meta['source_contract']!=self.contract
                or meta['draws']!=DRAWS or meta['n']!=len(self)
                or any(meta[k] for k in ('F_reads','TX_reads','target_reads'))):
            raise ValueError('Wrong LoRa action bank source contract')
        self.actions={k:np.load(p/f'{k}.npy',allow_pickle=False,mmap_mode='r')
                      for k in ('received','kernel','coefficients','variance')}
        self.full_bank=self.read_full_action() if full_action else None
        self.is_L=np.tile(np.arange(COUNT)<L_COUNT,DRAWS)
        self.quantile=np.random.default_rng(2026112219).random((20,len(self)),dtype=np.float32)

    def __len__(self):
        return DRAWS*COUNT

    def ready(self):
        return all(len(a)==len(self) and a.dtype==np.float32 for a in self.actions.values())

    def read_full_action(self):
        from .full_dura_action_targets import ACTION_BANK
        root=self.root/ACTION_BANK
        proof=read(root/'complete.json')
        if proof!=dict(status='SOURCE_FULL_ACTION_READY',source_contract=self.contract,n=COUNT,
                       draws=2,chirps=6,exact_noisy_view_replay=True,fit_probes=4,roles=list(ROLES),
                       TX_reads=0,F_reads=0,V_reads=0,target_reads=0):
            raise ValueError('Full action bank does not match the LoRa source contract')
        scene,main,variance=(np.load(root/f'{k}.npy',mmap_mode='r',allow_pickle=False)
                             for k in ('L_scene','main','variance'))
        if (scene.shape!=(L_COUNT,) or scene.dtype!=np.int64 or scene.min()<1 or scene.max()>6
                or main.shape!=(2,COUNT,6,2,256) or main.dtype!=np.float32
                or variance.shape!=(2,COUNT,6) or variance.dtype!=np.float32):
            raise ValueError('Full action bank dimensions differ from the source chirps')
        return scene,main,variance

    def full_action_batch(self,packet,chirp,labeled,epoch):
        scenes,main,variance=self.full_bank
        draw=(epoch+1)%2
        l=packet[labeled]
        u=packet[~labeled]-L_COUNT
        observed=np.empty((len(packet),2,256),np.float32)
        if draw:
            observed[labeled]=self.l_second[l,chirp[labeled]]
            observed[~labeled]=self.u_second[u,chirp[~labeled]]
        else:
            observed[labeled]=self.l[scenes[l],l,chirp[labeled]]
            observed[~labeled]=self.u_first[u,chirp[~labeled]]
        kernel=np.zeros((len(packet),2,64),np.float32)
        kernel[:,0,0]=1.
        coef=np.zeros((len(packet),4,256),np.float32)
        coef[:,:2]=main[draw,packet,chirp]
        return dict(received=observed,kernel=kernel,coefficients=coef,
                    variance=variance[draw,packet,chirp].copy())

    def batch(self,indices,epoch,coarse=False,action=False):
        packet=indices%COUNT
        chirp=(packet+2*(indices//COUNT))%6
        labeled=packet<L_COUNT
        ground=np.empty((len(indices),2,256),np.float32)
        ground[labeled]=self.l[0,packet[labeled],chirp[labeled]]
        ground[~labeled]=self.u[packet[~labeled]-L_COUNT,chirp[~labeled]]
        q=self.quantile[epoch-1,indices]
        identity=q<.15
        if action:
            if self.full_bank is None:
                a={k:v[indices].copy() for k,v in self.actions.items()}
            else:
                a=self.full_action_batch(packet,chirp,labeled,epoch)
            a['received'][identity]=ground[identity]
            a['kernel'][identity]=0.
            a['kernel'][identity,0,0]=1.
            a['coefficients'][identity]=0.
            a['coefficients'][identity,0]=1.
            a['variance'][identity]=1e-8
            if self.observed_phase:
                ground,a['coefficients']=in_observed_phase(ground,a['received'],a['coefficients'])
            return dict(clean=ground,**a)
        observed=ground.copy()
        real=q>=.75 if coarse and not self.full_coarse else ~identity
        ls=real&labeled
        us=real&~labeled
        if epoch%2:
            scene=(indices[ls]+epoch)%6+1
            observed[ls]=self.l[scene,packet[ls],chirp[ls]]
            observed[us]=self.u_first[packet[us]-L_COUNT,chirp[us]]
        else:
            observed[ls]=self.l_second[packet[ls],chirp[ls]]
            observed[us]=self.u_second[packet[us]-L_COUNT,chirp[us]]
        if coarse and not self.full_coarse:
            residual=(q>=.15)&~real
            observed[residual]=self.actions['received'][indices[residual]]
        if self.observed_phase:
            ground=in_observed_phase(ground,observed)
        return dict(clean=ground,**{'coarse' if coarse else 'received':observed})
