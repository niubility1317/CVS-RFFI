"""Original PL plus paired correlation loss; reuse native L and U raw forwards."""
from contextlib import contextmanager
from pathlib import Path
from types import MethodType
import numpy as np
import torch
from torch.nn import functional as F
from experiments.cric_dura_comparison.io import read,tensor
from experiments.cric_dura_comparison.f_model import PREFIX
from .history import KEYS
from .selective_covariance import correlation,consistency

METHODS=('CovSel-U','CovRandom-U','CovFull-U','CovSel-LU')
MASKS={'CovSel-U':'selective','CovRandom-U':'random','CovFull-U':'full','CovSel-LU':'selective'}
RECIPE=dict(levels=['time_down','t2.pool'],ridge=1e-3,ramp_updates=444,positive_threshold=.95,positive_weight=1.,
    U_scope='all_hidden_U',L_pairs=64,mask_mode='natural_support_capped',extra_U_student_forwards=1,
    extra_L_forwards=0,new_inference_parameters=0,EMA_updates=0,
    coefficient_rule='min(1,.1*median(primary_norm)/max_method_median(covariance_norm));9trainL/U batches,noV',
    replica_schedule='new_U_epoch_mod3_plus1',scene_schedule='(newcall_minus1+new_U_epoch_div3)_mod3',
    rng='extraAs_private_fork_seed2026102627_plus_globalUcall')


def schedule(call):
    if type(call) is not int or not 1<=call<=6660:raise ValueError('Fixed thirty new U epochs required')
    epoch=(call-1)//222
    return epoch%3+1,((call-1)+epoch//3)%3


def combine(u,l,nu,nl,scope):
    if scope not in ('U','LU') or nu<=0 or nl<=0:raise ValueError('Positive physical pair counts and declared gradient scope required')
    return (nu*u+nl*(l if scope=='LU' else l.detach()))/(nu+nl)


class Exposure:
    def __init__(self,metadata):
        if 'y' in metadata:raise ValueError('U true TX must be absent')
        rows=list(zip(*(metadata[k].tolist() for k in KEYS)))
        if len(rows)!=56700 or len(set(rows))!=56700:raise ValueError('Complete unique original U required')
        self.lookup={row:i for i,row in enumerate(rows)};self.visits=torch.zeros(56700,dtype=torch.long)
        self.calls=self.seen=self.accepted=0;self.by_rx={};self.by_replica=[0]*3;self.by_scene=[0]*3

    def indices(self,batch,role):
        if role!='U_s' or (batch[1]!=-1).any() or batch[3]['tx_label_visible'].any():raise ValueError('Only hidden TX sourceU may enter covariance')
        rows=list(zip(*(batch[3][k].tolist() for k in KEYS)))
        if len(set(rows))!=len(rows) or any(row not in self.lookup for row in rows):raise ValueError('Foreign/duplicate physical U')
        return torch.tensor([self.lookup[row] for row in rows],dtype=torch.long)

    def state(self):
        return dict(calls=self.calls,seen=self.seen,accepted=self.accepted,unique_seen=int((self.visits>0).sum()),
            min_visits=int(self.visits.min()),max_visits=int(self.visits.max()),covariance_U_seen=self.seen,
            covariance_L_pairs=64*self.calls,student_U_packets=2*self.seen,teacher_U_packets=self.seen,
            by_source_RX=self.by_rx,by_replica=self.by_replica,by_scene=self.by_scene,
            U_truth_read=False,target_reads=0,model_EMA_updates=0,new_inference_parameters=0,default_head_joint=True)


class SourceViews:
    def __init__(self,c,data):
        from experiments.pamf_channel_pl.u_realization_cache import RUN,namespace
        from experiments.pamf_channel_pl.restored_source import SCENES
        contract=read(c['source_contract']);root=Path(c['project'])/'runs'/RUN;done=read(root/'complete.json')
        if (done['status']!='SOURCE_U_REALIZATION_CACHE_COMPLETE' or done['U_physical_records']!=56700
            or any(done[k] for k in ('F_weight_reads','F_output_reads','F_forward_packets','U_truth_read','V_reads','target_reads','optimizer_updates','G_updates'))):
            raise ValueError('Completed source-only view cache required')
        physical=np.stack([data.metadata[k] for k in KEYS],1);self.lookup={tuple(row):i for i,row in enumerate(physical)};self.arrays={}
        for replica in (1,2,3):
            p=root/('fresh'+str(replica));header=read(p/'complete.json')
            if (header['source_contract']!=contract or header['G']!=c['source_G'] or header['namespace']!=namespace(replica)
                or header['U_seen']!=56700 or header['scenes']!=list(SCENES)):
                raise ValueError('Source As view provenance differs')
            with np.load(p/'physical_index.npz',allow_pickle=False) as a:
                if not np.array_equal(a['physical'],physical) or list(a['ids'])!=contract['role_ids']['U_s']:raise ValueError('As/raw physical alignment differs')
            values=np.load(p/'channel_U.npy',mmap_mode='r',allow_pickle=False)
            if values.shape!=(3,56700,2,256) or values.dtype!=np.float32:raise ValueError('Finite FP32 paired As cache required')
            self.arrays[replica]=values

    def take(self,metadata,replica,scene,device):
        rows=zip(*(metadata[k].tolist() for k in KEYS));ids=[self.lookup[tuple(row)] for row in rows]
        values=np.array(self.arrays[replica][scene,ids],copy=True)
        if not np.isfinite(values).all():raise ValueError('Nonfinite cached sourceAs')
        return tensor(values,device)


def load_masks(c):
    p=torch.load(c['covariance_masks'],map_location='cpu',weights_only=False)
    if (p['schema']!='pamf_sig1_selective_covariance_masks_v2_natural_support' or p['fit_role']!='L_s'
        or p['source_contract']!=read(c['source_contract']) or p['source_prefix']!=c['shared_SIG1_E20']
        or p['recipe']['model_seed']!=c['model_seed'] or p['U_truth_read'] or p['V_used'] or p['target_used']):
        raise ValueError('Masks must come from same-contract,same-basis labeled-source E20')
    for level,channels,count in [('time_down',48,94),('t2.pool',96,294)]:
        for name,n in [('selective',count),('random',count),('full',channels*(channels-1)//2)]:
            m=p['masks'][level][name]
            if m.dtype!=torch.bool or m.shape!=(channels,channels) or torch.tril(m).any() or int(m.sum())!=n:raise ValueError('Actual fitted mask shape/count differs')
    return p['masks']


class Objective:
    def __init__(self,c):
        self.c=c;self.masks=load_masks(c);self.history=None;self.cache=None;self.maps={};self.phase=None;self.last={};self.candidates={}

    @contextmanager
    def observe(self,student):
        self.maps={};self.L_forwards=0;self.phase=None;original=student.forward_identity_only;handles=[]
        def capture(level):
            def hook(module,inputs,out):
                if self.phase is not None:
                    maps=self.maps.setdefault(self.phase,{})
                    if level in maps:raise ValueError('Repeated map in one actual source forward')
                    value=correlation(out,RECIPE['ridge'])
                    if self.c.get('covariance_calibration'):value.retain_grad()
                    maps[level]=value
            return hook
        for level in RECIPE['levels']:handles.append(student.get_submodule(PREFIX+level).register_forward_hook(capture(level)))
        def forward(model,x,*args,**kwargs):
            old=self.phase
            if self.phase is None and model.training and len(x)==128:
                self.L_forwards+=1
                self.phase={2:'L_As',4:'L_raw'}.get(self.L_forwards)
            try:return original(x,*args,**kwargs)
            finally:self.phase=old
        student.forward_identity_only=MethodType(forward,student)
        try:yield
        finally:
            student.forward_identity_only=original
            for handle in handles:handle.remove()
            self.phase=None

    def plain(self,student,device):
        data=student._cd_U_loader.dataset
        if data.role!='U_s' or 'y' in data.metadata:raise ValueError('Hidden sourceU only')
        if self.history is None:self.history=Exposure(data.metadata)
        if self.cache is None:self.cache=SourceViews(self.c,data)
        try:batch=next(student._cd_U_iter)
        except StopIteration:student._cd_U_iter=iter(student._cd_U_loader);batch=next(student._cd_U_iter)
        self.ids=self.history.indices(batch,data.role);self.batch=batch;x=batch[0].to(device)
        student.eval()
        with torch.no_grad():confidence,pseudo=student(x).softmax(1).max(1)
        student.train();accepted=confidence>=.95;self.phase='U_raw'
        try:logits=student.forward_identity_only(x,domain_labels=batch[2].to(device))['tx_logits']
        finally:self.phase=None
        student._cd_U_seen+=len(x);student._cd_U_accepted+=int(accepted.sum());self.accepted=accepted
        value=(F.cross_entropy(logits,pseudo,reduction='none')*accepted).mean()
        self.last=dict(COV_PL_loss=value.detach(),COV_raw_PL_accepted=int(accepted.sum()))
        return value

    def extra(self,student,kwargs):
        if self.L_forwards!=5 or set(self.maps)!={'L_As','L_raw','U_raw'}:raise ValueError('Unchanged five native extraL and original Uraw required')
        call=student._feature_l_steps+self.c.get('step_offset',0)-4440
        if call!=self.history.calls+1:raise ValueError('Continuous original E20+10+20 source trajectory required')
        replica,scene=schedule(call);device=kwargs['labels'].device;n=len(self.batch[0])
        view=self.cache.take(self.batch[3],replica,scene,device);self.phase='U_As'
        try:
            with torch.random.fork_rng(devices=[device.index if device.index is not None else torch.cuda.current_device()] if device.type=='cuda' else []):
                torch.random.default_generator.manual_seed(2026102627+call)
                if device.type=='cuda':torch.cuda.manual_seed(2026102627+call)
                student.forward_identity_only(view,domain_labels=self.batch[2].to(device))
        finally:self.phase=None
        mask=(torch.arange(128,device=device)+student._feature_l_steps).remainder(2)!=0
        lrx=kwargs['extra'][1]['rx_i'].to(device)[mask];urx=self.batch[3]['rx_i'].to(device)
        methods=METHODS if self.c.get('covariance_calibration') else (self.c['paper_method'],);losses={}
        for method in methods:
            values=[]
            for level in RECIPE['levels']:
                m=self.masks[level][MASKS[method]].to(device)
                u=consistency(self.maps['U_raw'][level],self.maps['U_As'][level],m,receivers=urx,source_rxs=self.c['source_rxs'])
                l=consistency(self.maps['L_raw'][level][mask],self.maps['L_As'][level][mask],m,receivers=lrx,source_rxs=self.c['source_rxs'])
                values.append(combine(u,l,n,64,'LU' if method=='CovSel-LU' else 'U'))
            losses[method]=torch.stack(values).mean()
        h=self.history;h.calls+=1;h.seen+=n;h.accepted+=int(self.accepted.sum());h.visits[self.ids]+=1
        h.by_replica[replica-1]+=n;h.by_scene[scene]+=n
        for rx in self.batch[3]['rx_i'].unique().tolist():
            chosen=self.batch[3]['rx_i']==rx;r=h.by_rx.setdefault(str(rx),dict(seen=0,accepted=0,covariance=0))
            r['seen']+=int(chosen.sum());r['covariance']+=int(chosen.sum());r['accepted']+=int((chosen.to(device)&self.accepted).sum())
        value=losses[self.c['paper_method']];weight=self.c['covariance_coefficient']*min(call/444.,1.)
        self.last.update(COV_loss=value.detach(),COV_weight=weight,COV_seen=h.seen,COV_calls=h.calls,COV_L_pairs=64*h.calls,
            COV_replica=replica,COV_scene=scene,COV_L_forwards_reused=5,COV_extra_U_forwards=1)
        self.candidates=losses if self.c.get('covariance_calibration') else {}
        return weight*value
