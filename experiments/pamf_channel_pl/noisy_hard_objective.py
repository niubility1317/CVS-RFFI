"""Cached noisy hard-pool labels, bounded losses and matched raw/H/G forwards."""
from contextlib import contextmanager
from pathlib import Path
import numpy as np
import torch
from experiments.cric_dura_comparison.io import read
from .noisy_hard_loss import mixed_loss
from .u_realization_cache import RUN as HG_RUN,KEYS

RECIPES=dict(NoisyCE=dict(q=0.,rho=0.,extra='none',uniform=False),
    NoisyScaledCE=dict(q=0.,rho=0.,extra='none',uniform=True),
    NoisyGCE=dict(q=.7,rho=0.,extra='none',uniform=False),
    NoisyGCE2Raw=dict(q=.7,rho=.25,extra='raw',uniform=False),
    NoisyGCEG=dict(q=.7,rho=.25,extra='G',uniform=False),
    NoisyGCEHG=dict(q=.7,rho=.25,extra='HG',uniform=False))
METHODS=tuple(RECIPES)


class Objective:
    def __init__(self,c):
        self.c=c;self.recipe=RECIPES[c['paper_method']];folder=Path(c['noisy_U_cache']);d=read(folder/'complete.json')
        self.contract=read(c['source_contract'])
        if d['status']!='NOISY_SOURCE_U_CACHE_COMPLETE' or d['source_contract']!=self.contract or d['teacher']!=c['raw_labeler'] or d['model_seed']!=c['model_seed']:
            raise ValueError('Own same-contract noisy U cache required')
        with np.load(folder/'noisy_U_targets.npz',allow_pickle=False) as a:
            if set(a.files)!={'physical','ids','label','candidate','teacher_confidence','original_confidence'} or list(a['ids'])!=self.contract['role_ids']['U_s']:
                raise ValueError('No U truth, target or foreign physical order permitted')
            self.physical=a['physical'].copy();self.label=a['label'].copy();self.candidate=a['candidate'].copy()
            expected=(a['teacher_confidence']>=.95)&(a['original_confidence']<.95)&(self.physical[:,1]!=1)
            if not np.array_equal(expected,self.candidate) or self.physical.shape!=(56700,5) or ((self.label<0)|(self.label>=6)).any():raise ValueError('Fixed noisy-pool rules differ')
        self.lookup={tuple(row):i for i,row in enumerate(self.physical.tolist())};self.views=None;self.seen=self.noisy_seen=self.extra_seen=0;self.self_used=self.disagreement=0;self.view_counts={};self.last={}

    def fresh_views(self,metadata):
        if self.views is not None:return self.views
        physical=np.stack([metadata[k] for k in KEYS],axis=1)
        if not np.array_equal(physical,self.physical):raise ValueError('Actual source U physical order differs')
        root=Path(self.c['project'])/'runs'/HG_RUN
        if read(root/'complete.json')['status']!='SOURCE_U_REALIZATION_CACHE_COMPLETE':raise ValueError('Complete frozen H/G streams required')
        arrays=[]
        for r in (1,2,3):
            folder=root/('fresh'+str(r));d=read(folder/'complete.json')
            if d['source_contract']!=self.contract or d['G']!=self.c['source_G'] or d['namespace']!='PAMF/realization_diversity/U/fresh'+str(r):raise ValueError('Same G/source streams required')
            with np.load(folder/'physical_index.npz',allow_pickle=False) as a:
                if not np.array_equal(a['physical'],self.physical) or list(a['ids'])!=self.contract['role_ids']['U_s']:raise ValueError('Paired H/G order differs')
            pair={k:np.load(folder/name,mmap_mode='r',allow_pickle=False) for k,name in [('H','channel_U.npy'),('G','restored_U.npy')]}
            if any(a.shape!=(3,56700,2,256) or a.dtype!=np.float32 for a in pair.values()):raise ValueError('Complete FP32 paired IQ streams required')
            arrays.append(pair)
        self.views=arrays;return arrays

    def __call__(self,student,device):
        try:batch=next(student._cd_U_iter)
        except StopIteration:student._cd_U_iter=iter(student._cd_U_loader);batch=next(student._cd_U_iter)
        if not bool((batch[1]==-1).all()) or bool(batch[3]['tx_label_visible'].any()):raise ValueError('U truth exposed')
        n=len(batch[0]);indices=np.asarray([self.lookup[tuple(int(batch[3][k][i]) for k in KEYS)] for i in range(n)],np.int64)
        x=batch[0].to(device);domain=batch[2].to(device);b=torch.tensor(self.candidate[indices].tolist(),device=device,dtype=torch.bool)
        target=torch.tensor(self.label[indices].tolist(),device=device,dtype=torch.long)
        student.eval()
        with torch.no_grad():confidence,pseudo=student(x).softmax(1).max(1)
        student.train();accepted=confidence>=.95;logits=student.forward_identity_only(x,domain_labels=domain)['tx_logits'];extra=None;count=int(b.sum())
        step=student._feature_l_steps+self.c['step_offset'];local=step-6661
        if local<0:raise ValueError('Noisy hard phase begins only after own E30')
        if self.recipe['rho'] and count:
            kind=self.recipe['extra'];value=x[b]
            if kind in ('G','HG'):
                kind='G' if kind=='G' or local%2 else 'H';view=(local//2)%9;r,s=divmod(view,3)
                data=self.fresh_views(student._cd_U_loader.dataset.metadata)[r][kind]
                a=np.ascontiguousarray(data[s,indices[self.candidate[indices]]]);value=torch.frombuffer(bytearray(a.tobytes()),dtype=torch.float32).reshape(a.shape).to(device)
                key=kind+'_fresh'+str(r+1)+'_scene'+str(s);self.view_counts[key]=self.view_counts.get(key,0)+count
            extra=student.forward_identity_only(value,domain_labels=domain[b])['tx_logits'];self.extra_seen+=count
        loss=mixed_loss(logits,pseudo,accepted,target,b,q=self.recipe['q'],rho=self.recipe['rho'],extra_logits=extra,uniform_attenuation=self.recipe['uniform'])
        student._cd_U_seen+=n;student._cd_U_accepted+=int(accepted.sum());self.seen+=n;self.noisy_seen+=count;self.self_used+=int((accepted&~b).sum());self.disagreement+=int((b&(pseudo!=target)).sum())
        self.last=dict(NPL_seen=self.seen,NPL_noisy_seen=self.noisy_seen,NPL_self_used=self.self_used,NPL_disagreement=self.disagreement,NPL_extra_packets=self.extra_seen,
            NPL_noisy_count_batch=count,NPL_q=self.recipe['q'],NPL_rho=self.recipe['rho'],NPL_uniform_attenuation=float(self.recipe['uniform']),NPL_loss=loss.detach())
        return loss


@contextmanager
def installed(c,holder):
    from experiments.cric_dura_comparison import t1_model
    from experiments.cric_dura_comparison.structural import NoSequence
    from experiments.pamf_repl_completion.objectives import loader_scope
    from .noisy_hard_source import validate_config
    old_install,old_plain,old_hook=t1_model.install,t1_model.plain_pl,t1_model.hook
    def hook(self,**kwargs):
        value,metrics=old_hook(self,**kwargs);metrics.update(self._npl.last);return value,metrics
    hook._accepts_labeled_inputs=True
    def build(model,cfg):
        validate_config(cfg);model=old_install(model,cfg);model.id_backbone.encoder.sequence_pool=NoSequence();model._npl=Objective(cfg);holder['model']=model
        model._feature_learning_report.update(no_sigtime=True,noisy_hard_recipe=model._npl.recipe,source_U_IQ_used=True)
        return model
    t1_model.install=build;t1_model.plain_pl=lambda self,device:self._npl(self,device);t1_model.hook=hook
    try:
        with loader_scope():yield
    finally:t1_model.install,t1_model.plain_pl,t1_model.hook=old_install,old_plain,old_hook
