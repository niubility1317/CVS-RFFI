"""Temporary training hooks: no persistent state, F, labels or extra G forward."""
import torch
from . import design as d
from experiments.pamf_r46_gn_response.normalization import norm_links

class FeatureHints:
    def __init__(self,teacher,student,maps,arm):
        if arm not in d.ARMS or teacher.training or any(p.requires_grad for p in teacher.parameters()):
            raise ValueError('Registered scope and frozen eval restoration teacher required')
        links=norm_links(teacher,student,maps)
        if len(links)!=168 or sum(not v.name.startswith('prior') for v in links)!=84:
            raise ValueError('Existing84context/84prior GN locations required')
        self.arm=arm;self.links=[v for v in links if arm=='all_hint' or not v.name.startswith('prior')]
        self.phase='idle';self.cache={};self.parts={};self.hooks=[]
        self.teacher=teacher;self.student=student
        try:
            for link in self.links:
                index=link.outputs.to(device=link.source.weight.device)
                def teacher_hook(module,inputs,output,link=link,index=index):
                    if self.phase=='idle':return
                    if self.phase!='teacher' or link.name in self.cache:raise ValueError('Exactly one teacher use before each child use required')
                    self.cache[link.name]=output.detach().index_select(1,index)
                def student_hook(module,inputs,output,link=link):
                    if self.phase=='idle':return
                    if self.phase!='student' or link.name not in self.cache or link.name in self.parts:
                        raise ValueError('Aligned single student use after teacher required')
                    target=self.cache[link.name]
                    if target.shape!=output.shape or output.ndim!=3 or output.device!=target.device:
                        raise ValueError('Matching native per-IQ retained channel coordinates required')
                    energy=target.square().mean((1,2)).clamp_min(d.ENERGY_FLOOR)
                    self.parts[link.name]=(output-target).square().mean((1,2))/energy
                self.hooks.extend((link.source.register_forward_hook(teacher_hook),link.target.register_forward_hook(student_hook)))
        except BaseException:
            self.close();raise

    def begin(self):
        if self.phase!='idle' or not self.hooks or self.teacher.training or any(p.requires_grad for p in self.teacher.parameters()):
            raise ValueError('Previous pair complete and frozen teacher required')
        self.cache.clear();self.parts.clear();self.phase='teacher'

    def teacher_done(self):
        if self.phase!='teacher' or set(self.cache)!={v.name for v in self.links}:raise ValueError('All registered teacher features required')
        self.phase='student'

    def finish(self):
        if self.phase!='student' or set(self.parts)!={v.name for v in self.links}:raise ValueError('All registered aligned student features required')
        context=torch.stack([v for name,v in self.parts.items() if not name.startswith('prior')]).mean(0)
        prior=torch.stack([v for name,v in self.parts.items() if name.startswith('prior')]).mean(0) if self.arm=='all_hint' else torch.zeros_like(context)
        result=dict(teacher_context_feature_nmse=context,teacher_prior_feature_nmse=prior)
        self.cache.clear();self.parts.clear();self.phase='idle'
        return result

    def close(self):
        for handle in self.hooks:handle.remove()
        self.hooks.clear();self.cache.clear();self.parts.clear();self.phase='closed'

    def __enter__(self):return self
    def __exit__(self,*args):self.close()
