"""Identity-only, unit-aware AdamW with exact raw-Hz model/checkpoint semantics.

Frequency masters/moments live in float64 u=f/fs; native parameters remain
float32 Hz at every forward. Dense parameters keep the declared AdamW recipe.
"""
import copy
import torch


class FrequencyAdamW:
    def __init__(self, model, groups, frequency_peak_lr, dense_weight_decay=1e-4):
        if not 0 < frequency_peak_lr <= 1e-5:
            raise ValueError('Frequency peak LR outside declared physical step range')
        self.param_groups=groups
        self.peak=float(frequency_peak_lr)
        self.lr=self.peak
        self.entries=[]
        for module_name,module in model.named_modules():
            if not module_name.startswith('id_backbone.'):
                continue
            if all(hasattr(module,k) for k in ('low_hz_','band_hz_','sample_rate')):
                fs=float(module.sample_rate)
                if not fs > 0:
                    raise ValueError('Invalid sample rate')
                for key in ('low_hz_','band_hz_'):
                    p=getattr(module,key)
                    if not p.requires_grad or p.dtype!=torch.float32 or not torch.isfinite(p).all() or bool((p<1.).any()):
                        raise ValueError('Expected trainable float32 identity frequency')
                    master=torch.nn.Parameter(p.detach().double()/fs)
                    if not torch.equal((master.detach()*fs).to(p.dtype),p.detach()):
                        raise ValueError('Initial Hz conversion changes native parameters')
                    self.entries.append(dict(name=module_name+'.'+key,p=p,master=master,fs=fs))
        if not self.entries:
            raise ValueError('No identity frequency parameters')
        frequency_ids={id(e['p']) for e in self.entries}
        all_ids=[id(p) for g in groups for p in g['params']]
        if len(set(all_ids))!=len(all_ids) or not frequency_ids<=set(all_ids):
            raise ValueError('Frequency/group ownership mismatch')
        dense=[dict(g,params=[p for p in g['params'] if id(p) not in frequency_ids]) for g in groups]
        if any(not g['params'] for g in dense):
            raise ValueError('Empty dense identity group')
        self.dense=torch.optim.AdamW(dense,weight_decay=dense_weight_decay)
        self.frequency=torch.optim.AdamW([e['master'] for e in self.entries],lr=self.lr,
                                        weight_decay=0.,eps=1e-8,betas=(.9,.999))
        self.steps=0
        self.last_metrics={}

    def zero_grad(self,set_to_none=True):
        for group in self.param_groups:
            for p in group['params']:
                if set_to_none:
                    p.grad=None
                elif p.grad is not None:
                    p.grad.zero_()
        self.frequency.zero_grad(set_to_none=set_to_none)

    def set_frequency_lr(self,lr):
        if not 0 <= lr <= self.peak:
            raise ValueError('Frequency LR outside frozen schedule')
        self.lr=float(lr)

    @torch.no_grad()
    def step(self):
        before=[]
        for e in self.entries:
            p,m,fs=e['p'],e['master'],e['fs']
            if not torch.equal((m*fs).to(p.dtype),p):
                raise ValueError('Native Hz parameters diverged from master (SAM must restore first)')
            if p.grad is not None and not torch.isfinite(p.grad).all():
                raise FloatingPointError('Nonfinite frequency gradient')
            before.append((p.detach().clone(),m.detach().clone()))
            m.grad=None if p.grad is None else p.grad.detach().double()*fs
        for original,dense in zip(self.param_groups,self.dense.param_groups):
            dense['lr']=original['lr']
        for group in self.frequency.param_groups:
            group['lr']=self.lr
        self.dense.step()
        self.frequency.step()
        rows=[]
        for e,(raw_before,master_before) in zip(self.entries,before):
            p,m,fs=e['p'],e['master'],e['fs']
            # Bound adaptive displacement in u; native abs/Nyquist synthesis
            # remains unchanged. A 1 Hz floor avoids abs'(0)=0 dead parameters.
            proposal=m.detach().clone()
            bounded=proposal.clamp(min=master_before-self.lr,max=master_before+self.lr)
            bounded=bounded.clamp_min(1./fs)
            m.copy_(bounded)
            p.copy_((m*fs).to(p.dtype))  # copy_ invalidates native filter caches.
            delta=p-raw_before
            active=torch.zeros_like(p,dtype=torch.bool) if p.grad is None else p.grad!=0
            active_count=int(active.sum())
            rows.append(dict(name=e['name'],sample_rate=fs,lr_u=self.lr,
                step_cap_Hz=fs*self.lr,mean_abs_delta_Hz=float(delta.abs().mean()),
                max_abs_delta_Hz=float(delta.abs().max()),active_gradient_count=active_count,
                active_gradient_zero_update_count=int((active&(delta==0)).sum()),
                active_gradient_zero_update_fraction=float((active&(delta==0)).sum())/active_count if active_count else None,
                bounded_coordinates=int((proposal!=bounded).sum()),master_dtype=str(m.dtype),native_dtype=str(p.dtype)))
        self.steps+=1
        self.last_metrics=dict(step=self.steps,frequency=rows,frequency_weight_decay=0.,
            initial_native_conversion_exact=True,coordinate='u=f/fs',native_parameter_unit='Hz')
        return self.last_metrics

    def state_dict(self):
        return dict(schema='frequency_adamw_v1',dense=self.dense.state_dict(),
            frequency=self.frequency.state_dict(),masters=[e['master'].detach().clone() for e in self.entries],
            names=[e['name'] for e in self.entries],sample_rates=[e['fs'] for e in self.entries],
            peak=self.peak,lr=self.lr,steps=self.steps,last_metrics=copy.deepcopy(self.last_metrics))

    def load_state_dict(self,state):
        if state['schema']!='frequency_adamw_v1' or state['names']!=[e['name'] for e in self.entries] or state['sample_rates']!=[e['fs'] for e in self.entries] or state['peak']!=self.peak:
            raise ValueError('Optimizer coordinate/state mismatch')
        if len(state['masters'])!=len(self.entries):
            raise ValueError('Incomplete frequency masters')
        for e,value in zip(self.entries,state['masters']):
            if value.dtype!=torch.float64 or not torch.equal((value.to(e['p'].device)*e['fs']).to(e['p'].dtype),e['p'].detach()):
                raise ValueError('Load native model first; frequency state must match exactly')
        self.dense.load_state_dict(state['dense']);self.frequency.load_state_dict(state['frequency'])
        # The caller schedules the outer groups, and step() copies their LR
        # into dense AdamW. Restore both views before the next optimizer step.
        for outer,inner in zip(self.param_groups,self.dense.param_groups):
            for key,value in inner.items():
                if key!='params':outer[key]=copy.deepcopy(value)
        with torch.no_grad():
            for e,value in zip(self.entries,state['masters']):e['master'].copy_(value)
        self.set_frequency_lr(state['lr']);self.steps=int(state['steps']);self.last_metrics=copy.deepcopy(state['last_metrics'])
