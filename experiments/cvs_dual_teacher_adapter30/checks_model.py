"""CPU synthetic checks of the real L hook; no checkpoint or source IQ reads."""
from copy import deepcopy

import torch
from torch import nn
from torch.nn import functional as F

from . import adapters, design as d, model as api


class FakeChannel:
    def __init__(self):
        self.calls = self.packets = 0
        self.coordinates = []

    def __deepcopy__(self, memo):
        result = type(self)(); memo[id(self)] = result; return result

    def apply(self, raw, meta, *, epoch, step):
        if epoch != 21 or step != 4441:
            raise AssertionError('Local first step must use cumulative E21/4441')
        self.calls += 1; self.packets += len(raw)
        self.coordinates.append((epoch,step))
        return raw.detach() + .125, [dict(scene='synthetic_fullAs',
            namespace=f'FcrossRX/train/e{epoch}/step{step}/L_S') for _ in raw]


class FakeTeachers:
    """A plain tensor fixture, deliberately not an nn.Module or real teacher."""
    def __init__(self):
        self.packets = dict(Tc=0,Ts=0); self.inputs = None

    def assert_frozen(self):
        return None

    def stats(self):
        return dict(packets=dict(self.packets))

    def same_iq_logits(self, first, satellite):
        self.inputs = (first.detach().clone(),satellite.detach().clone())
        self.packets['Tc'] += len(first); self.packets['Ts'] += len(satellite)
        index = torch.arange(len(first),device=first.device) % 6
        clean = F.one_hot(index,6).to(first.dtype)*2.
        other = F.one_hot((index+1)%6,6).to(first.dtype)*.5
        return dict(clean=clean,satellite=clean*.2+other)


def _student_type(native_block, native_head):
    class Encoder(nn.Module):
        def __init__(self):
            super().__init__(); self.core = nn.Module()
            core = self.core.id_backbone = nn.Module()
            core.stem = nn.Conv1d(2,48,1,bias=False)
            core.t1 = native_block(48,72,k=5,pool=2,drop=.1)
            core.t2 = native_block(72,96,k=5,pool=2,drop=.1)
            core.t3 = native_block(96,96,k=3,pool=1,drop=.1)
            core.projection = nn.Linear(96,160)
            core.physical = nn.Linear(96,160)
            core.cls_head = native_head(160,6)

        def features(self, x):
            b = self.core.id_backbone
            h = b.t3(b.t2(b.t1(F.avg_pool1d(b.stem(x),2)))).mean(-1)
            return b.cls_head.components(b.projection(h),b.physical(h))[2]

    class SyntheticStudent(nn.Module):
        def __init__(self):
            super().__init__(); self.id_backbone = nn.Module()
            self.id_backbone.encoder = Encoder()
            self.dom_backbone = nn.Module(); self.dom_backbone.cls_head = nn.Linear(3,2)
            self._feature_learning_report = {}; self._feature_l_steps = 0
            # Exercise strict F206-before-adapter ordering without constructing F.
            for i in range(206-len(self.state_dict())):
                self.register_buffer('synthetic_padding_'+str(i),torch.tensor(float(i)))
            if len(self.state_dict()) != 206:
                raise AssertionError('Synthetic parent must have exactly 206 state entries')

        def forward_identity_only(self, x, domain_labels=None):
            z = self.id_backbone.encoder.features(x)
            return dict(tx_logits=self.get_submodule(adapters.CORE).cls_head.classify(z))

        def forward(self,x):
            return self.forward_identity_only(x)['tx_logits']
    return SyntheticStudent


def run(native_block=None, native_head=None):
    if native_block is None:
        from model import DSConvBlock1d as native_block
    if native_head is None:
        from experiments.cvs_residual_identity.model import ResidualPhysicalHead as native_head
    if (native_block.__name__ != 'DSConvBlock1d'
            or native_head.__name__ != 'ResidualPhysicalHead'):
        raise ValueError('Use the actual native block and unchanged default head classes')
    old_threads = torch.get_num_threads(); torch.set_num_threads(2)
    try:
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(27310)
            return _run(native_block,native_head)
    finally:
        torch.set_num_threads(old_threads)


def _run(native_block,native_head):
    Student = _student_type(native_block,native_head)
    negatives = 0
    def reject(call,kind=ValueError):
        nonlocal negatives
        try:call()
        except kind:negatives += 1
        else:raise AssertionError('Invalid L/lineage/configuration accepted')

    raw = torch.randn(128,2,256); ordinary = raw + .03125
    labels = torch.arange(128,dtype=torch.long)%6
    metadata = dict(eq_i=torch.ones(128,dtype=torch.long),rx_i=torch.ones(128,dtype=torch.long),
        day_i=torch.ones(128,dtype=torch.long),sig_i=torch.arange(128),base_index=torch.arange(128))
    reports = {}
    for row in d.rows():
        c = d.config(row); m = Student(); before = deepcopy(m.state_dict())
        native_W = m.get_submodule(adapters.CORE).cls_head
        rng = torch.get_rng_state().clone()
        api._attach_runtime(m,c,channel=FakeChannel())
        if not torch.equal(rng,torch.get_rng_state()) or m.get_submodule(adapters.CORE).cls_head is not native_W:
            raise AssertionError('Runtime/adapter install consumed RNG or replaced the default head')
        original = set(m._dta_original_parameter_names)
        trainable = {n for n,p in m.named_parameters() if p.requires_grad}
        expected = (set(api.adapter_parameter_names(m)) if c['adapter']['enabled'] else
            {n for n in original if n.startswith('id_backbone.') and n not in m._dta_head_names})
        if (trainable != expected or any(p.requires_grad for n,p in m.named_parameters() if n in m._dta_head_names)
                or any(not torch.equal(value,m.state_dict()[name]) for name,value in before.items())):
            raise AssertionError('Actual original/head/adapter training scope differs')
        if c['adapter']['enabled'] and sum(p.numel() for p in m.parameters() if p.requires_grad) != 4768:
            raise AssertionError('Only 4768 adapter parameters may train')
        ema = deepcopy(m)
        if (ema._dta_channel is m._dta_channel or ema._dta_channel.calls
                or ema.id_backbone.encoder._dta_witness is not ema._dta_witness
                or ema.id_backbone.encoder.features.__self__ is not ema.id_backbone.encoder
                or ema.id_backbone.encoder._counted_features.__self__ is not ema.id_backbone.encoder
                or ema._source_labeled_feature_hook.__self__ is not ema):
            raise AssertionError('Own EMA copy must have independent empty runtime/correct bound methods')
        teacher = api.set_teachers(m,FakeTeachers()); state_keys = set(m.state_dict())
        m.train(); first = api.capture_L(m,raw,ordinary,labels,1)
        if not torch.equal(first,ordinary if c['first_view']=='ordinary' else raw):
            raise AssertionError('Actual source-return first IQ differs from the fixed row')
        reject(lambda: api.capture_L(m,raw,ordinary,labels,1))
        output = m.forward_identity_only(first)
        native_CE = F.cross_entropy(output['tx_logits'],labels,label_smoothing=.01)
        m._feature_l_steps = 1
        delta,_ = api.hook(m,model=m,labeled_output=output,labels=labels,extra=(None,metadata),
            domains=None,optimizer_step=1,identity_loss=native_CE,labeled_count=128,labeled_inputs=raw)
        total = native_CE+delta; rec = m._dta_records[0]
        expected_total = sum(c['loss'][k]*rec[v] for k,v in (
            ('clean_CE_weight','clean_CE'),('satellite_CE_weight','As_CE'),
            ('clean_KD_weight','clean_KL'),('satellite_KD_weight','satellite_KL')))
        if (abs(float(total.detach())-expected_total)>2e-5
                or m._dta_steps != 1 or m._dta_L_packets != (256 if c['As'] else 128)
                or m.id_backbone.encoder._full_train_packets != m._dta_L_packets
                or rec['encoder_first_calls']!=1 or rec['extra_encoder_calls']!=int(c['As'])
                or m._dta_channel.calls!=1 or m._dta_channel.packets!=128
                or m._dta_teacher_packets!={'Tc':128,'Ts':128}
                or teacher.stats()['packets']!={'Tc':128,'Ts':128}
                or not torch.equal(teacher.inputs[0],first)
                or not torch.equal(teacher.inputs[1],raw+.125)
                or set(m.state_dict())!=state_keys or any('teacher' in n for n in m.state_dict())
                or m._dta_witness['raw'] is not None):
            raise AssertionError('Real hook views/loss/counters/shadows/default-head state differ')
        if not c['As'] and (rec['As_CE'] or rec['satellite_KL'] or rec['lossstats']['satellite']['student_forward']):
            raise AssertionError('NoAs cannot run/use a second student slot')
        total.backward()
        head_grad = [n for n,p in m.named_parameters() if n in m._dta_head_names and p.grad is not None]
        if head_grad or not any(p.grad is not None and bool(p.grad.count_nonzero()) for p in m.parameters() if p.requires_grad):
            raise AssertionError('Trainable feature gradient must pass the frozen original default head')
        opt = torch.optim.SGD([p for p in m.parameters() if p.requires_grad],lr=.001)
        opt.step(); opt.zero_grad(set_to_none=True)
        if any(not torch.equal(before[n],m.state_dict()[n]) for n in m._dta_head_names):
            raise AssertionError('All original classifier tensors must remain unchanged')
        if c['adapter']['enabled']:
            adapters.assert_frozen_state(m,before)
        saved = deepcopy(m.state_dict()); reloaded = Student()
        api._attach_runtime(reloaded,c,channel=FakeChannel()); reloaded.load_state_dict(saved,strict=True)
        if any(not torch.equal(t,reloaded.state_dict()[n]) for n,t in saved.items()):
            raise AssertionError('Strict reconstructed new-state reload differs')
        payload = dict(epoch=10,a1_ema_successful_updates=2220,run_id=c['run_id'],candidate_id=c['row_id'],
            phase1_lineage=api.phase1_lineage(c),checkpoint_selection='final_only',model=saved,ema_model=saved)
        api.check_final_payload(payload,c)
        for field,bad in (('epoch',30),('a1_ema_successful_updates',6660),('candidate_id','B0-s2026092701')):
            mutated = dict(payload); mutated[field]=bad
            reject(lambda mutated=mutated: api.check_final_payload(mutated,c))
        mutated = deepcopy(payload); mutated['phase1_lineage']['head_frozen']=False
        reject(lambda: api.check_final_payload(mutated,c))
        bad_c = deepcopy(c); bad_c['loss']['clean_KD_weight'] += .001
        reject(lambda: d.validate(bad_c))
        reports[c['arm']] = dict(student_L_packets=m._dta_L_packets,Tc_packets=128,Ts_packets=128,
            channel_packets=128,first_input=c['first_view'],student_As_forward=c['As'],
            trainable_parameters=sum(p.numel() for p in m.parameters() if p.requires_grad),
            default_head_frozen=True,strict_reload=True,loss_formula_matches=True)

    c = d.config(d.rows()[-1]); m = api._attach_runtime(Student(),c,channel=FakeChannel()).train()
    reject(lambda: api.capture_L(m,raw,ordinary,torch.full_like(labels,-1),1))
    reject(lambda: api.capture_L(m,raw,ordinary,labels,2))
    reject(lambda: api.capture_L(m,raw[:127],ordinary[:127],labels[:127],1))
    reject(lambda: api.capture_L(m,raw.double(),ordinary.double(),labels,1))
    bad = raw.clone();bad[0,0,0]=float('nan')
    reject(lambda: api.capture_L(m,bad,ordinary,labels,1))
    api.capture_L(m,raw,ordinary,labels,1)
    reject(lambda: m.forward_identity_only(ordinary))
    reject(lambda: api.set_teachers(m,nn.Linear(2,6)))

    # Invalid coordinates are rejected before any physical channel dependency is used.
    channel = api.ContinuationPractical.__new__(api.ContinuationPractical)
    for epoch,step in ((20,4440),(21,4442-222),(30,6661),(31,6660),(22,4441)):
        reject(lambda epoch=epoch,step=step: channel.apply(raw,metadata,epoch=epoch,step=step))
    for key,bad_value in (('eq_i',2),('rx_i',2),('day_i',4)):
        bad_meta = dict(metadata);bad_meta[key]=torch.full((128,),bad_value,dtype=torch.long)
        reject(lambda bad_meta=bad_meta: channel.apply(raw,bad_meta,epoch=21,step=4441))
    bad_meta=dict(metadata);del bad_meta['base_index']
    reject(lambda: channel.apply(raw,bad_meta,epoch=21,step=4441))

    # Test the public install order with a synthetic F206 fixture, no native/CP loader.
    saved_prepare,saved_channel = api._prepare_F,api.ContinuationPractical
    parent = Student(); parent_state = deepcopy(parent.state_dict())
    payload = dict(epoch=20,a1_ema_successful_updates=4440,run_id=d.PARENT_RUN,
        candidate_id=d.PARENT_ROW,model=parent_state)
    try:
        api._prepare_F=lambda model,config:model
        api.ContinuationPractical=FakeChannel
        installed=api.install(Student(),c,payload)
        if any(not torch.equal(v,installed.state_dict()[n]) for n,v in parent_state.items()):
            raise AssertionError('Strict parent must load before attaching new zero-B adapters')
        bad_parent=dict(payload);bad_parent['epoch']=10
        reject(lambda: api.install(Student(),c,bad_parent))
    finally:
        api._prepare_F,api.ContinuationPractical=saved_prepare,saved_channel
    return dict(status='PASS',rows=reports,protocol_negatives=negatives,
        native_block=native_block.__name__,default_head=native_head.__name__,
        strict_parent206_before_adapter=True,local10_cumulative30_checked=True,
        own_EMA_deepcopy_empty_channel=True,teacher_not_registered=True,
        CPU_RNG_preserved=True,synthetic_optimizer_steps=6,real_training_updates=0,
        checkpoints_loaded=0,real_IQ_packets=0,physical_channel_calls=0,full_F_model_constructed=False)


if __name__ == '__main__':
    import json
    print(json.dumps(run(),allow_nan=False))
