"""Default-head F continuation with view-specific teachers and feature adapters."""
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from types import MethodType

import numpy as np
import torch
from torch.nn import functional as F

from . import adapters, design as d
from .distillation import objective


def phase1_lineage(c):
    return dict(schema='F_dualTeacher_adapter30_lineage_v1', run_id=c['run_id'],
        row_id=c['row_id'], model_seed=c['model_seed'], parent_run_id=d.PARENT_RUN,
        parent_row_id=d.PARENT_ROW, parent_epoch=20, parent_updates=4440,
        segment_epochs=10, segment_updates=2220, cumulative_epochs=30,
        cumulative_updates=6660, head_frozen=True,
        adapter_enabled=c['adapter']['enabled'], selection='fixed_FINAL30')


def parent_payload(c):
    """The original parent lineage/role check, before loading either parent state."""
    d.validate(c)
    old = d.parent_config(); source = Path(d.PARENT_SOURCE)
    done, frozen = d.read(source/'completion.json'), d.read(source/'source_frozen.json')
    initial, contract = d.read(source/'initialization.json'), d.read(source/'source_contract.json')
    if (done['config'] != old or done['epoch'] != 20 or done['optimizer_steps'] != 4440
            or done['target_access'] or done['head_fit_updates']
            or frozen['config'] != old or frozen['selection'] != 'final_only'
            or frozen['target_access'] or not initial['scratch_only']
            or initial['checkpoint_sources'] or initial['ancestors'] or initial['target_contact']
            or contract['role_ids'] != d.read(d.SOURCE)['role_ids']):
        raise ValueError('Fixed B0 FINAL20 parent provenance or physical source roles differ')
    cp = torch.load(source/'final_ssdg.pth', map_location='cpu', weights_only=False)
    if (cp['epoch'] != 20 or cp['a1_ema_successful_updates'] != 4440
            or cp['run_id'] != d.PARENT_RUN or cp['candidate_id'] != d.PARENT_ROW
            or cp['args']['baseline_ckpt'] or cp['args']['teacher_ckpt']
            or cp['checkpoint_selection'] != 'final_only'
            or len(cp['model']) != 206 or cp.get('ema_model') is None
            or set(cp['model']) != set(cp['ema_model'])):
        raise ValueError('Own parent student/EMA FINAL20 payload differs')
    return cp


class ContinuationPractical:
    """Original full physics with only the new cumulative E21..30 boundary.

    The original Practical implementation and its E1..20 guard are untouched.
    Namespace/seed/session/config/normalization/propagation are retained exactly.
    """
    def __init__(self):
        from experiments.cvs_fullsource_concat20.channel import Practical, serial
        from experiments.cvs_fullsource_concat20 import design as original
        self.base = Practical('train')
        self.seeds, self.scenes = original.CHANNEL_SEEDS, original.TRAIN_SCENES
        self.serial = serial

    def __getattr__(self, key):
        return getattr(self.base, key)

    def __deepcopy__(self, memo):
        result = type(self)(); memo[id(self)] = result; return result

    def audit(self):
        return dict(self.base.audit(), cumulative_epoch_range=[21,30],
                    cumulative_step_range=[4441,6660], original_physics_unchanged=True)

    def summary(self):
        return dict(self.base.summary(), cumulative_epoch_range=[21,30],
                    cumulative_step_range=[4441,6660], fresh_each_step=True)

    def apply(self, x, meta, *, epoch, step):
        if (epoch not in range(21,31) or step not in range(4441,6661)
                or epoch != (step-1)//222+1 or x.shape != (128,2,256)
                or x.dtype != torch.float32 or not bool(torch.isfinite(x).all())):
            raise ValueError('Finite actual source L128 and cumulative E21..30 only')
        keys = ('eq_i','rx_i','day_i','sig_i','base_index')
        if any(k not in meta or len(meta[k]) != len(x) for k in keys):
            raise ValueError('Original L physical metadata required')
        values = {k:meta[k].detach().cpu().tolist() if hasattr(meta[k],'detach') else list(meta[k])
                  for k in keys}
        if (set(map(int,values['rx_i']))-set(d.FIT)
                or set(map(int,values['day_i']))-{1,2,3} or set(map(int,values['eq_i']))!={1}):
            raise ValueError('Original source RX/day/equalized roles required')
        raw = np.asarray(x.detach().cpu().tolist(), dtype=np.float64)
        out, records = [], []
        namespace = f'FcrossRX/train/e{epoch}/step{step}/L_S'
        p = self.base
        for i in range(len(x)):
            pid = tuple(int(values[k][i]) for k in keys)
            rx, day = int(values['rx_i'][i]), int(values['day_i'][i])
            session = f'rx{rx}:day{day}:eq{int(values["eq_i"][i])}'
            rng = np.random.default_rng(p.ch.stable_seed(self.seeds['train'],namespace,'scene',pid))
            scene = str(rng.choice(self.scenes)); cfg = p.configs[scene]
            seed = p.ch.stable_seed(self.seeds['train'],namespace,scene,pid)
            hw = p.ch.receiver_for_session(cfg,self.seeds['receiver'],session)
            hardware = asdict(hw)
            if p.sessions.setdefault(session,hardware) != hardware:
                raise ValueError('Hardware changed within physical receiver session')
            normalized, _ = p.ch.normalize_input(raw[i,0]+1j*raw[i,1])
            signal, info = p.residual.ResidualChannel(cfg,seed,hw).process(normalized)
            if not np.isfinite(signal).all() or info['channel_equalization_applied']:
                raise ValueError('Original finite full residual/noEQ output required')
            p.coverage[(rx,day,scene,info['state_end'],info['receiver_lock_status'])] += 1
            out.append(np.stack((signal.real,signal.imag)))
            records.append(self.serial(dict(info,scene=scene,sample_id=pid,
                                            session_id=session,namespace=namespace)))
        p.calls += 1; p.packets += len(x)
        a = np.ascontiguousarray(out,dtype=np.float32)
        return torch.frombuffer(bytearray(a.tobytes()),dtype=torch.float32).reshape(a.shape).to(x.device), records


def head_parameter_names(model):
    names = set()
    for path, module in model.named_modules():
        if path.split('.')[-1] == 'cls_head':
            names.update(path+'.'+name for name,_ in module.named_parameters())
    return tuple(sorted(names))


def adapter_parameter_names(model):
    return tuple(model._dta_adapter_names)


def set_teachers(model, bundle):
    if isinstance(bundle, torch.nn.Module) or not all(callable(getattr(bundle,key,None))
            for key in ('same_iq_logits','stats','assert_frozen')):
        raise ValueError('Teachers must use the plain frozen same-IQ holder')
    bundle.assert_frozen()
    model.__dict__['_dta_teachers'] = bundle
    return bundle


def capture_L(model, raw, ordinary, labels, step):
    if (not model.training or not torch.is_grad_enabled()
            or step != model._dta_steps+1 or step not in range(1,2221)
            or raw.shape != (128,2,256) or ordinary.shape != raw.shape
            or raw.dtype != torch.float32 or ordinary.dtype != raw.dtype
            or raw.device != ordinary.device or not bool(torch.isfinite(raw).all())
            or not bool(torch.isfinite(ordinary).all()) or labels.shape != (128,)
            or labels.dtype != torch.long or bool(((labels<0)|(labels>5)).any())
            or model._dta_witness['raw'] is not None):
        raise ValueError('One ordered visible L128 raw/ordinary call required')
    first = ordinary if model._dta_config['first_view'] == 'ordinary' else raw
    model._dta_witness.update(raw=raw.detach().clone(), ordinary=ordinary.detach(),
        first=first.detach().clone(), labels=labels.detach().clone(), As=None,
        actual_first=None, actual_As=None, phase='first', step=step, first_seen=0, extra_seen=0)
    return first


def counted(self, x):
    key = '_full_train_packets' if self.training else '_full_eval_packets'
    setattr(self,key,getattr(self,key)+len(x))
    witness = self._dta_witness
    if self.training and witness['raw'] is not None:
        phase = witness['phase']
        if phase not in ('first','extra'):
            raise ValueError('Unknown actual labeled student slot')
        expected = witness['first'] if phase == 'first' else witness['As']
        counter = 'first_seen' if phase == 'first' else 'extra_seen'
        if expected is None or not torch.equal(x,expected) or witness[counter]:
            raise ValueError('Actual encoder must consume each declared L IQ exactly once')
        witness[counter] += 1
        witness['actual_first' if phase == 'first' else 'actual_As'] = x.detach()
    value = self._counted_features(x)
    if value.shape != (len(x),160) or not bool(torch.isfinite(value).all()):
        raise ValueError('Finite actual joint160 default-head representation required')
    self._joint160_seen_calls += 1; self._joint160_seen_packets += len(x)
    self._joint160_last_shape = list(value.shape)
    return value


def hook(self, *, model, labeled_output, labels, extra, domains, optimizer_step,
         identity_loss, labeled_count, labeled_inputs):
    c, witness = self._dta_config, self._dta_witness
    if (self is not model or not self.training or not torch.is_grad_enabled()
            or optimizer_step != self._feature_l_steps or optimizer_step != self._dta_steps+1
            or optimizer_step != witness['step'] or labeled_count != 128
            or labels.shape != (128,) or witness['raw'] is None
            or witness['phase'] != 'first' or witness['first_seen'] != 1 or witness['extra_seen']
            or not torch.equal(labels,witness['labels'])
            or not torch.equal(labeled_inputs,witness['raw'])
            or not isinstance(extra,(tuple,list)) or len(extra)<2):
        raise ValueError('Real ordered L update, raw hook and actual first encoder must agree')
    first = labeled_output['tx_logits']
    ce = F.cross_entropy(first,labels,label_smoothing=.01)
    if not torch.allclose(ce.detach(),identity_loss.detach(),atol=1e-6,rtol=1e-5):
        raise ValueError('Original all128 native smoothed identity CE differs')
    cumulative = 4440+optimizer_step; epoch = (cumulative-1)//222+1
    as_iq, records = self._dta_channel.apply(witness['raw'],extra[1],epoch=epoch,step=cumulative)
    if as_iq.shape != (128,2,256) or len(records) != 128 or not bool(torch.isfinite(as_iq).all()):
        raise ValueError('Every visible L requires one complete unchanged fullAs')
    second = None
    if c['As']:
        witness.update(phase='extra',As=as_iq.detach())
        second = self.forward_identity_only(as_iq,domain_labels=domains)['tx_logits']
        if witness['extra_seen'] != 1 or not torch.equal(witness['actual_As'],as_iq):
            raise ValueError('Actual As student and Ts must receive identical fullAs IQ')
    bundle = self.__dict__['_dta_teachers']
    if bundle is None:
        raise ValueError('Frozen teachers must be preloaded before native student scopes')
    teachers = bundle.same_iq_logits(witness['actual_first'],as_iq)
    bundle.assert_frozen()
    loss_options = {key:c['loss'][key] for key in (
        'clean_temperature','satellite_temperature','clean_teacher_scale','clean_student_scale',
        'satellite_teacher_scale','satellite_student_scale','clean_KD_weight','satellite_KD_weight',
        'teacher_correct_mask','temperature_compensation','clean_CE_weight','satellite_CE_weight')}
    total, info = objective(first,second,labels,teachers['clean'],teachers['satellite'],**loss_options)
    n = 1+int(c['As'])
    if witness['extra_seen'] != int(c['As']):
        raise ValueError('NoAs must not execute a second student forward')
    self._dta_steps += 1; self._dta_L_packets += n*128
    for key in ('Tc','Ts'):
        self._dta_teacher_packets[key] += 128
    record = dict(info, lossstats=info, step=optimizer_step, epoch=(optimizer_step-1)//222+1,
        cumulative_step=cumulative, cumulative_epoch=epoch, first_input=c['first_view'],
        L_forwards=n, student_L_packets=n*128, As_packets=128 if c['As'] else 0,
        student_As_packets=128 if c['As'] else 0, channel_generated_packets=128,
        fresh_As_packets=128, Tc_packets=128, Ts_packets=128,
        clean_teacher_shadow=c['loss']['clean_KD_weight']==0.,
        satellite_teacher_shadow=c['loss']['satellite_KD_weight']==0.,
        first_encoder_input_verified=True,
        ordinary_first_forward_verified=c['first_view']=='ordinary',
        raw_first_forward_verified=c['first_view']=='raw',
        encoder_first_calls=witness['first_seen'],
        extra_encoder_calls=witness['extra_seen'], hook_raw_MSE=0., encoder_first_MSE=0.,
        Tc_same_first_IQ=True, Ts_same_fullAs_IQ=True,
        As_student_same_IQ=True if c['As'] else None,
        As_scene_counts=dict(Counter(r['scene'] for r in records)),
        As_namespaces=sorted({r['namespace'] for r in records}),
        default_head_frozen=True, student_confidence_gate=False, teacher_registered=False)
    self._dta_records.append(record)
    witness.update(raw=None,ordinary=None,first=None,labels=None,As=None,actual_first=None,
                   actual_As=None,phase=None)
    return total-identity_loss, {'dualTeacher/'+key:value for key,value in record.items()
                                if isinstance(value,(int,float))}


hook._accepts_labeled_inputs = True


def _attach_runtime(model, c, channel=None):
    d.validate(c)
    head = model.get_submodule(adapters.CORE).cls_head
    if head.weight.shape != (6,160) or head.scale != 30.:
        raise ValueError('Original default six-class joint160 cosine30 head required')
    original = tuple(name for name,_ in model.named_parameters())
    if c['adapter']['enabled']:
        adapters.attach(model,seed=c['model_seed'])
        adapter_names = tuple(name for name,_ in adapters.adapter_named_parameters(model))
    else:
        model.requires_grad_(False); model.id_backbone.requires_grad_(True)
        adapter_names = ()
    head_names = head_parameter_names(model)
    for name,p in model.named_parameters():
        if name in head_names:
            p.requires_grad_(False)
    model._dta_original_parameter_names = original
    model._dta_adapter_names = adapter_names
    model._dta_head_names = head_names
    model._dta_config, model._dta_parent_meta = c, phase1_lineage(c)
    model._dta_steps, model._dta_L_packets = 0, 0
    model._dta_records = []; model._dta_teacher_packets = dict(Tc=0,Ts=0)
    model.__dict__['_dta_teachers'] = None
    model._dta_channel = ContinuationPractical() if channel is None else channel
    model._dta_witness = dict(raw=None,ordinary=None,first=None,labels=None,As=None,
        actual_first=None,actual_As=None,phase=None,step=0,first_seen=0,extra_seen=0)
    model._source_labeled_feature_hook = MethodType(hook,model)
    encoder = model.id_backbone.encoder
    encoder._counted_features = encoder.features
    encoder._full_train_packets = 0; encoder._full_eval_packets = 0
    encoder._joint160_seen_calls = 0; encoder._joint160_seen_packets = 0
    encoder._joint160_last_shape = None; encoder._dta_witness = model._dta_witness
    encoder.features = MethodType(counted,encoder)
    model._feature_learning_report.update(dualTeacher=dict(phase1_lineage(c)),
        adapter=adapters.adapter_state_report(model) if adapter_names else None,
        original_F_trainable=c['original_F_trainable'], default_head_frozen=True,
        teacher_registered=False, new_U_mechanism=False, head_fit_updates=0)
    return model


def _prepare_F(model,c):
    from experiments.cvs_rdrop_sinc_units import model as af
    from experiments.cvs_source_identity_readout.input_policy import apply
    response = model.id_backbone.encoder.response
    if type(response).__name__ == 'ExcitationResponse':
        model._identity_input_policy = apply(model)
    elif type(response).__name__ != 'NoExplicitCFO':
        raise ValueError('Original reference-response identity path required')
    return af.configure(af.original.old.install(model,
        af.original.d.legacy_config(c['model_seed'])),d.model_config())


def install(model,c,parent):
    """Native structure -> pure F -> strict parent206 -> optional adapter6."""
    d.validate(c)
    model = _prepare_F(model,c)
    if (parent['epoch'] != 20 or parent['a1_ema_successful_updates'] != 4440
            or parent['run_id'] != d.PARENT_RUN or parent['candidate_id'] != d.PARENT_ROW
            or len(parent['model']) != 206 or len(model.state_dict()) != 206):
        raise ValueError('Load the fixed parent F206 before installing feature adapters')
    model.load_state_dict(parent['model'],strict=True)
    return _attach_runtime(model,c)


def build_structure(c,device='cpu'):
    d.validate(c)
    from experiments.cvs_rdrop_sinc_units import model as af
    model = af.configure(af.original.old.build(
        af.original.d.legacy_config(c['model_seed']),'cpu'),d.model_config())
    return _attach_runtime(model,c).to(device)


def source_domain_count(args):
    from experiments.cvs_rdrop_sinc_units.model import source_domain_count as original
    return original(args)


def classifier(model):
    return model.get_submodule(adapters.CORE).cls_head


def check_final_payload(cp,c):
    """Check local/cumulative lineage before strict reconstructed-state loading."""
    if (cp['epoch'] != 10 or cp['a1_ema_successful_updates'] != 2220
            or cp['run_id'] != c['run_id'] or cp['candidate_id'] != c['row_id']
            or cp['phase1_lineage'] != phase1_lineage(c)
            or cp['checkpoint_selection'] != 'final_only'
            or len(cp['model']) != 206+6*int(c['adapter']['enabled'])
            or cp.get('ema_model') is None or set(cp['model']) != set(cp['ema_model'])):
        raise ValueError('New checkpoint local10/cumulative30 identity differs')
    return cp


def own_model(c,device='cpu'):
    d.validate(c)
    out = Path(c['output_root'])
    done,frozen = d.read(out/'completion.json'),d.read(out/'source_frozen.json')
    initial,contract = d.read(out/'initialization.json'),d.read(out/'source_contract.json')
    if (done['config'] != c or done['epoch'] != 30 or done['segment_epoch'] != 10
            or done['parent_epoch'] != 20 or done['optimizer_steps'] != 2220
            or done['cumulative_optimizer_steps'] != 6660 or done['target_access']
            or done['head_fit_updates'] or frozen['config'] != c or frozen['epoch'] != 30
            or frozen['selection'] != 'fixed_FINAL30' or frozen['target_access']
            or initial['scratch_only'] is not False or initial['target_contact']
            or initial['checkpoint_sources'] != c['checkpoint_sources']
            or contract['role_ids'] != d.read(d.SOURCE)['role_ids']):
        raise ValueError('Own FINAL30 continuation/parent/default-head source lineage differs')
    cp = torch.load(out/'final_ssdg.pth',map_location='cpu',weights_only=False)
    check_final_payload(cp,c)
    model = build_structure(c,device)
    model.load_state_dict(cp['model'],strict=True)
    return model.eval().requires_grad_(False)
