"""Only synthetic source roles and models; no target or private truth."""
from contextlib import contextmanager
from copy import deepcopy
import importlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn
from torch.nn import functional as F


def api():
    return importlib.import_module('experiments.pamf_phase3_open_world.source_style_train')


class Mask29(nn.Linear):
    def forward(self,x):
        value=super().forward(x); mask=torch.ones(29,device=x.device); mask[27]=0
        return value*mask


class Reader(nn.Module):
    def __init__(self):
        super().__init__();self.core=nn.Module();self.core.id_backbone=nn.Module();b=self.core.id_backbone
        b.sinc=nn.Module();b.sinc.low_coordinate=nn.Parameter(torch.full((24,1),.01));b.sinc.band_coordinate=nn.Parameter(torch.full((24,1),.02))
        b.time_fuse=nn.Sequential(nn.Conv1d(2,48,1),nn.GroupNorm(6,48),nn.ReLU())
        b.time_down=nn.AvgPool1d(2);b.t2=nn.Module();b.t2.pool=nn.Sequential(nn.Conv1d(48,96,1),nn.AvgPool1d(4))
        b.cls_head=nn.Module();h=b.cls_head;h.weight=nn.Parameter(torch.randn(6,160)*.1);h.scale=30.
        h.base_norm=nn.LayerNorm(160);h.pa_norm=nn.LayerNorm(160);h.gain=nn.Parameter(torch.tensor(.2))
        self.middle=nn.Linear(96,160);self.response=Mask29(512,29);self.response_projection=nn.Linear(29,160)
        self.response_gain=nn.Parameter(torch.tensor(.2));self.sequence_pool=nn.Linear(512,160)
        self.padding=nn.Parameter(torch.zeros(232993-sum(p.numel() for p in self.parameters())))
        self.register_buffer('unchanged',torch.tensor(1.))

    def features(self,x):
        b=self.core.id_backbone;h=b.cls_head;flat=x.flatten(1)
        z=self.middle(b.t2.pool(torch.tanh(b.time_down(b.time_fuse(x)))).mean(2))
        z=z+(b.sinc.low_coordinate.sum()+b.sinc.band_coordinate.sum())*x[:,0].mean(1)[:,None]
        return h.base_norm(z)+h.gain.tanh()*h.pa_norm(z.square())+self.response_gain.tanh()*self.response_projection(self.response(flat))+self.sequence_pool(flat)

    def classify_features(self,z):
        return 30*F.linear(F.normalize(z,dim=1,eps=.0001),F.normalize(self.core.id_backbone.cls_head.weight,dim=1,eps=.0001))


@contextmanager
def fixture(root,mp):
    from test_pamf_phase3_native_identity_train import fixture as old
    with old(root,mp) as (identity,c,pm,b,parent,pcd,events):
        from experiments.pamf_phase3_open_world import star_signal_open_model as signal
        torch.manual_seed(9);parent.id_backbone.encoder=Reader().eval().requires_grad_(False);b['reader']=parent.id_backbone.encoder
        pure=signal.SignalOpen(signal.HeadlessSignal(parent.id_backbone.encoder)).eval().requires_grad_(False)
        pure_identity=dict(run_id='20261030-phase3-star-signal-open-s2026092701-r03',arm='E_SIGNAL',signal_parameters=334721,target_reads=0,projection_updates=2220)
        pm['model_identities']['E0']=pure_identity;b['model']=pure;b['model_identity']=pure_identity
        m=api();mp.setattr(m,'STEPS',2);mp.setattr(m,'COUNTS',dict(L=256,OE=150))
        mp.setattr(m.body.base,'load_trained_model',lambda *args:(deepcopy(pure),pure_identity))
        c=dict(c,run_id=m.RUN_ID,output_root=str(root/'style/train'))
        rng=np.random.default_rng(4);banks={};roles={}
        for role,n in m.COUNTS.items():
            iq=rng.normal(size=(n,2,256)).astype(np.float32);domain=np.arange(n)%15
            meta=dict(rx_i=np.asarray([1,3,4,6,8])[domain//3].astype(np.int64),day_i=(domain%3+1).astype(np.int64),domain=domain.astype(np.int64))
            if role=='L':meta['y']=(np.arange(n)//15%6).astype(np.int64)
            with torch.no_grad():z=pure(torch.tensor(iq.tolist()));teacher=parent.id_backbone.encoder.classify_features(parent.id_backbone.encoder.features(torch.tensor(iq.tolist())))
            bank=dict(iq=iq,ids=np.asarray([role+str(i) for i in range(n)]),metadata=meta,parent_z=np.asarray(z.tolist(),np.float32))
            if role=='OE':bank['teacher_u']=np.asarray(z.tolist(),np.float32)
            banks[role]=bank;roles[role]=dict(teacher_logits=np.asarray(teacher.tolist(),np.float32)) if role=='L' else {}
        manifest=dict(parent_F0_identity=pm['parent_F0_identity'],dura_identity=pm['dura_identity'],source_only=True)
        def source(*args):
            assert events==['checkpoint'];events.append('L_OE');return banks,manifest,roles
        mp.setattr(m,'_source_cache',source)
        mp.setattr(m.body.base,'close_role',lambda *args:None)
        yield m,c,pm,b,parent,banks,events


def test_moments_detach_OE_exact_passthrough_content_gradient_and_finally_cleanup():
    m=api();core=m.core;torch.manual_seed(2)
    h=torch.randn(8,3,7,requires_grad=True);partner=torch.tensor([2,3,0,1,4,5,6,7]);valid=torch.tensor([True]*4+[False]*4)
    lam=torch.full((8,1,1),.3,requires_grad=True)
    mixed=core.mix_moments(h,partner,valid,lam,1e-6)
    assert torch.equal(mixed[4:],h[4:])
    mixed[0].sum().backward();assert h.grad[0].norm()>0 and h.grad[2].abs().sum()==0 and lam.grad is None
    r=Reader().eval();plan=core.style_plan(2,4);before=[len(r.get_submodule(path)._forward_hooks) for path in core.SITES]
    with pytest.raises(RuntimeError,match='intentional'):
        with core.style_context(r,plan,np.random.default_rng(1),force=True):
            raise RuntimeError('intentional')
    assert before==[len(r.get_submodule(path)._forward_hooks) for path in core.SITES]
    with core.style_context(r,plan,np.random.default_rng(1),force=True) as observed:r.features(torch.randn(8,2,256))
    assert all(v['invocations']==1 and v['OE_exact_passthrough'] for v in observed.values())
    assert before==[len(r.get_submodule(path)._forward_hooks) for path in core.SITES]


def test_source_sampler_sameTX_differentRX_and_full444_matched_trajectory(tmp_path,monkeypatch):
    with fixture(tmp_path,monkeypatch) as (m,c,pm,b,parent,banks,events):
        a=m.core.SourceSampler(banks['L'],banks['OE']);b=m.core.SourceSampler(banks['L'],banks['OE'])
        for step in range(1,445):
            x=a.sample(step);y=b.sample(step)
            assert all(np.array_equal(x[k],y[k]) for k in ('A','B','OE'))
            meta=banks['L']['metadata'];assert np.array_equal(meta['y'][x['A']],meta['y'][x['B']])
            assert np.all(meta['rx_i'][x['A']]!=meta['rx_i'][x['B']])
        bad=deepcopy(banks['L']);bad['metadata']['rx_i'][:]=1
        with pytest.raises(ValueError,match='partner'):m.core.SourceSampler(bad,banks['OE'])


def test_two_source_students_joint_body_head_projection_strict_load(tmp_path,monkeypatch,subtests):
    with fixture(tmp_path,monkeypatch) as (m,c,pm,b,parent,banks,events):
        backbone=parent.id_backbone.encoder.core.id_backbone
        assert list(backbone.time_down.parameters())==[] and sum(p.numel() for p in backbone.time_fuse.parameters())>0
        monkeypatch.setattr(torch,'from_numpy',lambda *a:pytest.fail('No NumPy ABI bridge'));monkeypatch.setattr(torch.Tensor,'numpy',lambda *a:pytest.fail('No NumPy ABI bridge'))
        before=m.identity._snapshot(parent);done=m.train(c,'cpu');out=Path(c['output_root'])
        assert events==['checkpoint','L_OE'] and done['source_only'] and done['total_optimizer_updates']==4 and done['target_U_reads']==0
        assert done['all_student_bodies_changed'] and done['all_student_heads_changed'] and done['all_student_projections_changed']
        assert done['real_parent_signal_forward_rows']==406 and done['real_student_forward_rows']==1280
        assert done['synthetic_student_forward_rows']==8
        smoke=m._read(out/'smoke.json')['native_open_style']
        assert smoke['gradient_norms']['shallow']>0
        assert all(v['shape'][0]==4 and v['valid_rows']==2 and v['OE_exact_passthrough'] for v in smoke['style_sites'].values())
        assert done['G_forward_rows']==done['real_F_forward_rows']==done['real_PCD_forward_rows']==0
        trace={a:[json.loads(v) for v in (out/a/'steps.jsonl').read_text().splitlines()] for a in m.ARMS}
        for source,style in zip(trace['SOURCE_NATIVE'],trace['STYLE_NATIVE']):
            assert all(source[k]==style[k] for k in ('A_indices','B_indices','OE_indices'))
            assert source['style_sites']=={} and all(v['invocations']==1 and v['OE_exact_passthrough'] for v in style['style_sites'].values())
            for row in (source,style):
                assert row['gradient_norms']['head']>0 and row['gradient_norms']['projection']>0 and row['gradient_norms']['shallow']>0 and row['gradient_norms']['Sinc']==0
                assert abs(row['loss_total']-sum(row[k] for k in ('crossset_metric','OE_KL','CE','KD'))-10*row['source_anchor'])<3e-5
        m.identity._unchanged(parent,before,'Parent')
        monkeypatch.setattr(m,'_source_cache',lambda *a:pytest.fail('Loader cannot read source IQ'))
        monkeypatch.setattr(m,'_backend',lambda *a:pytest.fail('Loader cannot load G/PCD'))
        for arm in m.ARMS:
            model,i=m.load_trained_model(c,arm,'cpu');assert i==done['identities'][arm] and i['signal_parameters']==335681 and i['source_only']
            assert sum(p.numel() for p in model.parameters())==335681 and all(not p.requires_grad for p in model.parameters())
            z,logits=model(torch.randn(3,2,256));assert z.shape==(3,160) and logits.shape==(3,6)
        cp=out/'STYLE_NATIVE/final.pt';saved=cp.read_bytes()
        for defect in ('missing_head','dtype','Sinc','buffer','ancestor','projection_unchanged'):
            with subtests.test(defect=defect):
                p=torch.load(cp,weights_only=False)
                if defect=='missing_head':p['network'].pop('reader.'+m.identity.READER_HEAD)
                elif defect=='dtype':p['network']['reader.'+m.identity.READER_HEAD]=p['network']['reader.'+m.identity.READER_HEAD].double()
                elif defect=='Sinc':p['network']['reader.'+m.identity.READER_SINC+'low_coordinate']+=1
                elif defect=='buffer':p['network']['reader.unchanged']+=1
                elif defect=='ancestor':p['parent_signal_identity']={'bad':True}
                else:
                    for k,v in b['model'].projection.state_dict().items():p['network']['projection.'+k]=v.clone()
                torch.save(p,cp)
                with pytest.raises(ValueError):m.load_trained_model(c,'STYLE_NATIVE','cpu')
                cp.write_bytes(saved)
        with pytest.raises(FileExistsError):m.train(c,'cpu')


def test_private_extra_inputs_wrong_source_and_failed_smoke_stop_before_values(tmp_path,monkeypatch,subtests):
    with fixture(tmp_path,monkeypatch) as (m,c,pm,b,parent,banks,events):
        for change in (dict(truth='private'),dict(U_root='forbidden'),dict(K=3),dict(steps=1)):
            with subtests.test(change=change),pytest.raises(ValueError):m.validate_config(dict(c,**change))
        monkeypatch.setattr(m,'_smoke',lambda *a:(_ for _ in ()).throw(ValueError('probe failed')))
        with pytest.raises(ValueError,match='probe failed'):m.train(c,'cpu')
        assert events==['checkpoint'] and not Path(c['output_root']).exists()
        b['model_identity']=dict(b['model_identity'],source_only=False)
        with pytest.raises(ValueError,match='pure.source'):m.train(c,'cpu')
