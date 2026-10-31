"""Bounded synthetic checks; no real IQ, checkpoints, or private truth."""
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
    return importlib.import_module('experiments.pamf_phase3_open_world.native_identity_train')


class Reader(nn.Module):
    def __init__(self):
        super().__init__()
        self.core = nn.Module(); self.core.id_backbone = nn.Module()
        b = self.core.id_backbone; b.sinc = nn.Module()
        b.sinc.low_coordinate = nn.Parameter(torch.full((24, 1), .01))
        b.sinc.band_coordinate = nn.Parameter(torch.full((24, 1), .02))
        b.t1 = nn.Linear(512, 32); b.t2 = nn.Linear(32, 160)
        b.cls_head = nn.Module(); b.cls_head.weight = nn.Parameter(torch.randn(6, 160) * .01)
        b.cls_head.scale = 30.
        with torch.no_grad():
            b.cls_head.weight[0].zero_(); b.cls_head.weight[0, 0] = 1.
        self.bn = nn.BatchNorm1d(160); self.dropout = nn.Dropout(.7)
        self.padding = nn.Parameter(torch.zeros(232993 - sum(p.numel() for p in self.parameters())))

    def features(self, x):
        b = self.core.id_backbone
        z = b.t2(torch.tanh(b.t1(x.flatten(1))))
        z = z + (b.sinc.low_coordinate.sum() + b.sinc.band_coordinate.sum()) * x[:, 0].mean(1)[:, None]
        offset = torch.zeros(160, device=x.device); offset[0] = 8.
        return self.dropout(self.bn(z + offset))

    def classify_features(self, z):
        return 30. * F.linear(F.normalize(z, dim=1, eps=.0001), F.normalize(self.core.id_backbone.cls_head.weight, dim=1, eps=.0001))


class Wrapper(nn.Module):
    def __init__(self):
        super().__init__(); self.id_backbone = nn.Module(); self.id_backbone.encoder = Reader()
        self.unused = nn.Parameter(torch.ones(667729))
        self.register_buffer('outside_buffer', torch.ones(2))


class PCD(nn.Module):
    def __init__(self):
        super().__init__(); self.linear = nn.Linear(512, 160)

    def forward(self, x):
        return F.normalize(self.linear(x.flatten(1)), dim=1)


@contextmanager
def fixture(root, mp):
    m = api(); old_threads = torch.get_num_threads(); torch.set_num_threads(1)
    try:
        mp.setattr(m, 'STEPS', 2); mp.setattr(m, 'L_COUNT', 256)
        torch.manual_seed(6); parent = Wrapper().eval().requires_grad_(False)
        native = dict(source_output=str(root/'original-F0'), source_contract=str(root/'contract'),
                      base_release=str(root/'base'), model_seed=2026092701)
        source = dict(native_config=native, IQ_root=str(root/'IQ'), output_root=str(root/'r03'))
        bodyref = dict(reference_config=dict(train_config=source))
        oldtrain = dict(output_root=str(root/'old-PCD'), U_public_root=str(root/'U/public'),
                        U_capsule_id='p3fresh-test', U_split_id='star-split')
        reference = dict(code_commit=m.REFERENCE_CODE, output_root=str(root/'old-predict'),
                         train_config=oldtrain, source_control_config=bodyref,
                         Q_conditions={k: dict(public_root=str(root/k/'public')) for k in ('clean_DURA', 'star_DURA')})
        c = dict(run_id=m.RUN_ID, code_commit='d'*40, reference_config=reference,
                 output_root=str(root/'native/train'), steps=2, lr=.0001)
        pm = dict(parent_F0_identity=dict(native), dura_identity=dict(deployment_id='dura_v1'),
                  loaded_dura_identity=dict(deployment_id='dura_v1'),
                  model_identities=dict(TARGET_PCD=dict(arm='TARGET_PCD', target_U_reads=576)))
        parent._phase3_parent_identity = dict(native)
        parent._phase3_source_handle = dict(classes=m.native.CLASSES, source_rxs=m.native.RX,
            source_days=[1,2,3], split_seed=392005, counts=dict(L_s=6300,U_s=56700,V=27000), source_contract=native['source_contract'])
        rng = np.random.default_rng(7); iq = rng.normal(size=(256,2,256)).astype(np.float32)
        with torch.no_grad(): teacher = parent.id_backbone.encoder.classify_features(parent.id_backbone.encoder.features(torch.tensor(iq.tolist())))
        L = dict(ids=np.asarray(['L'+str(i) for i in range(256)]), restored=iq,
                 metadata=dict(y=(np.arange(256)%6).astype(np.int64)),
                 teacher_logits=np.asarray(teacher.tolist(), np.float32),
                 manifest=dict(count=256,parent_F0_identity=pm['parent_F0_identity'],dura_identity=pm['dura_identity']))
        pcd = PCD().eval().requires_grad_(False); r03 = PCD().eval().requires_grad_(False); G = nn.Linear(1,1).eval().requires_grad_(False)
        b = dict(reader=parent.id_backbone.encoder, model=r03, dura=(G, lambda x:(x.clone(),)),
                 reader_identity=pm['parent_F0_identity'], dura_identity=pm['loaded_dura_identity'])
        events = []; state = dict(test=np.ones(2)); tau = np.zeros((2,6), np.float64)
        mp.setattr(m.reg.fresh, 'validate_config', lambda x:x)
        mp.setattr(m, '_reference', lambda config:(pm,state,tau,{}))
        mp.setattr(m, '_backend', lambda config,device:b)
        mp.setattr(m, '_load_PCD', lambda config,device:(pcd,pm['model_identities']['TARGET_PCD']))
        mp.setattr(m.native, 'make_trainable_model', lambda config,device:deepcopy(parent).to(device))
        mp.setattr(m, '_native_config', lambda config:native)
        def loadL(config, reference):
            assert events==['checkpoint']; events.append('L'); return L
        mp.setattr(m, '_L', loadL)
        mp.setattr(m.cache, 'close_role', lambda x:None)
        def checkpoint(*args):
            events.append('checkpoint')
            return dict(status='PASS', actual_F0_r03_DURA_PCD_checkpoints=True,
                        synthetic_F_forward_rows=2,synthetic_signal_forward_rows=4,G_synthetic_forward_rows=2,real_data_reads=0,optimizer_updates=0)
        mp.setattr(m.reg, '_smoke', checkpoint)
        oldroot=Path(oldtrain['output_root']);oldroot.mkdir()
        manifest=dict(count=576,capsule_id='p3fresh-test',split_id='star-split')
        (oldroot/'complete.json').write_text(json.dumps(dict(status=m.fresh.STATUS,code_commit=m.REFERENCE_CODE,
            config=oldtrain,source_only=False,target_U_reads=576,original_models_unchanged=True,U_manifest=manifest)),encoding='utf-8')
        Uids=np.asarray(['p3fresh-'+format(i,'024x') for i in range(576)])
        views=rng.normal(size=(3,576,2,256)).astype(np.float32)
        np.savez(oldroot/'U_features.npz',ids=Uids,views=views,parent_z=np.zeros((576,160),np.float32),neighbors=np.zeros(576,np.int64))
        def public(config):
            assert Path(c['output_root'],'SOURCE_ID/complete.json').is_file()
            events.append('U');return manifest,Uids,np.zeros((576,2,256),np.float32)
        mp.setattr(m, '_public_U', public)
        mp.setattr(m.reg.fresh, '_raw', lambda *args:np.ones((len(args[1]),2),np.float64))
        yield m,c,pm,b,parent,pcd,events
    finally:
        torch.set_num_threads(old_threads)


def test_CE_KD_and_masked_PL_are_exact_and_teacher_detached():
    m=api();torch.manual_seed(2)
    logits=torch.randn(5,6,requires_grad=True);teacher=torch.randn(5,6,requires_grad=True);y=torch.arange(5)
    r=m.source_loss(logits,y,teacher)
    expected=F.cross_entropy(logits,y)+4*F.kl_div(F.log_softmax(logits/2,dim=1),F.softmax(teacher.detach()/2,dim=1),reduction='batchmean')
    torch.testing.assert_close(r['loss'],expected);r['loss'].backward();assert teacher.grad is None and logits.grad.norm()>0
    a=torch.randn(4,6,requires_grad=True);b=torch.randn(4,6,requires_grad=True);pseudo=torch.tensor([0,1,2,3]);mask=torch.tensor([True,False,True,False])
    pl=m.pseudo_loss(a,b,pseudo,mask)
    torch.testing.assert_close(pl,(.5*(F.cross_entropy(a,pseudo,reduction='none')+F.cross_entropy(b,pseudo,reduction='none'))*mask).sum()/4)
    pl.backward();assert a.grad[1].abs().sum()==0 and b.grad[3].abs().sum()==0
    assert m.pseudo_loss(a,b,pseudo,torch.zeros(4,dtype=torch.bool)).item()==0
    logits=np.zeros((4,6),np.float32);logits[:,0]=[20,20,0,20]
    mask,labels,_=m.fixed_mask(logits,np.asarray([1.,-1.,1.,0.]))
    np.testing.assert_array_equal(mask,[True,False,False,True]);np.testing.assert_array_equal(labels,np.zeros(4,np.int64))


def test_full_native_reader_two_independent_arms_and_strict_load(tmp_path,monkeypatch,subtests):
    with fixture(tmp_path,monkeypatch) as (m,c,pm,b,parent,pcd,events):
        monkeypatch.setattr(torch,'from_numpy',lambda *args:pytest.fail('No NumPy ABI bridge'))
        monkeypatch.setattr(torch.Tensor,'numpy',lambda *args:pytest.fail('No NumPy ABI bridge'))
        before={k:v.detach().clone() for k,v in parent.state_dict().items()}
        done=m.train(c,'cpu');out=Path(c['output_root'])
        assert events==['checkpoint','L','U']
        assert done['SOURCE_completed_before_U_open'] and done['source_sampling_matched'] and done['total_optimizer_updates']==4
        assert done['target_Q_reads']==done['private_truth_reads']==done['U_truth_reads']==done['OE_reads']==0
        assert done['G_forward_rows']==done['real_r03_forward_rows']==0 and done['real_F_forward_rows']==576
        assert done['real_PCD_forward_rows']==576 and done['source_L_reads']==256
        assert done['all_student_heads_changed'] and done['all_student_bodies_changed'] and done['all_student_Sinc_unchanged']
        traces={a:[json.loads(x) for x in (out/a/'steps.jsonl').read_text().splitlines()] for a in m.ARMS}
        for s,t in zip(traces['SOURCE_ID'],traces['TARGET_ID']):
            assert s['L_indices']==t['L_indices'] and not s['U_indices'] and len(t['U_indices'])==64
            for row in (s,t):
                assert row['gradient_norms']['head']>0 and row['gradient_norms']['shallow']>0 and row['gradient_norms']['middle']>0
                assert row['gradient_norms']['Sinc']==0 and row['source_indices_equal']
                assert abs(row['loss_total']-row['CE']-row['KD']-row['PL'])<2e-5
        assert any(r['U_accepted']>0 for r in traces['TARGET_ID'])
        assert done['real_student_forward_rows']==sum(r['student_forward_rows'] for rows in traces.values() for r in rows)
        assert all(torch.equal(v,parent.state_dict()[k]) for k,v in before.items()) and all(p.grad is None for p in parent.parameters())
        assert all(p.grad is None for p in pcd.parameters())
        monkeypatch.setattr(m,'_L',lambda *args:pytest.fail('Reload must not read L'))
        monkeypatch.setattr(m,'_U',lambda *args:pytest.fail('Reload must not read U'))
        monkeypatch.setattr(m,'_backend',lambda *args:pytest.fail('Reload must not load r03/G'))
        for arm in m.ARMS:
            model,identity=m.load_trained_model(c,arm,'cpu')
            assert identity==done['identities'][arm] and identity['reader_parameters']==232993
            assert identity['identity_weight_source_only']==(arm=='SOURCE_ID') and identity['system_PCD_has_target_U_ancestor']
            assert not any(p.requires_grad for p in model.parameters()) and m.native_logits(model,torch.randn(2,2,256)).shape==(2,6)
        cp=out/'TARGET_ID/final.pt';saved=cp.read_bytes()
        for defect in ('head_missing','dtype','shape','nan','Sinc','buffer','ancestor','head_unchanged'):
            with subtests.test(defect=defect):
                payload=torch.load(cp,weights_only=False);network=payload['reader_state']
                if defect=='head_missing':network.pop(m.READER_HEAD)
                elif defect=='dtype':network[m.READER_HEAD]=network[m.READER_HEAD].double()
                elif defect=='shape':network[m.READER_HEAD]=network[m.READER_HEAD][:-1]
                elif defect=='nan':network[m.READER_HEAD][0,0]=torch.nan
                elif defect=='Sinc':network[m.READER_SINC+'low_coordinate']+=1
                elif defect=='buffer':network['bn.running_mean']+=1
                elif defect=='head_unchanged':network[m.READER_HEAD]=parent.id_backbone.encoder.state_dict()[m.READER_HEAD].clone()
                else:payload['parent_F0_identity']={'wrong':True}
                torch.save(payload,cp)
                with pytest.raises(ValueError):m.load_trained_model(c,'TARGET_ID','cpu')
                cp.write_bytes(saved)
        with pytest.raises(FileExistsError):m.train(c,'cpu')


def test_empty_PL_mask_skips_strong_forwards_without_changing_SOURCE_match(tmp_path,monkeypatch):
    with fixture(tmp_path,monkeypatch) as (m,c,pm,b,parent,pcd,events):
        teacher=m._teacher
        def empty(*args):
            result=teacher(*args);result['mask'][:]=False;return result
        monkeypatch.setattr(m,'_teacher',empty)
        done=m.train(c,'cpu');out=Path(c['output_root'])
        for row in (json.loads(line) for line in (out/'TARGET_ID/steps.jsonl').read_text().splitlines()):
            assert row['PL']==row['U_accepted']==row['student_U_forward_rows']==0 and row['strong_forward_skipped']
            assert row['student_forward_rows']==128
        assert done['real_student_forward_rows']==2*128*2
        left=torch.load(out/'SOURCE_ID/final.pt',weights_only=False)['reader_state']
        right=torch.load(out/'TARGET_ID/final.pt',weights_only=False)['reader_state']
        assert all(torch.equal(left[k],right[k]) for k in left)


def test_illegal_fields_ancestry_failed_smoke_and_partial_source_preserved(tmp_path,monkeypatch,subtests):
    with fixture(tmp_path,monkeypatch) as (m,c,pm,b,parent,pcd,events):
        for change in (dict(truth='private'),dict(Q_root='forbidden'),dict(unknown_count=4),dict(steps=1),
                       dict(output_root=c['reference_config']['train_config']['output_root']),
                       dict(reference_config=dict(c['reference_config'],code_commit='0'*40))):
            with subtests.test(change=change),pytest.raises(ValueError):m.validate_config(dict(c,**change))
        smoke=m._student_smoke
        monkeypatch.setattr(m,'_student_smoke',lambda *args:(_ for _ in ()).throw(ValueError('probe failed')))
        with pytest.raises(ValueError,match='probe failed'):m.train(c,'cpu')
        assert events==['checkpoint'] and not Path(c['output_root']).exists()
        monkeypatch.setattr(m,'_student_smoke',smoke);events.clear()
        monkeypatch.setattr(m,'_loop',lambda *args:(_ for _ in ()).throw(ValueError('source interrupted')))
        monkeypatch.setattr(m,'_public_U',lambda *args:pytest.fail('No U before source completion'))
        with pytest.raises(ValueError,match='source interrupted'):m.train(c,'cpu')
        assert not Path(c['output_root'],'SOURCE_ID/complete.json').exists() and not Path(c['output_root'],'complete.json').exists()
        with pytest.raises(ValueError):m.load_trained_model(c,'SOURCE_ID','cpu')
