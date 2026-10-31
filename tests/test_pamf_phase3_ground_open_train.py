"""Ground-clean source training, ordinary student PL, and strict reader states."""
from contextlib import contextmanager
from copy import deepcopy
import importlib
import json
from pathlib import Path
import numpy as np
import pytest
import torch
from torch.nn import functional as F


def api():return importlib.import_module('experiments.pamf_phase3_open_world.ground_open_train')


def test_PL_uses_detached_student_confidence_and_full_batch_denominator():
    m=api();prob=torch.tensor([.96,.94,.951,.949],dtype=torch.float32)
    values=torch.zeros(4,6);values[:,0]=torch.log(5*prob/(1-prob));logits=values.requires_grad_()
    loss,info=m.pseudo_loss(logits)
    expected=F.cross_entropy(logits,torch.zeros(4,dtype=torch.long),reduction='none')
    torch.testing.assert_close(loss,(expected[0]+expected[2])/4)
    assert info['accepted']==2 and info['count']==4 and info['pseudo_histogram']==[2,0,0,0,0,0]
    assert info['acceptance_rate']==.5 and info['threshold']==.95
    loss.backward();assert logits.grad[0].norm()>0 and logits.grad[2].norm()>0
    assert logits.grad[1].abs().sum()==logits.grad[3].abs().sum()==0
    low=torch.zeros(4,6,requires_grad=True);zero,empty=m.pseudo_loss(low)
    zero.backward();assert zero.item()==0 and empty['accepted']==0 and low.grad.abs().sum()==0


def public_meta(n=150):
    d=np.arange(n,dtype=np.int64)%15
    return dict(ids=np.asarray(['u'+str(i) for i in range(n)]),domain=d,rx_i=np.asarray([1,3,4,6,8],np.int64)[d//3],day_i=d%3+1)


def test_Us_sampler_only_public_domains_balances128_and_rejects_labels(subtests):
    m=api();meta=public_meta();a=m.UsSampler(meta);b=m.UsSampler(meta)
    for step in (1,2):
        indices=a.sample(step);np.testing.assert_array_equal(indices,b.sample(step))
        assert indices.dtype==np.int64 and indices.shape==(128,)
        assert np.all(np.bincount(meta['domain'][indices],minlength=15)>=8)
    for field in ('y','tx','sig_i','base_index'):
        with subtests.test(field=field),pytest.raises(ValueError):m.UsSampler(dict(meta,**{field:np.zeros(150,np.int64)}))


def test_RXPAIR_source_objective_is_shared_and_OE_never_uses_true_labels():
    m=api();torch.manual_seed(3);L=torch.randn(4,6,requires_grad=True);OE=torch.randn(3,6,requires_grad=True);teacher=torch.randn(4,6,requires_grad=True)
    y=torch.tensor([0,1,2,3]);ld=torch.tensor([0,1,0,1]);od=torch.tensor([0,1,0])
    result=m.source_loss(L,y,teacher,ld,OE,od)
    ce=F.cross_entropy(L,y);kd=4*F.kl_div(F.log_softmax(L/2,dim=1),F.softmax(teacher.detach()/2,dim=1),reduction='batchmean')
    mask=ld[:,None]==od[None];rank=F.softplus((torch.logsumexp(OE,1)[None]-torch.logsumexp(L,1)[:,None])[mask]).mean()
    torch.testing.assert_close(result['total'],ce+kd+rank);assert result['uniform'].item()==0 and result['pair_count']==6
    result['total'].backward();assert teacher.grad is None and L.grad.norm()>0 and OE.grad.norm()>0


@contextmanager
def fixture(root,mp):
    from test_pamf_phase3_native_identity_train import fixture as inherited
    with inherited(root,mp) as (native,oldc,pm,b,parent,pcd,events):
        m=api();data=m._data();mp.setattr(m,'STEPS',2);mp.setattr(m,'COUNTS',dict(L=270,OE=90));mp.setattr(m,'US_COUNT',150)
        mp.setattr(data,'COUNTS',dict(L=270,OE=90,Us=150,V=180,novel=45))
        r03=dict(arm='E_SIGNAL',source_only=True);pm['model_identities']['E0']=r03;b['model_identity']=r03
        c=dict(oldc,run_id=m.RUN_ID,data_root=str(root/'ground_data'),output_root=str(root/'ground_train'))
        native_config=native._source_config(c)['native_config'];manifests={};rng=np.random.default_rng(36)
        data_identity=dict(parent_F0_identity=pm['parent_F0_identity'],dura_identity=pm['dura_identity'],loaded_dura_identity=pm['loaded_dura_identity'])
        for role,n in [('L',270),('OE',90),('Us',150)]:
            ids=np.asarray([role+'_'+str(i) for i in range(n)]);domain=np.arange(n,dtype=np.int64)%15
            meta=dict(ids=ids,rx_i=np.asarray([1,3,4,6,8],np.int64)[domain//3],day_i=domain%3+1,domain=domain)
            if role=='L':
                domain=np.tile(np.repeat(np.arange(15,dtype=np.int64),3),6)
                meta.update(y=np.repeat(np.arange(6,dtype=np.int64),45),rx_i=np.asarray([1,3,4,6,8],np.int64)[domain//3],day_i=domain%3+1,domain=domain)
            clean=rng.normal(size=(n,2,256)).astype(np.float32)
            with torch.no_grad():
                iq=b['dura'][1](torch.tensor(clean.tolist()))[0];teacher=b['reader'].classify_features(b['reader'].features(iq)) if role=='L' else None
            manifests[role]=data._export(Path(c['data_root'])/'roles'/role,role,np.asarray(iq.tolist(),np.float32),ids,meta,c,data_identity,teacher=teacher)
        source=dict(reader_config={k:native_config[k] for k in ('source_output','base_release','model_seed')},source_contract=native_config['source_contract'])
        marker=dict(schema=data.SCHEMA,status=data.STATUS,run_id=m.RUN_ID,code_commit=c['code_commit'],source_only=True,phase3_data_status='VALIDATED_ONCE',
            capsule_id=data.CAPSULE_ID,split_id=data.SPLIT_ID,source_config=source,roles=manifests,counts=dict(data.COUNTS),**data_identity,test_only_synthetic_data=True)
        data._write(Path(c['data_root'])/'complete.json',marker);events.clear();batches=[]
        initial=m._initial_model
        def construct(*args):
            student,ancestor=initial(*args);student.register_forward_pre_hook(lambda model,args:batches.append(len(args[0])));return student,ancestor
        mp.setattr(m,'_initial_model',construct)
        actual_load=data.load_role
        def load(path,role):
            assert role in ('L','OE','Us') and Path(c['output_root'],'smoke.json').is_file()
            if role=='Us':assert Path(c['output_root'],'rows/SUP/complete.json').is_file() and not Path(c['output_root'],'rows/SSL/complete.json').exists()
            events.append(role);return actual_load(path,role)
        mp.setattr(data,'load_role',load);mp.setattr(data,'load_query',lambda *a:pytest.fail('Training cannot read sourceQ/CAL/target queries'))
        yield m,c,pm,b,parent,events,batches


def test_actual_two_arms_smoke_source_before_Us_matched_streams_and_strict_CP(tmp_path,monkeypatch,subtests):
    with fixture(tmp_path,monkeypatch) as (m,c,pm,b,parent,events,batches):
        monkeypatch.setattr(torch,'from_numpy',lambda *a:pytest.fail('Use verified list bridge'))
        monkeypatch.setattr(torch.Tensor,'numpy',lambda *a:pytest.fail('Use verified list bridge'))
        initial=m._snapshot(parent);done=m.train(c,'cpu');out=Path(c['output_root'])
        assert events==['L','OE','Us'] and batches==[2,4,6,256,256,384,384]
        assert done['total_optimizer_updates']==4 and done['real_student_forward_rows']==1280
        assert done['real_student_forward_rows_by_arm']==dict(SUP=512,SSL=768) and done['source_L_OE_sampling_matched']
        assert done['source_U_GT_reads']==done['CAL_reads']==done['source_Q_reads']==done['target_U_reads']==0
        assert done['real_F_teacher_forward_rows']==done['G_forward_rows']==done['real_parent_signal_forward_rows']==0
        assert all(torch.equal(value,parent.state_dict()[key]) for key,value in initial.items())
        traces={arm:[json.loads(x) for x in (out/'rows'/arm/'steps.jsonl').read_text().splitlines()] for arm in m.ARMS}
        assert all(a['L_indices']==b['L_indices'] and a['OE_indices']==b['OE_indices'] for a,b in zip(*traces.values()))
        assert all(r['Us_indices']==[] and r['PL']==r['PL_count']==0 for r in traces['SUP'])
        assert all(len(r['Us_indices'])==128 and r['PL_denominator']==128 and sum(r['pseudo_histogram'])==r['PL_accepted'] for r in traces['SSL'])
        assert all(r['gradient_norms']['Sinc']==0 and all(r['gradient_norms'][k]>0 for k in ('head','body','shallow','middle')) for t in traces.values() for r in t)
        assert all(abs(r['total']-(r['CE']+r['KD']+r['OE']+r['PL']))<1e-6 for t in traces.values() for r in t)
        assert done['identities']['SUP']['source_U_s_reads']==0 and done['identities']['SSL']['source_U_s_reads']==150
        monkeypatch.setattr(m._data(),'load_role',lambda *a:pytest.fail('Strict loading must not read role arrays'))
        for arm in m.ARMS:
            model,i=m.load_trained_model(c,arm,'cpu')
            assert i['reader_parameters']==232993 and i['trainable_parameters']==232945 and i['identity_head_updates']==2
            assert i['ground_clean_training'] and i['projection_updates']==0 and not model.training and not any(p.requires_grad for p in model.parameters())
        path=out/'rows/SSL/final.pt';payload=torch.load(path,map_location='cpu',weights_only=False)
        for defect in ('parent','head','Sinc','buffer','data_identity'):
            with subtests.test(defect=defect):
                bad=deepcopy(payload)
                if defect=='parent':bad['original_parent_identity']={'wrong':'F0'}
                elif defect=='data_identity':bad['ground_data_identity']=dict(bad['ground_data_identity'],capsule_id='G10')
                else:
                    key=m.core.HEAD_WEIGHT if defect=='head' else next(k for k in bad['network'] if k.startswith(m.core.SINC_PREFIX)) if defect=='Sinc' else next(k for k in bad['network'] if k.endswith('running_mean'))
                    if defect=='head':del bad['network'][key]
                    else:bad['network'][key]=bad['network'][key]+1
                torch.save(bad,path)
                with pytest.raises(ValueError):m.load_trained_model(c,'SSL','cpu')
        torch.save(payload,path)
        with pytest.raises(FileExistsError):m.train(c,'cpu')


def test_illegal_roles_ancestor_and_failed_smoke_never_open_source_IQ(tmp_path,monkeypatch,subtests):
    with fixture(tmp_path,monkeypatch) as (m,c,pm,b,parent,events,batches):
        for patch in (dict(CAL_root='private'),dict(U_truth='private'),dict(steps=1),dict(data_root=c['output_root']),dict(output_root=c['data_root'])):
            with subtests.test(config=next(iter(patch))),pytest.raises(ValueError):m.validate_config(dict(c,**patch))
        role=Path(c['data_root'])/'roles/L/manifest.json';before=role.read_bytes();bad=json.loads(before);bad['input_path']='G10_restored_P1'
        role.write_text(json.dumps(bad),encoding='utf-8')
        with subtests.test(old_G10_teacher_route=True),pytest.raises(ValueError):m.train(c,'cpu')
        assert events==[] and not Path(c['output_root']).exists();role.write_bytes(before)
        smoke=m._smoke;monkeypatch.setattr(m,'_smoke',lambda *a:(_ for _ in ()).throw(ValueError('actual smoke failed')))
        with pytest.raises(ValueError,match='actual smoke failed'):m.train(c,'cpu')
        assert events==[] and not Path(c['output_root']).exists()
        monkeypatch.setattr(m,'_smoke',smoke);b['reader_identity']={'wrong':'F0'}
        with pytest.raises(ValueError):m.train(c,'cpu')
        assert events==[] and not Path(c['output_root']).exists()
