from copy import deepcopy
from types import MethodType
import pytest
import torch
from torch import nn
from torch.nn import functional as F
from experiments.pamf_channel_pl.plan import PROTOCOLS,SEEDS
from experiments.pamf_peer_supervised.plan import rows,make_config,validate_config,validate_runtime_context
from experiments.pamf_peer_supervised.objective import consensus_mask,PeerObjective,installed
from experiments.pamf_peer_supervised.priority import owner_matches,worker_item,defer


def context():
    cases=[]
    for p in PROTOCOLS:
        for s in SEEDS:
            cfg=dict(model_seed=s,ablation='F21',run_id='original',row_id=p+'-ordinary_PL-s'+str(s),
                source_contract=p+'/contract.json',source_cache=p+'/cache',source_rxs=[1,3,4,6,8],
                source_days=[1,2,3],num_classes=10 if p=='ManyRx' else 6,regime='native_cric',own_EMA=True)
            cases.append(dict(protocol=p,row_id=cfg['row_id'],model_seed=s,source_config=cfg,
                parent_source='old/'+cfg['row_id']+'/segment20',expected_source_contract=p+'/contract.json'))
    return dict(cases=cases)


def config(method='PeerT1U2'):
    return make_config(context(),next(r for r in rows() if r['method']==method),'PROJECT','commit')


def test_missing_historical_score_job_handle_is_rejected_before_release():
    c=dict(context(),reference={'handles':{}},consensus_source_root='/source',previous_runtime='/old/runtime.json',
        queue_priority_authorization='human_defer_unstarted_21_preserve_healthy_and_all_GPU')
    with pytest.raises(ValueError,match='reference_score_jobs'):validate_runtime_context(c)
    c['reference_score_jobs']='/actual_previous_owner/jobs.json'
    assert validate_runtime_context(c) is c


def test_three_variants_have_three_students_and_unmodified_original_prefixes():
    assert len(rows())==9
    for r in rows():
        c=make_config(context(),r,'PROJECT','commit');validate_config(c)
        assert c['trajectory']==[20,10,20] and c['optimizer_updates']==11100
        assert {p['model_seed'] for p in c['peers']}==set(SEEDS)-{r['model_seed']}
        assert c['hard_pl']['weight']==(1. if r['method']=='PeerT1U1' else 2.)
        assert c['repl_row']['structure']=='full' and not any(c['repl_row']['blocks'].values())


@pytest.mark.parametrize('bad',['truth','EMA','head','budget','AsKL','peer_seed','peer_contract','peer_epoch','class_space','extra_U'])
def test_forbidden_inputs_or_recipe_changes_block_before_loading(bad):
    c=config()
    if bad=='truth':c['target_truth']='target'
    elif bad=='EMA':c['own_EMA']=True
    elif bad=='head':c['posthoc_head_fit']=True
    elif bad=='budget':c['optimizer_updates']=9000
    elif bad=='AsKL':c['peer_supervised']['As_KL']=1.
    elif bad=='peer_seed':c['peers'][0]['model_seed']=c['model_seed']
    elif bad=='peer_contract':c['peers'][0]['source_config']['source_contract']='other'
    elif bad=='peer_epoch':c['peers'][0]['parent_source']='old/segment30'
    elif bad=='class_space':c['num_classes']=10
    else:c['hard_pl_no_uas']=False
    with pytest.raises(ValueError):validate_config(c)


@pytest.mark.parametrize('classes',[6,10])
def test_consensus_never_replaces_own_label_or_accepts_own_low_confidence(classes):
    pseudo=torch.tensor([0,0,0,0]);confidence=torch.tensor([.99,.99,.90,.99])
    a=torch.zeros(4,classes);b=a.clone();a[:,0]=b[:,0]=10.
    a[1,0]=b[1,0]=0.;a[1,1]=b[1,1]=10.;b[3]=0.
    assert consensus_mask(confidence,pseudo,[a,b]).tolist()==[True,False,False,False]
    assert pseudo.tolist()==[0,0,0,0]


class Peer(nn.Module):
    def __init__(self,disagree=False):
        super().__init__();self.w=nn.Parameter(torch.tensor(7.));self.register_buffer('probe',torch.tensor(1.))
        self.calls=0;self.disagree=disagree
    def forward(self,x):
        assert not self.training and not torch.is_grad_enabled();self.calls+=1
        z=torch.zeros(len(x),6);z[:,0]=self.w
        if self.disagree:z[1::2,0]=0.;z[1::2,1]=self.w
        return z


class Student(nn.Module):
    def __init__(self):
        super().__init__();self.id_backbone=nn.Module();self.id_backbone.encoder=nn.Module()
        self.id_backbone.encoder.core=nn.Module();core=nn.Module();self.id_backbone.encoder.core.id_backbone=core
        core.time_down=nn.Conv1d(2,4,1);core.t2=nn.Module();core.t2.pool=nn.Conv1d(4,4,1);core.cls_head=nn.Linear(4,6)
        self._feature_l_steps=1;self._cd_U_seen=self._cd_U_accepted=0;self.inputs=[];self.physical_inputs=[]
    def _cd_practical(self,x,meta,epoch,step):self.physical_inputs.append(x.detach().clone());return x*1.5
    def forward(self,x):
        if not self.training:
            z=torch.zeros(len(x),6);z[:,0]=7.;return z
        core=self.id_backbone.encoder.core.id_backbone
        h=core.t2.pool(core.time_down(x));return core.cls_head(F.dropout(h.mean(-1),p=.1,training=True))
    def forward_identity_only(self,x,domain_labels=None):
        self.inputs.append(x.detach().clone());return dict(tx_logits=self(x))


@pytest.mark.parametrize('method',['PeerT1U2','PeerPL2','PeerT1U1'])
def test_actual_U_branch_keeps_peers_frozen_and_joint_head_gradients(method):
    torch.manual_seed(13);c=config(method);model=Student().train();peers=[Peer().eval().requires_grad_(False),Peer(True).eval().requires_grad_(False)]
    x=torch.randn(12,2,8);meta=dict(rx_i=torch.tensor([1,3,4,6]*3),tx_label_visible=torch.zeros(12,dtype=torch.bool))
    model._cd_U_loader=[(x,torch.full((12,),-1),torch.zeros(12,dtype=torch.long),meta)]
    model._cd_U_iter=iter(model._cd_U_loader);obj=PeerObjective(c,peers)
    loss=obj.loss(model,'cpu');loss.backward();s=obj.state_dict();full=method!='PeerPL2'
    assert s['accepted']==6 and s['raw_accepted']==12 and s['peer_views']==24
    assert len(model.inputs)==(6 if full else 1) and s['student_views']==12*(6 if full else 1)
    assert len(model.physical_inputs)==(1 if full else 0)
    assert all(p.grad is None for peer in peers for p in peer.parameters()) and all(peer.calls==1 for peer in peers)
    assert not any(id(p) in {id(q) for peer in peers for q in peer.parameters()} for p in model.parameters())
    assert model.id_backbone.encoder.core.id_backbone.cls_head.weight.grad.abs().sum()>0
    assert all(not m._forward_hooks for m in model.modules()) and model.training and torch.isfinite(loss)
    if full:assert s['As_CE_fraction_of_physical_packet']==.4 and obj.metrics['HPL_As_KL_weight']==0.


def test_visible_U_or_non_source_RX_rejected_before_peer_forward():
    c=config();peers=[Peer().eval().requires_grad_(False) for _ in range(2)];obj=PeerObjective(c,peers)
    model=Student().train();x=torch.ones(8,2,8);y=torch.full((8,),-1);domain=torch.zeros(8,dtype=torch.long)
    for meta in (dict(rx_i=torch.ones(8),tx_label_visible=torch.ones(8,dtype=torch.bool)),
                 dict(rx_i=torch.full((8,),99),tx_label_visible=torch.zeros(8,dtype=torch.bool))):
        model._cd_U_loader=[(x,y,domain,meta)];model._cd_U_iter=iter(model._cd_U_loader)
        with pytest.raises(ValueError):obj.loss(model,'cpu')
    assert all(peer.calls==0 for peer in peers)


def test_bound_hook_and_loader_scope_are_restored_after_error(monkeypatch):
    from experiments.cric_dura_comparison import t1_model,f_data
    def native(self,*,model):assert self is model;return torch.tensor(1.),{'L':768}
    native._accepts_labeled_inputs=True
    monkeypatch.setattr(t1_model,'hook',native);before=(t1_model.plain_pl,t1_model.hook,f_data.context)
    c=config();obj=type('Objective',(),{'metrics':{'peer':1}})();holder={};owner=object()
    with pytest.raises(RuntimeError):
        with installed(c,holder,obj):
            value,metrics=MethodType(t1_model.hook,owner)(model=owner)
            assert value==1 and metrics=={'L':768,'peer':1} and t1_model.hook._accepts_labeled_inputs
            raise RuntimeError('probe')
    assert before==(t1_model.plain_pl,t1_model.hook,f_data.context)


def test_only_exact_controller_and_worker_identity_can_be_adopted(monkeypatch):
    import experiments.pamf_peer_supervised.priority as priority
    rt=dict(python='PY',release_root='/release',run_root='/run',runtime_file='/release/runtime.json')
    info=dict(cwd='/release',argv=['PY','-u','-m','experiments.pamf_channel_pl.training_runner','dispatch','--config','/release/runtime.json'])
    assert owner_matches(info,rt)
    assert not owner_matches(dict(info,cwd='/other'),rt)
    job=dict(id='train__ROW',kind='train',gpu=5,config='/run/configs/train__ROW.json')
    wi=dict(cwd='/release',argv=['PY','-u','-m','experiments.pamf_channel_pl.training_runner','worker','--kind','train','--config',job['config']],gpu='5')
    monkeypatch.setattr(priority,'process_info',lambda pid:wi)
    assert worker_item(job,rt,123)['pid']==123
    wi['gpu']='6'
    with pytest.raises(ValueError):worker_item(job,rt,123)
    assert defer(dict(lane='old',row_id='later'),False,{'later'})
    assert not defer(dict(lane='old',row_id='healthy'),False,{'later'})
    assert not defer(dict(lane='new',row_id='new'),False,{'later'})
    assert not defer(dict(lane='old',row_id='later'),True,{'later'})


def test_handoff_only_stops_controller_and_preserves_all_99_jobs(tmp_path,monkeypatch):
    import json
    from pathlib import Path
    import experiments.pamf_peer_supervised.priority as p
    oldroot=tmp_path/'old';root=tmp_path/'new';oldroot.mkdir();root.mkdir();fakeproc=tmp_path/'proc';fakeproc.mkdir()
    def put(path,value):path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value),encoding='utf-8')
    oldrt=dict(python='PY',release_root='/oldrelease',run_root=str(oldroot),runtime_file='/oldrelease/runtime.json',
        run_id=p.OLD_RUN,code_commit=p.OLD_COMMIT,launch_owner=p.OLD_OWNER)
    rt=dict(run_root=str(root),previous_runtime=str(tmp_path/'oldrt.json'),runtime_file='/newrelease/runtime.json',
        code_commit='new',queue_priority_authorization='human_defer_unstarted_21_preserve_healthy_and_all_GPU')
    put(Path(rt['previous_runtime']),oldrt);jobs=[];states={};infos={12:dict(cwd='/oldrelease',argv=['PY','-u','-m','experiments.pamf_channel_pl.training_runner','dispatch','--config',oldrt['runtime_file']])}
    for i in range(33):
        for kind in ('train','predict','score'):
            key=kind+'__ROW'+str(i);cfg=oldroot/'configs'/(key+'.json');completion=oldroot/key/'complete.json'
            put(cfg,dict(output_root=str(oldroot/key)))
            job=dict(id=key,kind=kind,row_id='ROW'+str(i),gpu=5+i%3,config=str(cfg),completion=str(completion),successful_statuses=['OK'])
            jobs.append(job);states[key]=dict(status='QUEUED')
            if i<6:put(completion,dict(status='OK'));states[key]=dict(status='COMPLETE')
            elif i<12 and kind=='train':
                pid=100+i;states[key]=dict(status='RUNNING',pid=pid);(fakeproc/str(pid)).mkdir()
                infos[pid]=dict(cwd='/oldrelease',gpu=str(job['gpu']),argv=['PY','-u','-m','experiments.pamf_channel_pl.training_runner','worker','--kind',kind,'--config',str(cfg)])
    put(oldroot/'jobs.json',jobs);put(oldroot/'pipeline_state.json',dict(pid=12,status='RUNNING',code_commit=p.OLD_COMMIT,launch_owner=p.OLD_OWNER,jobs=states))
    monkeypatch.setattr(p,'Path',lambda name:fakeproc if str(name)=='/proc' else Path(name))
    monkeypatch.setattr(p,'process_info',lambda pid:infos.get(pid))
    kills=[]
    def kill(pid,sig):kills.append(pid);del infos[pid]
    monkeypatch.setattr(p.os,'kill',kill)
    converted,statuses,active,children,deferred,_=p.handoff(rt)
    assert kills==[12] and len(active)==len(children)==6 and len(deferred)==21 and len(converted)==99
    assert {v['pid'] for v in active.values()}==set(range(106,112))
    assert [j['config'] for j in converted]==[j['config'] for j in jobs]
    assert [j['completion'] for j in converted]==[j['completion'] for j in jobs]
    assert all(j['execution_release']=='/oldrelease' for j in converted)
    assert all(infos[pid] for pid in range(106,112))
