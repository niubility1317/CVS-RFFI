from copy import deepcopy
import json
from pathlib import Path
import random
import numpy as np
import pytest
import torch
from cvsrffi.xuc_fusion.runtime import resolve_args,synthetic_source
from cvsrffi.xuc_fusion.tickets import TicketStream
from cvsrffi.xuc_fusion.control import SourceObserver,ReliableController,recovery_quality
from cvsrffi.xuc_fusion.objective import FusionObjective
from cvsrffi.xuc_fusion.native import train_native
from cvsrffi.game_tracking.runtime import build_model
from cvsrffi.game_tracking.legacy.losses import PrototypeMemoryBank
from cvsrffi.game_tracking.step_context import prepare_context,Core90Objective
from cvsrffi.game_tracking.solvers import GameSolver
ROOT=Path(__file__).resolve().parents[1]
MATRIX=json.loads((ROOT/'configs/matrix15.json').read_text(encoding='utf-8'))
RECIPE=json.loads((ROOT/'configs/core90_recipe_reference.json').read_text(encoding='utf-8'))

def setup(rowid,epoch=1):
    row=deepcopy(next(r for r in MATRIX['runs'] if r['id']==rowid))
    args=resolve_args(RECIPE,row,dataset='synthetic',output='unused',device='cpu',synthetic=True,test_epochs=200,test_steps=1)
    args.num_classes=6
    torch.manual_seed(123);source=synthetic_source();model=build_model(args,15,torch.device('cpu')).train()
    proto=PrototypeMemoryBank(6,15,momentum=args.proto_momentum,margin=args.proto_margin,domain_align_weight=args.proto_domain_align_weight,
        push_weight=args.proto_push_weight,min_count=args.proto_min_count)
    proto._lazy_init(160,torch.device('cpu'),torch.float32)
    stream=TicketStream(source,args,row);stream.next_epoch=epoch;ticket,_=stream.choose();batch,u=stream.batches(ticket)
    args.game_evidence_version=2
    ctx=prepare_context(batch,u,model,None,args,ticket.epoch,ticket.batch_index,ticket.weights,torch.Generator().manual_seed(ticket.seed),None,
                        exposure_record=ticket.exposure(batch[3]['sample_id']))
    ctx.grid_plan=ticket.grid_plan;ctx.rx=batch[3]['rx_i'];ctx.day=batch[3]['day_i'];ctx.field_components=[];ctx.audit_gradients=True;ctx.fusion_telemetry={}
    return row,args,source,model,proto,stream,ticket,ctx

@pytest.mark.parametrize('epoch',[1,80,131])
def test_full_core90_objective_preserved(epoch):
    row,args,_,model,proto,_,_,ctx=setup('M00',epoch)
    left=deepcopy(ctx);right=deepcopy(ctx)
    torch.manual_seed(92);a=Core90Objective(model,args,proto)(left)
    torch.manual_seed(92);b=FusionObjective(model,args,proto,row)(right)
    torch.testing.assert_close(a,b,rtol=0,atol=0)
    assert left.origin_terms==right.origin_terms
    if epoch==131:assert 'unlabeled_entropy' in right.origin_terms

def test_full_xu_extragradient_pseudo_stage_and_single_optimizer_commit():
    row,args,source,model,proto,stream,ticket,ctx=setup('M12',131)
    _,u=stream.batches(ticket)
    assert (u[1]==-1).all() and 'tx_i' not in u[3]
    ctx.base_mask=torch.ones_like(ctx.base_mask);args.pseudo_strong_agreement=False
    optimizer=torch.optim.AdamW(model.parameters(),lr=args.lr)
    solver=GameSolver(model,optimizer,mode='extragradient',nonfinite='raise',max_grad_norm=5.)
    result=solver.step(lambda:FusionObjective(model,args,proto,row)(ctx))
    assert result.accepted and result.field_evaluations==2
    assert ctx.field_components[0]==ctx.field_components[1]
    assert {'x_cross_rx','u_normalized','unlabeled_ce','sat_cls'}<=set(ctx.field_components[0])
    assert ctx.fusion_telemetry['x_weighted_identity_grad_norm']>0
    assert ctx.fusion_telemetry['u_weighted_identity_grad_norm']>0
    assert ctx.fusion_telemetry['normalized_valid_blocks']==4
    assert {int(s['step']) for s in optimizer.state.values()}=={1}
    assert ticket.ticket_id not in stream.consumed
    stream.commit(ticket)
    with pytest.raises(ValueError):stream.commit(ticket)

def evidence(step,lag=.001,imb=.6,valid=True):
    return dict(observation_id=step,step=step,recovery={'valid':valid},geometry_valid=True,
        geometry=dict(identity=.9,margin=.5,tx_main_energy=.2,unit_interaction=.01,leo_cosine=.99),
        G_lag=lag,readability=.2,direction_imbalance=imb,difficulty={})

@pytest.mark.parametrize('lag,expected',[(.007,'NORMAL'),(.001,'CORRECT'),(.02,'CATCHUP')])
def test_cstar_confirmed_actions_and_middle_band(lag,expected):
    c=ReliableController()
    for step in [0,250,500,750,1000,1250]:c.observe(evidence(step,lag))
    d=c.decide(1250);assert d['action']==expected
    if expected!='NORMAL':
        c.commit(d,1250,True);assert c.decide(1250)['action']=='NORMAL'
    assert c.decide(1261)['action']=='NORMAL'

def test_cstar_duplicate_observations_cannot_confirm():
    c=ReliableController()
    for step in [0,250,500]:c.observe(evidence(step))
    c.observe(evidence(750));before=deepcopy(c.state_dict())
    for _ in range(30):assert not c.observe(evidence(750))
    assert c.state_dict()==before and c.decide(750)['action']=='NORMAL'
    c.observe(evidence(1000,valid=False));assert c.last_kind is None and c.action_streak==0

def test_recovery_paired_capture_uncertainty():
    assert recovery_quality(2,1,[-.3,-.2,-.2,-.1],['a','a','b','b'])['valid']
    assert not recovery_quality(2,2,[-.3,-.2],['a','b'])['valid']
    assert not recovery_quality(2,1,[-.3,.5],['a','b'])['valid']
    assert not recovery_quality(2,1,[-.3,-.3],['a','a'])['valid']

def test_passive_observer_restores_model_and_rng():
    _,args,source,model,_,_,_,_=setup('M05')
    state=deepcopy(model.state_dict());rng=torch.get_rng_state().clone();npstate=np.random.get_state();pystate=random.getstate()
    flags=[m.training for m in model.modules()]
    result=SourceObserver(source,args).observe(model,step=0,epoch=1,adv_weight=.01)
    assert result['source_only'] and result['recovery_steps']==40 and result['recovery']['independent_groups']==45
    assert torch.equal(rng,torch.get_rng_state()) and pystate==random.getstate()
    assert np.array_equal(npstate[1],np.random.get_state()[1])
    assert flags==[m.training for m in model.modules()]
    assert all(torch.equal(v,model.state_dict()[k]) for k,v in state.items())

def test_ticket_multiset_weights_and_coverage_are_equal():
    row,args,source,_,_,_,_,_=setup('M12');args.epochs=20
    fixed=deepcopy(row);fixed['ticket_curriculum_enabled']=False
    streams=[TicketStream(source,args,r) for r in (row,fixed)]
    observed=[];orders=[]
    for stream in streams:
        records={};order=[]
        while True:
            ticket,coverage=stream.choose(capability=True,difficulty={'leo_clear_weak':.5,'leo_rain_weak':2})
            if ticket is None:break
            records[ticket.ticket_id]=(ticket.epoch,ticket.labeled,ticket.unlabeled,ticket.channel_seed,ticket.selected,ticket.scene,ticket.weights)
            order.append(ticket.ticket_id);window=stream.commit(ticket)
            if window:assert window['membership_verified'] and len(window['ticket_ids'])<=10*stream.steps
        observed.append(records);orders.append(order)
    assert observed[0]==observed[1] and len(observed[0])==20
    assert orders[0]!=orders[1]

@pytest.mark.parametrize('rid',['M09','M10'])
def test_native_arguments_and_import_isolation(rid):
    from SSDG import train_ssdg as native
    from cvsrffi import schedule
    import post_stage_cli
    for module in (native,schedule,post_stage_cli):assert Path(module.__file__).is_relative_to(ROOT)
    ref=json.loads((ROOT/'configs/a1_native_recipe_reference.json').read_text(encoding='utf-8'))
    args=train_native(ref,rid,'unused','unused','unused',True)
    assert args['epochs']==200 and args['seed']==392005 and args['amp'] is False
    assert args['from_scratch'] and args['a1_scratch_only'] and args['a1_source_screen_only']
    assert not args['baseline_ckpt'] and not args['teacher_ckpt']
    assert args['a1_ecrs_cross_rx_weight']==(.05 if rid=='M10' else 0)

def test_gpu_admission_reserves_uninitialized_children(monkeypatch):
    from scripts import dispatch_xuc15 as d
    class Child:
        pid=44
        def poll(self):return None
    def output(argv,**kwargs):
        return '0, GPU-a, 20000\n1, GPU-b, 20000\n' if '--query-gpu=index,uuid,memory.free' in argv else 'GPU-a, 11\nGPU-b, 22\nGPU-b, 33\n'
    monkeypatch.setattr(d.subprocess,'check_output',output)
    assert d.available_gpu({'M00':{'process':Child(),'gpu':0}}) is None
    assert d.available_gpu({})==0

def test_recovery_cannot_retrain_healthy_rows(tmp_path):
    from scripts.dispatch_xuc15 import retry_allowed
    for status in ['TRAINING_COMPLETE','PREDICTIONS_FIXED','SCORED']:
        assert not retry_allowed({'status':status},tmp_path)
    assert retry_allowed({'status':'TRAIN_FAILED'},tmp_path)
    (tmp_path/'completion.json').write_text(json.dumps({'epochs':200}))
    assert not retry_allowed({'status':'TRAIN_FAILED'},tmp_path)

def test_all_matrix_commands_and_source_builder_use_frozen_runs():
    from scripts.dispatch_xuc15 import train_command
    for row in MATRIX['runs']:
        argv=train_command(row,Path('/project'),Path('/project/runs/new'))
        assert '--source-contract' in argv and '--row' in argv
        assert argv[argv.index('--row')+1]==row['id']
        assert '--baseline_ckpt' not in argv
        assert ('train_xuc_native.py' in argv[2])==(row['id'] in ('M09','M10'))
