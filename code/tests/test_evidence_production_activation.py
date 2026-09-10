"""Production-size technical acceptance; synthetic IQ never supports accuracy claims."""
import json
import sys
import time
from types import SimpleNamespace
from pathlib import Path
import pytest
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from test_evidence_integration import model_args
from post_stage_common import build_baseline_model
from cvsrffi.evidence_head_training import attach_evidence_head,initialize_from_source,supervised_evidence_loss


def test_extended_contract_checks_actual_seed_mapping_equalization_and_ratios(tmp_path):
    from cvsrffi.evidence_head_training import validate_data_contract
    def records(rx,start,count,day=1):
        return [SimpleNamespace(tx_i=0,rx_i=rx,day_i=day,sig_i=i,eq_i=0) for i in range(start,start+count)]
    def loader(index):return SimpleNamespace(dataset=SimpleNamespace(index=index,eq_list=[1]))
    source=[records(0,0,7),records(0,7,63),records(0,70,30)]
    target=records(1,0,2,day=2)
    pid=lambda i:f'{i.tx_i}:{i.rx_i}:{i.day_i}:{i.sig_i}'
    roles={key:list(map(pid,value)) for key,value in zip(('L_s','U_s','V','target'),[*source,target])}
    receipt={'seed':392005,'requested_labeled_ratio':.07,'requested_unlabeled_ratio':.63,
             'requested_source_val_ratio':.30,'realized_rho_tolerance':.001,'realized_source_val_tolerance':.001}
    ctx={'dataset_id':'fixture','train_loader':loader(source[0]),'unlabeled_loader':loader(source[1]),
         'val_loader':loader(source[2]),'named_test_loaders':{'target':loader(target)},
         'class_id_to_tx':['physical-tx'],
         'split_info':{'test':{'test_rxs_idx':[1]},'source_split_receipt':receipt}}
    contract={'dataset_id':'fixture','source_receivers':[0],'target_receivers':[1],'roles':roles,
              'split_seed':392005,'equalized':1,'source_days':[1],'actual_target_days':[2],
              'tx_mapping':['physical-tx'],'ratios':{'L_s':.07,'U_s':.63,'V':.30},
              'source_selection':'final_only','checkpoint_initialization':'from_scratch','upstream':[]}
    path=tmp_path/'contract.json'
    path.write_text(json.dumps(contract),encoding='utf-8')
    assert validate_data_contract(ctx,path)==contract
    for key,value,error in [('split_seed',1,'seed'),('equalized',0,'equalized'),
                            ('source_days',[0],'source days'),('actual_target_days',[0],'target days'),
                            ('tx_mapping',[0],'TX mapping'),('ratios',{'L_s':.1,'U_s':.6,'V':.3},'ratio'),
                            ('upstream',['old-checkpoint'],'upstream'),('unexpected',True,'keys')]:
        path.write_text(json.dumps({**contract,key:value}),encoding='utf-8')
        with pytest.raises(ValueError,match=error):validate_data_contract(ctx,path)


def test_frozen_h4_tx_sorted_source_has_competing_support_episodes():
    from cvsrffi.evidence_pipeline import fit_frozen_head
    torch.set_num_threads(2)
    args=model_args();model=build_baseline_model(args,torch.device('cpu'))
    # Production builders enumerate TXs; naive contiguous slices each have one class.
    x=torch.randn(18,2,256);y=torch.arange(3).repeat_interleave(6)
    ids=[f'{int(label)}:0:0:{i}' for i,label in enumerate(y)]
    contract={'dataset_id':'SYNTHETIC_SORTED','roles':{'L_s':ids}}
    payload={'baseline_args':vars(args),'model':model.state_dict(),'training_data_contract':contract,
             'checkpoint_lineage':{'from_scratch':True,'upstream':[],
                                   'selection':'final_only','target_feedback':False}}
    _,trained=fit_frozen_head(payload,{'x':x,'y':y,'physical_ids':ids},contract,
                            {'variant':'H4','covariance_rank':2},epochs=1,batch_size=6)
    assert trained['history'][0]['evidence/support_effective_queries']>0
    assert trained['history'][0]['evidence/support_loss']>0


def test_ssdg_production_satellite_schedule_keeps_ce_view_separate(tmp_path):
    from SSDG.train_ssdg import build_arg_parser,_prepare_concat_sat_batch_for_training,_loss_weights
    from cvsrffi.core90_evidence_profile import core90_arguments
    from concat_sat_channel_aug import ConcatSatChannelAugment
    from cvsrffi.eval import apply_sat_channel_for_scenario
    args=build_arg_parser().parse_args(core90_arguments('fixture.pkl','contract.json',str(tmp_path),
        variant='H0',device='cpu',source_rxs='0,1',source_days='0,1',target_rxs='2',target_days='2'))
    aug=ConcatSatChannelAugment(scenarios=args.sat_train_scenarios.split(','),
        schedule=args.sat_view_schedule,p=1.,seed=392005,apply_fn=apply_sat_channel_for_scenario)
    torch.set_num_threads(2)
    x=torch.randn(6,2,256);y=torch.arange(6);domain=torch.arange(6)%2
    stages=[(1,.30,{'leo_clear_weak'}),(41,.60,{'leo_low_elev_weak','leo_rain_weak'}),
            (80,.60,{'leo_low_elev_weak','leo_rain_weak'}),
            (91,.80,{'leo_clear_weak','leo_low_elev_weak','leo_rain_weak'})]
    for epoch,probability,scenes in stages:
        applied=0
        for batch_idx in range(12):
            clean,labels,domains,view,info=_prepare_concat_sat_batch_for_training(
                aug,x,y,domain,args=args,epoch=epoch,batch_idx=batch_idx)
            assert torch.equal(clean,x) and torch.equal(labels,y) and torch.equal(domains,domain)
            assert info['expanded']==0 and view.view_prob==probability
            if view.applied:
                applied+=1
                assert view.scenario in scenes and not torch.equal(view.x,x)
        assert applied>0
    weights=_loss_weights(args,{})
    assert (args.sat_cons_start_epoch,weights['sat_cls'],weights['sat_cons'])==(80,.68,0.)


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA unavailable')
def test_frozen_h4_production_batch_gradient_and_memory():
    torch.set_num_threads(2)
    torch.manual_seed(392005)
    args=model_args();args.num_classes=6
    model=build_baseline_model(args,torch.device('cuda'))
    model.requires_grad_(False)
    attach_evidence_head(model,dict(variant='H4',covariance_rank=4))
    x=torch.randn(256,2,256,device='cuda')
    y=torch.arange(256,device='cuda')%6
    initialize_from_source(model,[(x[:24],y[:24])])
    model.eval();model.evidence_head.train()
    torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize()
    started=time.perf_counter()
    out=model(x,y_tx=y,return_aux=True)
    extra,stats=supervised_evidence_loss(model,out,y,{'physical_sample_id':[f'p{i}' for i in range(256)]},256)
    loss=torch.nn.functional.cross_entropy(out['tx_logits'],y)+extra
    loss.backward();torch.cuda.synchronize()
    assert torch.isfinite(loss)
    assert stats['evidence/support_queries']>0
    assert model.evidence_head.response.mean.grad.abs().sum()>0
    assert model.evidence_head.raw_diagonal.grad.abs().sum()>0
    assert all(p.grad is None for p in model.id_backbone.parameters())
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
    print(json.dumps({'variant':'H4','batch':256,'classes':6,'feature_dim':model.evidence_head.feature_dim,
                      'forward_backward_seconds':time.perf_counter()-started,
                      'peak_cuda_allocated_mib':torch.cuda.max_memory_allocated()/1024**2}))
