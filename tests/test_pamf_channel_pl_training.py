from copy import deepcopy
import pytest
from experiments.pamf_channel_pl.training_plan import make_config,validate_config,TRAIN_RUN
from experiments.pamf_channel_pl.policy import RECIPE


def case(protocol='F0'):
    return dict(protocol=protocol,row_id=protocol+'-ordinary_PL-s2026092701',model_seed=2026092701,
        parent_source='PROJECT/old/'+protocol+'/segment20',expected_source_contract='source.json',
        source_config=dict(model_seed=2026092701,ablation='F21',source_contract='source.json',source_cache='source-cache',
            source_rxs=[1],source_days=[1,2,3],num_classes=10 if protocol=='ManyRx' else 6,regime='native_cric',
            own_EMA=True,method_id='pamf_t1_e50',hard_pl=dict(weight=6),parent_source='stale',step_offset=30))


def test_new_row_uses_own_E20_and_exact_ordinary_PL_scaffold():
    original=case();c=make_config(original,'PROJECT','commit')
    assert c['run_id']==TRAIN_RUN and c['shared_prefix']==original['parent_source']
    assert c['channel_pl']==RECIPE and c['plain_PL_weight']==1. and not c['own_EMA']
    assert c['optimizer_updates']==11100 and c['trajectory']==[20,10,20]
    assert c['repl_row']['L_A'] and c['repl_row']['L_B'] and c['repl_row']['structure']=='full'
    assert not any(k.startswith('hard_pl') for k in c)
    assert all(k not in c for k in ('parent_source','step_offset','epoch_offset','segment_epochs'))
    assert original['source_config']['own_EMA']
    validate_config(c)


@pytest.mark.parametrize('bad',['target','EMA','Uweight','threshold','budget','head','As','style','seed','classes','run'])
def test_rejects_unmatched_training_or_query_inputs(bad):
    c=make_config(case(),'PROJECT','commit')
    if bad=='target':c['tests']={'query':'forbidden'}
    elif bad=='EMA':c['own_EMA']=True
    elif bad=='Uweight':c['plain_PL_weight']=6.
    elif bad=='threshold':c['channel_pl']['channel_threshold']=.9
    elif bad=='budget':c['optimizer_updates']=999
    elif bad=='head':c['posthoc_head_fit']=True
    elif bad=='As':c['satellite_concat']=True
    elif bad=='style':c['hard_pl_mixstyle']={}
    elif bad=='seed':c['model_seed']=7
    elif bad=='classes':c['num_classes']=10
    else:c['run_id']='other'
    with pytest.raises(ValueError):validate_config(c)


def test_manyrx_retains_ten_class_native_head():
    c=make_config(case('ManyRx'),'PROJECT','commit');validate_config(c)
    assert c['num_classes']==10


def test_real_smoke_loader_respects_active_native_256_U_scope(monkeypatch):
    import torch
    from experiments.cric_dura_comparison import f_data
    from experiments.pamf_channel_pl.training import source_context
    from experiments.pamf_channel_pl.objective import installed
    contract=dict(classes=list(range(6)),source_rxs=[1],source_days=[1,2,3],counts=dict(L_s=256,U_s=1024,V=256))
    class Source(torch.utils.data.Dataset):
        def __init__(self,root,role,c):self.role=role;self.x=torch.zeros(c['counts'][role],2,256)
        def __len__(self):return len(self.x)
        def __getitem__(self,i):return self.x[i],-1 if self.role=='U_s' else 0,0,{'tx_label_visible':self.role!='U_s'}
    monkeypatch.setattr(f_data,'read',lambda _:contract)
    monkeypatch.setattr(f_data,'SourceDataset',Source)
    c=make_config(case(),'PROJECT','commit')
    assert f_data.context(c)['unlabeled_loader'].batch_size==128
    with installed(c,{}):
        ctx=source_context(c)
        assert ctx['unlabeled_loader'].batch_size==256
        batch=next(iter(ctx['unlabeled_loader']))
        assert len(batch[0])==256 and bool((batch[1]==-1).all())
