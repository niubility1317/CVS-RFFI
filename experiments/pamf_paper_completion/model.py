"""Graph changes isolated to this run; old owners and recipes are untouched."""
from contextlib import contextmanager
from types import MethodType

def source_arguments(original,c,device):
    a=original(c,device)
    # The ordinary trainer normally derives this from the data loader before
    # construction. The standalone no-query smoke constructs the same adapter.
    a.input_len=256
    a.use_unlabeled=a.use_ema_teacher=False;a.lambda_u=a.lambda_ent=0.
    return a

def shuffled_without_time(self,h):
    import torch
    from torch.nn import functional as F
    from experiments.cvs_feature_sequence50.features import path_signature
    u=torch.tanh(self.local_projection(h)).transpose(1,2);b,t,_=u.shape
    windows=u.reshape(b*4,t//4,6)
    return F.pad(path_signature(windows[:,self.window_order],False),(0,7)).reshape(b,112)

def structure(model,row,seed):
    from torch import nn
    from experiments.cvs_rdrop_sinc_units.x3_features import CompositePool,DescriptorView
    from experiments.cric_dura_comparison.supervised_physics import install
    ref=model.id_backbone.encoder;arm=row['structure'];n=row['signature_views']
    if n==1:ref.sequence_pool=CompositePool(ref.sequence_pool.channels,'SIG_TIME',seed)
    if arm=='no_signature':install(model,'NO_SIGNATURE',seed)
    elif arm=='moment':
        ref.sequence_pool.views=nn.ModuleList([DescriptorView(ref.sequence_pool.channels,'COV',seed,i) for i in range(n)])
    elif arm in ('no_pa','no_reference','reference_phase'):
        install(model,{'no_pa':'NO_PA','no_reference':'NO_REFERENCE','reference_phase':'WRONG_REFERENCE'}[arm],seed)
    elif arm not in ('full','no_time'):raise ValueError('Unknown structure')
    if row['permuted']:
        if arm in ('no_signature','moment'):raise ValueError('Order-free control must keep its exact historical graph')
        install(model,'SHUFFLE_TIME',seed)
    if arm=='no_time':
        if not row['permuted']:raise ValueError('Current no-time control retains the fixed permutation')
        for v in ref.sequence_pool.views:v.forward=MethodType(shuffled_without_time,v)
    expected=177025 if arm=='no_signature' else (196001 if n==1 else 232993)
    expected-=11040 if arm=='no_pa' else 12800 if arm=='no_reference' else 0
    expected+=160*(ref.core.id_backbone.cls_head.weight.shape[0]-6)
    actual=sum(p.numel() for p in ref.parameters())
    if actual!=expected:raise ValueError(('Deploy graph parameter count',arm,actual,expected))
    if arm!='no_signature':
        if len(ref.sequence_pool.views)!=n or ref.sequence_pool.output_projection.weight.shape[1]!=112*n:raise ValueError('Wrong signature views')
        if row['permuted'] and any(v.window_order.tolist()!=[0,5,3,6,2,1,4,7] for v in ref.sequence_pool.views):raise ValueError('Wrong fixed temporal order')
    model._feature_learning_report.update(paper_completion=True,structure=arm,signature_views=n,permuted=row['permuted'],deploy_parameters=actual)
    return model

@contextmanager
def installed(c,holder,ssl_enabled=False):
    from experiments.cric_dura_comparison import t1_model,f_train
    from experiments.pamf_repl_completion.objectives import loader_scope,portable_scope,objective
    from experiments.pamf_ssl_research.hard_pl_train import installed as hard_scope
    from .plan import validate_config
    old_install,old_args=t1_model.install,f_train.arguments
    def build(net,cfg):
        validate_config(cfg)
        return structure(old_install(net,cfg),cfg['paper_row'],cfg['model_seed'])
    def args(cfg,device):
        return source_arguments(old_args,cfg,device)
    t1_model.install,f_train.arguments=build,args
    try:
        with loader_scope():
            if ssl_enabled:
                with portable_scope(),hard_scope(c,holder,objective_cls=objective):yield
            else:yield
    finally:t1_model.install,f_train.arguments=old_install,old_args

@contextmanager
def reader_scope(row):
    from experiments.cric_dura_comparison import f_model
    old=f_model.install
    f_model.install=lambda net,c:structure(old(net,c),row,c['model_seed'])
    try:yield
    finally:f_model.install=old

def load_reader(folder,contract,row,device):
    from experiments.cric_dura_comparison.frozen_inference import load_F
    with reader_scope(row):return load_F(folder,contract,device,require_no_unlabeled=row['learning_regime']=='supervised')

def ce_build(variant,seed):
    from .plan import CE_VARIANT
    if variant!=CE_VARIANT:raise ValueError('Only registered three-view CE graph')
    from experiments.standard_ce_baselines.sigtime import SIGTimeCVS
    from experiments.cvs_rdrop_sinc_units.x3_features import CompositePool
    model=SIGTimeCVS(seed)
    model.sequence_pool=CompositePool(model.sequence_pool.channels,'SIG_TIME_X3',seed)
    model._architecture.update(name='PAMF-CE Three-view SIG-TIME',signature_views=3,temporal_permutation=False,
                               extra_feature_parameters=sum(p.numel() for p in model.sequence_pool.parameters()))
    if sum(p.numel() for p in model.parameters())!=232993:raise ValueError('E200 graph differs from three-view PAMF')
    return model

@contextmanager
def ce_scope():
    from pathlib import Path
    from .plan import CE_VARIANT
    import experiments.standard_ce_baselines as package
    # The fixed native releases provide their dependencies; this run's trainer
    # and prediction checks must come from the committed current code.
    package.__path__.insert(0,str(Path(__file__).resolve().parents[1]/'standard_ce_baselines'))
    from experiments.standard_ce_baselines import source,predict
    saved=(source.build,source.SUPPORTED,predict.build,predict.SUPPORTED)
    source.build=predict.build=ce_build
    source.SUPPORTED=predict.SUPPORTED=(*source.SUPPORTED,CE_VARIANT)
    try:yield
    finally:source.build,source.SUPPORTED,predict.build,predict.SUPPORTED=saved
