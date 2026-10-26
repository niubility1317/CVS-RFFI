"""Single-view structure and CE/ordinary-PL hooks, without global old-run edits."""
from contextlib import contextmanager
from types import MethodType
from .plan import ARMS,validate_config

def structure(model,arm,seed):
    import torch
    from torch import nn
    from experiments.cvs_rdrop_sinc_units.x3_features import CompositePool,DescriptorView
    from experiments.cric_dura_comparison.supervised_physics import install as physical
    if arm not in ARMS:raise ValueError('Unknown single-view structure')
    ref=model.id_backbone.encoder
    ref.sequence_pool=CompositePool(ref.sequence_pool.channels,'SIG_TIME',seed)
    if arm=='moment_pooling':
        ref.sequence_pool.views=nn.ModuleList([DescriptorView(ref.sequence_pool.channels,'COV',seed,0)])
    elif arm!='full':
        old={'no_time':'NO_TIME','no_signature':'NO_SIGNATURE','no_pa':'NO_PA','no_reference':'NO_REFERENCE',
             'shuffled_time':'SHUFFLE_TIME','random_reference_phase':'WRONG_REFERENCE'}[arm]
        physical(model,old,seed)
    count=sum(p.numel() for p in ref.parameters())
    expected={'no_signature':177025,'no_pa':184961,'no_reference':183201}.get(arm,196001)+160*(ref.core.id_backbone.cls_head.weight.shape[0]-6)
    if count!=expected:raise ValueError(('Single-view parameter count differs',arm,count,expected))
    if arm!='no_signature' and (len(ref.sequence_pool.views)!=1 or tuple(ref.sequence_pool.output_projection.weight.shape)!=(160,112)):
        raise ValueError('One actual learned descriptor view required')
    model._feature_learning_report.update(single_view_paper=True,structure=ARMS[arm][0],signature_views=0 if arm=='no_signature' else 1,deploy_parameters=count)
    return model

def ce_hook(self,**kw):
    from experiments.practical_full_2x2.pamf_model import hook,plain_pl
    correction,metrics=hook(self,**kw)
    c=self._cd_config;epoch=(kw['optimizer_step']+c.get('step_offset',0)-1)//222+1
    if c['U_use'] and epoch>=14:correction=correction+plain_pl(self,kw['labeled_output']['tx_logits'].device)
    metrics.update(Paper_CE_only=1.,Paper_PL_seen=self._cd_U_seen,Paper_PL_accepted=self._cd_U_accepted,
        T1_PL_seen=self._cd_U_seen,T1_PL_accepted=self._cd_U_accepted,CRIC_supcon_weight=0.,T1_ordinary_KL_weight=0.,T1_As_KL_weight=0.)
    return correction,metrics
ce_hook._accepts_labeled_inputs=True

def loader_configuration(c):
    # F15 is a model/loss compatibility ID that historically also disabled U.
    # Build a U256 loader through the ordinary-PL loader route, without changing
    # the actual F15 CE model, objective, checkpoint identity or permissions.
    cfg=dict(c,regime='native_cric' if c['U_use'] else 'matched_sat_supervised')
    if c['U_use'] and c['paper_row']['family']=='branch':cfg['ablation']='F21'
    return cfg

@contextmanager
def installed(c):
    from experiments.cric_dura_comparison import t1_model,f_train,f_data
    old_install,old_args=t1_model.install,f_train.arguments
    old_data,old_context=f_data.context,f_train.context
    def build(model,cfg):
        validate_config(cfg)
        if cfg['paper_row']['family']=='branch':
            from experiments.practical_full_2x2.pamf_model import install
            model=install(model,cfg);model._source_labeled_feature_hook=MethodType(ce_hook,model)
        else:model=old_install(model,cfg)
        return structure(model,cfg['paper_row']['arm'],cfg['model_seed'])
    def args(cfg,device):
        a=old_args(cfg,device);a.input_len=256
        a.use_unlabeled=a.use_ema_teacher=False;a.lambda_u=a.lambda_ent=0.
        return a
    t1_model.install,f_train.arguments=build,args
    def context(cfg):return old_data(loader_configuration(cfg))
    f_data.context=f_train.context=context
    try:
        yield
    finally:
        t1_model.install,f_train.arguments=old_install,old_args
        f_data.context,f_train.context=old_data,old_context

@contextmanager
def reader_scope(row):
    from experiments.cric_dura_comparison import f_model
    old=f_model.install
    def build(model,c):return structure(old(model,c),row['arm'],row['model_seed'])
    f_model.install=build
    try:yield
    finally:f_model.install=old

def load_reader(folder,contract,row,device):
    from experiments.cric_dura_comparison.frozen_inference import load_F
    with reader_scope(row):return load_F(folder,contract,device,require_no_unlabeled=row['learning_regime']=='supervised')
