from contextlib import contextmanager
from pathlib import Path
import shutil,time
from experiments.cric_dura_comparison.io import read,write
from .plan import validate_config

@contextmanager
def seed_scope():
    """Reuse immutable native defaults; new scratch seeds change RNG, not weights."""
    from experiments.cric_dura_comparison import f_train
    old_config,old_args=f_train.original_config,f_train.arguments
    def config(seed):
        # runtime.installed also initializes the core/head from this field.
        # Argument construction alone is insufficient to set the actual RNG.
        value=dict(old_config(2026092701 if seed>=2026092704 else seed))
        value['model_seed']=seed
        return value
    def args(c,device):
        template=dict(c,model_seed=2026092701) if c['model_seed']>=2026092704 else c
        a=old_args(template,device);a.seed=c['model_seed']
        return a
    f_train.original_config,f_train.arguments=config,args
    try:yield
    finally:f_train.original_config,f_train.arguments=old_config,old_args

@contextmanager
def single_base(c,reader=False):
    from experiments.cric_dura_comparison import f_model,t1_model
    target=f_model if reader else t1_model;old=target.install
    def build(model,cfg):
        model=old(model,cfg)
        if c['view_comparison']['view_count']==1:
            from experiments.cvs_rdrop_sinc_units.x3_features import CompositePool
            ref=model.id_backbone.encoder
            ref.sequence_pool=CompositePool(ref.sequence_pool.channels,'SIG_TIME',cfg['model_seed'])
        return model
    target.install=build
    try:yield
    finally:target.install=old

@contextmanager
def installed(c,holder,enabled):
    from experiments.pamf_repl_completion.objectives import installed as original
    with seed_scope(),single_base(c),original(c,holder,enabled):yield

def assert_structure(model,c):
    ref=model.id_backbone.encoder;views=c['view_comparison']['view_count'];pool=ref.sequence_pool
    expected=232993 if views==3 else 196001
    count=sum(p.numel() for p in ref.parameters())
    if len(pool.views)!=views or tuple(pool.output_projection.weight.shape)!=(160,112*views) or count!=expected:
        raise ValueError(('Signature view architecture differs',views,count,expected))
    orders=[v.window_order.tolist() for v in pool.views]
    if any(o!=orders[0] or len(o)!=8 or sorted(o)!=list(range(8)) or o[0]!=0 or o[-1]!=7 or o==list(range(8)) for o in orders):
        raise ValueError('Shared fixed endpoint-preserving temporal permutation required')
    return dict(view_count=views,parameters=count,order=orders[0],projection=list(pool.output_projection.weight.shape))

def train(c):
    validate_config(c)
    from experiments.cric_dura_comparison.f_train import train as segment
    root=Path(c['output_root']);root.mkdir(parents=True,exist_ok=False);write(root/'resolved_config.json',c)
    previous=None;segments=[];states=[];started=time.monotonic()
    for epochs,offset in ((20,0),(10,20),(20,30)):
        folder=root/f'segment{epochs+offset}';cfg=dict(c,output_root=str(folder),segment_epochs=epochs,epoch_offset=offset,step_offset=offset*222,parent_source=previous)
        holder={}
        with installed(cfg,holder,offset>=20):segment(cfg)
        if offset>=20:
            state=holder['objective'].state_dict()
            if state['calls']!=epochs*222 or state['EMA_updates'] or state['U_truth_read'] or state['student_views']!=6*state['seen']:
                raise ValueError('Actual complete original REPL execution differs')
            write(folder/'repl_state.json',state);states.append(state)
        done=read(folder/'complete.json')
        if done['status']!='SOURCE_FROZEN' or done['updates']!=epochs*222 or done['target_access'] or done['U_truth_read'] or done['head_fit_updates']:
            raise ValueError('Incomplete own-contract source segment')
        segments.append(dict(folder=str(folder),**done));previous=str(folder)
        write(root/'progress.json',dict(status='SEGMENT_COMPLETE',epoch=epochs+offset,updates=(epochs+offset)*222),exclusive=False)
    shutil.copyfile(Path(previous)/'final_ssdg.pth',root/'final_ssdg.pth');write(root/'source_contract.json',read(c['source_contract']))
    write(root/'complete.json',dict(status='SOURCE_FROZEN',config=c,epoch=50,updates=11100,segments=segments,
        initialization='same_contract_scratch20_own10_own20',ancestors=[s['folder'] for s in segments],source_contract=c['source_contract'],
        target_access=False,U_truth_read=False,head_fit_updates=0,actual_REPL={k:sum(s[k] for s in states) for k in ('calls','seen','accepted','student_views')},
        seconds=time.monotonic()-started,cumulative_training_seconds=sum(s['seconds'] for s in segments),historical_posthoc_selection_disclosed=True))

def load_reader(folder,contract,device):
    from experiments.pamf_repl_completion.source import load_reader as original
    d=read(Path(folder)/'complete.json');c=d['config'];validate_config(c)
    if any(s['config']['view_comparison']!=c['view_comparison'] for s in d['segments']):raise ValueError('Own architecture ancestry mismatch')
    with seed_scope(),single_base(c,reader=True):return original(folder,contract,device)
