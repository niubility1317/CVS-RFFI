"""Fixed final E50 or E200; scratch and same-row continuations only."""
from pathlib import Path
import shutil,time
from experiments.cric_dura_comparison.io import read,write
from .plan import validate_config

def train(c):
    validate_config(c)
    if c['paper_row']['family']=='architecture':return train_ce(c)
    from experiments.cric_dura_comparison.f_train import train as segment
    from .model import installed
    root=Path(c['output_root']);root.mkdir(parents=True,exist_ok=False);write(root/'resolved_config.json',c)
    previous=None;segments=[];states=[];started=time.monotonic()
    for n,offset in ((20,0),(10,20),(20,30)):
        folder=root/f'segment{offset+n}';cfg=dict(c,output_root=str(folder),segment_epochs=n,epoch_offset=offset,step_offset=offset*222,parent_source=previous)
        holder={};enabled=c['paper_row']['learning_regime']=='ssl' and offset>=20
        with installed(cfg,holder,enabled):segment(cfg)
        if enabled:
            state=holder['objective'].state_dict()
            if state['calls']!=n*222 or state['EMA_updates'] or state['U_truth_read'] or state['student_views']!=6*state['seen']:raise ValueError('Incomplete original SSL')
            write(folder/'repl_state.json',state);states.append(state)
        done=read(folder/'complete.json')
        if (done['status']!='SOURCE_FROZEN' or done['updates']!=n*222 or done['target_access'] or done['U_truth_read'] or done['head_fit_updates']):raise ValueError('Incomplete/illegal source segment')
        segments.append(dict(folder=str(folder),**done));previous=str(folder)
        write(root/'progress.json',dict(status='SEGMENT_COMPLETE',epoch=offset+n,updates=(offset+n)*222),exclusive=False)
    shutil.copyfile(Path(previous)/'final_ssdg.pth',root/'final_ssdg.pth');write(root/'source_contract.json',read(c['source_contract']))
    progress=[read(Path(s['folder'])/'progress.json') for s in segments]
    u=dict(plain_PL_seen=sum(s.get('PL_seen',0) for s in progress),plain_PL_accepted=sum(s.get('PL_accepted',0) for s in progress),
           full_SSL={k:sum(s[k] for s in states) for k in ('calls','seen','accepted','student_views')},U_truth_read=False,EMA_updates=0)
    write(root/'complete.json',dict(status='SOURCE_FROZEN',config=c,epoch=50,updates=11100,segments=segments,
        initialization='same_contract_scratch20_own10_own20',ancestors=[s['folder'] for s in segments],source_contract=c['source_contract'],
        target_access=False,U_truth_read=False,head_fit_updates=0,actual_U=u,seconds=time.monotonic()-started,
        cumulative_training_seconds=sum(s['seconds'] for s in segments)))

def train_ce(c):
    from experiments.standard_ce_baselines.source import train as native
    from .model import ce_scope
    with ce_scope():native(c)
    root=Path(c['output_root']);d=read(root/'completion.json')
    if d['status']!='SOURCE_TRAINED' or d['epoch']!=200 or d['steps']!=10000 or d['target_access']:raise ValueError('E200 final missing')
    write(root/'freeze.json',dict(status='FROZEN',epoch=200,variant=c['variant'],model_seed=c['model_seed'],checkpoint=str(root/'last.pt'),target_access=False,target_score_used=False))
    write(root/'complete.json',dict(status='SOURCE_FROZEN',epoch=200,updates=10000,target_access=False,U_truth_read=False,head_fit_updates=0,config=c))
