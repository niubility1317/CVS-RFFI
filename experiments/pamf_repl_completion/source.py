"""Own-contract E20/10/20 training, sharing only identical ordinary-PL prefixes."""
from contextlib import contextmanager
from pathlib import Path
import shutil
import time
from experiments.cric_dura_comparison.io import read,write
from .plan import validate_config

def is_repl(c):
    return c['repl_row']['U'] and c['repl_row']['method']!='ordinary_PL'

def parent(c,path):
    import torch
    done=read(Path(path)/'complete.json');pc=done['config']
    if (done['status']!='SOURCE_FROZEN' or done['updates']!=done['epoch']*222 or done['target_access']
        or done['U_truth_read'] or done['head_fit_updates'] or pc['model_seed']!=c['model_seed']
        or pc['ablation']!=c['ablation'] or read(pc['source_contract'])!=read(c['source_contract'])):
        raise ValueError('CHECKPOINT_DATA_CONTRACT_MISMATCH:own prefix')
    r=pc.get('repl_row',dict(L_A=True,L_B=True,structure='full'))
    if any(r[k]!=c['repl_row'][k] for k in ('L_A','L_B','structure')):raise ValueError('Prefix objective/architecture mismatch')
    cp=torch.load(Path(path)/'final_ssdg.pth',map_location='cpu',weights_only=False)
    if (cp['run_id']!=pc['run_id'] or cp['candidate_id']!=pc['row_id'] or cp['epoch']!=done['epoch']
        or cp['args']['teacher_ckpt'] or cp['args']['use_ema_teacher'] or cp['args']['use_unlabeled']
        or cp['a1_ema_successful_updates']!=done['updates']):raise ValueError('CHECKPOINT_PROVENANCE_UNVERIFIED:own prefix payload')
    return dict(folder=str(path),**done)

def train(c,prefix_only=False):
    validate_config(c)
    from experiments.cric_dura_comparison.f_train import train as segment
    from .objectives import installed
    root=Path(c['output_root']);start=time.perf_counter();states=[];segments=[]
    if prefix_only:
        cfg=dict(c,segment_epochs=20,epoch_offset=0,step_offset=0,parent_source=None)
        with installed(cfg,{},False):segment(cfg)
        return
    root.mkdir(parents=True,exist_ok=False);write(root/'resolved_config.json',c)
    previous=c.get('shared_prefix')
    if previous:segments.append(parent(c,previous))
    recipe=((10,20),(20,30)) if previous else ((20,0),(10,20),(20,30))
    for epochs,offset in recipe:
        folder=root/('segment%d'%(offset+epochs));cfg=dict(c,output_root=str(folder),segment_epochs=epochs,
            epoch_offset=offset,step_offset=offset*222,parent_source=previous)
        holder={};enabled=is_repl(c) and offset>=20
        with installed(cfg,holder,enabled):segment(cfg)
        if enabled:
            state=holder['objective'].state_dict()
            if state['calls']!=epochs*222 or state['EMA_updates'] or state['U_truth_read'] or state['student_views']!=6*state['seen']:
                raise ValueError('Complete selected REPL U objective differs')
            if not c['repl_row']['blocks']['style'] and any(state['U_style']['applied'].values()):raise ValueError('Disabled style executed')
            write(folder/'repl_state.json',state);states.append(state)
        done=read(folder/'complete.json');segments.append(dict(folder=str(folder),**done));previous=str(folder)
        write(root/'progress.json',dict(status='SEGMENT_COMPLETE',epoch=offset+epochs,updates=(offset+epochs)*222,
            seconds=time.perf_counter()-start),exclusive=False)
    if len(segments)!=3 or sum(s['updates'] for s in segments)!=11100:raise ValueError('Full E50 trajectory required')
    actual={k:sum(s[k] for s in states) for k in ('calls','seen','accepted','student_views')}
    actual.update(blocks=c['repl_row']['blocks'],U_enabled=c['U_use'],EMA_updates=0,U_truth_read=False,
        default_head_joint=True,new_inference_parameters=0,states=[str(root/('segment%d'%n)/'repl_state.json') for n in (30,50)] if states else [])
    seconds=time.perf_counter()-start
    shutil.copyfile(Path(previous)/'final_ssdg.pth',root/'final_ssdg.pth')
    write(root/'source_contract.json',read(c['source_contract']))
    write(root/'complete.json',dict(status='SOURCE_FROZEN',method_id='pamf_repl' if is_repl(c) else c['repl_row']['method'],
        scaffold_id='pamf_t1_e50',epoch=50,updates=11100,segments=segments,config=c,source_contract=c['source_contract'],
        initialization='same_contract_scratch20_own10_own20',ancestors=[s['folder'] for s in segments],
        target_access=False,U_truth_read=False,head_fit_updates=0,actual_REPL=actual,seconds=seconds,
        cumulative_training_seconds=sum(s['seconds'] for s in segments),reused_prefix_seconds=segments[0]['seconds'] if c.get('shared_prefix') else 0.,
        historical_posthoc_selection_disclosed=True))

@contextmanager
def reader_structure(c):
    from experiments.cric_dura_comparison import f_model
    original=f_model.install
    def install(model,cfg):
        model=original(model,cfg);row=cfg.get('repl_row',{});arm=row.get('structure','full')
        if arm!='full':
            from experiments.cric_dura_comparison.supervised_physics import install as structure
            model=structure(model,arm,cfg['model_seed'])
        return model
    f_model.install=install
    try:yield
    finally:f_model.install=original

def load_reader(folder,contract,device,require_no_unlabeled=False):
    from experiments.cric_dura_comparison.frozen_inference import load_F
    c=read(Path(folder)/'complete.json')['config']
    with reader_structure(c):return load_F(folder,contract,device,require_no_unlabeled=require_no_unlabeled)
