"""Scratch, own-seed, contract-bound 20+10+20 training."""
from pathlib import Path
import shutil
import time
from experiments.cric_dura_comparison.io import read,write
from .plan import validate_config
from .model import installed

def saved_segments(c):
    """Reuse only this exact row's immutable, fully saved native segments."""
    parent=c.get('resume_source_root')
    if not parent:return []
    base=Path(parent)
    from .plan import RUN
    if base.parts[-5:]!=('runs',RUN,'F',c['row_id'],'source'):
        raise ValueError('Only the original own-row source may resume')
    original=read(base/'resolved_config.json')
    if Path(original['output_root'])!=base:raise ValueError('Wrong original output')
    for k,v in original.items():
        if k not in ('output_root','code_commit') and c.get(k)!=v:
            raise ValueError('Resume changes original source recipe: '+k)
    result=[];previous=None
    for length,offset in ((20,0),(10,20),(20,30)):
        folder=base/f'segment{offset+length}'
        if not (folder/'complete.json').exists():break
        done=read(folder/'complete.json');cfg=done['config']
        expected=dict(original,output_root=str(folder),segment_epochs=length,epoch_offset=offset,
                      step_offset=offset*222,parent_source=previous)
        if (any(cfg.get(k)!=v for k,v in expected.items()) or done['status']!='SOURCE_FROZEN'
            or done['updates']!=length*222 or done['target_access'] or done['U_truth_read']
            or done['head_fit_updates'] or not (folder/'final_ssdg.pth').is_file()):
            raise ValueError('Saved own-row segment is incomplete or incompatible')
        result.append(dict(folder=str(folder),**done));previous=str(folder)
    return result

def train(c):
    validate_config(c)
    reused=saved_segments(c)
    from experiments.cric_dura_comparison.f_train import train as segment
    root=Path(c['output_root']);root.mkdir(parents=True,exist_ok=False);write(root/'resolved_config.json',c)
    previous=reused[-1]['folder'] if reused else None;chain=list(reused);started=time.monotonic()
    for length,offset in ((20,0),(10,20),(20,30))[len(reused):]:
        folder=root/f'segment{offset+length}'
        cfg=dict(c,output_root=str(folder),segment_epochs=length,epoch_offset=offset,step_offset=offset*222,parent_source=previous)
        with installed(cfg):segment(cfg)
        done=read(folder/'complete.json')
        if done['status']!='SOURCE_FROZEN' or done['updates']!=length*222 or done['target_access'] or done['U_truth_read'] or done['head_fit_updates']:
            raise ValueError('Incomplete source segment or illegal permissions')
        chain.append(dict(folder=str(folder),**done));previous=str(folder)
        write(root/'progress.json',dict(status='SEGMENT_COMPLETE',epoch=offset+length,updates=(offset+length)*222),exclusive=False)
    shutil.copyfile(Path(previous)/'final_ssdg.pth',root/'final_ssdg.pth')
    write(root/'source_contract.json',read(c['source_contract']))
    write(root/'complete.json',dict(status='SOURCE_FROZEN',config=c,epoch=50,updates=11100,segments=chain,
        source_contract=c['source_contract'],initialization='same_contract_scratch20_own10_own20',ancestors=[x['folder'] for x in chain],
        target_access=False,U_truth_read=False,head_fit_updates=0,seconds=time.monotonic()-started,
        cumulative_training_seconds=sum(x['seconds'] for x in chain)))
