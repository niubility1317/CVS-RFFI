"""Scratch, own-seed, contract-bound 20+10+20 training."""
from pathlib import Path
import shutil
import time
from experiments.cric_dura_comparison.io import read,write
from .plan import validate_config
from .model import installed

def train(c):
    validate_config(c)
    from experiments.cric_dura_comparison.f_train import train as segment
    root=Path(c['output_root']);root.mkdir(parents=True,exist_ok=False);write(root/'resolved_config.json',c)
    previous=None;chain=[];started=time.monotonic()
    for length,offset in ((20,0),(10,20),(20,30)):
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
