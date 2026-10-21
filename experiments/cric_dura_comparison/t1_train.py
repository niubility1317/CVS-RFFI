"""Same-contract scratch20 + strict own10 + strict own20; fixed final student."""
from pathlib import Path
import shutil
from .io import read,write


def train(c,device='cuda:0',smoke_steps=None):
    from .f_train import train as segment
    root=Path(c['output_root']);root.mkdir(parents=True,exist_ok=False)
    parent=None;segments=[]
    for epochs,offset in ((20,0),(10,20),(20,30)):
        folder=root/('segment%02d'%(offset+epochs))
        cfg=dict(c,output_root=str(folder),segment_epochs=epochs,epoch_offset=offset,step_offset=offset*222,parent_source=parent)
        segment(cfg,device=device,smoke_steps=smoke_steps)
        done=read(folder/'complete.json');segments.append(dict(folder=str(folder),**done));parent=str(folder)
        write(root/'progress.json',dict(status='SEGMENT_COMPLETE',epoch=offset+epochs,updates=(offset+epochs)*222,
            seconds=sum(s['seconds'] for s in segments)),exclusive=False)
    shutil.copyfile(Path(parent)/'final_ssdg.pth',root/'final_ssdg.pth')
    write(root/'source_contract.json',read(c['source_contract']))
    write(root/'complete.json',dict(status='LOCAL_SYNTHETIC_SMOKE_ONLY' if smoke_steps is not None else 'SOURCE_FROZEN',
        method_id='pamf_t1_e50',epoch=50,updates=11100,segments=segments,source_contract=c['source_contract'],config=c,
        initialization='same_contract_scratch20_own10_own20',ancestors=[s['folder'] for s in segments],
        head_fit_updates=0,U_truth_read=False,target_access=False,seconds=sum(s['seconds'] for s in segments)))
