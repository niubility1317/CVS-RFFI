"""Reuse original native50 runner, disabling all separately fitted probes."""
import argparse
from pathlib import Path
from types import SimpleNamespace
import torch
from experiments.cvs_f_rxmargin50.source import train

def run(c):
    from . import design as d,model as api
    from .observer import Observer
    from experiments.cvs_f_txgeometry20.source import validate as source_metrics
    d.validate(c);profile=d.read(c['runtime_profile'])
    def final_source(m,loader,device):
        value=source_metrics(m,loader,device)
        value.update(source_val_count=value['count'],source_val_accuracy=value['accuracy'],source_val_worst_rx=value['worst_RX'])
        return value
    view=SimpleNamespace(**{key:getattr(d,key) for key in ('validate','make_args','read','write','FULL_FP32_POLICY','ROOT','UPDATES')},SOURCE=Path(profile['dataset']['source_contract']),source_validation=final_source)
    observer=Observer(c);train(c,design=view,model_api=api,observer=observer);observer.finish()
    out=Path(c['output_root']);done=d.read(out/'completion.json');cp=torch.load(out/'final_ssdg.pth',map_location='cpu',weights_only=False)
    from .baseline import assert_recipe
    assert_recipe(SimpleNamespace(**cp['args']))
    if done['optimizer_steps']!=11100 or done['logged_steps']!=11100 or done['head_fit_updates'] or done['target_access'] or cp['a1_ema_successful_updates']!=11100 or cp['epoch']!=50 or not cp.get('ema_model'):raise ValueError('Original50/defaulthead/ownEMA budget differs')
    initial=torch.load(out/'initial_smoke.pt',map_location='cpu',weights_only=False)
    W='id_backbone.encoder.core.id_backbone.cls_head.weight'
    if torch.equal(initial['model'][W],cp['model'][W]):raise ValueError('Default head failed joint training')
    d.write(out/'local_source_complete.json',dict(status='SOURCE_FROZEN',config=c,epoch=50,optimizer_steps=11100,source_commit=(d.ROOT/'release_commit.txt').read_text().strip(),sourceV=done['final_source_metrics'],head_fit_updates=0,source_roles='EXACT_MATCH',initialization='scratch',ancestors=[],U_TX_truth_hidden=True,registered_U_stage_exposure=963900,actual_source_packet_reads={k:observer.ctx[v].dataset.samples_read for k,v in [('L_s','train_loader'),('U_s','unlabeled_loader'),('V','val_loader')]},elapsed_seconds=done['elapsed_seconds'],peak_cuda_MiB=torch.cuda.max_memory_allocated()/1024**2,satellite_training=False,default_head_joint=True,target_access=False))
    return done

if __name__=='__main__':
    from . import design as d
    p=argparse.ArgumentParser();p.add_argument('--row',required=True);p.add_argument('--profile',required=True);p.add_argument('--metadata',required=True);p.add_argument('--base',required=True);a=p.parse_args()
    run(d.config(next(r for r in d.rows() if r['row_id']==a.row),a.base,a.profile,a.metadata))
