"""Bounded scratch-checkpoint and original-channel execution smoke, no query."""
import argparse
import json
import sys
from copy import deepcopy
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
import torch.nn.functional as F
from scripts.train_rc4_practical import build_args, resolved_config, native
from cvsrffi.original_leo import FusedCleanSatelliteForward
from cvsrffi.practical_adapter import PRACTICAL as ORIGINAL, practical_config, set_smoke_context
import cvsrffi.practical_adapter as adapter
from cvsrffi.eval import apply_sat_channel_for_scenario, make_sat_config
from cvsrffi.checkpoint_loading import build_exact_ssdg_model_from_checkpoint


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--device',default='cpu');a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    a.output.mkdir(parents=True)
    args=build_args(a.config,'unused-source',str(a.output/'unused-train'),'unused-contract','unused-inputs','unused-truth','smoke')
    device=torch.device(a.device);torch.manual_seed(392005);torch.set_num_threads(2)
    model_args=native.merge_checkpoint_args({'model':None,'args':{},'stats':{},'split_info':None},args,input_len=256,num_domains=15)
    model_args=native._apply_model_cli_args(model_args,args)
    model=native.build_baseline_model(model_args,device)
    payload={'epoch':100,'args':resolved_config(args),'model':model.state_dict()}
    path=a.output/'scratch_smoke.pth';torch.save(payload,path)
    model,_=build_exact_ssdg_model_from_checkpoint(torch.load(path,map_location='cpu',weights_only=False),input_len=256,device=device)
    model.train();x=torch.randn(4,2,256,device=device);y=torch.tensor([0,1,2,3],device=device);d=torch.arange(4,device=device)
    calls=[];hook=model.register_forward_pre_hook(lambda _m,inputs:calls.append(list(inputs[0].shape)))
    results=[]
    for scene in ORIGINAL[1:]:
        cfg=practical_config(scene,args)
        set_smoke_context(len(x))
        sat,_=apply_sat_channel_for_scenario(x,scene,args,gen=torch.Generator(device=device).manual_seed(17),return_meta=False)
        assert torch.isfinite(sat).all() and not torch.equal(x,sat)
        fused=FusedCleanSatelliteForward(model,sat)
        clean=fused(x,y_tx=y,domain_labels=d,grl_lambda=1.,return_aux=True)
        loss=F.cross_entropy(clean['tx_logits'],y)+.68*F.cross_entropy(fused.satellite_output['tx_logits'],y)
        model.zero_grad(set_to_none=True);loss.backward()
        assert torch.isfinite(loss) and all(torch.isfinite(v.grad).all() for v in model.parameters() if v.grad is not None)
        results.append({'scene':scene,'channel_model':cfg.processing_route,'equalization_configured':cfg.equalization_enabled,'equalizer_method':cfg.equalizer_method,'channel_evidence':adapter._last_meta,'loss':float(loss.detach()),'forward_shape':calls[-1]})
    hook.remove();assert calls==[[8,2,256]]*3
    ema=deepcopy(model).eval()
    for param in ema.parameters():param.requires_grad=False
    called=[]
    def checked_channel(iq,scene,*values,**kwargs):
        assert scene in ORIGINAL[1:]
        called.append(scene)
        return apply_sat_channel_for_scenario(iq,scene,*values,**kwargs)
    model.zero_grad(set_to_none=True)
    student=model(x,y_tx=y,domain_labels=d,grl_lambda=1.,return_aux=True)
    with torch.no_grad():teacher=ema(x,y_tx=None,domain_labels=d,return_aux=True)
    labeled=native._compute_daot_labeled_step(model=model,ema_model=ema,student_clean=student,
        x_clean=x,y_clean=y,d_clean=d,args=args,epoch=21,batch_idx=1,apply_sat_fn=checked_channel,prototype_matrix=None)
    unlabeled=native._compute_daot_unlabeled_step(model=model,ema_model=ema,teacher_clean=teacher,student_strong=student,
        x_unlabeled=x,d_unlabeled=d,args=args,epoch=21,batch_idx=1,apply_sat_fn=checked_channel,prototype_matrix=None)
    daot_loss=labeled['loss']+unlabeled['loss']
    assert torch.isfinite(daot_loss) and daot_loss>0
    daot_loss.backward()
    assert all(torch.isfinite(v.grad).all() for v in model.parameters() if v.grad is not None)
    assert called and set(called)=={'practical_high'}
    from torch.utils.data import Dataset,DataLoader
    class SourceValidation(Dataset):
        def __len__(self):return 7
        def __getitem__(self,i):return torch.randn(2,256),i%6,0,{'base_index':i,'rx_i':1,'day_i':1}
    args.sat_train_protocol_scenario_list=list(ORIGINAL[1:]);args.eval_max_batches=0
    validation=native._evaluate_source_val_tail_geometry(model,
        {'val_loader':DataLoader(SourceValidation(),batch_size=3),'domain_label_map':{0:0}},device,args)
    assert isinstance(validation,dict)
    # Full batch time/metadata proves the actual route, including EQ configured vs applied.
    import time
    set_smoke_context(128); started=time.monotonic()
    apply_sat_channel_for_scenario(torch.randn(128,2,256,device=device),ORIGINAL[1],args,gen=torch.Generator(device=device).manual_seed(27))
    batch_seconds=time.monotonic()-started
    (a.output/'acceptance.json').write_text(json.dumps({'status':'PASS','batch128_channel_seconds':batch_seconds,'scratch_checkpoint':str(path),'query_inputs':0,
        'target_inputs':0,'daot':args.use_adv3b02_daot_stn,'rc4':args.fasttrust_rc4,'results':results,
        'daot_labeled_loss':float(labeled['loss'].detach()),'daot_unlabeled_loss':float(unlabeled['loss'].detach()),
        'daot_scenes_called':called,'source_validation_small_last_batch':'PASS'},indent=2),encoding='utf-8')
    print('PASS genuine concat, practical, DAOT, source validation, scratch checkpoint, no query')

if __name__=='__main__':main()
