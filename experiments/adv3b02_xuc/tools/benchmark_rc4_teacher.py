"""Random-initialized EMA-equivalent teacher: exact outputs and bounded CUDA timing."""
from pathlib import Path
from copy import deepcopy
import argparse,json,statistics,sys,time
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'code'))
from scripts.train_rc4_practical import native,build_args

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--batch-size',type=int,default=512);p.add_argument('--repeats',type=int,default=15)
    a=p.parse_args();path=Path(a.output)
    if path.exists():raise FileExistsError(path)
    if not torch.cuda.is_available():raise RuntimeError('CUDA benchmark requires local CUDA')
    torch.set_num_threads(1);torch.manual_seed(913)
    args=build_args(ROOT/'configs/rc4_practical_residual_noeq_20260918.json','unused','unused','unused','unused','unused','synthetic_teacher')
    args.a1_runtime_fast=False
    ma=native.merge_checkpoint_args({'model':None,'args':{},'stats':{},'split_info':None},args,input_len=256,num_domains=15)
    model=native.build_baseline_model(native._apply_model_cli_args(ma,args),torch.device('cuda')).eval()
    for v in model.parameters():v.requires_grad_(False)
    x=torch.randn(a.batch_size,2,256,device='cuda');domains=torch.arange(len(x),device='cuda')%15
    result={'scope':'random model eval EMA teacher component, not training step','torch':torch.__version__,'device':torch.cuda.get_device_name(0),'batch_size':len(x),'repeats':a.repeats,'cases':[]}
    def call(enabled):
        args.a1_rc4_teacher_identity_only=enabled
        return native._forward_rc4_teacher(model,x,domain_labels=domains,grl_lambda=.7,args=args,is_ema_teacher=True)
    with torch.no_grad():
        for amp in [False,True]:
            state=deepcopy(model.state_dict());rng=torch.get_rng_state().clone();gpu_rng=torch.cuda.get_rng_state_all()
            with torch.autocast('cuda',dtype=torch.float16,enabled=amp):
                old,new=call(False),call(True)
                for k in ['z_id','tx_logits']:torch.testing.assert_close(old[k],new[k],rtol=0,atol=0)
                for _ in range(5):call(False);call(True)
                times={False:[],True:[]}
                for rep in range(a.repeats):
                    for enabled in ([False,True] if rep%2==0 else [True,False]):
                        torch.cuda.synchronize();start=time.perf_counter();call(enabled);torch.cuda.synchronize();times[enabled].append(time.perf_counter()-start)
            assert all(torch.equal(v,model.state_dict()[k]) for k,v in state.items())
            assert torch.equal(rng,torch.get_rng_state()) and all(torch.equal(u,v) for u,v in zip(gpu_rng,torch.cuda.get_rng_state_all()))
            result['cases'].append({'amp':amp,'full_seconds':times[False],'identity_seconds':times[True],'full_median_s':statistics.median(times[False]),'identity_median_s':statistics.median(times[True]),'speedup':statistics.median(times[False])/statistics.median(times[True]),'exact_logits_z_state_rng':True})
    result['status']='VERIFIED';path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding='utf-8',newline='\n') as f:f.write(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='cases'}));print([(x['amp'],x['speedup']) for x in result['cases']])

if __name__=='__main__':main()
