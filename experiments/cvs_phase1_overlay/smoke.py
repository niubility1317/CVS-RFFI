"""Disposable synthetic checks of the real four-arm forward/backward/channel paths."""
import argparse
from pathlib import Path
import tempfile
import sys
import torch
from experiments.cvs_phase1_overlay.contract import ROOT,ARMS,build_model,write
sys.path[:0]=[str(ROOT),str(ROOT/'code')]
from experiments.cvs_phase1_overlay.augmentation import OriginalLEO,loss_for_batch
from experiments.cvs_equivariant_identity.precision import numerical_context,FULL_FP32_POLICY


def smoke(device='cpu'):
    torch.set_num_threads(2);torch.manual_seed(8137);device=torch.device(device)
    results=[]
    with tempfile.TemporaryDirectory(prefix='reference-overlay-smoke-') as td, numerical_context(FULL_FP32_POLICY):
        x=torch.randn(8,2,256,device=device);y=torch.tensor([0,0,1,1,2,2,3,3],device=device);d=torch.tensor([0,1]*4,device=device)
        batch=dict(meta=[dict(sample_id='synthetic-'+str(i),rx_i=i%2,day_i=1) for i in range(8)])
        for arm in ARMS:
            m=build_model(arm).to(device).train();opt=torch.optim.AdamW(m.parameters(),lr=2e-4)
            aug=OriginalLEO(Path(td)/arm) if 'leo' in arm else None
            observed=set()
            # Exercise native channel in each registered scheduling phase.
            if aug:
                for epoch in [1,41,91]:
                    for step in range(1,25):
                        view=aug(x,batch,epoch,step)
                        if view.applied:
                            assert view.x.shape==x.shape and torch.isfinite(view.x).all()
                            assert not torch.equal(view.x,x)
                            observed.add(view.scenario)
                    if epoch==1:assert 'practical_high' in observed
                assert observed=={'practical_high','practical_mid','practical_low_urban'}
            for epoch in [79,80]:
                opt.zero_grad(set_to_none=True)
                loss,info=loss_for_batch(m,x,y,d,batch,aug,epoch,1);loss.backward()
                grads=[p.grad for p in m.parameters() if p.grad is not None]
                assert grads and all(torch.isfinite(g).all() for g in grads)
                assert (info['satellite_ce'] is not None)==(aug is not None and epoch>=80)
                opt.step()
            m.eval()
            with torch.no_grad():before=m(x)
            p=Path(td)/(arm+'.pt');torch.save(m.state_dict(),p)
            loaded=build_model(arm).to(device).eval();loaded.load_state_dict(torch.load(p,map_location=device,weights_only=True),strict=True)
            with torch.no_grad():after=loaded(x)
            torch.testing.assert_close(before,after,atol=0,rtol=0)
            results.append(dict(arm=arm,parameters=sum(p.numel() for p in m.parameters()),loss=float(loss.detach()),actual_channel_scenes=sorted(observed),checkpoint_roundtrip='EXACT',forward_backward='PASS'))
    return dict(status='PASS',device=str(device),torch=torch.__version__,rows=results,real_data_access=False,checkpoint_ancestors=[])


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--device',default='cpu');p.add_argument('--output',required=True);a=p.parse_args()
    result=smoke(a.device);write(a.output,result);print(result)
