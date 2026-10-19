"""Focused budget, truth-negative, default-head and source-objective checks."""
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile


def main():
    os.environ['CUDA_VISIBLE_DEVICES']=''
    path=Path(__file__).with_name('pipeline.py')
    spec=importlib.util.spec_from_file_location('dg_pipeline',path)
    p=importlib.util.module_from_spec(spec);spec.loader.exec_module(p)
    matrix=p.read(path.with_name('matrix.json'))
    assert len(matrix['rows'])==24 and matrix['allowed_gpus']==[0,1]
    assert len({r['row_id'] for r in matrix['rows']})==24
    for row in matrix['rows']:
        p.safe_row(row)
        for field in ('truth','resume','teacher','target_inputs'):
            bad=dict(row,**{field:'forbidden'})
            try:p.safe_row(bad)
            except ValueError:pass
            else:raise AssertionError('Forbidden source input accepted')
        bad=dict(row,epochs=row['epochs']+1)
        try:p.safe_row(bad)
        except ValueError:pass
        else:raise AssertionError('Wrong budget accepted')
    import numpy as np
    with tempfile.TemporaryDirectory(prefix='phase1-dg-negative-') as td:
        root=Path(td);p.write(root/'complete.json',dict(status='INCOMPLETE'))
        try:p.score(dict(prediction_output=td,truth=str(root/'TRUTH_MUST_REMAIN_CLOSED')))
        except ValueError as e:assert 'Truth closed' in str(e)
        else:raise AssertionError('Incomplete prediction reached truth')
    if '--models' in sys.argv:
        import torch
        artifacts=Path(os.environ['DG_AUTHOR_LOCAL'])
        results=[]
        for family in ('riei','drift','ibrff','sdg'):
            c=next(dict(r) for r in matrix['rows'] if r['family']==family and r['variant']=='full')
            c.update(ibrff_model_root=str(artifacts/'ibrff_pilot_model'),sdg_model_root=str(artifacts/'sdg_model_source'))
            p.seed(c['model_seed']);net=p.build(c)
            x=torch.randn(12,2,256);y=torch.arange(12)%6;rx=torch.arange(12)%5
            if family in ('riei','drift'):
                optimizer=torch.optim.SGD(list(net.ec.parameters())+list(net.rc.parameters()),lr=c['lr']) if family=='riei' else torch.optim.Adam(net.parameters(),lr=c['lr'])
                fed=torch.optim.SGD(net.fed.parameters(),lr=c['lr']) if family=='riei' else None
                before=next(net.parameters()).detach().clone()
                loss=p.ordinary_step(net,c,(x,y,rx),optimizer,fed)
                assert not torch.equal(before,next(net.parameters()).detach())
                if family=='drift':
                    from baselines.drift.losses import negative_mse_separation
                    assert float(negative_mse_separation(torch.ones(2,256),torch.zeros(2,256)))==-256.
            else:
                optimizer=torch.optim.Adam(net.parameters(),lr=c['lr'])
                logits=p.forward(net,c,x);loss=torch.nn.functional.cross_entropy(logits,y)
                if family=='ibrff':
                    from utils.hsic import hsic_normalized_cca
                    hidden,_=net(x);loss=loss+.04*hsic_normalized_cca(hidden.flatten(1),x.flatten(1))
                loss.backward();p.finite(net,loss);optimizer.step()
                if family=='sdg':
                    a=p.author_sdg(c)
                    embedding,logits=net(torch.cat((x,x.flip(-1)),0))
                    contrast=a.SupConLoss()
                    objective=torch.nn.functional.cross_entropy(logits[12:],y)+contrast(embedding.unsqueeze(1),y.repeat(2),adv=False)+100.*contrast(embedding.unsqueeze(1),torch.arange(2).repeat_interleave(12),adv=True)
                    optimizer.zero_grad();objective.backward();p.finite(net,objective);optimizer.step()
                    other=p.build(dict(c,model_seed=2026092702))
                    first=p.build(c)
                    assert not torch.equal(next(other.parameters()),next(first.parameters())), 'SDG import overwrote model seed'
                    ds=a.AugDataset(np.random.randn(12,2,256).astype('float32'),np.arange(12)%6,a.transforms.ToTensor(),[2],[1])
                    for i in range(len(ds)):
                        views,label=ds[i];assert len(views)==2 and all(tuple(v.shape)==(2,256) for v in views)
            net.eval()
            with torch.no_grad():expected=p.forward(net,c,x)
            with tempfile.TemporaryDirectory(prefix='phase1-dg-checkpoint-') as td:
                checkpoint=Path(td)/'real.pt';p.save(checkpoint,net,c,{'same_contract':True},1,1,0.)
                payload=p.load(checkpoint,c,{'same_contract':True})
                copy=p.build(c);copy.load_state_dict(payload['state_dict'],strict=True);copy.eval()
                with torch.no_grad():assert torch.equal(expected,p.forward(copy,c,x))
                try:p.load(checkpoint,c,{'wrong_contract':True})
                except ValueError:pass
                else:raise AssertionError('Mismatched checkpoint contract accepted')
            results.append(family)
        print('MODEL_DEFAULT_HEAD_GRADIENT_CHECKPOINT_PASS',results)
    print('BUDGET_SOURCE_NEGATIVE_TRUTH_NEGATIVE_PASS rows=24 gpus=0,1 seeds=3')


if __name__=='__main__':main()
