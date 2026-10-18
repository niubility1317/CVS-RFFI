"""Focused source split, unchanged-author loss, checkpoint and truth isolation checks."""
import argparse,contextlib,io,pathlib,tempfile
from unittest import mock
import numpy as np
import torch
import pipeline as p

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--model-root',required=True);parser.add_argument('--candidate',required=True);parser.add_argument('--output',required=True);args=parser.parse_args()
    torch.set_num_threads(2);checks=[];c=p.read_json(args.candidate)
    with tempfile.TemporaryDirectory(prefix='sdg-check-') as tmp:
        root=pathlib.Path(tmp);rt={'model_root':args.model_root,'candidate':args.candidate,'run_root':str(root),'device':'cpu'}
        a=p.author(rt)
        # Compare the exact author get_dataset split; target reads are synthetic.
        x=np.random.RandomState(19).randn(6000,2,256);y=np.repeat(np.arange(6),1000);calls=[]
        def source_only(runtime,conf,day):
            calls.append(day)
            if day!=1:raise AssertionError('Query accessed during source split')
            return x,y
        with mock.patch.object(p,'block',side_effect=source_only):train,val=p.source_data(rt,c)
        with mock.patch.dict(a.get_dataset.__globals__,{'load_dataset_from_file':lambda *args,**kwargs:(x,y)}):expected=a.get_dataset('WiSig')
        assert np.array_equal(train[0],expected['train'][0]) and np.array_equal(train[1],expected['train'][1])
        assert np.array_equal(val[0],expected['val'][0]) and np.array_equal(val[1],expected['val'][1]) and calls==[1]
        ti,vi,si=p.source_indices();assert set(ti).isdisjoint(vi) and set(ti).isdisjoint(si) and set(vi).isdisjoint(si)
        checks.append('author_source_split_exact_3780_1620_600_no_target_read')
        a.setup_seed(2023)
        for op in a.augmentations:
            out=op(x[0].astype(np.float32));assert out.shape==(2,256) and np.isfinite(out).all(),op.__name__
        checks.append('all_seven_unchanged_author_augmentations_work_on_numpy123')
        conf=argparse.Namespace(main_aug_depth=[2],aux_aug_depth=[1],lambda_con=[1.,100.])
        dl=a.DataLoader(a.AugDataset(x[:12],np.arange(12)%6,a.transforms.ToTensor(),[2],[1]),batch_size=32,shuffle=True)
        a.setup_seed(2023);net=p.build(rt);before={k:v.clone() for k,v in net.state_dict().items()}
        opt=torch.optim.Adam(net.parameters(),lr=.001,weight_decay=0);loss=(torch.nn.NLLLoss(),a.SupConLoss())
        with mock.patch.object(torch.cuda,'is_available',return_value=True),mock.patch.object(torch.Tensor,'cuda',lambda self,*args,**kwargs:self),contextlib.redirect_stdout(io.StringIO()):
            a.train(net,loss,dl,opt,1,conf)
        assert all(torch.isfinite(v).all() for v in net.state_dict().values())
        assert any(not torch.equal(v,before[k]) for k,v in net.state_dict().items())
        assert net.fc.out_features==6 and sum(v.numel() for v in net.parameters())==129894
        checks.append('unchanged_author_train_with_original_losses_finite_updates_default_head')
        path=root/'source.pt';p.save(path,net,c,1,.5)
        restored=p.build(rt);restored.load_state_dict(p.load(path,c,'cpu')['state_dict'],strict=True)
        assert all(torch.equal(v,restored.state_dict()[k]) for k,v in net.state_dict().items())
        checks.append('state_dict_roundtrip_exact')
        bad=dict(c,source_days=[2])
        try:p.load(path,bad,'cpu')
        except ValueError:pass
        else:raise AssertionError('Wrong source contract accepted')
        checks.append('checkpoint_source_contract_mismatch_rejected')
        with mock.patch.object(p.np,'load',side_effect=AssertionError('Truth read')):
            try:p.score(root)
            except FileNotFoundError:pass
            else:raise AssertionError('Scorer ran before prediction completion')
        try:p.prepare_query(rt)
        except ValueError:pass
        else:raise AssertionError('Query exported before training freeze')
        checks.append('query_export_after_freeze_and_truth_after_predictions_only')
        p.write_json(root/'immutable.json',{'first':True})
        try:p.write_json(root/'immutable.json',{})
        except FileExistsError:pass
        else:raise AssertionError('Output overwritten')
        checks.append('immutable_artifact_preserved')
    result={'status':'PASS','numpy':np.__version__,'checks':checks,'input':'synthetic IQ only; no author weights, real source or target read'}
    p.write_json(args.output,result);print(__import__('json').dumps(result,indent=2))
if __name__=='__main__':main()
