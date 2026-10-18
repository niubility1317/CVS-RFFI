"""Focused prediction routing, real model state roundtrips and truth isolation."""
import argparse,importlib,json,os,pathlib,pickle,subprocess,sys,tempfile
from unittest import mock
import evaluate as p

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--method',choices=['asknet','wavemlp','difl']);parser.add_argument('--source-root',required=True);parser.add_argument('--evidence',required=True);parser.add_argument('--output');a=parser.parse_args()
    if a.method is None:
        rows=[]
        for method in ['asknet','wavemlp','difl']:
            cmd=[sys.executable,'-X','utf8',__file__,'--method',method,'--source-root',a.source_root,'--evidence',a.evidence]
            result=subprocess.run(cmd,capture_output=True,text=True,encoding='utf-8');result.check_returncode()
            rows.append(json.loads(result.stdout.splitlines()[-1]))
        out={'status':'PASS','rows':rows,'input':'synthetic only; no original weights or real source/target read'}
        p.write_json(a.output,out);print(json.dumps(out,indent=2));return
    p.bootstrap({});np=p.np;torch=p.torch;method=a.method
    source=pathlib.Path(a.source_root)/method/('WiSig' if method=='asknet' else '')
    sys.path.insert(0,str(source));evidence=p.read_json(a.evidence)['methods'][method]
    configs=[v['config'] for v in evidence['resolved_configs']]
    if method=='difl':
        from Network1 import my_resnet
        net=my_resnet()
    else:
        native=importlib.import_module('main');cfg=argparse.Namespace(**configs[0])
        net=native.build_model(cfg,6) if method=='asknet' else native._build_model_from_conf(cfg,tx_num=6,wavelet_kernel_size=cfg.wavelet_kernel_size)
    x=torch.randn(8,2,256);net.eval()
    with torch.no_grad():net(x)
    with tempfile.TemporaryDirectory(prefix='old-eval-check-') as temp:
        root=pathlib.Path(temp);path=root/'frozen.pt';torch.save(net.state_dict(),path)
        row={'model_root':str(source),'checkpoint':str(path)};rt={'device':'cpu'}
        with mock.patch.object(p,'checked_row',return_value=(row,{},configs)):restored=p.model(rt,method)
        with torch.no_grad():
            actual=p.logits(restored,x,method);native_out=restored(x)
            expected=native_out[0] if method=='wavemlp' else native_out[1] if method=='difl' else native_out
        assert actual.shape==(8,6) and torch.equal(actual,expected)
        assert all(torch.equal(v,restored.state_dict()[k]) for k,v in net.state_dict().items())
        with mock.patch.object(p.np,'load',side_effect=AssertionError('Truth accessed')):
            try:p.score(root)
            except FileNotFoundError:pass
            else:raise AssertionError('Truth read before prediction completion')
        p.write_json(root/'fixed.json',{})
        try:p.write_json(root/'fixed.json',{})
        except FileExistsError:pass
        else:raise AssertionError('Output overwritten')
        if method=='asknet':
            data=root/'data';folder=data/'author_layout/ManySig/non_equalized/date1';folder.mkdir(parents=True)
            blocks=[np.random.RandomState(i).randn(100,256,2).astype(np.float32) for i in range(6)]
            with (folder/'rx_2-19_data.pkl').open('wb') as h:pickle.dump({'data':blocks},h)
            os.environ['WISIG_DATA_ROOT']=str(data/'author_layout')
            original_x,_=native.load_single_dataset('ManySig',3,1,6,'non_equalized')
            assert np.array_equal(p.raw_source({'data_root':str(data)}),original_x[:8])
    print(json.dumps({'method':method,'status':'PASS','original_head_routing':True,'actual_model_state_roundtrip':True,'truth_not_read_without_complete_predictions':True,'immutable_output':True,'raw_normalization_exact_check':method=='asknet'}))
if __name__=='__main__':main()
