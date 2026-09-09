"""Actual predictor and independent scorer on a tiny synthetic target fixture."""
import argparse
import json
from pathlib import Path
import sys
from unittest.mock import patch
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from check_a1_mechanism_screen import args_for, build
from run_a1_mechanism_screen import mechanism_matrix
from cvsrffi.a1_periodic_target import evaluate_checkpoint
from cvsrffi.truth_last import build_truth_sidecar, stable_sample_id
from scripts import predict_phase1_truth_last as prediction


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--device',default='cuda:0')
    args=parser.parse_args()
    folder=args.output.with_suffix('')
    folder.mkdir(parents=True,exist_ok=False)
    inputs=folder/'synthetic_inputs';inputs.mkdir()
    truth=folder/'synthetic_truth.json'
    build_truth_sidecar([{'physical_id':f'fixture:{i}','label':i%6} for i in range(12)],
        output_path=truth,split_binding='synthetic_check_only')
    x=torch.randn(12,2,256)
    class Fixture(torch.utils.data.Dataset):
        def __init__(self, root): assert Path(root)==inputs
        def __len__(self): return len(x)
        def __getitem__(self,i):
            return x[i],-1,0,{'physical_sample_id':stable_sample_id(f'fixture:{i}',split_binding='synthetic_check_only')}
    row=next(r for r in mechanism_matrix()['rows'] if r['id']=='E2_RESPONSE_PAIR')
    cfg=args_for(row); model=build(cfg,torch.device(args.device))
    checkpoint=folder/'this_check_fresh.pth'
    torch.save({'model':model.state_dict(),'args':vars(cfg)},checkpoint)
    before={n:v.detach().clone() for n,v in model.state_dict().items()}
    with patch.object(prediction,'_OpaqueTargetDataset',Fixture):
        seconds=evaluate_checkpoint(checkpoint,output=folder/'target_epochs/E080',
            input_package=inputs,truth=truth,run_id='synthetic_check',row_id=row['id'],
            device=args.device,expected_records=48)
    assert all(torch.equal(v,model.state_dict()[n]) for n,v in before.items())
    summary=json.loads((folder/'target_epochs/E080/score.json').read_text())
    assert len(summary['metrics'])==4 and summary['record_count']==48
    result={'status':'PASS','same_training_model_unchanged':True,'records':48,
        'scenarios':list(summary['metrics']),'seconds':seconds,'historical_checkpoints':0,
        'boundary':'synthetic fixture; actual predictor and separate CPU scorer; no performance claim'}
    args.output.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result))


if __name__=='__main__':main()
