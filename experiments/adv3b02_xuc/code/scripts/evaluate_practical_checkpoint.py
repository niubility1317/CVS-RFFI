"""Frozen historical checkpoint evaluation; no training or checkpoint selection."""
import argparse,gc,json,os,subprocess,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from cvsrffi.xuc_fusion.checkpoints import load_model
from scripts.predict_phase1_truth_last import main as predict

def main():
    p=argparse.ArgumentParser()
    for key in ('checkpoint','recipe','input-package','truth','output','run-id','row-id','scenarios'):
        p.add_argument('--'+key,required=True)
    p.add_argument('--epoch',type=int,required=True)
    a=p.parse_args();output=Path(a.output)
    if output.exists():raise FileExistsError(output)
    torch.set_num_threads(1)
    # Actual frozen checkpoint smoke on synthetic IQ, before target access.
    model,_=load_model(Path(a.checkpoint),torch.device('cpu'),expected_epoch=a.epoch)
    model.eval()
    with torch.no_grad():
        value=model(torch.zeros(2,2,256),y_tx=None,return_aux=True)['tx_logits']
        if value.shape!=(2,6) or not torch.isfinite(value).all():raise ValueError('Checkpoint smoke failed')
    del model,value;gc.collect()
    start=time.time()
    predict(['--checkpoint',a.checkpoint,'--recipe',a.recipe,'--input-package',a.input_package,
        '--output-root',a.output,'--run-id',a.run_id,'--row-id',a.row_id,'--mode','predict',
        '--device','cuda:0','--num-workers','0','--batch-size','256','--expected-epoch',str(a.epoch),
        '--scenarios',a.scenarios,'--identity-only','--practical-cpu-pipeline','--practical-prefetch'])
    command=[sys.executable,str(Path(__file__).with_name('score_phase1_truth_last.py')),
        '--predictions',str(output/'predictions.json'),'--truth',a.truth,
        '--output',str(output/'score.json'),'--scenarios',a.scenarios]
    with (output/'scorer.log').open('x') as log:
        subprocess.run(command,env=dict(os.environ,CUDA_VISIBLE_DEVICES=''),stdout=log,stderr=subprocess.STDOUT,check=True)
    count=json.loads((output/'score.json').read_text())['record_count']
    expected=len(json.loads((Path(a.input_package)/'manifest.json').read_text())['sample_ids'])*len(a.scenarios.split(','))
    if count!=expected:raise ValueError('Incomplete coverage')
    (output/'completion.json').write_text(json.dumps(dict(status='ARTIFACTS_COMPLETE',checkpoint=a.checkpoint,
        epoch=a.epoch,record_count=count,seconds=time.time()-start,feeds_training=False),indent=2)+'\n')

if __name__=='__main__':main()
