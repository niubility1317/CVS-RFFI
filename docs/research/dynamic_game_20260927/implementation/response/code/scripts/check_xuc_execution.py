"""Bounded source-synthetic execution and own-checkpoint reconstruction; never query."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from cvsrffi.xuc_fusion.checkpoints import load_model

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--device',default='cpu')
    p.add_argument('--rows',default='M00,M08,M12');a=p.parse_args()
    release=Path(__file__).resolve().parents[2]
    if a.output.exists():raise FileExistsError(a.output)
    a.output.mkdir(parents=True);results=[]
    for row in a.rows.split(','):
        out=a.output/row
        subprocess.run([sys.executable,str(release/'code/scripts/train_xuc.py'),'--matrix',str(release/'configs/matrix15.json'),
            '--recipe',str(release/'configs/core90_recipe_reference.json'),'--row',row,'--dataset','synthetic',
            '--output',str(out),'--device',a.device,'--synthetic'],cwd=release,check=True)
        model,checkpoint=load_model(out/'final_ssdg.pth',torch.device(a.device),expected_epoch=1)
        with torch.no_grad():value=model(torch.ones(4,2,256,device=a.device),return_aux=True)['tx_logits']
        if value.shape!=(4,6) or not torch.isfinite(value).all():raise ValueError('reconstructed forward failed')
        result=dict(row=row,checkpoint_reconstruction='STRICT_PASS',shape=list(value.shape),target_contact=False,
                    actual_modules={n:getattr(m,'__file__',None) for n,m in sys.modules.items() if n in ('SSDG.train_ssdg','post_stage_cli','cvsrffi.xuc_fusion.runtime')})
        results.append(result)
    (a.output/'acceptance.json').write_text(json.dumps(dict(status='PASS',rows=results,source_synthetic_only=True),indent=2),encoding='utf-8')
    print('XUC_EXECUTION_CHECK_PASS',flush=True)
if __name__=='__main__':main()
