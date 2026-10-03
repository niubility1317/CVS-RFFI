"""Check all eight real frozen checkpoints before any query load."""
import argparse
from pathlib import Path
import torch
from experiments.cvs_validdual_clean.contracts import expected_rows,prediction_config,read,write,SOURCE_CONTRACT
from experiments.cvs_validdual_clean.provenance import frozen_source_matrix,validate_payload
def smoke(output):
    if output.exists():raise FileExistsError('Preserve checkpoint smoke evidence')
    frozen_source_matrix();torch.set_num_threads(2);expected=read(SOURCE_CONTRACT);rows=[]
    for rid in sorted(expected_rows()):
        c=prediction_config(rid);root=Path(c['source_output'])
        done,initial,contract,resolved=[read(root/name) for name in ('completion.json','initialization.json','source_contract.json','resolved_config.json')]
        payload=torch.load(root/'last.pt',map_location='cpu',weights_only=False)
        model=validate_payload(c,done,initial,contract,expected,resolved,payload).eval()
        with torch.no_grad():scores=model(torch.zeros(2,2,256))
        if scores.shape!=(2,6) or not torch.isfinite(scores).all():raise ValueError('Frozen checkpoint smoke failed')
        rows.append(dict(row_id=rid,status='VERIFIED',parameters=sum(p.numel() for p in model.parameters()),query_access=False))
    write(output,dict(status='VERIFIED',rows=rows,target_access=False));print('VERIFIED8CHECKPOINTS_NO_QUERY')
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();smoke(a.output)
