"""Resolve a prepared row by default. Execution is explicit and never automatic."""
import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'code'))
from cvsrffi.xuc_fusion.joint_config import make_row
from cvsrffi.xuc_fusion.runtime import resolve_args,train

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,required=True)
    p.add_argument('--dataset',default='ManySig.pkl')
    p.add_argument('--output',required=True)
    p.add_argument('--source-contract')
    p.add_argument('--device',default='cuda:0')
    p.add_argument('--execute',action='store_true',help='Only after separate user authorization to start experiments')
    a=p.parse_args();doc=json.loads(a.config.read_text(encoding='utf-8'))
    row=make_row(doc['row']['id'],doc['row']['joint'])
    recipe=json.loads((ROOT/'configs/core90_recipe_reference.json').read_text(encoding='utf-8'))
    args=resolve_args(recipe,row,dataset=a.dataset,output=a.output,device=a.device)
    if not a.execute:
        print(json.dumps(dict(status='VALIDATED_NOT_LAUNCHED',row=row,resolved=vars(args)),ensure_ascii=False));return 0
    if not a.source_contract:raise ValueError('formal training requires the existing source physical-role contract')
    return train(args,row,a.source_contract)
if __name__=='__main__':raise SystemExit(main())
