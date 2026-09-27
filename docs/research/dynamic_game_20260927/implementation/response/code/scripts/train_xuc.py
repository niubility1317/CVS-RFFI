from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import argparse
import json
from cvsrffi.xuc_fusion.runtime import resolve_args,train


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--matrix',type=Path,required=True);p.add_argument('--recipe',type=Path,required=True)
    p.add_argument('--row',required=True);p.add_argument('--dataset',required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--source-contract');p.add_argument('--device',default='cuda:0')
    p.add_argument('--synthetic',action='store_true');p.add_argument('--test-epochs',type=int,default=1);p.add_argument('--test-steps',type=int,default=1)
    a=p.parse_args();matrix=json.loads(a.matrix.read_text(encoding='utf-8'))
    row=next(r for r in matrix['runs'] if r['id']==a.row)
    if row['pipeline']!='core90':raise ValueError('native rows must use native bridge')
    args=resolve_args(json.loads(a.recipe.read_text(encoding='utf-8')),row,dataset=a.dataset,output=a.output,
                      device=a.device,synthetic=a.synthetic,test_epochs=a.test_epochs,test_steps=a.test_steps)
    return train(args,row,a.source_contract)


if __name__=='__main__':raise SystemExit(main())
