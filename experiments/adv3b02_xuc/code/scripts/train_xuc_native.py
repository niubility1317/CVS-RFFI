import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from cvsrffi.xuc_fusion.native import train_native

def main():
    p=argparse.ArgumentParser()
    for name in ['reference','row','dataset','output','source-contract']:p.add_argument('--'+name,required=True)
    p.add_argument('--validate-only',action='store_true')
    a=p.parse_args()
    result=train_native(json.loads(Path(a.reference).read_text(encoding='utf-8')),a.row,a.dataset,a.output,a.source_contract,a.validate_only)
    if a.validate_only:print(json.dumps(result,ensure_ascii=False,default=str))
    return 0
if __name__=='__main__':raise SystemExit(main())
