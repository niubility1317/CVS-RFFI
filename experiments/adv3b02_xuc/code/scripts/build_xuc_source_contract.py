import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from cvsrffi.game_tracking.data import build_source
from cvsrffi.game_tracking.runtime import json_write
from cvsrffi.xuc_fusion.runtime import resolve_args,validate_source_contract

def main():
    p=argparse.ArgumentParser()
    for key in ['matrix','recipe','dataset','output']:p.add_argument('--'+key,required=True)
    a=p.parse_args();out=Path(a.output)
    if out.exists():raise FileExistsError(out)
    matrix=json.loads(Path(a.matrix).read_text(encoding='utf-8'))
    recipe=json.loads(Path(a.recipe).read_text(encoding='utf-8'))
    args=resolve_args(recipe,matrix['runs'][0],dataset=a.dataset,output=out.parent,device='cpu')
    source=build_source(args);validate_source_contract(source)
    json_write(out,source.info)
    print(json.dumps(dict(status='VERIFIED',counts=source.info['counts'],roles='source_only_contiguous',target_access=False)))
if __name__=='__main__':main()
