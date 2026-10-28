import json
import os
from pathlib import Path

def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))

def write(path, value, exclusive=True):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x' if exclusive else 'w',encoding='utf-8') as f:
        json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False)
        f.write('\n')

def atomic(path,value):
    path=Path(path);tmp=path.with_name(path.name+'.'+str(os.getpid())+'.tmp')
    write(tmp,value);os.replace(tmp,path)

def activate(rt):
    """Retain the exact Phase1 implementation, including its immutable dependencies."""
    import sys
    import experiments
    root=Path(__file__).resolve().parents[2]
    old=Path(rt['phase1_release'])
    experiments.__path__=[str(root/'experiments'),str(old/'experiments')]
    from experiments.pamf_repl_view_shuffle.runner import activate as native
    native(rt)
    experiments.__path__=[str(root/'experiments'),str(old/'experiments')]+list(experiments.__path__)
    sys.path[:0]=[str(root),str(root/'code')]

def require(condition,message):
    if not condition:raise ValueError(message)

def validate_split(split,manifest,count):
    allowed={'protocol_schema','phase2_data_status','capsule_id','split_id','receiver','scenario',
             'k','registered_classes','support_indices','support_labels','query_indices','support_seed'}
    require(set(split)==allowed,'Unexpected split input, including potential query truth')
    require(split['protocol_schema']=='p2_min_v1' and split['phase2_data_status']=='VALIDATED_ONCE'
            and split['capsule_id']==manifest['capsule_id'],'Wrong capsule/split contract')
    classes=split['registered_classes'];sx=split['support_indices'];qy=split['query_indices'];y=split['support_labels'];k=split['k']
    require(classes and len(classes)==len(set(classes)),'Duplicate registered classes')
    require(type(k) is int and k>0 and len(sx)==k*len(classes) and len(y)==len(sx),'Invalid K/support')
    require(all(type(i) is int and 0<=i<count for i in sx+qy),'Invalid physical position')
    require(len(sx)==len(set(sx)) and len(qy)==len(set(qy)) and not set(sx)&set(qy),'Support/query overlap')
    require(set(y)==set(range(len(classes))) and all(y.count(c)==k for c in range(len(classes))),'Invalid labels/class balance')
    return split
