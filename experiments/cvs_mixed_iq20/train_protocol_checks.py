import copy
import json
from . import train_design as d


def run():
    c=d.config(d.rows()[0]);cases=[('epochs',50),('updates',11100),('parent_checkpoint','old.pth'),
        ('checkpoint_sources',['old.pth']),('total_epochs',40),('label_epochs',20),('pseudo_epochs',0),
        ('source_roles',dict(L_s=5580,U_s=56700,V=27000)),('test_views',['clean']),
        ('separate_head_training',True),('model_seed',2026092702)]
    for key,value in cases:
        changed=copy.deepcopy(c);changed[key]=value
        try:d.validate(changed)
        except ValueError:pass
        else:raise ValueError('Fixed contract/budget mutation accepted: '+key)
    altered=copy.deepcopy(d.config(d.rows()[-1]));altered['training_mechanism']['compose_A_s_A_g']=True
    try:d.validate(altered)
    except ValueError:pass
    else:raise ValueError('Unregistered cascade accepted')
    from . import train_evaluate as e
    import os
    env=os.environ.get('CUDA_VISIBLE_DEVICES');os.environ['CUDA_VISIBLE_DEVICES']='0'
    try:
        try:e.score(d.rows()[0]['row_id'])
        except ValueError as exc:
            if 'CPU scorer' not in str(exc):raise
        else:raise ValueError('GPU scorer accepted')
    finally:
        if env is None:os.environ.pop('CUDA_VISIBLE_DEVICES',None)
        else:os.environ['CUDA_VISIBLE_DEVICES']=env
    result=dict(status='PASS',negative_cases=13,real_IQ_reads=0,truth_reads=0,checkpoint_loads=0,optimizer_steps=0)
    print(json.dumps(result));return result
