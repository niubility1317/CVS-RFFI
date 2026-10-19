"""Focused negative and actual factory checks; no source IQ or target access."""
import argparse,copy,importlib,json,sys
from pathlib import Path
from . import design as d

def reject(fn):
 try:fn()
 except (ValueError,KeyError):return
 raise AssertionError('Invalid matrix/config accepted')

def check(identifier):
 r=d.row(identifier)
 from .binding import configured,modules,factory_seed
 b=configured(r);c=b['config'](30);m=b['design'];m.validate(c)
 a=m.make_args(c,'cpu')
 assert (a.seed,c['model_seed'])==(r['model_seed'],r['model_seed'])
 assert a.epochs==10 and a.label_epochs==0 and a.pseudo_epochs==10
 assert c['updates']==2220 and c['test_views']==['clean']
 assert not a.teacher_ckpt and not a.freeze_backbone
 for key,val in [('model_seed',2026092701),('epochs',9),('parent_source','wrong'),('row_id','other')]:
  bad=copy.deepcopy(c);bad[key]=val;reject(lambda:m.validate(bad))
 if r['family'] not in ('pa','fir'):
  c20=b['config'](20);m20=b.get('twenty',m);m20.validate(c20);a20=m20.make_args(c20,'cpu')
  assert (a20.epochs,a20.label_epochs,a20.pseudo_epochs,a20.seed)==(20,13,7,r['model_seed'])
  assert not a20.baseline_ckpt and not a20.teacher_ckpt and c20['updates']==4440
  assert Path(c['parent_source'])==Path(c20['output_root'])
 if r['family']=='fir':
  parent=b['twenty'].config(next(x for x in b['twenty'].rows() if x['arm']=='B01'))
  assert parent['seed_completion']['group']=='HALF_AS'
  assert c['parent_source']==parent['output_root']
 full=importlib.import_module('experiments.cvs_fullsource_concat20.design');before=full.SEED
 with factory_seed(r['model_seed']):assert full.base_model_config()['model_seed']==r['model_seed']
 assert full.SEED==before
 b,api=modules(r)
 import torch
 torch.set_num_threads(2)
 model=api.build(c20 if r['family']=='u' else c,'cpu').eval()
 assert model._selected_factory_seed==r['model_seed']
 with torch.no_grad():z=model.forward_identity_only(torch.zeros(2,2,256))['tx_logits']
 assert z.shape==(2,6) and torch.isfinite(z).all()
 assert full.SEED==before
 print(json.dumps(dict(status='PASS',row_id=identifier,actual_factory_seed=model._selected_factory_seed,NN=2,query_read=False)))

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--row');a=p.parse_args()
 assert len(d.rows())==16 and len({r['row_id'] for r in d.rows()})==16
 assert d.GPU_IDS==(4,6) and d.MAX_GPU_WORKERS==3 and d.VIEWS==('clean',)
 reject(lambda:d.row('B01-s2026092701'));reject(lambda:d.row('AS_FULL-s2026092702'))
 if a.row:check(a.row)
 else:print('PASS exact16-row matrix, two GPUs, three workers/card, clean-only')
