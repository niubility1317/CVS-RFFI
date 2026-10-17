"""Protocol negatives and completion-before-source/CP for same-row mechanism."""
import importlib.util,json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
SOURCE=Path(__file__).with_name('cvs_boundary_E30_source_20261017.py')
spec=importlib.util.spec_from_file_location('boundary_source30',SOURCE);s=importlib.util.module_from_spec(spec);spec.loader.exec_module(s)
class SourceMechanismTests(unittest.TestCase):
 def handles(self):
  roles=dict(role='V',counts=dict(L_s=6300,U_s=56700,V=27000),V_panel=1800,held_RX=None,extra_Ldev=False,U_truth_used=False,target_access=False,updates=0)
  obs=dict(status='SOURCE_OBSERVATIONS_COMPLETE',unique_V=1800,draws=2,scenes=list(s.SCENES),physical_draws=10800,target_access=False,training_updates=0,formal_test_changed=False,conditions=['full']+['original_control']*9+['edge_boundary'])
  return roles,obs
 def test_existing_source_handles(self):
  s.validate_handles(*self.handles())
 def test_role_and_control_negatives(self):
  for field,value in [('role','L_s'),('held_RX',8),('extra_Ldev',True),('U_truth_used',True),('target_access',True),('updates',1),('V_panel',720),('counts',dict(L_s=5580,U_s=56700,V=27000))]:
   r,o=self.handles();r[field]=value
   with self.subTest(field=field),self.assertRaises(ValueError):s.validate_handles(r,o)
  for field,value in [('draws',3),('target_access',True),('training_updates',1),('formal_test_changed',True),('conditions',['full']+['control']*9+['edge_total'])]:
   r,o=self.handles();o[field]=value
   with self.subTest(field=field),self.assertRaises(ValueError):s.validate_handles(r,o)
 def test_incomplete_BCTX_blocks_any_source_or_CP(self):
  with tempfile.TemporaryDirectory() as td:
   base=Path(td);roots={arm:base/arm for arm in ('B00','B01','BCTX')}
   views=['clean','practical_high','practical_mid','practical_low_suburban','practical_high_urban','practical_mid_urban','practical_low_urban']
   for arm,root in roots.items():
    rows={'prediction/complete.json':dict(status='PREDICTIONS_COMPLETE',count=168000,views=views,truth_read=False,query_fit=False,all_parameters_frozen=True,head_fit_updates=0),'scoring/complete.json':dict(status='SCORED_COMPLETE',views=views if arm!='BCTX' else views[:-1],truth_last=True),'source/completion.json':dict(epoch=30,optimizer_steps=6660,target_access=False,head_fit_updates=0,config=dict(epochs=30,label_epochs=20,pseudo_epochs=10,updates=6660,held_RX=None,source_roles=dict(L_s=6300,U_s=56700,V=27000)))}
    for name,q in rows.items():
     p=root/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(q),encoding='utf8')
   with patch.object(s,'roots',return_value=roots),patch.object(s,'panel') as panel,patch('sys.argv',['source','--commit','synthetic']),patch.object(Path,'mkdir') as mkdir,self.assertRaises(ValueError):s.main()
   panel.assert_not_called();mkdir.assert_not_called()
if __name__=='__main__':unittest.main()
