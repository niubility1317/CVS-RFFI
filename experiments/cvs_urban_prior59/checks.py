"""Focused next-run checks; synthetic tensors only, no checkpoint/data access."""
import argparse,copy,json,sys,types,unittest
from pathlib import Path

def local_dependencies(path):
 root=Path(__file__).resolve().parents[2];sys.path[:0]=[str(root),str(root/'code')]
 package=types.ModuleType('experiments');package.__path__=[str(Path(path)/'experiments'),str(root/'experiments')];sys.modules['experiments']=package

class Checks(unittest.TestCase):
 def test_sampler_equal_zero_and_boundary_parity(self):
  from . import sampler as s
  import numpy as np
  for seed in range(2000):
   legacy=int(np.random.default_rng(seed).choice(3));got,record=s.choose(np.random.default_rng(seed),s.NOMINAL);zero,_=s.choose(np.random.default_rng(seed),s.bounded_mix(0.))
   self.assertEqual(got,s.SCENES[legacy]);self.assertEqual(zero,got);self.assertEqual(record['legacy_equal_scene'],got)
  class Edge:
   def __init__(self,j,v):self.j=j;self.v=v
   def choice(self,n):return self.j
   def random(self):return self.v
  for j in range(3):
   for v in (0.,float(np.nextafter(1.,0.))):self.assertEqual(s.choose(Edge(j,v),s.NOMINAL)[0],s.SCENES[j])
 def test_sampler_prior_support_and_rng_isolation(self):
  from . import sampler as s
  import numpy as np
  state=np.random.get_state();rng=np.random.default_rng(59);counts={scene:0 for scene in s.SCENES}
  for _ in range(20000):counts[s.choose(rng)[0]]+=1
  for scene,p in zip(s.SCENES,s.URBAN):self.assertLess(abs(counts[scene]/20000-p),.015)
  after=np.random.get_state();self.assertEqual(state[0],after[0]);np.testing.assert_array_equal(state[1],after[1]);self.assertEqual(state[2:],after[2:]);self.assertEqual(s.bounded_mix(1.),s.URBAN)
 def test_invalid_prior(self):
  from . import sampler as s
  for prior in [(0.,.5,.5),(-.1,.5,.6),(.2,.3,.4),(.2,.8),(.2,.4,float('nan')),(True,0.,0.)]:
   with self.assertRaises(ValueError):s.weights(prior)
  for u in [-.1,1.,float('nan')]:
   with self.assertRaises(ValueError):s.inverse_cdf(u,s.URBAN)
  for eta in [-.1,1.1,True,float('inf')]:
   with self.assertRaises(ValueError):s.bounded_mix(eta)
 def test_exact_tail_contract(self):
  from . import design as d
  c=d.config(d.rows()[0]);d.validate(c);self.assertEqual((c['epochs'],c['updates'],c['epoch_offset'],c['step_offset'],c['L_NN_per_step']),(10,2220,20,4440,768));self.assertEqual(c['parent_source'],d.old_config()['parent_source']);self.assertEqual(Path(c['parent_source']).parts[-2:],('B01-s2026092701','stage20'));self.assertEqual(c['scene_prior']['probabilities'],[.2,.4,.4]);self.assertIsNone(c['scene_prior']['q']);self.assertFalse(c['new_Ag']);self.assertTrue(c['As']);self.assertEqual(c['total_L_loss_scale'],1.25)
  args=d.make_args(c,'cpu');self.assertEqual((args.epochs,args.label_epochs,args.pseudo_epochs,args.lr),(10,0,10,.0002));self.assertFalse(args.from_scratch);self.assertEqual(args.baseline_ckpt,c['parent_source']+'/final_ssdg.pth');self.assertFalse(args.teacher_ckpt)
  for key,value in [('model_seed',2026092702),('epochs',20),('updates',4440),('new_Ag',True),('parent_source','/wrong'),('L_NN_per_step',512)]:
   wrong=copy.deepcopy(c);wrong[key]=value
   with self.assertRaises(ValueError):d.validate(wrong)
  wrong=copy.deepcopy(c);wrong['scene_prior']['probabilities']=[1/3]*3
  with self.assertRaises(ValueError):d.validate(wrong)
  with self.assertRaises(ValueError):d.config(d.rows()[0],20)
 def test_final_checkpoint_and_data_boundary(self):
  from . import design as d,protocol as p
  c=d.config(d.rows()[0]);roles={'L_s':['fixtureL'],'U_s':['fixtureU'],'V':['fixtureV']};done=dict(config=c,epoch=10,optimizer_steps=2220,target_access=False);frozen=dict(config=c,selection='final_only',target_access=False);initial=dict(scratch_only=False,target_contact=False,checkpoint_sources=c['checkpoint_sources'],source_roles='EXACT_MATCH');p.final_metadata(c,done,frozen,initial,dict(role_ids=roles),roles)
  cp=dict(epoch=10,a1_ema_successful_updates=2220,run_id=c['run_id'],candidate_id=c['row_id'],checkpoint_selection='final_only',ema_model={},args=dict(baseline_ckpt=c['parent_source']+'/final_ssdg.pth',teacher_ckpt=''));p.checkpoint(c,cp)
  for key,value in [('run_id',d.REFERENCE_RUN),('candidate_id','B01-s2026092701'),('epoch',20),('ema_model',None)]:
   wrong=copy.deepcopy(cp);wrong[key]=value
   with self.assertRaises(ValueError):p.checkpoint(c,wrong)
  wrong=copy.deepcopy(cp);wrong['args']['teacher_ckpt']='/external'
  with self.assertRaises(ValueError):p.checkpoint(c,wrong)
  with self.assertRaises(ValueError):p.final_metadata(c,done,frozen,initial,dict(role_ids={}),roles)
  wrong=copy.deepcopy(done);wrong['target_access']=True
  with self.assertRaises(ValueError):p.final_metadata(c,wrong,frozen,initial,dict(role_ids=roles),roles)
 def test_source_failure_still_requires_all7(self):
  from unittest.mock import patch
  from . import design as d,report
  c=d.config(d.rows()[0]);native=dict(source_val_count=27000,source_val_accuracy=.98);control=dict(source_val_count=27000,source_val_accuracy=.9835925925925926);current=dict(config=c,NN=12600,target_access=False,formal_scores_read=False,training_updates=0,clean_accuracy=.999,source_accuracies=[[.9,.75,.61],[.9,.75,.61]])
  old=dict(config=d.old_config(),target_access=False,formal_scores_read=False,source_accuracies=[[.9,.74,.60],[.9,.74,.60]])
  def read(path):
   if 'URBAN-s' in str(path):return current if 'source_qualification' in str(path) else dict(target_access=False,final_source_metrics=native)
   return old if 'B01-s' in str(path) else dict(target_access=False,final_source_metrics=control)
  with patch.object(d,'read',side_effect=read),patch.object(d,'write') as written:
   value=report.source_decision()
  self.assertFalse(value['source_qualified']);self.assertFalse(value['checks']['fullV_clean_within_0p1pp']);self.assertTrue(value['formal_evaluation_always_required']);self.assertFalse(value['formal_scores_read']);self.assertAlmostEqual(value['fullV_clean_vs_NoAs30_pp'],-.35925925925926);self.assertEqual(written.call_count,1)
 def meta(self,n):
  import torch
  return dict(eq_i=torch.ones(n,dtype=torch.long),rx_i=torch.tensor([1,3,4,6,8]*((n+4)//5))[:n],day_i=torch.arange(n)%3+1,sig_i=torch.arange(n)+2000,base_index=torch.arange(n))
 def test_actual_original_equal_parity_and_rng(self):
  import numpy as np,torch
  from . import channel as c,sampler as s
  old=c.Original();new=c.Practical(prior=s.NOMINAL);x=torch.randn((5,2,256),generator=torch.Generator().manual_seed(59));meta=self.meta(5);state=torch.random.get_rng_state().clone();ns=np.random.get_state()
  a,ra=old.apply(x,meta,30,6660);b,rb=new.apply(x,meta,30,6660)
  self.assertTrue(torch.equal(a,b));self.assertEqual([r['scene'] for r in ra],[r['scene'] for r in rb]);self.assertTrue(torch.equal(state,torch.random.get_rng_state()));np.testing.assert_array_equal(ns[1],np.random.get_state()[1]);self.assertTrue(torch.isfinite(b).all())
  again,rc=new.apply(x,meta,30,6660);self.assertTrue(torch.equal(b,again));_,changed=new.apply(x,meta,30,6659);self.assertNotEqual(rb[0]['namespace'],changed[0]['namespace']);
  for adapter in (old,new):
   with self.assertRaisesRegex(ValueError,'zero-energy'):adapter.apply(torch.zeros(5,2,256),meta,30,6660)
 def test_full_state_lock_support_no_filter(self):
  from unittest.mock import patch
  import torch
  from . import channel as c
  channel=c.Practical();cases=[(state,lock) for state in ['LOS','PARTIAL','BLOCKED'] for lock in ['locked','degraded','unlocked']];it=iter(cases)
  class Probe:
   def __init__(self,*args):pass
   def process(self,x):
    state,lock=next(it);return x,dict(state_end=state,receiver_lock_status=lock,channel_equalization_applied=False)
  with patch.object(channel.residual,'ResidualChannel',Probe):y,records=channel.apply(torch.ones(9,2,256),self.meta(9),21,4441)
  self.assertEqual(len(y),9);self.assertEqual([(r['state_end'],r['receiver_lock_status']) for r in records],cases);self.assertEqual(channel.packets,9);self.assertEqual(sum(channel.coverage.values()),9);self.assertFalse(channel.summary()['state_filter']);self.assertFalse(channel.summary()['receiver_lock_filter'])
 def test_actual_sixL_sameIQ_and_defaulthead(self):
  from .preflight import synthetic
  result=synthetic();self.assertEqual(result['NN'],768);self.assertEqual(result['As_IQ'],64);self.assertTrue(result['same_generated_IQ_each_pair']);self.assertTrue(result['default_head_joint_gradient']);self.assertEqual(result['cumulative_epoch'],30)

def run():
 import torch
 torch.set_num_threads(2)
 suite=unittest.defaultTestLoader.loadTestsFromTestCase(Checks);result=unittest.TextTestRunner(verbosity=2).run(suite)
 if not result.wasSuccessful():raise RuntimeError('Focused urban-prior checks failed')
 return dict(status='PASS',tests=result.testsRun,real_IQ=0,checkpoint_reads=0,target_access=False,optimizer_updates=0)

if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('--dependencies',type=Path,required=True);args=parser.parse_args();local_dependencies(args.dependencies)
 name='experiments.cvs_urban_prior59';spec=__import__('importlib.util',fromlist=['util']).spec_from_file_location(name,Path(__file__).with_name('__init__.py'),submodule_search_locations=[str(Path(__file__).parent)]);package=__import__('importlib.util',fromlist=['util']).module_from_spec(spec);sys.modules[name]=package;spec.loader.exec_module(package);__package__=name
 print(json.dumps(run()))
