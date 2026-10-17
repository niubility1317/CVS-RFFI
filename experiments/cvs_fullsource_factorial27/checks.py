"""Focused fixed-map, composition, full-role and actual defaulthead graph tests."""
import copy,io,json,math,tempfile,unittest
from dataclasses import asdict
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from . import design as d,model as api
from .attenuation import attenuate,LocalAttenuation
from .observer import Observer
from .qualification import decide
from experiments.cvs_fullsource_concat30_25 import design as reference,model as baseline
from SSDG import train_ssdg as native
class Checks(unittest.TestCase):
 def test_protocol_and_source_verdict(self):
  self.assertEqual([r['arm'] for r in d.rows()],['B10','B11'])
  for row in d.rows():
   c=d.config(row);d.validate(c);a=d.make_args(c,'cpu')
   self.assertEqual((a.epochs,a.label_epochs,a.pseudo_epochs),(30,20,10));self.assertTrue(c['Ag']);self.assertEqual(c['As'],row['arm']=='B11')
   for change in [dict(epochs=20),dict(updates=4440),dict(held_RX=1),dict(source_RX=[3,4,6,8]),dict(source_roles=dict(L_s=5580,U_s=56700,V=27000)),dict(checkpoint_sources=['old']),dict(checkpoint_selection='target_best'),dict(local_attenuation=dict(c['local_attenuation'],width=32)),dict(local_attenuation=dict(c['local_attenuation'],relative_L2_max=.2)),dict(Ag=False),dict(As=not c['As']),dict(test_views=['clean'])]:
    with self.subTest(change=change),self.assertRaises(ValueError):d.validate(dict(c,**change))
  good=np.full((2,2),.70);ref=np.full((2,2),.68)
  self.assertTrue(decide('B11',.98,.98,good,ref)['source_replication_evidence'])
  self.assertFalse(decide('B11',.97,.98,good,ref)['source_replication_evidence'])
  bad=good.copy();bad[1,1]=.68;self.assertFalse(decide('B11',.98,.98,bad,ref)['source_replication_evidence'])
  self.assertTrue(decide('B10',.98,.98,np.full((2,2),.3),ref)['source_replication_evidence'])
  with self.assertRaises(ValueError):decide('B11',.98,.98,good[:1],ref)
 def test_fixed_map_all_origins_and_RNG(self):
  torch.set_num_threads(2);g=torch.Generator().manual_seed(2701);x=torch.randn(241,2,256,generator=g);starts=torch.arange(241)
  out,stats=attenuate(x,starts)
  for i,start in enumerate(starts.tolist()):
   w=torch.zeros(256,dtype=torch.float64)
   for k in range(16):w[start+k]=math.sin(math.pi*(k+.5)/16)**2
   energy=float(x[i].double().square().sum());weighted=float((x[i].double()*w).square().sum());depth=min(.5,.1*(1-2e-6)*math.sqrt(energy/weighted));expected=x[i].double()*(1-depth*w)
   self.assertTrue(torch.allclose(out[i].double(),expected,atol=3e-7,rtol=2e-7));mask=torch.ones(256,dtype=torch.bool);mask[start:start+16]=False;self.assertTrue(torch.equal(out[i,:,mask],x[i,:,mask]))
  self.assertLessEqual(max(stats['relative_L2']),.1+1e-7)
  zero,zstats=attenuate(torch.zeros(8,2,256),torch.arange(8));self.assertEqual(zstats['relative_L2'],[0.]*8);self.assertEqual(int(zero.count_nonzero()),0)
  rx=torch.tensor(([1,3,4,6,8]*26)[:128]);meta=dict(eq_i=torch.ones(128,dtype=torch.long),rx_i=rx,day_i=torch.arange(128)%3+1,sig_i=torch.arange(128),base_index=torch.arange(128));sample=x[:128]
  a=LocalAttenuation(20261017111);b=LocalAttenuation(20261017111);before=torch.get_rng_state().clone();npbefore=np.random.get_state()
  y,_,st=a.apply(sample,meta,1);z,_,zt=b.apply(sample,dict(meta,sig_i=meta['sig_i'].flip(0)),1)
  self.assertTrue(torch.equal(y,z));self.assertEqual(st,zt);self.assertTrue(torch.equal(before,torch.get_rng_state()));self.assertTrue(np.array_equal(npbefore[1],np.random.get_state()[1]))
  _,_,nextst=a.apply(sample,meta,2);self.assertNotEqual(st['starts'],nextst['starts'])
  a.calls=6660
  for step in [0,6661,True,1.]:
   with self.subTest(step=step),self.assertRaises(ValueError):a.apply(sample,meta,step)
 def test_actual_native_concat_composition(self):
  from leo_practical.channel import Config,receiver_for_session,normalize_input,stable_seed
  from leo_practical.residual import ResidualChannel
  from cvsrffi import practical_adapter as pa
  torch.set_num_threads(2);x=torch.randn(128,2,256,generator=torch.Generator().manual_seed(2702));y=torch.arange(128)%6;rx=torch.tensor(([1,3,4,6,8]*26)[:128]);day=torch.arange(128)%3+1;domains=torch.tensor([{1:0,3:3,4:6,6:9,8:12}[int(r)]+int(t)-1 for r,t in zip(rx,day)]);meta=dict(eq_i=torch.ones(128,dtype=torch.long),rx_i=rx,day_i=day,sig_i=torch.arange(128),base_index=torch.arange(128));calls=[]
  def physical(v,scene,args,gen=None,return_meta=True):
   cfg=Config(fs_hz=25000000.,fc_hz=2450000000.,scenario=scene,processing_route='residual',mode='post_sync',equalization_enabled=False,reference_snr_db=30.,boundary='reflect');raw=np.asarray(v.tolist(),np.float64);result=[];records=[]
   for j in range(128):
    wave,_=normalize_input(raw[j,0]+1j*raw[j,1]);receiver=receiver_for_session(cfg,2026092701,'synthetic_rx'+str(int(rx[j]))+'_day'+str(int(day[j])));seed=stable_seed(202610172201,'synthetic source',scene,str(j));out,record=ResidualChannel(cfg,seed,receiver).process(wave);self.assertFalse(record['channel_equalization_applied']);records.append(record);result.append(np.stack((out.real,out.imag)).tolist())
   pa._last_meta=dict(config=asdict(cfg),records=records);out=torch.tensor(result,dtype=v.dtype);calls.append((v.clone(),out.clone(),scene));return out,dict()
  with tempfile.TemporaryDirectory() as td:
   for row in d.rows():
    c=d.config(row);o=Observer(c);o.out=Path(td)/row['arm'];o.out.mkdir();o.sequence=io.StringIO();o.physics=io.StringIO();o.local_log=io.StringIO();o.record('L_s',(x,y,domains,meta));before=len(calls)
    with o.native_scope(native):
     obj=native.ConcatSatChannelAugment(scenarios=d.TRAIN_SCENES,p=float(c['As']),seed=202610172201,apply_fn=physical)
     result=obj.expand(x,y,domains,args=d.make_args(c,'cpu'),epoch=1,batch_idx=1)
     starts=np.random.default_rng(np.random.SeedSequence([20261017111,1])).integers(0,241,size=128).tolist();modified,_=attenuate(x,torch.tensor(starts))
     self.assertTrue(torch.equal(result.x[:128],x));self.assertTrue(torch.equal(result.y,torch.cat((y,y))));self.assertTrue(torch.equal(result.d_raw,torch.cat((domains,domains))))
     if c['As']:
      self.assertEqual(len(calls),before+1);self.assertTrue(torch.equal(calls[-1][0],modified));self.assertTrue(torch.equal(result.x[128:],calls[-1][1]));self.assertEqual(o.sat_packets,128)
      other,_=physical(x,result.scenario,None);after,_=attenuate(other,torch.tensor(starts));self.assertFalse(torch.allclose(after,result.x[128:]))
     else:self.assertEqual(len(calls),before);self.assertTrue(torch.equal(result.x[128:],modified));self.assertEqual(o.sat_packets,0)
     with self.assertRaises(ValueError):obj.expand(x,y,domains,args=None,epoch=1,batch_idx=2)
    self.assertEqual(o.local.packets,128)
 def test_real_defaulthead_fourforward_graph(self):
  torch.set_num_threads(2);r=d.rows()[0];c=d.config(r);m=api.build(c,'cpu').train();control=baseline.build(reference.config(reference.rows()[0]),'cpu');initial={k:v.clone() for k,v in m.state_dict().items()}
  self.assertEqual(len(initial),206);self.assertEqual(sum(p.numel() for p in m.parameters()),900722);self.assertTrue(all(torch.equal(v,control.state_dict()[k]) for k,v in initial.items()))
  x=torch.randn(128,2,256,generator=torch.Generator().manual_seed(2703));starts=torch.arange(128)%241;modified,_=attenuate(x,starts);xc=torch.cat((x,modified));y=torch.arange(128)%6;yy=torch.cat((y,y));rx=torch.tensor(([1,3,4,6,8]*26)[:128]);day=torch.arange(128)%3+1;domains=torch.tensor([{1:0,3:3,4:6,6:9,8:12}[int(r)]+int(t)-1 for r,t in zip(rx,day)]);dc=torch.cat((domains,domains));meta=dict(rx_i=rx,day_i=day)
  m._full_calls=1110;m._feature_l_steps=1110
  with m._phase1_labeled_feature_context():out=m(xc,y_tx=yy,return_aux=True)
  ce=F.cross_entropy(out['tx_logits'],yy,label_smoothing=.01);correction,metrics=m._source_labeled_feature_hook(model=m,labeled_output=out,labels=y,extra=(domains,meta),domains=dc,optimizer_step=1111,identity_loss=ce,labeled_count=128);loss=ce+correction
  self.assertTrue(torch.isfinite(loss));self.assertEqual(m.id_backbone.encoder._full_train_packets,512);self.assertEqual(m._full_extra_packets,256);self.assertEqual(m._full_supcon_applied,1);self.assertEqual(len(m._full_records[-1]['details']),4)
  loss.backward();chosen=[(n,p) for n,p in m.named_parameters() if n.endswith(('cls_head.weight','low_coordinate','local_projection.weight'))];self.assertTrue(all(p.requires_grad and p.grad is not None and torch.isfinite(p.grad).all() and p.grad.norm()>0 for n,p in chosen));self.assertIsNone(m._full_pending)
  self.assertTrue(all(torch.equal(v,m.state_dict()[k]) for k,v in initial.items()))
if __name__=='__main__':
 result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Checks))
 if not result.wasSuccessful():raise SystemExit(1)
 print(json.dumps(dict(status='PASS',tests=result.testsRun,synthetic_NN=512,synthetic_native_physical_calls=256,real_IQ=0,CP=0,optimizer_updates=0,target_access=False)))
