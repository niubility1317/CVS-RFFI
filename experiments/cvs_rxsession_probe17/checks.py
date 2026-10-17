"""Focused negative protocols,scientific sign controls and real native syntheticphysics."""
import copy,json,sys
from pathlib import Path
import numpy as np
from . import design as d
from .physics import session,measure
from .analyze import source_metadata,metrics,qualify

def reject(f):
 try:f()
 except (ValueError,KeyError,TypeError):return
 raise AssertionError('Illegal configuration accepted')
def main():
 n=0
 for key,val in [('model_epochs',20),('L',6300),('held_RX1',1260),('target_access',True),('U',1),('V',1),('head_fit_updates',1),('CHANNEL_SEED',1),('banks',['original']),('scenes',['practical_low_urban']),('automatic_training',True),('expected_NN',1)]:
  c=d.config(d.rows()[0]);c[key]=val;reject(lambda:d.validate(c));n+=1
 for args in [(1,1,1,'original'),(3,0,1,'original'),(3,1,0,'original'),(3,1,1,'chosen'),(True,1,1,'original')]:reject(lambda:session(*args));n+=1
 visible=dict(physical_id='opaque-fit-id',TX=0,RX=3,day=1,eq=1);assert source_metadata([visible],{'opaque-fit-id'})['opaque-fit-id']==visible
 for change in [dict(physical_id='held-id'),dict(physical_id=[1,3,1,0,3000]),dict(TX=6),dict(RX=1),dict(day=0),dict(eq=0),dict(TX=True)]:
  reject(lambda:source_metadata([dict(visible,**change)],{'opaque-fit-id'}));n+=1
 reject(lambda:source_metadata([visible,visible],{'opaque-fit-id'}));n+=1
 reject(lambda:source_metadata([visible],{'opaque-fit-id','missing'}));n+=1
 good=dict(TX=0,prediction=0,logits=[2.,0.,0.,0.,0.,0.],margin_over30=2/30)
 for change in [dict(prediction=1),dict(TX=6),dict(logits=[1.,2.]),dict(logits=[float('nan')]*6),dict(margin_over30=1.)]:
  q=dict(good,**change);reject(lambda:metrics([q]));n+=1
 assert metrics([good])['accuracy']==1
 groups={sc:dict(count=5040,banks={b:dict(accuracy=.8 if b=='original' else .78,disagreement_vs_original=0 if b=='original' else .04) for b in d.BANKS}) for sc in d.SCENES}
 assert all(d.criteria(groups).values()) and d.criteria(groups)==qualify(groups)
 sciences=0
 for sc in d.SCENES:
  for k,value in [('accuracy',.799),('disagreement_vs_original',.029)]:
   g=copy.deepcopy(groups)
   for b in d.BANKS[1:]:g[sc]['banks'][b][k]=value
   assert not all(d.criteria(g).values()) and d.criteria(g)==qualify(g);sciences+=1
  g=copy.deepcopy(groups);g[sc]['banks']['bank1']['accuracy']=.81;assert not all(d.criteria(g).values());sciences+=1
 expected=dict(run_id=d.REFERENCE_RUN,arm='ALL',model_seed=d.MODEL_SEED)
 roles={};start=0
 for key,size in [('L_s',5040),('U_s',45360),('V',21600)]:roles[key]=list(range(start,start+size));start+=size
 contract=dict(held_RX=1,source_receivers=list(d.FIT),native_role_comparison='EXACT_MATCH_CARRIER_THEN_SOURCE_FOLD',role_ids=roles);carrier=dict(role_ids=copy.deepcopy(roles))
 done=dict(config=expected,epoch=20,optimizer_steps=4440,initialization='scratch',checkpoint_sources=[],ancestors=[],head_fit_updates=0,target_access=False,target_evaluated=False)
 init=dict(scratch_only=True,checkpoint_sources=[],ancestors=[],target_contact=False,source_roles='EXACT_MATCH');freeze=dict(epoch=20,selection='final_only',target_access=False,checkpoint=d.REFERENCE+'/final_ssdg.pth');result=dict(actual_steps=4440,L_forward_packets=4546560,As_training_observations=568320,default_head_joint=True,head_fit_updates=0,target_access=False,U_truth_read=False,checkpoint_sources=[])
 assert d.verify_reference(done,init,freeze,contract,carrier,result,expected)==contract
 for key,val in [('native_role_comparison','EXACT_MATCH'),('held_RX',3),('source_receivers',[1,3,4,6,8])]:
  bad=copy.deepcopy(contract);bad[key]=val;reject(lambda:d.verify_reference(done,init,freeze,bad,carrier,result,expected));n+=1
 for key,val in [('ancestors',['old']),('target_evaluated',True),('optimizer_steps',4441),('initialization','resume')]:
  bad=copy.deepcopy(done);bad[key]=val;reject(lambda:d.verify_reference(bad,init,freeze,contract,carrier,result,expected));n+=1
 bad=copy.deepcopy(contract);bad['role_ids']['L_s'][0]=bad['role_ids']['U_s'][0];reject(lambda:d.verify_reference(done,init,freeze,bad,carrier,result,expected));n+=1
 sys.path.insert(0,str(d.ROOT/'experiments/adv3b02_xuc/code'))
 import leo_practical.channel as ch,leo_practical.residual as residual
 from dataclasses import replace
 x=np.exp(2j*np.pi*.123*np.arange(256))*(1+.1*np.sin(np.arange(256)))
 total=0
 for sc in d.SCENES:
  cfg=ch.Config(fs_hz=25000000.,fc_hz=2462000000.,scenario=sc,processing_route='residual',mode='post_sync',equalization_enabled=False)
  for k in range(3):
   y,m=measure(ch,residual,cfg,x,d.CHANNEL_SEED+k,3,1,1);assert set(y)==set(d.BANKS) and m['original_replay_MSE']==0;total+=5
   assert any(not np.array_equal(y['original'],y[b]) for b in d.BANKS[1:])
  for invalid in (np.zeros(256),np.ones(255),np.full(256,np.nan)):
   reject(lambda:measure(ch,residual,cfg,invalid,1,3,1,1));n+=1
  for bad in (replace(cfg,equalization_enabled=True),replace(cfg,reference_snr_db=31.),replace(cfg,fs_hz=1e6)):
   reject(lambda:measure(ch,residual,bad,x,1,3,1,1));n+=1
 print(json.dumps(dict(status='PASS',protocol_negatives=n,scientific_counterexamples=sciences,synthetic_native_process_calls=total,real_IQ=0,checkpoint=0,NN=0,updates=0,target_access=False)))
if __name__=='__main__':main()
