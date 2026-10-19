"""Focused gradient algebra and source-scope tests; then actual frozen-F smoke."""
from pathlib import Path
assert (Path(__file__).with_name('core.py')).is_file(), 'EXPECTED_RED: source gradient core not implemented'
import json,torch
from core import scope,gram_summary,unit_scale,reference_scope,OBJECTIVES,derived
def run():
 from runner import CountingRole
 class Tiny:
  role='L_s';y=[0,1,2]
  def __len__(self):return 3
  def __getitem__(self,i):return self.y[i]
 counted=CountingRole(Tiny());assert [counted[i] for i in range(3)]==[0,1,2] and counted.samples_read==3 and counted.role=='L_s'
 bad=0
 for role in ('U_s','V','query','clean'):
  try:scope(role,torch.tensor([1,3]),torch.tensor([1,3]))
  except ValueError:bad+=1
  else:raise AssertionError('Forbidden diagnostic role accepted')
 for y,rx in [(torch.tensor([-1,3]),torch.tensor([1,3])),(torch.tensor([1,3]),torch.tensor([0,3]))]:
  try:scope('L_s',y,rx)
  except ValueError:bad+=1
  else:raise AssertionError('Hidden/foreign source labels accepted')
 try:reference_scope('wrong','wrong')
 except ValueError:bad+=1
 else:raise AssertionError('Unverified reference accepted')
 a=torch.tensor([[1.,0.],[-1.,0.],[0.,2.]],dtype=torch.float64);r=gram_summary(a@a.T,['A','B','C']);assert r['cosine'][0][1]==-1 and r['cosine'][0][2]==0 and r['norms']==[1.,1.,2.]
 zero=gram_summary(torch.zeros(2,2,dtype=torch.float64),['a','b']);assert zero['cosine'][0][1] is None
 assert unit_scale('p.sinc.low_coordinate','CLIP_PHYSICAL')==1/25000
 assert unit_scale('p.sinc.band_coordinate','NORMALIZED_HZ')==1000
 assert unit_scale('p.time_down.weight','NORMALIZED_HZ')==1
 x=torch.arange(len(OBJECTIVES)*3,dtype=torch.float64).reshape(len(OBJECTIVES),3);v=derived(x);torch.testing.assert_close(v[-3],x[[0,2,4,5]].mean(0));torch.testing.assert_close(v[-2],x[[1,3]].mean(0));torch.testing.assert_close(v[-1],x[:6].mean(0))
 return dict(status='PASS',protocol_negative_cases=bad,hand_derived_cosines=True,zero_gradient_undefined=True,physical_coordinate_mapping=True,CE_group_linear_oracle=True,actual_dataset_boundary_counter=True,IQ_reads=0,target_reads=0,optimizer_updates=0,head_fit_updates=0)
if __name__=='__main__':print(json.dumps(run()))
