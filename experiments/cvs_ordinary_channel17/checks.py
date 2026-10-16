"""Focused native augmentation/RNG/dependency and fixedscientific counterexamples;no model."""
import copy,importlib.util,json,math,random,sys
from types import SimpleNamespace
import torch
from . import design as d
from .paired import paired_views
from .decision import qualify

def run():
 torch.set_num_threads(2);native=d.ROOT/'experiments/adv3b02_xuc/code';sys.path.insert(0,str(native));import DataAugmentation as module
 spec=importlib.util.spec_from_file_location('ordinary_channel_local_schedule',native/'cvsrffi/schedule.py');schedule=importlib.util.module_from_spec(spec);spec.loader.exec_module(schedule);args=SimpleNamespace(**d.read(d.ROOT/'experiments/cvs_ordinary_channel17/local_schedule_args.json'));base=d.read(d.ROOT/'experiments/cvs_ordinary_channel17/reference_aug.json');aug=module.build_augmentor(**base);schedule.configure_augmentor_for_epoch(aug,base,13,args)
 x=torch.randn(32,2,256,generator=torch.Generator().manual_seed(202610172001));y=torch.arange(32)%6;original=x.clone();ordinary=0;replay=0
 for case in range(3):
  active=copy.deepcopy(aug)
  for key in ('p_time_shift','p_amp_scale','p_phase_rot','p_cfo','p_phase_noise','p_dc_offset','p_bandedge_taper','p_pa'):setattr(active,key,1. if case==1 else 0.)
  active.p_multipath=active.p_awgn=0. if case==0 else 1.;random.seed(202610172010+case);torch.manual_seed(202610172010+case);py=random.getstate();ts=torch.random.get_rng_state();native_o,native_sp=active(x.clone(),labels=y,no_pa=False,return_pa_strength=True);pyend=random.getstate();tsend=torch.random.get_rng_state();random.setstate(py);torch.random.set_rng_state(ts);origmp,origawgn=module._apply_multipath,module._apply_awgn;views,strength,record=paired_views(active,x,y)
  assert torch.equal(native_o,views['full']) and torch.equal(native_sp,strength) and random.getstate()==pyend and torch.equal(torch.random.get_rng_state(),tsend) and torch.equal(x,original) and module._apply_multipath is origmp and module._apply_awgn is origawgn
  ordinary+=5*len(x);replay+=record['kernel_replay_random_scalars']
  if case==0:assert all(torch.equal(v,x) for v in views.values()) and not record['events']['full']
  if case==2:
   assert all(e['kind'] in ('MP','AWGN') for e in record['events']['full']);mp=record['events']['full'][0]['parameters'];h=torch.complex(torch.tensor(mp['kernel_real']),torch.tensor(mp['kernel_imag']));z=torch.complex(x[:,0],x[:,1]);pad=len(h)-1;zr=torch.nn.functional.pad(z.real.unsqueeze(1),(pad,0),mode='reflect');zi=torch.nn.functional.pad(z.imag.unsqueeze(1),(pad,0),mode='reflect');hr=h.real.view(1,1,-1);hi=h.imag.view(1,1,-1);mz=torch.complex((torch.nn.functional.conv1d(zr,hr)-torch.nn.functional.conv1d(zi,hi)).squeeze(1),(torch.nn.functional.conv1d(zr,hi)+torch.nn.functional.conv1d(zi,hr)).squeeze(1));nfull=torch.complex(views['full'][:,0],views['full'][:,1])-mz;nnomp=torch.complex(views['no_MP'][:,0],views['no_MP'][:,1])-z;snr=torch.tensor(record['events']['full'][1]['parameters']['snr_db']);sig1=torch.sqrt(mz.abs().square().mean(1,keepdim=True)/(10**(snr/10))/2);sig0=torch.sqrt(z.abs().square().mean(1,keepdim=True)/(10**(snr/10))/2);assert float((nfull/sig1-nnomp/sig0).abs().max())<3e-4 and not torch.allclose(sig1,sig0,rtol=1e-3,atol=1e-5);assert torch.equal(views['no_both'],x)
 negatives=0
 bad=[(None,y),(x[:0],y[:0]),(x[:,0],y),(x[:,:,:255],y),(x.double(),y),(x.to(torch.int64),y),(x.requires_grad_(),y)];x=x.detach();bad.extend([(x,None),(x,y.float()),(x,y[:-1]),(x,torch.full_like(y,6)),(x,torch.full_like(y,-1)),(torch.full_like(x,float('nan')),y),(torch.zeros_like(x),y)])
 for bx,by in bad:
  py=random.getstate();ts=torch.random.get_rng_state()
  try:paired_views(aug,bx,by)
  except (ValueError,TypeError):negatives+=1
  else:raise AssertionError('Invalid input accepted')
  assert random.getstate()==py and torch.equal(ts,torch.random.get_rng_state())
 for change in ({'target_access':True},{'primary_arm':'no_MP'},{'optimizer_updates':1}):
  c=d.config();c.update(change)
  try:d.validate(c)
  except ValueError:negatives+=1
  else:raise AssertionError('Changed source protocol accepted')
 raw=dict(accuracy=.998,macro_F1=.998,**{'by_'+k:{'0':dict(accuracy=.998)} for k in ('RX','TX','day')});full=dict(accuracy=.99,macro_F1=.99,**{'by_'+k:{'0':dict(accuracy=.99)} for k in ('RX','TX','day')});primary=dict(accuracy=.997,macro_F1=.997,raw_correct_retention=.995,**{'by_'+k:{'0':dict(accuracy=.997)} for k in ('RX','TX','day')});assert qualify(raw,full,primary,[.5,.6,.7,.8])['decision']=='PASS';counter=0
 for key,value in [('accuracy',.99),('macro_F1',.98),('raw_correct_retention',.98)]:
  z=copy.deepcopy(primary);z[key]=value;assert qualify(raw,full,z,[.5,.6,.7,.8])['decision']=='FAIL';counter+=1
 for key in ('RX','TX','day'):
  z=copy.deepcopy(primary);z['by_'+key]['0']['accuracy']=.97;assert qualify(raw,full,z,[.5,.6,.7,.8])['decision']=='FAIL';counter+=1
 assert qualify(raw,full,primary,[-.3,-.2,1,1])['decision']=='FAIL';counter+=1
 result=dict(status='PASS',input_config_negatives=negatives,scientific_counterexamples=counter,synthetic_ordinary_generations=ordinary,synthetic_kernel_replay_random_scalars=replay,source_IQ_reads=0,checkpoint_loads=0,NN=0,updates=0,canonical_full_output_RNG=True,allfourarms_nativeparameters_PAmasks_RNG_matched=True,total_AWGN_power_dependency_verified=True,no_channel_identity_verified=True,modulefunctions_restored=True)
 print(json.dumps(result));return result
if __name__=='__main__':run()
