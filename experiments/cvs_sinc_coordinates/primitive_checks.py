import copy,json,torch
from .coordinate import CoordinateSinc
def checks():
 from experiments.cvs_phase1_overlay.model import native_modules
 native_modules();from model import SincConv1d
 torch.set_num_threads(2);torch.manual_seed(5049)
 base=SincConv1d(24,79,sample_rate=25e6,dataset='wisig',input_len=256)
 before=torch.get_rng_state().clone();raw=CoordinateSinc(base,'RAW_HZ');center=CoordinateSinc(base,'CENTER_25K');assert torch.equal(before,torch.get_rng_state())
 assert sum(p.numel() for p in raw.parameters())==sum(p.numel() for p in center.parameters())==48
 x=torch.randn(3,2,256)
 base.eval();raw.eval();center.eval()
 with torch.no_grad():
  expected=base.forward_iq_pair(x);assert torch.equal(expected,raw.forward_iq_pair(x)) and torch.equal(expected,center.forward_iq_pair(x))
  same=center._filters(torch.device('cpu'),torch.float32).clone()
  center.low_coordinate[4,0]=.01
  changed=center._filters(torch.device('cpu'),torch.float32).clone();assert not torch.equal(same,changed)
  center.low_coordinate.zero_();restored=center._filters(torch.device('cpu'),torch.float32).clone();assert torch.equal(same,restored)
  center.band_coordinate[4,0]=.01;assert not torch.equal(same,center._filters(torch.device('cpu'),torch.float32))
  center.band_coordinate.zero_()
 # The same native physical function, with chain-rule scaled coordinate grads.
 raw.train();center.train();raw.zero_grad();center.zero_grad()
 raw.forward_iq_pair(x).square().mean().backward();center.forward_iq_pair(x).square().mean().backward()
 for a,b in [(raw.low_coordinate,center.low_coordinate),(raw.band_coordinate,center.band_coordinate)]:
  assert a.grad is not None and b.grad is not None and torch.isfinite(b.grad).all()
  assert torch.allclose(b.grad,25000*a.grad,atol=1e-5,rtol=2e-5)
 # Constant centered coordinates stay exactly0 under the actual EMA rule.
 from experiments.cvs_phase1_stack.runtime import installed
 from experiments.cvs_legacy_no_sat import design as ld
 c=ld.config(ld.rows()[0])
 teacher=copy.deepcopy(center).eval().requires_grad_(False)
 with installed(c) as native:
  for step in range(11100):native._update_ema_model(teacher,center,.999,versioned=True)
 assert torch.equal(teacher.low_coordinate,center.low_coordinate) and torch.equal(teacher.band_coordinate,center.band_coordinate)
 negative=0
 for mode,scale in [('bad',25000.),('CENTER_25K',1.)]:
  try:CoordinateSinc(base,mode,scale)
  except ValueError:negative+=1
  else:raise AssertionError('Unregistered coordinate accepted')
 return dict(status='PASS',synthetic_only=True,initial_native_IQ_features_exact=True,trainable_Sinc_parameters=48,initialization_RNG_preserved=True,cache_refresh_after_low_and_band_update=True,chain_rule_scale=25000.,stationary_center_EMA_exact=True,negative_cases=negative,head_fit=False)
if __name__=='__main__':print(json.dumps(checks()))
