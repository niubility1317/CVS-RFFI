"""Only training boundary changes. Native receiver processing stays coupled."""
from contextlib import contextmanager
from dataclasses import replace
def boundary_for_call(call):
 if type(call) is not int or not 1<=call<=6660:raise ValueError('One registered labeled call in E30 required')
 return 'reflect' if call%2 else 'edge'
@contextmanager
def training_boundary(pa,boundary):
 if boundary not in ('reflect','edge'):raise ValueError('Registered two training boundaries only')
 original=pa.practical_config
 def configured(scene,args):
  ctx=pa._context.get()
  if ctx is None or not ctx[2].startswith('source_dynamic_E') or not ctx[2].endswith('_L'):raise ValueError('Boundary intervention is labeled-source training only')
  cfg=original(scene,args)
  expected=dict(boundary='reflect',processing_route='residual',mode='post_sync',equalization_enabled=False,reference_snr_db=30.,fs_hz=25000000.,fc_hz=2450000000.,quality_adaptation_enabled=True,phase_tracking_enabled=True,iq_compensation_enabled=True)
  if any(getattr(cfg,k)!=v for k,v in expected.items()):raise ValueError('Complete original noeq practical required')
  return replace(cfg,boundary=boundary)
 pa.practical_config=configured
 try:yield
 finally:pa.practical_config=original
