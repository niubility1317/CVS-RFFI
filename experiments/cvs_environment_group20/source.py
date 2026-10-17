"""Existing RX6 role isolation/ownEMA/final20/sevenreadouts,unchangeddefaulthead."""
from . import design as d,model as api
from experiments.cvs_f_additive_rx6 import source as existing
from experiments.cvs_feature_sequence50.source import train as shared_train
existing.d=d # One isolated worker process;reuse exact verifiedRX6 role machinery.
class Observer(existing.Observer):
 def epoch(self,m,ctx,rows,out,device):
  e=len(rows);s=e*222;expected=0 if self.c['arm']=='RAW' else s*128
  if m._feature_l_steps!=s or m._joint_calls!=s or m._joint_L_packets!=s*1024 or m._additive_calls!=s or m._additive_L_packets!=s*256 or m._supcon_calls!=s or m._supcon_applied!=max(0,e-5)*222 or m._supcon_pending is not None or m._supcon_handles or self.optimizer._sinc_physical_clip_calls!=s or self.optimizer._sinc_unit_steps!=s or m._group_steps!=s or m._group_generated!=s*128 or sum(m._group_coverage.values())!=s*128 or m._joint_As_packets!=expected or sum(m._joint_coverage.values())!=expected or self.teacher._group_steps or self.reader._group_steps:raise ValueError('Exact E20/8L/As/ownEMA/source budgets required')
  d.write(out/'group_progress.json',dict(epoch=e,steps=s,generated=m._group_generated,As_consumed=m._joint_As_packets,L_NN=m._joint_L_packets,extra_L_NN=m._additive_L_packets,supcon=m._supcon_applied,extra_U=False,target_access=False))
 def _final_record(self,done,payload,views):
  r=super()._final_record(done,payload,views);m=self.model
  if m._group_steps!=4440 or m._group_generated!=568320 or sum(m._group_coverage.values())!=568320 or self.teacher._group_steps or self.reader._group_steps:raise ValueError('Complete source group generation/noextraU required')
  d.write(self.out/'group_coverage.json',dict(generated=m._group_generated,consumed=m._joint_As_packets,cells=[dict(TX=k[0],RX=k[1],day=k[2],scene=k[3],state=k[4],lock=k[5],count=v) for k,v in sorted(m._group_coverage.items())],physical_initializations=1136640,physical_processes=568320,allL_supervised=True,confidence_gate=False,U_truth_read=False))
  r.update(status='SOURCE_PROPAGATION_GROUP_E20_COMPLETE',group_steps=m._group_steps,group_generated=m._group_generated,As=self.c['arm']!='RAW',Ag=False);r['views']={v:{k:x for k,x in q.items() if k not in ('physical_ids','labels','predictions','physics','margin_over30','logits')} for v,q in views.items()};d.write(self.out/'satmix_result.json',r);return r

def train(c):
 o=Observer(c);shared_train(c,design=d,model_api=api,observer=o);return o.finish()
