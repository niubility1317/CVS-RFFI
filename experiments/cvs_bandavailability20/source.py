"""Original E20 source trainer,automatic frozen clean+sixpractical defaulthead readouts."""
from . import design as d,model as api
from experiments.cvs_additive_as20.source import Observer as Parent
from experiments.cvs_as_timing20.source import Observer as SameReader
from experiments.cvs_feature_sequence50.source import train as shared_train
class Observer(Parent):
 evaluate=SameReader.evaluate
 scene=SameReader.scene
 def epoch(self,m,ctx,rows,out,device):
  super().epoch(m,ctx,rows,out,device);steps=len(rows)*222
  if m._availability_calls!=steps or m._availability_generated!=steps*128 or sum(m._availability_coverage.values())!=steps*128 or self.teacher._availability_calls or self.reader._availability_calls:raise ValueError('ExactL-only maskshadow/operator budget required')
  d.write(out/'availability_progress.json',dict(epoch=len(rows),steps=steps,generated=m._availability_generated,unique_L=len({i for i,k in m._availability_packets}),pair_cell_count=len(m._availability_coverage),extra_U=False,target_access=False))
 def _final_record(self,done,payload,views):
  result=super()._final_record(done,payload,views);m=self.model
  if m._availability_generated!=568320 or len(m._availability_coverage)!=432 or len({i for i,k in m._availability_packets})!=5040 or len(m._availability_packets)!=30240:raise ValueError('FullTX/RX/day/pair andeveryL×sixpair coverage required')
  d.write(self.out/'availability_coverage.json',dict(cells=[dict(TX=k[0],RX=k[1],day=k[2],pair=k[3],count=n) for k,n in sorted(m._availability_coverage.items())],packet_pair_exposures=[dict(base_index=k[0],pair=k[1],count=n) for k,n in sorted(m._availability_packets.items())],generated=m._availability_generated,IQ_views_consumed=568320 if self.c['arm']=='MASK' else 0,no_U_augmentation=True))
  result.update(status='SOURCE_BAND_AVAILABILITY_E20_COMPLETE',Ag_candidate=self.c['arm']=='MASK',Ag=False,As=False,availability_generated=m._availability_generated,availability_steps=m._availability_calls);result['views']={v:{k:x for k,x in q.items() if k not in ('physical_ids','labels','predictions','physics','margin_over30','logits')} for v,q in views.items()};d.write(self.out/'satmix_result.json',result);return result

def train(c):
 o=Observer(c);shared_train(c,design=d,model_api=api,observer=o);return o.finish()
