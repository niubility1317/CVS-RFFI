"""Original same-row final20 default-head source seven-view evaluation."""
from . import design as d,model as api
from experiments.cvs_scene_cycle20.source import Observer as Parent
from experiments.cvs_feature_sequence50.source import train as shared_train
class Observer(Parent):
 def epoch(self,m,ctx,rows,out,device):
  super().epoch(m,ctx,rows,out,device);steps=len(rows)*222;active=max(0,len(rows)-5)*222;ns=self.c['supcon_view_routing']['satellite_positions']
  if m._route_calls!=steps or m._route_applied!=active or m._route_sat_appearances!=active*ns or m._route_ord_appearances!=active*(128-ns) or m._route_handles:raise ValueError('Actual existing SupCon view budget differs')
  d.write(out/'supcon_views_progress.json',dict(epoch=len(rows),steps=steps,route_applied=active,As_descriptors=m._route_sat_appearances,ordinary_descriptors=m._route_ord_appearances,supcon_terms=4*active,extra_NN_forwards=0,new_loss_terms=0,handles_remaining=0,target_access=False))
 def _final_record(self,done,payload,views):
  result=super()._final_record(done,payload,views);m=self.model;ns=self.c['supcon_view_routing']['satellite_positions']
  if m._route_applied!=3330 or m._route_sat_appearances!=3330*ns or m._route_ord_appearances!=3330*(128-ns) or m._route_handles or self.teacher._route_calls or self.reader._route_calls:raise ValueError('L-only routing and zero teacher/reader updates required')
  result.update(status='SOURCE_SUPCON_VIEWS_E20_COMPLETE',supcon_view_policy=self.c['supcon_view_routing']['policy'],route_calls=m._route_calls,route_applied=m._route_applied,route_As_appearances=m._route_sat_appearances,route_ordinary_appearances=m._route_ord_appearances,route_new_NN_forwards=0,route_new_loss_terms=0,route_new_teacher_forwards=0)
  d.write(self.out/'supcon_views_audit.json',dict(config=self.c['supcon_view_routing'],calls=m._route_calls,applied=m._route_applied,As_appearances=m._route_sat_appearances,ordinary_appearances=m._route_ord_appearances,phase_terms=4*m._route_applied,default_head_joint=True,teacher_calls=0,reader_calls=0,extra_NN_forwards=0,new_loss_terms=0,target_access=False))
  d.write(self.out/'satmix_result.json',result);return result
def train(c):
 observer=Observer(c);shared_train(c,design=d,model_api=api,observer=observer);return observer.finish()
