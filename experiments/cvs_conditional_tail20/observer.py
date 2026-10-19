from . import design as d,model as api
from experiments.cvs_F_nontrue20.observer import Observer as Original
from experiments.cvs_feature_sequence50.source import train as shared_train

class Observer(Original):
 def initialize(self,m,c,out):
  super().initialize(m,c,out);d.write(self.out/'second_actual_channel.json',m._second_channel.audit())
 def epoch(self,m,ctx,rows,out,device):
  super().epoch(m,ctx,rows,out,device)
  if m._second_channel.packets!=len(rows)*222*128 or m._second_channel.calls!=len(rows)*222:raise ValueError('Second complete draw budget differs')
 def finish(self):
  result=super().finish();m=self.model
  if m._nontrue_L_packets!=4440*384 or m._second_channel.packets!=568320 or m._second_channel.sessions!=m._nontrue_channel.sessions:raise ValueError('Full two-draw/three-forward/session execution differs')
  result.update(second_channel=m._second_channel.summary(),external_teacher_NN=0,second_draw_packets=568320,conditional_pairs=568320,As_packets=1136640,student_L_packets=1704960)
  d.write(self.out/'training_result.json',result);return result

def train(c):
 o=Observer(c);shared_train(c,design=d,model_api=api,observer=o);return o.finish()
