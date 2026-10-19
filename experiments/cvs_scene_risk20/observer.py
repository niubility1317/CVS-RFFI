from . import design as d,model as api
from experiments.cvs_F_nontrue20.observer import Observer
from experiments.cvs_feature_sequence50.source import train as shared_train
def train(c):
 o=Observer(c);shared_train(c,design=d,model_api=api,observer=o);result=o.finish()
 d.write(o.out/'scene_risk_state.json',dict(logq=o.model._risk_logq.tolist(),steps=o.model._nontrue_steps,optimizer_state='native final_ssdg.pth',EMA='native ownEMA',future_continuation_automatic=False))
 return result
