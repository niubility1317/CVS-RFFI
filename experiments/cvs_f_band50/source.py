import torch
from . import design as d,model as api
from .observer import Observer
from experiments.cvs_feature_sequence50.source import train as original
from experiments.cvs_rdrop_sinc_units.feature_check import measure
def train(c):
 observer=Observer(c);original(c,design=d,model_api=api,observer=observer);observer.finish()
 from .provenance import source_proof,validate_checkpoint
 p=source_proof(c);ck=torch.load(p/'final_ssdg.pth',map_location='cpu',weights_only=False);validate_checkpoint(ck,c);initial=torch.load(p/'initial_smoke.pt',map_location='cpu',weights_only=False)
 feature=measure(d.base_config(c['model_seed']),initial['model'],ck['model'],ck['ema_model']);feature.update(row_id=c['row_id'],new_sinc_policy=c['arm']);d.write(p/'source_feature_witness.json',feature)
 return d.read(p/'completion.json')
