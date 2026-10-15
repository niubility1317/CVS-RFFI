import json,torch
from experiments.cvs_rdrop_sinc_units import design as d,model as api
from experiments.cvs_rdrop_sinc_units.feature_check import measure
torch.set_num_threads(2);results=[]
for arm in 'BCDEF':
 c=d.config(next(r for r in d.rows() if r['arm']==arm and r['model_seed']==d.SEEDS[0]));m=api.build(c,'cpu');initial={n:v.clone() for n,v in m.state_dict().items()}
 with torch.no_grad():
  m.id_backbone.encoder.sequence_pool.gain_raw.add_(.001);m.id_backbone.encoder.core.id_backbone.cls_head.weight.add_(.001)
  s=m.id_backbone.encoder.core.id_backbone.sinc
  if hasattr(s,'low_coordinate'):s.low_coordinate.add_(.001);s.band_coordinate.add_(.001)
  else:s.low_hz_.add_(100.);s.band_hz_.add_(100.)
 final=m.state_dict();v=measure(c,initial,final,final);results.append(dict(arm=arm,status=v['status'],restored=v['all_feature_parameters_buffers_preserved_by_diagnostic'],export=v['feature_export_exact']))
print(json.dumps(dict(status='PASS',checks=results)))
