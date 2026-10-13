"""Count actual time-block Conv1d MACs; synthetic functional profile only."""
import argparse,json
from pathlib import Path
import torch
from experiments.cvs_time_basis50 import design as d
from experiments.cvs_time_basis50.model import build
from experiments.cvs_time_factorial50.model import build as compact_build
from experiments.cvs_feature_restitution.features import layout

def profile(model):
 core,_,_=layout(model);rows=[];handles=[]
 for layer in ('t1','t2','t3'):
  for name,module in getattr(core,layer).named_modules():
   if not isinstance(module,torch.nn.Conv1d):continue
   def hook(module,inputs,output,layer=layer,name=name):
    batch=output.shape[0];multiplications=output.numel()//batch*(module.in_channels//module.groups)*module.kernel_size[0]
    rows.append(dict(layer=layer,module=name,input_shape_per_packet=list(inputs[0].shape[1:]),output_shape_per_packet=list(output.shape[1:]),kernel=module.kernel_size[0],groups=module.groups,Conv1d_MAC_per_packet=multiplications))
   handles.append(module.register_forward_hook(hook))
 try:
  with torch.no_grad():z=model.id_backbone.encoder.features(torch.zeros(2,2,256))
  assert z.shape==(2,160) and torch.isfinite(z).all()
 finally:
  for h in handles:h.remove()
 return dict(identity_parameters=sum(p.numel() for p in model.id_backbone.parameters()),time_parameters=sum(p.numel() for l in ('t1','t2','t3') for p in getattr(core,l).parameters()),time_Conv1d_MAC_per_packet=sum(r['Conv1d_MAC_per_packet'] for r in rows),time_Conv1d_MAC_by_layer={l:sum(r['Conv1d_MAC_per_packet'] for r in rows if r['layer']==l) for l in ('t1','t2','t3')},layers=rows)

def run(output):
 torch.set_num_threads(2);seed=d.SEEDS[0]
 models={'BASE_REF':build(d.config(d.rows()[0]),'cpu').eval(),'SINGLE_BASIS':build(d.config(d.rows()[1]),'cpu').eval(),'COMPACT_LINEAR':compact_build(d.compact_config(seed),'cpu').eval()}
 values={arm:profile(m) for arm,m in models.items()}
 if values['SINGLE_BASIS']['time_parameters']-values['COMPACT_LINEAR']['time_parameters']!=72:raise ValueError('Actual matched time count differs')
 for layer in ('t1','t2','t3'):
  a=[r for r in values['SINGLE_BASIS']['layers'] if r['layer']==layer]
  b=[r for r in values['COMPACT_LINEAR']['layers'] if r['layer']==layer]
  assert a[0]['input_shape_per_packet']==b[0]['input_shape_per_packet'] and a[-1]['output_shape_per_packet']==b[-1]['output_shape_per_packet']
 result=dict(status='VERIFIED',models=values,single_minus_compact_MAC=values['SINGLE_BASIS']['time_Conv1d_MAC_per_packet']-values['COMPACT_LINEAR']['time_Conv1d_MAC_per_packet'],single_to_compact_MAC_ratio=values['SINGLE_BASIS']['time_Conv1d_MAC_per_packet']/values['COMPACT_LINEAR']['time_Conv1d_MAC_per_packet'],scope='Only actual Conv1d multiplications/accumulations in modified t1/t2/t3; excludes unchanged Sinc/frequency/PA, norm/activation/add/pooling, FFT and memory/latency; MAC is not measured execution speed',IQ='two synthetic zero packets, length256; no dataset/truth/checkpoint read',model_updates=0,head_fit_updates=0)
 output.parent.mkdir(parents=True,exist_ok=True);output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');print(json.dumps({k:result[k] for k in ('status','single_minus_compact_MAC','single_to_compact_MAC_ratio')}));return result
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();run(a.output)
