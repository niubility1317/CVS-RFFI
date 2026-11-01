import torch,json
torch.set_num_threads(2);torch.manual_seed(512);out=[]
for shape in [(16,48,64,17),(16,48,32,33)]:
 for a in ('nchw','nhwc'):
  for b in ('nchw','nhwc'):
   x=torch.randn(shape).contiguous(memory_format=torch.channels_last if a=='nhwc' else torch.contiguous_format).requires_grad_();g=torch.randn(shape).contiguous(memory_format=torch.channels_last if b=='nhwc' else torch.contiguous_format)
   torch.backends.mkldnn.enabled=False;yr=torch.nn.functional.gelu(x);gr=torch.autograd.grad(yr,x,g,retain_graph=False)[0];torch.backends.mkldnn.enabled=True;y=.5*x*(1.+torch.erf(x*.7071067811865476));gg=torch.autograd.grad(y,x,g,retain_graph=False)[0]
   out.append({'shape':shape,'input':a,'grad':b,'forward_max_error':float((y-yr).abs().max()),'backward_max_error':float((gg-gr).abs().max())});assert torch.allclose(y,yr,atol=2e-6,rtol=2e-5) and torch.allclose(gg,gr,atol=2e-6,rtol=2e-5)
print(json.dumps({'status':'PASS','comparisons':out},allow_nan=False))
