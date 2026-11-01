import torch,json
torch.set_num_threads(2);torch.backends.mkldnn.enabled=True;torch.manual_seed(123)
results=[]
for shape in [(16,48,64,17),(16,48,32,33)]:
 for infmt in ('nchw','nhwc'):
  for gradfmt in ('nchw','nhwc'):
   for mode in ('none','tanh','erf'):
    violations=[]
    for i in range(8):
     x=torch.randn(shape);x=x.contiguous(memory_format=torch.channels_last if infmt=='nhwc' else torch.contiguous_format).requires_grad_();gout=torch.zeros(shape).contiguous(memory_format=torch.channels_last if gradfmt=='nhwc' else torch.contiguous_format)
     y=(.5*x*(1+torch.erf(x*0.7071067811865476)) if mode=='erf' else torch.nn.functional.gelu(x,approximate=mode));grad=torch.autograd.grad(y,x,grad_outputs=gout)[0];v=float(grad.detach().double().abs().nan_to_num(nan=0,posinf=1e300,neginf=1e300).max());bad=int((~torch.isfinite(grad)).sum())
     if v!=0 or bad:violations.append({'iteration':i,'max':v,'nonfinite':bad})
    results.append({'shape':shape,'input':infmt,'gradout':gradfmt,'mode':mode,'violations':violations})
print(json.dumps({'torch':torch.__version__,'threads':2,'mkldnn':True,'results':results},allow_nan=False))
