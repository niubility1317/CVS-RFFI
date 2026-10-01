"""Measure synthetic resource costs on a disposable clone, never update training state."""
import copy
import gc
import time
import torch
import torch.nn.functional as F


def synchronize(device):
    if device.type=='cuda':torch.cuda.synchronize(device)


def resource_profile(model,device):
    clone=copy.deepcopy(model).to(device);clone.eval()
    results=dict(hardware=torch.cuda.get_device_name(device) if device.type=='cuda' else 'cpu',
        precision='float32',torch_version=torch.__version__,total_parameters=sum(p.numel() for p in clone.parameters()),
        trainable_parameters=sum(p.numel() for p in clone.parameters() if p.requires_grad),
        resident_state_bytes=sum(v.numel()*v.element_size() for v in clone.state_dict().values()),
        input_shape=[2,256],warmup=5,inference_repetitions=20,training_repetitions=10,
        mac_scope='Conv1d and matrix multiplication only; excludes FFT, normalization, pooling, physical lifts and elementwise arithmetic',
        benchmark_state_used_for_training=False)
    with torch.no_grad():
        for batch in (1,128):
            x=torch.randn(batch,2,256,device=device)
            for _ in range(5):clone(x)
            synchronize(device);tic=time.perf_counter()
            for _ in range(20):clone(x)
            synchronize(device)
            results['inference_batch'+str(batch)+'_ms']=1000*(time.perf_counter()-tic)/20
        x=torch.randn(1,2,256,device=device)
        with torch.autograd.profiler.profile(record_shapes=True,use_cuda=False) as prof:clone(x)
    macs=0;fft=0
    for event in prof.function_events:
        shapes=event.input_shapes
        if event.name=='aten::mm' and len(shapes)>=2 and len(shapes[0])==2 and len(shapes[1])==2:
            macs+=shapes[0][0]*shapes[0][1]*shapes[1][1]
        elif event.name=='aten::addmm' and len(shapes)>=3 and len(shapes[1])==2 and len(shapes[2])==2:
            macs+=shapes[1][0]*shapes[1][1]*shapes[2][1]
        if event.name.startswith('aten::_fft'):fft+=1
    from torch.utils._python_dispatch import TorchDispatchMode
    class ConvolutionCounter(TorchDispatchMode):
        def __init__(self):super().__init__();self.macs=0
        def __torch_dispatch__(self,func,types,args=(),kwargs=None):
            out=func(*args,**(kwargs or {}))
            if str(func)=='aten.convolution.default' and len(args[1].shape)==3:
                self.macs+=out.numel()*args[1].shape[1]*args[1].shape[2]
            return out
    counter=ConvolutionCounter()
    with torch.no_grad(),counter:clone(x)
    results.update(conv_linear_macs_per_sample=macs+counter.macs,fft_calls_per_sample=fft)
    clone.train();x=torch.randn(128,2,256,device=device);y=torch.arange(128,device=device)%6
    optimizer=torch.optim.AdamW(clone.parameters(),lr=.0002,weight_decay=.0001)
    def update():
        optimizer.zero_grad(set_to_none=True);loss=F.cross_entropy(clone(x),y);loss.backward();optimizer.step()
    for _ in range(5):update()
    if device.type=='cuda':torch.cuda.reset_peak_memory_stats(device)
    synchronize(device);tic=time.perf_counter()
    for _ in range(10):update()
    synchronize(device)
    results['training_batch128_ms']=1000*(time.perf_counter()-tic)/10
    results['gradient_used_parameters']=sum(p.numel() for p in clone.parameters() if p.grad is not None)
    results['benchmark_peak_cuda_allocated_bytes']=torch.cuda.max_memory_allocated(device) if device.type=='cuda' else None
    results['benchmark_peak_note']='Disposable clone plus original frozen model resident; actual source-training peak is separately logged'
    del clone,optimizer,x,y;gc.collect()
    return results
