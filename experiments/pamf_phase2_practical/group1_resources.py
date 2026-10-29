"""Actual stage wall/memory/optimizer telemetry, including closed-form stages."""
import os
import threading
import time
import torch
from pathlib import Path
from .io import atomic


def process_rss():
    if os.name!='nt':return int(Path('/proc/self/statm').read_text().split()[1])*os.sysconf('SC_PAGE_SIZE')
    import ctypes
    from ctypes import wintypes
    class Counters(ctypes.Structure):
        _fields_=[('cb',wintypes.DWORD),('PageFaultCount',wintypes.DWORD)]+[(k,ctypes.c_size_t) for k in
            ('PeakWorkingSetSize','WorkingSetSize','QuotaPeakPagedPoolUsage','QuotaPagedPoolUsage',
             'QuotaPeakNonPagedPoolUsage','QuotaNonPagedPoolUsage','PagefileUsage','PeakPagefileUsage')]
    counter=Counters();counter.cb=ctypes.sizeof(counter)
    get=ctypes.windll.kernel32.GetCurrentProcess;get.restype=ctypes.c_void_p
    query=ctypes.windll.psapi.GetProcessMemoryInfo
    query.argtypes=(ctypes.c_void_p,ctypes.POINTER(Counters),wintypes.DWORD)
    if not query(get(),ctypes.byref(counter),counter.cb):raise ctypes.WinError()
    return int(counter.WorkingSetSize)


def tensor_bytes(value,seen=None):
    seen=set() if seen is None else seen
    if torch.is_tensor(value):
        if id(value) in seen:return 0
        seen.add(id(value));return value.numel()*value.element_size()
    if isinstance(value,dict):return sum(tensor_bytes(v,seen) for v in value.values())
    if isinstance(value,(tuple,list)):return sum(tensor_bytes(v,seen) for v in value)
    return 0


class Measure:
    def __init__(self,path,context=None,model=None,optimizers=False,optimizer_instances=()):
        self.path=path;self.context=context or {};self.model=model;self.track_optimizers=optimizers
        self.result={};self.steps=0;self.updated=set();self.max_optimizer_bytes=0;self.saved={};self.step_updated_max=0
        self.optimizer_instances=optimizer_instances;self.saved_instances={}

    def record_optimizer(self,opt):
        self.steps+=1;current=0
        for group in opt.param_groups:
            for p in group['params']:
                if p.grad is not None:self.updated.add((id(p),p.numel()));current+=p.numel()
        self.step_updated_max=max(self.step_updated_max,current)
        self.max_optimizer_bytes=max(self.max_optimizer_bytes,tensor_bytes(opt.state))

    def __enter__(self):
        self.cuda=torch.cuda.is_available() and os.environ.get('CUDA_VISIBLE_DEVICES','')!=''
        if self.cuda:
            torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
        base=process_rss();self.cpu_started=os.times()
        self.result=dict(self.context,started_at=time.time(),physical_GPU=os.environ.get('CUDA_VISIBLE_DEVICES'),
            GPU_name=torch.cuda.get_device_name(0) if self.cuda else None,torch_version=str(torch.__version__),
            cuda_version=torch.version.cuda,CPU_threads=torch.get_num_threads(),
            CUDA_allocated_baseline=torch.cuda.memory_allocated() if self.cuda else 0,
            CPU_RSS_baseline=base,CPU_RSS_sampled_peak=base,RSS_sampling_interval_seconds=.05,
            precision='FP32_TF32off_except_MILKR_FP64',concurrency='observed_shared_GPU_run',query_in_fit=False)
        self.event=threading.Event()
        self.start=time.perf_counter()
        def sample():
            last=time.monotonic()
            while not self.event.wait(.05):
                self.result['CPU_RSS_sampled_peak']=max(self.result['CPU_RSS_sampled_peak'],process_rss())
                if time.monotonic()-last>=5:
                    snapshot=dict(self.result,status='RUNNING',wall_seconds=time.perf_counter()-self.start,
                        optimizer_steps=self.steps if self.track_optimizers else self.result.get('native_optimizer_steps'),
                        updated_at=time.time(),optimizer_state_peak_bytes=self.max_optimizer_bytes,
                        CUDA_allocated_peak=torch.cuda.max_memory_allocated() if self.cuda else 0,
                        CUDA_reserved=torch.cuda.max_memory_reserved() if self.cuda else 0)
                    atomic(self.path,snapshot);last=time.monotonic()
        self.thread=threading.Thread(target=sample,daemon=True);self.thread.start()
        if self.track_optimizers:
            probe=self
            for cls in (torch.optim.Adam,torch.optim.AdamW,torch.optim.SGD,torch.optim.RMSprop):
                original=cls.step;self.saved[cls]=original
                def step(opt,*args,_original=original,**kwargs):
                    value=_original(opt,*args,**kwargs);probe.record_optimizer(opt)
                    return value
                cls.step=step
            for opt in self.optimizer_instances:
                if 'step' not in opt.__dict__:continue
                original=opt.step;self.saved_instances[opt]=original
                def scheduled(*args,_opt=opt,_original=original,**kwargs):
                    value=_original(*args,**kwargs);probe.record_optimizer(_opt);return value
                scheduled.__dict__.update(getattr(original,'__dict__',{}))
                opt.step=scheduled
        return self

    def __exit__(self,kind,value,tb):
        if self.cuda:torch.cuda.synchronize()
        elapsed=time.perf_counter()-self.start
        for cls,original in self.saved.items():cls.step=original
        for opt,original in self.saved_instances.items():opt.step=original
        self.event.set();self.thread.join(timeout=1)
        allocated=torch.cuda.max_memory_allocated() if self.cuda else 0
        self.result.update(status='FAILED' if kind else 'COMPLETE',error=str(value) if kind else None,
            ended_at=time.time(),wall_seconds=elapsed,optimizer_tracking_enabled=self.track_optimizers,
            optimizer_steps=self.steps if self.track_optimizers else self.result.get('native_optimizer_steps'),
            actual_updated_parameter_count=sum(n for _,n in self.updated),optimizer_state_peak_bytes=self.max_optimizer_bytes,
            maximum_single_optimizer_step_updated_parameters=self.step_updated_max,
            updated_parameter_count_scope='sum_unique_tensor_instances_with_grad_multiple_source_segments_may_construct_new_instances',
            CUDA_allocated_peak=allocated,CUDA_allocated_delta=max(0,allocated-self.result['CUDA_allocated_baseline']),
            CUDA_reserved=torch.cuda.max_memory_reserved() if self.cuda else 0,
            model_parameters=sum(p.numel() for p in self.model.parameters()) if self.model is not None else None,
            model_state_bytes=tensor_bytes(self.model.state_dict()) if self.model is not None else None,
            CPU_RSS_final=process_rss(),CPU_user_seconds=os.times().user-self.cpu_started.user,
            CPU_system_seconds=os.times().system-self.cpu_started.system,
            energy_joules=None,energy_reason='shared_GPU_not_attributable',
            MACs=None,FLOPs=None,operation_count_reason='not_measured_do_not_substitute_parameter_count')
        atomic(self.path,self.result)
        return False
