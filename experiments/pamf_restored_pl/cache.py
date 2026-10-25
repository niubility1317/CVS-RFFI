"""One shared IQ-only U cache and frozen raw-teacher targets; no target handles."""
import time
from pathlib import Path
from experiments.cric_dura_comparison.io import read,write
from experiments.pamf_channel_pl.restored_source import SCENES,snapshot,unchanged,source_channel,validate_context


def build(c):
    if set(c)-{'source_context','output_root','frozen_sources','run_id','code_commit'}:
        raise ValueError('Source cache worker rejects target/evaluation handles')
    from experiments.pamf_repl_completion.runner import activate
    activate(c)
    import numpy as np
    import torch
    from experiments.pamf_channel_pl.restored_dose_source import load_sources
    from experiments.cric_dura_comparison.f_data import SourceDataset
    from experiments.cric_dura_comparison.external_dura import fp32_copy
    from experiments.pamf_stage2_inference.engine import InferenceEngine
    context=c['source_context'];validate_context(context)
    out=Path(c['output_root']);out.mkdir(parents=True,exist_ok=False);start=time.perf_counter()
    models,configs,payloads,contract,g=load_sources(context,'cuda:0');del payloads
    engine=InferenceEngine(g,reuse=True,graph=True,native=False)
    labeled=SourceDataset(configs[0]['source_cache'],'L_s',contract)
    with torch.inference_mode():
        x=labeled.x[:2].to('cuda:0');restored=engine(x)[0].clone()
        if not torch.allclose(g(x),restored,atol=1e-5,rtol=1e-5) or any(m(restored).shape!=(2,6) for m in models):
            raise ValueError('Actual source-only frozen F/G smoke failed')
    del labeled,x,restored
    write(out/'smoke.json',dict(status='PASS',role='L_s',physical_packets=2,target_reads=0,optimizer_updates=0))
    before=[snapshot(m) for m in models];gbefore=snapshot(g)
    data=SourceDataset(configs[0]['source_cache'],'U_s',contract);n=len(data)
    if n!=56700:raise ValueError('Original F0 U role required')
    loader=torch.utils.data.DataLoader(data,batch_size=256,shuffle=False,num_workers=0)
    iq=np.lib.format.open_memmap(out/'restored_U.npy',mode='w+',dtype=np.float32,shape=(3,n,2,256))
    target=np.empty((n,6),np.float32);accepted=np.empty(n,bool);offset=0
    with torch.inference_mode():
        for i,batch in enumerate(loader):
            if not bool((batch[1]==-1).all()) or bool(torch.as_tensor(batch[3]['tx_label_visible']).any()):
                raise ValueError('Cache U TX truth must remain hidden')
            x=batch[0].to('cuda:0');count=len(x)
            probs=torch.stack([m(x).softmax(1) for m in models]);conf,label=probs.max(-1)
            mask=(conf>=.95).all(0)&(label==label[:1]).all(0)
            target[offset:offset+count]=fp32_copy(probs.mean(0))
            accepted[offset:offset+count]=np.asarray(mask.cpu().tolist(),bool)
            for k,scene in enumerate(SCENES):
                h=source_channel(x,batch[3],scene,contract,namespace='PAMF/restored_dose/U/fixed')
                y=engine(h)[0].clone()
                if not bool(torch.isfinite(y).all()):raise ValueError('Nonfinite source restored cache')
                iq[k,offset:offset+count]=fp32_copy(y)
            offset+=count
            if i%10==0 or offset==n:
                write(out/'progress.json',dict(status='SOURCE_U_CACHE',U_seen=offset,total=n,accepted=int(accepted[:offset].sum()),
                    seconds=time.perf_counter()-start,target_reads=0,U_truth_read=False),exclusive=False)
                print('RESTORED_U_CACHE',offset,n,flush=True)
    iq.flush();del iq
    if offset!=n or not unchanged(g,gbefore) or any(not unchanged(m,b) for m,b in zip(models,before)):
        raise ValueError('Incomplete U cache or frozen parameters/buffers changed')
    keys=('eq_i','rx_i','day_i','sig_i','base_index')
    physical=np.stack([data.metadata[k] for k in keys],axis=1)
    if len({tuple(x) for x in physical.tolist()})!=n:raise ValueError('Unique U physical index required')
    np.savez(out/'raw_teacher_targets.npz',probability=target,accepted=accepted,physical=physical,ids=data.metadata['ids'])
    write(out/'complete.json',dict(status='SOURCE_U_CACHE_COMPLETE',source_contract=contract,G=context['G'],
        source_cases=context['cases'],model_seeds=[2026092701,2026092702,2026092703],teacher_epoch=30,
        U_seen=n,accepted=int(accepted.sum()),acceptance=float(accepted.mean()),scenes=list(SCENES),
        source_namespace='PAMF/restored_dose/U/fixed',teacher_F_packet_forwards=3*n,G_packet_forwards=3*n,
        source_V_reads=0,U_truth_read=False,target_reads=0,F_G_updates=0,head_fit_updates=0,
        parameters_and_buffers_unchanged=True,seconds=time.perf_counter()-start,code_commit=c['code_commit']))


class SourceCache:
    def __init__(self,c):
        import numpy as np
        root=Path(c['source_cache_output']);done=read(root/'complete.json')
        if (done['status']!='SOURCE_U_CACHE_COMPLETE' or done['source_contract']!=read(c['source_contract'])
                or done['teacher_epoch']!=30 or done['model_seeds']!=[2026092701,2026092702,2026092703]
                or done['U_seen']!=56700 or done['U_truth_read'] is not False or done['target_reads']
                or done['source_V_reads'] or done['F_G_updates'] or not done['parameters_and_buffers_unchanged']):
            raise ValueError('Same-contract frozen source-only U cache required')
        with np.load(root/'raw_teacher_targets.npz',allow_pickle=False) as data:
            self.probability=data['probability'].copy();self.accepted=data['accepted'].copy();physical=data['physical'].copy()
            if list(data['ids'])!=done['source_contract']['role_ids']['U_s']:raise ValueError('U cache physical order differs')
        self.index={tuple(v):i for i,v in enumerate(physical.tolist())}
        self.iq=np.load(root/'restored_U.npy',mmap_mode='r',allow_pickle=False)
        if (self.probability.shape!=(56700,6) or self.accepted.shape!=(56700,) or len(self.index)!=56700
                or self.iq.shape!=(3,56700,2,256) or not np.isfinite(self.probability).all()
                or not np.allclose(self.probability.sum(1),1.,atol=1e-5)):
            raise ValueError('Aligned finite source targets and three restored scenes required')

    def take(self,metadata,scene_index,device,with_IQ):
        import numpy as np
        import torch
        if bool(torch.as_tensor(metadata['tx_label_visible']).any()):raise ValueError('U truth exposed')
        keys=('eq_i','rx_i','day_i','sig_i','base_index')
        rows=zip(*(metadata[k].detach().cpu().tolist() for k in keys))
        try:indices=[self.index[tuple(v)] for v in rows]
        except KeyError as e:raise ValueError('Foreign physical sample in source U cache') from e
        def tensor(value):
            a=np.ascontiguousarray(value,np.float32)
            return torch.frombuffer(bytearray(a.tobytes()),dtype=torch.float32).reshape(a.shape).to(device)
        q=tensor(self.probability[indices]);mask=torch.tensor(self.accepted[indices].tolist(),device=device,dtype=torch.bool)
        x=tensor(self.iq[scene_index,indices]) if with_IQ else None
        return q,mask,x
