"""Label-blind frozen EMA predictions; exec the independent scorer only at completion."""
import gzip,json,os,sys,time
from pathlib import Path
import numpy as np
from . import design as d

def run():
    import torch
    from .views import generate
    from experiments.cvs_urban_branch50.model import own_model
    from experiments.cvs_urban_branch50.evaluate import observations
    from experiments.cvs_urban_sigtime.channel import tensor_iq,audit
    from experiments.cvs_equivariant_identity.precision import numerical_context,actual_flags,FULL_FP32_POLICY
    if d.BASE.exists():raise FileExistsError('Existing target diagnostic must be reconciled, never resubmitted')
    torch.set_num_threads(2);started=time.perf_counter();c=d.parent_config()
    with numerical_context(FULL_FP32_POLICY):
        model=own_model(c,'cpu')
        checkpoint=torch.load(Path(c['output_root'])/'final_ssdg.pth',map_location='cpu',weights_only=False)
        if 'ema_model' not in checkpoint:raise ValueError('The exact source diagnostic EMA is absent')
        model.load_state_dict(checkpoint['ema_model'],strict=True);del checkpoint
        model.eval().requires_grad_(False)
        before={k:v.clone() for k,v in model.state_dict().items()}
        with torch.no_grad():smoke=model.forward_identity_only(torch.zeros(2,2,256))['tx_logits']
        if smoke.shape!=(2,6) or not torch.isfinite(smoke).all():raise ValueError('Frozen real EMA/default-head smoke failed')
        d.BASE.mkdir();out=d.BASE/'prediction';out.mkdir()
        d.write(d.BASE/'source_smoke.json',dict(status='PASS',query_IQ_read=False,truth_read=False,forward_packets=2))
        print('SMOKE PASS exact A0 final50 EMA/default head; no target IQ or truth used for smoke',flush=True)
        ids,manifest=observations();positions=d.sample_positions(ids);selected=ids[positions]
        canonical=np.load(d.FORMAL_VIEWS/(d.SCENE+'.npy'),mmap_mode='r',allow_pickle=False)
        clean=np.load(manifest['clean_ref'],mmap_mode='r',allow_pickle=False)
        if canonical.shape!=(168000,2,256) or canonical.dtype!=np.float32 or clean.shape!=canonical.shape or clean.dtype!=np.float32:
            raise ValueError('Original validated target array contract differs')
        d.write(out/'resolved_config.json',dict(config=d.configuration(),pid=os.getpid(),cwd=os.getcwd(),python=sys.executable,
            torch_version=torch.__version__,threads=torch.get_num_threads(),CUDA_VISIBLE_DEVICES=os.environ.get('CUDA_VISIBLE_DEVICES'),
            backend_flags=actual_flags(),actual_engine=audit(),formal_manifest=str(d.FORMAL_VIEWS/'manifest.json'),
            channel_config=next(v['config'] for v in manifest['evidence'] if v['scene']==d.SCENE),
            parent_checkpoint=c['output_root']+'/final_ssdg.pth',model_component='ema_model',
            commit=(d.ROOT/'release_commit.txt').read_text().strip(),truth_read=False,query_fit=False))
        scores={v:np.empty((d.COUNT,6),np.float32) for v in d.VIEWS};max_error=0.
        with gzip.open(out/'channel_metadata.jsonl.gz','xt',encoding='utf-8') as trace:
            with torch.no_grad():
                for start in range(0,d.COUNT,90):
                    stop=min(start+90,d.COUNT);arrays={v:[] for v in d.VIEWS}
                    for i in range(start,stop):
                        position=positions[i];waves,meta=generate(clean[position],canonical[position],selected[i],manifest)
                        max_error=max(max_error,meta['full_replay_max_abs_error'])
                        for view in d.VIEWS:arrays[view].append(waves[view])
                        trace.write(json.dumps(meta,allow_nan=False)+'\n')
                    for view in d.VIEWS:
                        x=tensor_iq(np.stack(arrays[view]),'cpu');logits=model.forward_identity_only(x)['tx_logits']
                        if logits.shape!=(stop-start,6) or not torch.isfinite(logits).all():raise ValueError('Invalid frozen EMA predictions')
                        scores[view][start:stop]=logits.tolist()
                    trace.flush();d.write(d.BASE/'progress.json',dict(predicted_physical_samples=stop,count=d.COUNT,truth_read=False))
                    print(json.dumps(dict(predicted_physical_samples=stop,count=d.COUNT,truth_read=False)),flush=True)
        if any(not torch.equal(v,model.state_dict()[k]) for k,v in before.items()) or any(p.grad is not None or p.requires_grad for p in model.parameters()):
            raise ValueError('Frozen EMA state or parameter boundary changed')
        for view in d.VIEWS:np.savez_compressed(out/(view+'.npz'),ids=selected,logits=scores[view])
        d.write(out/'complete.json',dict(status='PREDICTIONS_COMPLETE',config=d.configuration(),count=d.COUNT,views=list(d.VIEWS),
            truth_read=False,query_fit=False,model_state_unchanged=True,all_parameters_frozen=True,
            optimizer_updates=0,EMA_updates=0,head_fit_updates=0,forward_packets=4*d.COUNT+2,
            max_original_full_replay_error=max_error,seconds=time.perf_counter()-started))
    print('ALL FOUR PREDICTIONS COMPLETE; starting independent truth-last scorer',flush=True)
    os.execve(sys.executable,[sys.executable,'-u','-m','experiments.cvs_target_hq.score'],dict(os.environ,CUDA_VISIBLE_DEVICES=''))

if __name__=='__main__':run()
