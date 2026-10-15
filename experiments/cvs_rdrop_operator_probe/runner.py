"""CPU-only, source-only, immutable diagnostic; never constructs an optimizer."""
import importlib.util
import json
import sys
import types
from pathlib import Path


def bootstrap(request):
    here=Path(__file__).resolve().parents[2]
    common,fold,parent,model=(Path(request[k]) for k in ('dependency_release','fold_release','parent_release','model_release'))
    sys.path[:0]=[str(common),str(common/'code'),str(common/'experiments/adv3b02_xuc/code')]
    for suffix,root in (('cvs_rdrop_operator_probe',here),('cvs_sigtime_rdrop_envelope20',parent),
        ('cvs_sigtime_rdrop_legacy',model),('cvs_rdrop_fold20',model),
        ('cvs_sigtime_locked20',fold),('cvs_rx_u_increment20',fold)):
        name='experiments.'+suffix
        package=types.ModuleType(name)
        package.__path__=[str(root/'experiments'/suffix)]
        sys.modules[name]=package


def run(request):
    import os,random,time,traceback
    import numpy as np
    import torch
    from experiments.cvs_rdrop_operator_probe import design as d
    from experiments.cvs_rdrop_operator_probe.measure import pair_gradients
    from experiments.cvs_sigtime_rdrop_envelope20 import model as api
    from experiments.cvs_sigtime_rdrop_envelope20.provenance import source_proof,validate_checkpoint
    from experiments.cvs_phase1_stack.runtime import installed
    from experiments.cvs_phase1_stack.fast_execution import configure
    from experiments.cvs_sigtime_heldcredit.held_credit import fresh_parameter_caches
    from experiments.cvs_u_schedule50.counterfactual import _preserve_rng
    from cvsrffi.xuc_fusion.native import role_ids_from_native,opaque_id
    d.validate(request)
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='' or torch.cuda.is_available():
        raise ValueError('CPU-only source diagnostic required')
    if (Path(d.PARENT_RELEASE)/'release_commit.txt').read_text().strip()!=d.PARENT_COMMIT:
        raise ValueError('Frozen source implementation differs')
    if d.parent.read(d.parent.BASE/'completion.json')['status']!='SOURCE_MATRIX_COMPLETE':
        raise ValueError('Source matrix must be terminal')
    receipts=d.parent.read(d.parent.BASE/'launch.json')['rows']
    for receipt in receipts:
        proc=Path('/proc')/str(receipt['pid'])
        try:
            stat=(proc/'stat').read_text().rsplit(')',1)[1].split()
            if int(stat[19])==receipt['start_ticks'] and stat[0] not in ('Z','X','x'):
                raise ValueError('Original source worker still live;preserve and wait')
        except FileNotFoundError:
            pass
    c=d.parent.config(d.ROW)
    source=source_proof(c)  # Actual role/ancestor proof before any checkpoint load.
    if str(source/'final_ssdg.pth')!=request['checkpoint']:
        raise ValueError('Only own fixed FULL final20 permitted')
    if d.BASE.exists():raise FileExistsError('Immutable diagnostic output exists;reconcile')
    d.BASE.mkdir()
    started=time.time()
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False
    torch.set_float32_matmul_precision('highest')
    try:
        payload=torch.load(source/'final_ssdg.pth',map_location='cpu',weights_only=False)
        validate_checkpoint(payload,c)
        m=api.build(c,'cpu')
        m.load_state_dict(payload['model'],strict=True)
        m.eval()
        saved={k:v.detach().clone() for k,v in m.state_dict().items()}
        buffers={k:v.detach().clone() for k,v in m.named_buffers()}
        counters={k:getattr(m,k) for k in ('_feature_l_steps','_legacy_calls','_envelope_calls')}
        def require_unchanged():
            if any(not torch.equal(value,m.state_dict()[key]) for key,value in saved.items()) or any(not torch.equal(value,dict(m.named_buffers())[key]) for key,value in buffers.items()) or any(p.grad is not None for p in m.parameters()) or any(getattr(m,key)!=value for key,value in counters.items()) or m._legacy_actual_L_input is not None:
                raise ValueError('Frozen model parameters,buffers,gradient slots or counters changed')
        with torch.no_grad(),fresh_parameter_caches(m):
            torch.manual_seed(2026101619)
            smoke=m.forward_identity_only(torch.randn(2,2,256))
            if smoke['id_feat_joint'].shape!=(2,160) or not torch.isfinite(smoke['tx_logits']).all():
                raise ValueError('Own real checkpoint no-query smoke failed')
        require_unchanged()
        print('SMOKE PASS fixed own FULL source checkpoint;query_read=false',flush=True)
        # Native data construction is redirected to this new diagnostic root.
        args=configure(d.parent.make_args(c,'cpu'))
        args.output_dir=str(d.BASE/'data')
        with installed(c) as native:
            ctx=native._build_ssdg_wisig_data(args,torch.device('cpu'))
            if ctx['named_test_loaders'] or role_ids_from_native(ctx)!=d.read(d.SOURCE)['role_ids']:
                raise ValueError('Only existing full source carrier handles allowed')
            ds=ctx['train_loader'].dataset
            if len(ds)!=6300 or getattr(ds,'hide',True):raise ValueError('Visible source L only')
            cells={}
            for i,item in enumerate(ds.index):
                rx,day,tx=int(item.rx_i),int(item.day_i),int(item.tx_i)
                if rx in d.parent.FIT:cells.setdefault((tx,rx,day),[]).append(i)
            expected={(y,r,day) for y in range(6) for r in d.parent.FIT for day in (1,2,3)}
            if set(cells)!=expected or any(len(indices)!=70 for indices in cells.values()):
                raise ValueError('Fixed fit L cells differ')
            selected=[index for key in sorted(cells) for index in cells[key][:2]]
            samples=[ds[index] for index in selected]
            if any(int(sample[1])!=int(ds.index[index].tx_i) for index,sample in zip(selected,samples)):
                raise ValueError('Visible source labels differ from native index')
            x=torch.stack([sample[0] for sample in samples])
            y=torch.tensor([int(sample[1]) for sample in samples],dtype=torch.long)
            ids=[tuple(int(sample[3][key]) for key in ('eq_i','rx_i','day_i','sig_i','base_index')) for sample in samples]
            if x.shape!=(144,2,256) or len(set(ids))!=144 or torch.bincount(y,minlength=6).tolist()!=[24]*6 or any(identity[1] not in d.parent.FIT for identity in ids) or not torch.isfinite(x).all():
                raise ValueError('Only144 unique clean fit L inputs')
            source_L=set(d.read(source/'source_contract.json')['role_ids']['L_s'])
            if not {opaque_id(ds.index[index]) for index in selected}<=source_L:
                raise ValueError('Diagnostic inputs escape actual trained source L')
            d.write(d.BASE/'selected_L.json',dict(role='L_s',physical_ids=ids,visible_labels=y.tolist(),selection=request['source_selection']))
            records=[]
            for context in d.CONTEXTS:
                for seed in d.DRAW_SEEDS:
                    with _preserve_rng():
                        aug_state=None
                        if context=='CLEAN':view=x
                        else:
                            random.seed(seed+43210);np.random.seed(seed+43210);torch.manual_seed(seed+43210)
                            cfg=native.build_aug_base_cfg(args)
                            augmentor=native.make_augmentor(cfg)
                            aug_state=native.configure_augmentor_for_epoch(augmentor,cfg,min(20,args.label_epochs),args)
                            view=torch.nan_to_num(augmentor(x,labels=y,no_pa=(not bool(args.aug_enable_pa_normal))),nan=0.,posinf=0.,neginf=0.)
                        if view.shape!=x.shape or not torch.isfinite(view).all():raise ValueError('Native augmentation changed source shape')
                        random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
                        m.train()
                        with fresh_parameter_caches(m):rows=pair_gradients(m,view,y)
                        m.eval()
                    require_unchanged()
                    batch=dict(context=context,dropout_seed=seed,augmentation_seed=seed+43210 if context!='CLEAN' else None,
                        source_packets=144,actual_forward_packets=288,augmentation_state=aug_state,records=rows)
                    d.write(d.BASE/(context+'-s'+str(seed)+'.json'),batch)
                    records.append(batch)
                    d.write(d.BASE/'progress.json',dict(status='RUNNING',completed_pairs=len(records),expected_pairs=8,
                        actual_forward_packets=288*len(records),optimizer_steps=0,head_fit_updates=0,target_access=False))
                    print('PAIR_COMPLETE '+context+' '+str(seed),flush=True)
        require_unchanged()
        result=dict(status='SOURCE_OPERATOR_GRADIENT_COMPLETE',run_id=d.RUN,config=request,
            execution_commit=(d.ROOT/'release_commit.txt').read_text().strip(),source_execution_commit=d.PARENT_COMMIT,
            parent_checkpoint=request['checkpoint'],contexts=list(d.CONTEXTS),dropout_draws_per_context=4,
            actual_gradient_records=sum(len(batch['records']) for batch in records),batches=records,
            model_seeds=1,source_physical_L_packets=144,actual_L_materializations=144,
            actual_identity_forward_packets=2304,synthetic_CP_smoke_packets=2,
            frozen_parameter_tensors=len(saved),all_parameter_and_buffer_values_unchanged=True,
            parameter_grad_slots_empty=True,optimizer_steps=0,EMA_updates=0,head_fit_updates=0,
            U_IQ_reads=0,V_IQ_reads=0,held_RX1_IQ_reads=0,target_IQ_reads=0,target_truth_reads=0,
            elapsed_seconds=time.time()-started,goal_completed=False)
        if result['actual_gradient_records']!=240:raise ValueError('Complete fixed operator records required')
        d.write(d.BASE/'completion.json',result)
        print('SOURCE_OPERATOR_GRADIENT_COMPLETE records=240 updates=0 target=false',flush=True)
    except Exception as error:
        d.write(d.BASE/'failure.json',dict(status='FAILED',error=repr(error),traceback=traceback.format_exc(),outputs_preserved=True,no_retry=True))
        raise


def main():
    request=json.loads(Path(sys.argv[1]).read_text(encoding='utf-8'))
    bootstrap(request)
    run(request)


if __name__=='__main__':main()
