"""Four fixed source checkpoints, one L materialization, no training or head fit."""
import json,sys,types
from pathlib import Path

def bootstrap(request):
    here=Path(__file__).resolve().parents[2]
    common,fold,parent,model=(Path(request[k]) for k in ('dependency_release','fold_release','parent_release','model_release'))
    sys.path[:0]=[str(common),str(common/'code'),str(common/'experiments/adv3b02_xuc/code')]
    for suffix,root in (('cvs_rdrop_feature_geometry',here),('cvs_sigtime_rdrop_completion20',parent),('cvs_sigtime_rdrop_legacy',model),('cvs_rdrop_fold20',model),('cvs_sigtime_locked20',fold),('cvs_rx_u_increment20',fold)):
        name='experiments.'+suffix;p=types.ModuleType(name);p.__path__=[str(root/'experiments'/suffix)];sys.modules[name]=p

def run(request):
    import os,time,traceback,gc
    import numpy as np
    import torch
    from experiments.cvs_rdrop_feature_geometry import design as d,measure
    from experiments.cvs_rdrop_feature_geometry.runtime import snapshot,require_unchanged,extract,feature_array
    from experiments.cvs_sigtime_rdrop_completion20 import model as api
    from experiments.cvs_sigtime_rdrop_completion20.provenance import source_proof,validate_checkpoint
    from experiments.cvs_phase1_stack.runtime import installed
    from experiments.cvs_phase1_stack.fast_execution import configure
    from experiments.cvs_sigtime_heldcredit.held_credit import fresh_parameter_caches
    from cvsrffi.xuc_fusion.native import role_ids_from_native,opaque_id
    d.validate(request)
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='' or torch.cuda.is_available():raise ValueError('CPU-only diagnostic required')
    if (Path(d.PARENT_RELEASE)/'release_commit.txt').read_text().strip()!=d.PARENT_COMMIT:raise ValueError('Frozen source code differs')
    matrix=d.read(d.parent.BASE/'completion.json')
    if matrix['status']!='SOURCE_MATRIX_COMPLETE' or matrix['run_id']!=d.parent.RUN or not matrix['source_only'] or matrix['target_access'] or matrix['head_fit_updates']!=0:raise ValueError('Complete source-only parent matrix required')
    for receipt in d.read(d.parent.BASE/'launch.json')['rows']:
        try:
            stat=(Path('/proc')/str(receipt['pid'])/'stat').read_text().rsplit(')',1)[1].split()
            if int(stat[19])==receipt['start_ticks'] and stat[0] not in ('Z','X','x'):raise ValueError('Parent worker still live;preserve and wait')
        except FileNotFoundError:pass
    # All actual roles/ancestors checked before deserializing any checkpoint.
    sources={r['arm']:source_proof(d.parent.config(r)) for r in d.parent.rows()}
    if any(str(sources[a]/'final_ssdg.pth')!=request['checkpoints'][a] for a in d.ARMS):raise ValueError('Fixed own final20 checkpoints required')
    contracts={a:d.read(out/'source_contract.json') for a,out in sources.items()}
    if any(v['role_ids']!=contracts['FULL']['role_ids'] for v in contracts.values()):raise ValueError('All four source training contracts must match')
    if d.BASE.exists():raise FileExistsError('Immutable diagnostic root exists;reconcile')
    d.BASE.mkdir();started=time.time();torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False
    torch.set_float32_matmul_precision('highest')
    try:
        results={};selection=None;x=None
        for position,row_settings in enumerate(request['rows']):
            row=row_settings['parent_row'];arm=row['arm'];c=d.parent.config(row)
            out=Path(row_settings['output_root']);out.mkdir();d.write(out/'config.json',row_settings)
            payload=torch.load(sources[arm]/'final_ssdg.pth',map_location='cpu',weights_only=False);validate_checkpoint(payload,c)
            m=api.build(c,'cpu').eval().requires_grad_(False);m.load_state_dict(payload['model'],strict=True)
            saved=snapshot(m)
            with torch.no_grad(),fresh_parameter_caches(m):
                torch.manual_seed(request['synthetic_seed']+position)
                smoke=m.forward_identity_only(torch.randn(2,2,256))
                if smoke['id_feat_joint'].shape!=(2,160) or not torch.isfinite(smoke['tx_logits']).all():raise ValueError('Own real checkpoint no-query smoke failed')
            require_unchanged(m,saved);print('SMOKE PASS own '+arm+' source checkpoint;query_read=false',flush=True)
            if x is None:
                args=configure(d.parent.make_args(c,'cpu'));args.output_dir=str(d.BASE/'data')
                if not args.a1_source_screen_only or args.use_concat_sat_channel_aug:raise ValueError('Source carrier only,no target or satellite loaders')
                with installed(c) as native:
                    ctx=native._build_ssdg_wisig_data(args,torch.device('cpu'))
                    carrier=d.read(d.SOURCE)['role_ids']
                    if ctx['named_test_loaders'] or role_ids_from_native(ctx)!=carrier:raise ValueError('Existing validated source carrier handles required')
                    ds=ctx['train_loader'].dataset
                    if len(ds)!=6300 or getattr(ds,'hide',True) or getattr(ds,'transform',None) is not None or not getattr(ctx['unlabeled_loader'].dataset,'hide',False):raise ValueError('Clean visible L and label-blind U handles required')
                    cells={}
                    for i,item in enumerate(ds.index):cells.setdefault((int(item.tx_i),int(item.rx_i),int(item.day_i)),[]).append(i)
                    expected={(y,rx,day) for y in range(6) for rx in (1,3,4,6,8) for day in (1,2,3)}
                    if set(cells)!=expected or any(len(v)!=70 for v in cells.values()):raise ValueError('Original L physical cells differ')
                    fit=[i for key in sorted(cells) if key[1] in d.parent.FIT for i in cells[key][:2]]
                    held=[i for key in sorted(cells) if key[1]==1 for i in cells[key]]
                    source_roles=contracts['FULL']['role_ids']
                    fit_tokens={opaque_id(ds.index[i]) for i in fit};held_tokens={opaque_id(ds.index[i]) for i in held}
                    if not fit_tokens<=set(source_roles['L_s']) or not held_tokens<=set(carrier['L_s']) or any(held_tokens&set(v) for v in source_roles.values()):raise ValueError('Fit L membership or all-path held source separation differs')
                    selected=fit+held;samples=[ds[i] for i in selected]
                    if any(int(sample[1])!=int(ds.index[i].tx_i) for i,sample in zip(selected,samples)):raise ValueError('Visible source TX label differs')
                    x=torch.stack([sample[0] for sample in samples])
                    y=np.array([int(sample[1]) for sample in samples],dtype=np.int64)
                    ids=[tuple(int(sample[3][k]) for k in ('eq_i','rx_i','day_i','sig_i','base_index')) for sample in samples]
                    if x.shape!=(1404,2,256) or len(set(ids))!=1404 or not torch.isfinite(x).all():raise ValueError('Unique finite1404source L inputs required')
                    rx=np.array([v[1] for v in ids],dtype=np.int64);day=np.array([v[2] for v in ids],dtype=np.int64)
                    measure.validate_metadata(y[:144],rx[:144],day[:144],True);measure.validate_metadata(y[144:],rx[144:],day[144:],False)
                    selection=dict(role='visible source L only',fit_count=144,held_count=1260,physical_ids=ids,visible_labels=y.tolist(),receiver=rx.tolist(),day=day.tolist(),selection=request['selection'])
                    d.write(d.BASE/'selected_L.json',selection)
                del ctx,ds,samples
            z=extract(m,x);require_unchanged(m,saved)
            array=feature_array(z)
            np.savez_compressed(out/'features.npz',features=array)
            rel=measure.relations(array,y[:144],rx[:144],day[:144],y[144:],rx[144:],day[144:])
            result=dict(status='SOURCE_FEATURE_GEOMETRY_COMPLETE',arm=arm,config=row_settings,actual_identity_forward_packets=1404,synthetic_CP_smoke_packets=2,state_tensors=len(saved['state']),
                all_parameters_buffers_gradslots_counters_unchanged=True,optimizer_steps=0,head_fit_updates=0,**rel)
            d.write(out/'result.json',result);results[arm]=result
            d.write(d.BASE/'progress.json',dict(status='RUNNING',complete=list(results),expected=list(d.ARMS),actual_identity_forward_packets=1404*len(results),actual_L_materializations=1404,optimizer_steps=0,head_fit_updates=0,target_access=False))
            print('FEATURE_GEOMETRY_COMPLETE '+arm,flush=True)
            del m,payload,saved,z;gc.collect()
        completion=dict(status='SOURCE_FEATURE_GEOMETRY_MATRIX_COMPLETE',run_id=d.RUN,config=request,execution_commit=(d.ROOT/'release_commit.txt').read_text().strip(),source_execution_commit=d.PARENT_COMMIT,
            model_seeds=1,model_conditions=4,results=results,actual_L_materializations=1404,fit_L=144,held_L=1260,actual_identity_forward_packets=5616,synthetic_CP_smoke_packets=8,
            optimizer_steps=0,EMA_updates=0,head_fit_updates=0,U_IQ_reads=0,V_IQ_reads=0,target_IQ_reads=0,target_truth_reads=0,all_models_unchanged=True,elapsed_seconds=time.time()-started,goal_completed=False)
        d.write(d.BASE/'completion.json',completion);print('SOURCE_FEATURE_GEOMETRY_MATRIX_COMPLETE models=4 updates=0 target=false',flush=True)
    except Exception as error:
        d.write(d.BASE/'failure.json',dict(status='FAILED',error=repr(error),traceback=traceback.format_exc(),outputs_preserved=True,no_automatic_retry=True));raise

def main():
    request=json.loads(Path(sys.argv[1]).read_text(encoding='utf-8'));bootstrap(request);run(request)

if __name__=='__main__':main()
