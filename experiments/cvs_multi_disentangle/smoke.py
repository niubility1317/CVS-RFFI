"""Synthetic production-loop smoke; never opens source/query/truth artifacts."""
import argparse
import contextlib
from copy import deepcopy
import io
import json
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader
from experiments.cvs_multi_disentangle import design as d
from experiments.cvs_multi_disentangle.checks import source_fixture
from experiments.cvs_multi_disentangle.runtime import installed
from experiments.cvs_multi_disentangle.source import incremental_writer
from experiments.cvs_phase1_stack.runtime import installed as legacy_installed, blind_sample
from experiments.cvs_phase1_stack.fast_execution import configure
from experiments.cvs_phase1_repair.design import cosine_lr
from experiments.cvs_equivariant_identity.precision import numerical_context, actual_flags

EPOCHS=(1,21,22,23,24,80,131,132,200)


class Synthetic(Dataset):
    def __init__(self,role):
        self.role=role;self.x,self.y,self.rx=source_fixture('cpu')
        self.offset={'L':0,'U':10000,'V':20000}[role]
        if role=='U': self.x=torch.roll(self.x,1,-1)
        if role=='V': self.x=.99*self.x
    def __len__(self):return len(self.x)
    def __getitem__(self,i):
        label=int(self.y[i]);receiver=int(self.rx[i]);identifier=self.offset+i
        metadata=dict(rx_i=receiver,day_i=1,eq_i=1,sig_i=identifier,base_index=identifier,
                      tx_i=label,tx=label,true_tx_i=label,tx_label=label,y_tx=label)
        sample=(self.x[i],label,0 if receiver==1 else 1,metadata)
        if self.role=='U':
            sample=blind_sample(sample)
            assert sample[1]==-1 and sample[3]['tx_label_visible'] is False
            assert not set(sample[3])&{'tx_i','tx','true_tx_i','tx_label','y_tx'}
        return sample


class Finished(Exception):pass


def _trainer(native):
    callback=native._multi_disentangle_step
    values=dict(zip(callback.__code__.co_freevars,[cell.cell_contents for cell in callback.__closure__]))
    return values['state']['trainer']


def _run(c,output,device,legacy=False,capture=None):
    arm=c['arm'];local=dict(c,output_root=str(output))
    args=configure(d.make_args(c,device));args.output_dir=str(output)
    args.batch_size=24;args.eval_batch_size=24
    args.source_val_heavy_eval_start_epoch=9999;args.rc4_calibration_update_epochs='1'
    args.daot_diagnostic_epochs='1';args.rc4_gradient_telemetry_epochs='1'
    seen=[];active=[];state={};validation_calls=0
    installation=legacy_installed(c) if legacy else installed(local)
    writer_context=contextlib.nullcontext if legacy else incremental_writer
    with numerical_context(d.FULL_FP32_POLICY),installation as native,writer_context(native):
        originals={key:getattr(native,key) for key in ('build_baseline_model','_build_ssdg_wisig_data',
            '_write_ssdg_epoch_telemetry','_should_run_source_val_heavy_eval','evaluate_loader','_detach_log_mapping')}
        def build(a,dev):
            model=originals['build_baseline_model'](a,dev);state['model']=model
            if capture is not None:
                capture['initial']={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
                capture['rng']=[]
            blob=io.BytesIO();torch.save(model.state_dict(),blob);blob.seek(0)
            clone=deepcopy(model);clone.load_state_dict(torch.load(blob,map_location=dev,weights_only=False),strict=True)
            model.eval();clone.eval()
            with torch.no_grad():
                zero=torch.zeros(2,2,256,device=dev)
                assert torch.equal(model(zero),clone(zero))
            return model
        def step_metrics(values):
            if capture is not None and 'train/loss' in values:
                capture['rng'].append((torch.random.get_rng_state().clone(),
                    [v.clone() for v in torch.cuda.get_rng_state_all()] if device.startswith('cuda') else []))
            return originals['_detach_log_mapping'](values)
        def data(*unused):
            labeled=DataLoader(Synthetic('L'),batch_size=24)
            hidden=DataLoader(Synthetic('U'),batch_size=24)
            validation=DataLoader(Synthetic('V'),batch_size=24)
            state['validation']=validation
            return dict(train_loader=labeled,probe_train_loader=labeled,unlabeled_loader=hidden,
                val_loader=validation,source_calibration_loader=validation,named_test_loaders={},
                domain_label_map={1:0,3:1},num_domains=2,input_len=256,class_id_to_tx=d.CLASSES,
                balanced_train_sampler=None,split_info=dict(labeled_size=24,unlabeled_size=24,
                    source_val_size=24,source_calibration_size=24,source_selection_size=24,
                    mode='synthetic',named_test_meta={},source_split_receipt={}))
        def evaluate(*a,**kw):
            nonlocal validation_calls
            model=a[0] if a else kw['model']
            before={k:v.clone() for k,v in model.state_dict().items()}
            engine=None if legacy else _trainer(native)
            generator=None if engine is None else engine.generator.get_state().clone()
            group_keys=None if engine is None else set(engine.statistics.groups)
            updates=None if engine is None else engine.updates
            result=originals['evaluate_loader'](*a,**kw)
            assert all(torch.equal(v,model.state_dict()[k]) for k,v in before.items()), 'Validation changed identity state'
            if engine is not None:
                assert torch.equal(generator,engine.generator.get_state())
                assert group_keys==set(engine.statistics.groups) and updates==engine.updates
            validation_calls+=1
            return result
        def telemetry(cp,jp,rows):
            originals['_write_ssdg_epoch_telemetry'](cp,jp,rows)
            row=rows[-1];epoch=int(row['epoch']);seen.append(epoch)
            assert abs(row['lr']-cosine_lr(args.lr,epoch))<1e-10, 'Actual native LR differs'
            if not legacy:
                engine=_trainer(native)
                assert row['multi_inference_parameters']==sum(v.numel() for v in state['model'].parameters())
                assert row['multi_auxiliary_parameters']==engine.cost['auxiliary_parameters']
                if row['multi_auxiliary_active_steps_epoch']:
                    active.append(epoch)
                    assert row['multi_source_L_only']==1 and row['multi_receiver_samplewise_counterfactual']==0
                    if arm not in ('legacy_cosine','fixed_interventions'):
                        assert row['multi_auxiliary_gradient_norm']>0
                        if arm!='unified':
                            for kind in engine.paths: assert row['multi_'+kind+'_gradient_norm']>0, (arm,kind,row)
                    if 'receiver' in engine.paths:
                        assert row['multi_receiver_relations']>0, (arm,row)
            if epoch==200: raise Finished()
        native.build_baseline_model=build;native._build_ssdg_wisig_data=data
        native._write_ssdg_epoch_telemetry=telemetry;native.evaluate_loader=evaluate
        native._detach_log_mapping=step_metrics
        native._should_run_source_val_heavy_eval=lambda *a:False
        native.range=lambda *a:iter(EPOCHS) if a==(1,201) else range(*a)
        try:
            with (output.parent/(arm+('-original' if legacy else '')+'.log')).open('x',encoding='utf-8') as log,contextlib.redirect_stdout(log):
                try:native.train(args)
                except Finished:pass
            assert tuple(seen)==EPOCHS and validation_calls>0
            if not legacy:
                engine=_trainer(native)
                if arm=='legacy_cosine': assert not active and engine.updates==0
                else: assert active and 200 in active, (arm,active)
                auxiliary=torch.load(output/'auxiliary_final.pth',map_location='cpu',weights_only=False)
                assert auxiliary['epoch']==200 and auxiliary['config']==local
                assert auxiliary['inference_uses_auxiliary'] is False and auxiliary['resume_supported'] is False
                assert auxiliary['target_access'] is False and not auxiliary['checkpoint_sources']
            model_state={k:v.detach().cpu().clone() for k,v in state['model'].state_dict().items()}
            return model_state,dict(arm=arm,status='PASS',epochs=seen,actual_auxiliary_epochs=active,
                                   validation_readonly=True,hidden_U=True,production_writer=True,
                                   native_updates=len(EPOCHS),synthetic=True)
        finally:
            del native.range
            for key,value in originals.items():setattr(native,key,value)


def smoke(output,device='cpu'):
    torch.set_num_threads(2);output=Path(output);output.mkdir(parents=True,exist_ok=False)
    rows=[];baseline=None;baseline_capture={};original_capture={}
    for arm in d.ARMS:
        c=d.config(next(row for row in d.rows() if row['arm']==arm))
        weights,result=_run(c,output/arm,device,capture=baseline_capture if arm=='legacy_cosine' else None)
        if arm=='legacy_cosine': baseline=weights
        rows.append(result)
    c=d.config(next(row for row in d.rows() if row['arm']=='legacy_cosine'))
    original,_=_run(c,output/'legacy_original',device,legacy=True,capture=original_capture)
    assert original.keys()==baseline.keys()
    assert all(torch.equal(v,original_capture['initial'][k]) for k,v in baseline_capture['initial'].items())
    for a,b in zip(baseline_capture['rng'],original_capture['rng']):
        assert torch.equal(a[0],b[0]) and all(torch.equal(v,w) for v,w in zip(a[1],b[1]))
    bitwise=all(torch.equal(v,baseline[k]) for k,v in original.items())
    repeat_difference=None;baseline_difference=None
    if not bitwise:
        assert device.startswith('cuda'), 'CPU baseline changed native training trajectory'
        repeat_capture={}
        repeat,_=_run(c,output/'legacy_original_repeat',device,legacy=True,capture=repeat_capture)
        for a,b in zip(original_capture['rng'],repeat_capture['rng']):
            assert torch.equal(a[0],b[0]) and all(torch.equal(v,w) for v,w in zip(a[1],b[1]))
        repeat_difference=max(float((v-repeat[k]).abs().max()) for k,v in original.items() if v.is_floating_point())
        baseline_difference=max(float((v-baseline[k]).abs().max()) for k,v in original.items() if v.is_floating_point())
        assert baseline_difference<=repeat_difference, 'GPU baseline difference exceeds observed native repeat variation'
    result=dict(status='PASS',device=device,rows=rows,baseline_training_bitwise_equivalent=bitwise,
                baseline_initial_weights_bitwise_equal=True,baseline_all_step_native_rng_bitwise_equal=True,
                gpu_native_repeat_max_difference=repeat_difference,gpu_baseline_max_difference=baseline_difference,
                gpu_comparison_rule='Measured native-repeat variation under unchanged registered backend; no relaxed fixed tolerance',
                target_access=False,formal_weights=False,full_budget_verified=False,
                synthetic_override='synthetic L/U/V and local output only; real production objective/writer chain',
                backend_flags=d.FULL_FP32_POLICY)
    d.write(output/'completion.json',result);print(json.dumps(result));return result


def summarize_gpu_evidence(output,arms_root,probe_root,cpu_root):
    """Preserve the failed bitwise run and combine its completed mechanism checks."""
    output=Path(output);arms_root=Path(arms_root);probe_root=Path(probe_root)
    probe=d.read(probe_root/'completion.json');cpu=d.read(Path(cpu_root)/'completion.json')
    assert cpu['status']=='PASS' and cpu['baseline_training_bitwise_equivalent'] is True
    assert probe['initial_weights_bitwise_equal'] and probe['all_step_native_rng_bitwise_equal']
    old_repeat=probe['comparisons'][0]['max_parameter_difference']
    new_difference=probe['comparisons'][1]['max_parameter_difference']
    assert new_difference<=old_repeat
    rows=[]
    from experiments.cvs_multi_disentangle.runtime import ARM_PATHS
    for arm in d.ARMS:
        metrics=[json.loads(line) for line in (arms_root/arm/'metrics_epoch.jsonl').read_text().splitlines()]
        assert tuple(int(row['epoch']) for row in metrics)==EPOCHS
        final=metrics[-1]
        if arm!='legacy_cosine':
            assert final['multi_auxiliary_active_steps_epoch']>0
            if arm!='fixed_interventions':
                assert final['multi_auxiliary_gradient_norm']>0
                if arm!='unified':
                    assert all(final['multi_'+kind+'_gradient_norm']>0 for kind in ARM_PATHS[arm])
            if 'receiver' in ARM_PATHS[arm]:assert final['multi_receiver_relations']>0
        rows.append(dict(arm=arm,status='PASS',epochs=list(EPOCHS),metrics=str(arms_root/arm/'metrics_epoch.jsonl')))
    result=dict(status='PASS',device='cuda:0',rows=rows,mechanism_integration='VERIFIED',
        original_run_status='FAILED_GPU_BITWISE_BASELINE_ASSERTION',original_run_preserved=str(arms_root),
        baseline_training_bitwise_equivalent=False,cpu_baseline_training_bitwise_equivalent=True,
        initial_weights_bitwise_equal=True,all_step_native_rng_bitwise_equal=True,
        gpu_old_old_max_difference=old_repeat,gpu_old_new_max_difference=new_difference,
        gpu_baseline_numerical_equivalence='Within observed original-native-repeat variation; no fixed tolerance relaxed',
        baseline_probe=str(probe_root/'completion.json'),cpu_smoke=str(Path(cpu_root)/'completion.json'),
        target_access=False,formal_weights=False,full_budget_verified=False)
    d.write(output/'completion.json',result);return result


def baseline_probe(output,device='cuda:0'):
    """Separate CUDA nondeterminism from changes to the registered baseline."""
    torch.set_num_threads(2);output=Path(output);output.mkdir(parents=True,exist_ok=False)
    c=d.config(next(row for row in d.rows() if row['arm']=='legacy_cosine'))
    captures=[{}, {}, {}];weights=[]
    for i,legacy in enumerate((True,True,False)):
        folder=output/('old1' if i==0 else 'old2' if i==1 else 'new')
        folder.mkdir()
        state,_=_run(c,folder/'source',device,legacy=legacy,capture=captures[i]);weights.append(state)
    assert all(all(torch.equal(v,captures[i]['initial'][k]) for k,v in captures[0]['initial'].items()) for i in (1,2))
    assert all(len(captures[i]['rng'])==len(EPOCHS) for i in range(3))
    for i in (1,2):
        for a,b in zip(captures[0]['rng'],captures[i]['rng']):
            assert torch.equal(a[0],b[0]) and all(torch.equal(v,w) for v,w in zip(a[1],b[1])), 'Native baseline random streams differ'
    comparisons=[]
    for i,j in ((0,1),(0,2),(1,2)):
        differences={k:float((v-weights[j][k]).abs().max()) for k,v in weights[i].items() if v.is_floating_point()}
        comparisons.append(dict(left=i,right=j,bitwise=all(torch.equal(v,weights[j][k]) for k,v in weights[i].items()),
            max_parameter_difference=max(differences.values()),
            differing_tensors=sum(v>0 for v in differences.values()),
            largest=sorted(differences.items(),key=lambda v:-v[1])[:5]))
    result=dict(device=device,initial_weights_bitwise_equal=True,all_step_native_rng_bitwise_equal=True,
                comparisons=comparisons,target_access=False,formal_weights=False)
    d.write(output/'completion.json',result);print(json.dumps(result));return result


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True);parser.add_argument('--device',default='cpu')
    parser.add_argument('--baseline-probe',action='store_true')
    parser.add_argument('--summarize-gpu',action='store_true');parser.add_argument('--arms-root');parser.add_argument('--probe-root');parser.add_argument('--cpu-root')
    args=parser.parse_args()
    if args.summarize_gpu:
        print(json.dumps(summarize_gpu_evidence(args.output,args.arms_root,args.probe_root,args.cpu_root)))
    else:(baseline_probe if args.baseline_probe else smoke)(args.output,args.device)
