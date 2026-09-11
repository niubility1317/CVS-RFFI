"""One explicitly authorized source-only anchored matrix, with no target stage."""
import argparse,json,os,subprocess,sys,time,traceback
from pathlib import Path
CODE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(CODE))
import torch
from cvsrffi.anchored_pipeline import CANDIDATES,validate_config
from cvsrffi.anchored_source import load_ground

EXPERTS=('A2','A3','A4','C_angle','C_angle_keep')
FIXED=('A5-0','A5-25','A5-50','A5-75','A5-100')


def write_json(path,value,exclusive=False):
    with Path(path).open('x' if exclusive else 'w',encoding='utf-8',newline='\n') as f:
        json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False)


def source_dataset(ds,contract,role,out_len):
    """Use the already frozen physical IDs; do not split or construct target rows."""
    from dataset_wisig import WiSigCompactDataset,WiSigIndex
    if role not in {'L_s','V'}:raise ValueError('only source L_s/V inputs may be materialized')
    if list(map(str,ds['tx_list']))!=contract['tx_mapping']:raise ValueError('TX mapping changed')
    data=WiSigCompactDataset(ds,out_len=out_len,crop_mode='center',normalize=True,center=False,
        equalized=contract['equalized'],rx_keep=contract['source_receivers'],day_keep=contract['source_days'],
        domain='rx_day',build_index=False)
    ids=contract['roles'][role]
    if len(ids)!=len(set(ids)):raise ValueError('duplicate source physical IDs')
    eq=data.eq_keep[0];index=[]
    for pid in ids:
        y,rx,day,sig=map(int,pid.split(':'))
        if rx not in contract['source_receivers'] or day not in contract['source_days']:
            raise ValueError('source role contains a forbidden receiver/day')
        index.append(WiSigIndex(y,rx,day,eq,sig))
    data.index=index
    return data


def prepare(plan,root):
    from dataset_wisig import load_wisig_compact_pkl
    anchor,payload,contract=load_ground(plan['ground'],plan['contract'],'cpu')
    del anchor
    if str(Path(plan['dataset']).resolve())!=str(Path(contract['dataset_id']).resolve()):raise ValueError('dataset identity changed')
    ds=load_wisig_compact_pkl(plan['dataset'])
    data=root/'inputs';data.mkdir()
    counts={}
    for role in ('L_s','V'):
        dataset=source_dataset(ds,contract,role,int(payload['baseline_args'].get('wisig_out_len',256)))
        examples=[dataset[i] for i in range(len(dataset))]
        value=dict(x=torch.stack([r[0] for r in examples]),y=torch.tensor([r[1] for r in examples]),physical_ids=contract['roles'][role])
        if role=='L_s':
            previous=torch.load(plan['previous_source'],map_location='cpu',weights_only=True)
            if previous['physical_ids']!=value['physical_ids'] or not torch.equal(previous['y'],value['y']) or not torch.equal(previous['x'],value['x']):
                raise ValueError('source IQ preprocessing differs from the previous experiment')
        torch.save(value,data/(role+'.pt'));counts[role]=len(dataset)
        print('[SOURCE-INPUT] '+json.dumps(dict(role=role,count=len(dataset),target_access=False)),flush=True)
        del examples,value
    write_json(root/'source_inputs.json',dict(counts=counts,source_roles=['L_s','V'],target_access=False,
        previous_L_s_bitwise_equal=True,checkpoint_lineage=payload['checkpoint_lineage'],contract_reused=True),True)


def smoke(plan,root):
    from cvsrffi.anchored_geometry import AnchoredMetricHead
    from cvsrffi.anchored_pipeline import FrozenAnchoredSystem
    from cvsrffi.anchored_calibration import FinalProbabilityCalibrator
    source=torch.load(root/'inputs/L_s.pt',map_location='cpu',weights_only=True)
    anchor,payload,contract=load_ground(plan['ground'],plan['contract'],'cuda:0')
    indices=[int(torch.where(source['y']==y)[0][0]) for y in range(anchor.class_count)]
    h,s0,valid=anchor.extract(source['x'][indices]);head=AnchoredMetricHead(anchor.w0,anchor.tau0).to('cuda:0')
    torch.testing.assert_close(head(h),s0,atol=1e-5,rtol=1e-5)
    loss=torch.nn.functional.cross_entropy(head(h.detach()),source['y'][indices].cuda());loss.backward()
    if not all(p.grad is None or torch.isfinite(p.grad).all() for p in head.parameters()):raise FloatingPointError('source smoke gradient')
    # Exercise export identity and the actual-action CPU path on the host runtime.
    system=FrozenAnchoredSystem(anchor,head.export_state(),fixed_action=0,architecture=payload['baseline_args'],class_labels=contract['tx_mapping'])
    from cvsrffi.evidence_conditions import received_conditions
    result=system.predict_features(h,s0,received_conditions(source['x'][indices])['quality'],valid)
    if not torch.equal(result['top_class'],s0.cpu().argmax(-1)):raise AssertionError('H0 endpoint changed')
    system.export_state()
    print('[REAL-CHECKPOINT-SMOKE] '+json.dumps(dict(status='PASS',rows=len(indices),target_access=False,formal_head_initialization='fresh')),flush=True)


def stage_command(plan,root,stage,candidate,output,*,cache=None,input=None,role='L_s'):
    cmd=[sys.executable,'-u',str(CODE/'scripts/run_core90_anchored_geometry.py'),'--config',plan['config'],
        '--stage',stage,'--candidate',candidate,'--seed',str(plan['seed']),'--ground',plan['ground'],
        '--contract',plan['contract'],'--device','cuda:0','--threads',str(plan['threads']),'--output',str(output)]
    if cache is not None:cmd+=['--cache',str(cache)]
    if input is not None:cmd+=['--input',str(input)]
    if stage in {'cache','profile'}:cmd+=['--source',str(root/'inputs'/(role+'.pt')),'--role',role]
    return cmd


def child(plan,root,label,command,gpu):
    # Do not add this worker while two GPU compute processes are already live.
    # Other experiments remain untouched; this run waits for capacity.
    while True:
        ids=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid','--format=csv,noheader'],text=True)
        uuid=dict(line.strip().split(', ',1) for line in ids.strip().splitlines())[str(gpu)]
        active=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True)
        if sum(line.split(',')[0].strip()==uuid for line in active.splitlines())<2:break
        print('[WAITING-GPU-CAPACITY] '+str(gpu)+' '+label,flush=True);time.sleep(30)
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS=str(plan['threads']),MKL_NUM_THREADS=str(plan['threads']),PYTHONUNBUFFERED='1')
    path=Path(plan['logs'])/(label+'.log')
    with path.open('x',encoding='utf-8') as f:process=subprocess.Popen(command,cwd=str(CODE.parent),env=env,stdout=f,stderr=subprocess.STDOUT)
    with (root/'commands.jsonl').open('a',encoding='utf-8') as f:
        f.write(json.dumps(dict(label=label,pid=process.pid,ppid=os.getpid(),gpu=gpu,argv=command,cwd=str(CODE.parent),log=str(path),time=time.time()))+'\n')
    return label,process


def wait_children(jobs):
    failures=[]
    for label,process in jobs:
        code=process.wait()
        if code:failures.append(dict(label=label,exit_code=code))
    if failures:raise RuntimeError('source stage failed; no automatic retry: '+json.dumps(failures))


def finish_candidate(plan,root,candidate,origin,gpu):
    cache=root/'cache'/('blocks_V' if candidate=='P1' else 'joint_V')
    target=root/'candidates'/candidate
    for stage,input in (('calibrate',origin),('export',target/'calibrate/system_state.pt'),('profile',target/'export/frozen_system.pt')):
        wait_children([child(plan,root,candidate+'_'+stage,stage_command(plan,root,stage,candidate,target/stage,cache=cache,input=input),gpu)])
    return dict(candidate=candidate,status='SOURCE_SYSTEM_FROZEN',bundle=str(target/'export/frozen_system.pt'))


def coordinate(plan,root):
    if root.exists() or Path(plan['logs']).exists():raise FileExistsError('preserve existing run/log root; reconcile instead of relaunch')
    root.mkdir(parents=True);Path(plan['logs']).mkdir(parents=True)
    write_json(root/'plan.json',plan,True);gpu=plan['gpus'][0]
    def status(stage,**kw):
        write_json(root/'status.json',dict(stage=stage,pid=os.getpid(),time=time.time(),seed=plan['seed'],target_access=False,**kw))
        print('[SOURCE-STAGE] '+stage,flush=True)
    def internal(phase):return [sys.executable,'-u',str(Path(__file__).resolve()),'--root',str(root),'--phase',phase]
    try:
        status('PREPARING_SOURCE_INPUTS');wait_children([child(plan,root,'prepare',internal('prepare'),gpu)])
        status('REAL_CHECKPOINT_SMOKE');wait_children([child(plan,root,'smoke',internal('smoke'),gpu)])
        status('EXTRACTING_SOURCE_CACHES')
        for candidate,layout in (('A4','joint'),('P1','blocks')):
            wait_children([child(plan,root,layout+'_'+role,stage_command(plan,root,'cache',candidate,root/'cache'/(layout+'_'+role),role=role),plan['gpus'][i]) for i,role in enumerate(('L_s','V'))])
        status('EXPERT_OOF_FITS',base_fit_budget=30,max_fit_budget=40)
        jobs=[child(plan,root,candidate+'_oof',stage_command(plan,root,'oof',candidate,root/'oof'/candidate,cache=root/'cache/joint_L_s'),plan['gpus'][i]) for i,candidate in enumerate(EXPERTS)]
        wait_children(jobs)
        status('SOURCE_CALIBRATION_AND_EXPORT');results=[]
        for candidate in EXPERTS:results.append(finish_candidate(plan,root,candidate,root/'oof'/candidate/'system_state.pt',gpu))
        for candidate in ('A0','A1','P1'):
            output=root/'candidates'/candidate/'fit';cache=root/'cache'/('blocks_L_s' if candidate=='P1' else 'joint_L_s')
            wait_children([child(plan,root,candidate+'_fit',stage_command(plan,root,'fit',candidate,output,cache=cache),gpu)])
            results.append(finish_candidate(plan,root,candidate,output/('partial_evidence.pt' if candidate=='P1' else 'system_state.pt'),gpu))
        status('FIXED_AND_CONDITIONAL_NESTED_FUSION')
        for candidate in (*FIXED,'A6'):
            output=root/'candidates'/candidate/'fuse'
            wait_children([child(plan,root,candidate+'_fuse',stage_command(plan,root,'fuse',candidate,output,cache=root/'cache/joint_L_s',input=root/'oof/A4'),gpu)])
            if candidate=='A6':
                activation=json.loads((output/'activation.json').read_text(encoding='utf-8'))
                if activation['status']=='NOT_ACTIVATED_NO_SOURCE_UTILITY':
                    results.append(dict(candidate=candidate,status=activation['status']));continue
            results.append(finish_candidate(plan,root,candidate,output/'system_state.pt',gpu))
        if {r['candidate'] for r in results}!=set(CANDIDATES):raise AssertionError('matrix missing candidate')
        write_json(root/'matrix_summary.json',dict(seed=plan['seed'],candidates=results,target_access=False,
            source_promotion='PENDING_ALL_HEAD_SEEDS',confirmation=False,base_expert_fits=30,max_expert_fits=40),True)
        status('SOURCE_SYSTEM_FROZEN',completed_candidates=len(results),source_promotion='PENDING_ALL_HEAD_SEEDS',confirmation=False)
    except Exception:
        status('TECHNICAL_FAILURE',traceback=traceback.format_exc(),artifacts_preserved=True)
        raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',required=True);parser.add_argument('--phase',choices=('coordinate','prepare','smoke'),default='coordinate')
    for key in ('ground','contract','dataset','previous-source','logs'):parser.add_argument('--'+key)
    parser.add_argument('--config',default=str(CODE/'configs/core90_anchored_geometry_v1.json'))
    parser.add_argument('--seed',type=int,default=392005);parser.add_argument('--gpus',default='0,1,2,3,5')
    parser.add_argument('--threads',type=int,default=4)
    args=parser.parse_args();root=Path(args.root).resolve()
    if args.phase=='coordinate':
        config=validate_config(json.loads(Path(args.config).read_text(encoding='utf-8')))
        if args.seed!=392005 or args.seed not in config['head_seeds']:raise ValueError('this launch is authorized for seed392005 only')
        gpus=list(map(int,args.gpus.split(',')))
        if len(gpus)!=len(EXPERTS) or len(set(gpus))!=len(gpus) or min(gpus)<0:raise ValueError('five distinct explicit GPU bindings required')
        if args.threads<1 or any(getattr(args,k) is None for k in ('ground','contract','dataset','previous_source','logs')):parser.error('explicit inputs/logs and positive threads required')
        for key in ('ground','contract','dataset','previous_source','config'):
            if not Path(getattr(args,key)).is_file():raise FileNotFoundError(getattr(args,key))
        plan={k:str(Path(getattr(args,k)).resolve()) for k in ('ground','contract','dataset','previous_source','config','logs')}
        plan.update(seed=args.seed,gpus=gpus,threads=args.threads,source_only=True,backbone='verified same-contract frozen H0',head_epochs=80,base_fit_budget=30,max_fit_budget=40)
        coordinate(plan,root)
    else:
        plan=json.loads((root/'plan.json').read_text(encoding='utf-8'));torch.set_num_threads(plan['threads'])
        {'prepare':prepare,'smoke':smoke}[args.phase](plan,root)


if __name__=='__main__':main()
