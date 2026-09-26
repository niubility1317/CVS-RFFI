"""Reusable comparison entry point: list, prepare, launch, and package.

The versioned matched presets keep scientific data roles fixed. Frequently
changed model seeds, budgets and batch sizes need only short CLI options.
"""
import argparse
import copy
import json
import hashlib
from pathlib import Path
import re
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
METHODS=['cvcnn_ce-ce','cvcnn_ce-pl','riei_fd-ce','riei_fd-pl','drift-ce','drift-pl','poster-ce','radionet-ce']
SOURCE='20260927-phase1-baselines-practical-manysig-m5-r01'


def prepare_evaluation(source_path, run_id):
    source=json.loads(Path(source_path).read_text(encoding='utf-8'))
    if not re.fullmatch(r'[a-zA-Z0-9][a-zA-Z0-9_-]+',run_id):
        raise ValueError('Unsafe evaluation run-id')
    spec=copy.deepcopy(source)
    base=source['execution']['remote_run_root'].rsplit('/runs/',1)[0]
    root=base+'/runs/'+run_id
    spec.update(run_id=run_id,group_id='phase1-final-clean-practical-target',stage='Phase1-final-evaluation',
        display_name='最后一轮权重：clean与三种residual_noeq目标域评估',
        description='冻结源模型的独立最终测试；全矩阵预测固定后评分，测试结果不用于选择或调参。',
        parent_run_ids=[source['run_id']],status='PLANNED',evaluation_seed=2026092709,augmentation_seed=2026092708,
        source_state=source['execution']['remote_run_root']+'/state.json')
    spec.pop('final_evaluation',None)
    spec['code'].update(commit='recorded_at_release',checkout=str(ROOT))
    spec['checkpoint'].update(initialization='registered_source_last.pt',sources=[source['execution']['remote_run_root']],
        selection_rule='exact final epoch per source row; never best_by_val/test',
        provenance_verdict='runtime exact source physical roles + scratch provenance + final epoch')
    spec['permissions'].update(regime='frozen_zero_adaptation',query_use='evaluation-only; prediction then separate truth scorer',claim_scope='Phase1 target clean and residual satellite DG')
    spec['execution'].update(remote_run_root=root,remote_log_root=root,launch_owner='comparison/final-evaluation/'+run_id,
        local_artifact_root='automation_reports/CV-SincNet/'+run_id,gpu_policy='CPU only, sequential predictions,2threads',
        stop_rule='Stop on technical/provenance/data failure; retain outputs; no performance-based retries',
        launch_command=f"{spec['code']['environment']} tools/comparison_final_eval.py launch --spec automation_reports/CV-SincNet/{run_id}/experiment.json --commit RELEASE_COMMIT")
    for row in spec['rows']:
        cfg=json.loads((ROOT/row['config_ref']).read_text(encoding='utf-8'))
        row.update(source_root=row['output_root'],model_method=cfg['method'],output_root=root+'/'+row['row_id'],
            log_path=root+'/'+row['row_id']+'.log',gpu=None,purpose='fixed final target evaluation',
            optimizer=None,lr=None,resolved_config_ref=root+'/'+row['row_id']+'/startup.json',
            expected_artifacts=['startup.json','checkpoint_provenance.json','predictions.npz','predictions_complete.json'])
        row['seeds'].update(data=2026092709,augmentation=2026092708,evaluation=2026092709,support=None)
        row['seed_notes']='Model/split inherited unchanged; fixed target scene allocation/evaluation seed2026092709; channel2026092708; no support/adaptation.'
        row['command']=f"{spec['code']['environment']} tools/comparison_final_eval.py predict --spec automation_reports/CV-SincNet/{run_id}/experiment.json --row {row['row_id']}"
    spec['expected_artifacts']=['data/capsule/manifest.json','state.json','results.json','*/predictions.npz']
    spec['metrics_plan'].update(metric_names=['accuracy','macro_f1','confusion'],dimensions=['method','model_seed','view','receiver'])
    spec['notes']=['Each target physical ID has one clean diagnostic and exactly one fixed residual satellite observation; not a Phase2 capsule.',
        'Scene assignment depends only on opaque physical ID and fixed seed;3satellite scenes are physically disjoint.',
        'Predictor cannot open raw dataset or truth. Source/target receiver and physical-role overlap checked by isolated builder.',
        'All rows must complete predictions before independent scorer opens truth; metrics never trigger reselection/retraining.',
        'Previously optimized392005 and fresh seeds must be reported separately.']
    draft_dir=ROOT/'configs/comparisons'/run_id
    draft_dir.mkdir(parents=True,exist_ok=False)
    draft=draft_dir/'registration.json'
    draft.write_text(json.dumps(spec,ensure_ascii=False,indent=2),encoding='utf-8')
    subprocess.run([sys.executable,str(ROOT/'tools/experiment_registry.py'),'validate',str(draft),'--launch-ready'],cwd=ROOT,check=True)
    subprocess.run([sys.executable,str(ROOT/'tools/experiment_registry.py'),'new','--spec',str(draft)],cwd=ROOT,check=True)
    return 'automation_reports/CV-SincNet/'+run_id+'/experiment.json'


def prepare(args):
    if not re.fullmatch(r'[a-zA-Z0-9][a-zA-Z0-9_-]+', args.run_id):
        raise ValueError('run-id must be a safe unique directory name')
    methods=METHODS if args.methods=='all' else args.methods.split(',')
    if not methods or len(set(methods))!=len(methods) or set(methods)-set(METHODS):
        raise ValueError('Unknown/duplicate method; use list')
    seeds=[int(s) for s in args.seeds.split(',')]
    gpus=[int(s) for s in args.gpus.split(',')]
    if len(seeds)!=len(set(seeds)) or not gpus or len(set(gpus))!=len(gpus) or min(gpus)<0 or args.epochs<1 or args.batch_size<1:
        raise ValueError('Invalid seeds/GPU/budget')
    template=json.loads((ROOT/f'automation_reports/CV-SincNet/{SOURCE}/experiment.json').read_text(encoding='utf-8'))
    cfg_dir=ROOT/'configs/comparisons'/args.run_id
    if cfg_dir.exists() or (ROOT/'automation_reports/CV-SincNet'/args.run_id).exists():
        raise FileExistsError('Experiment already exists; no overwrites or implicit resume')
    cfg_dir.mkdir(parents=True)
    spec=copy.deepcopy(template)
    base=template['execution']['remote_run_root'].rsplit('/runs/',1)[0]
    runroot=base+'/runs/'+args.run_id
    logs=base+'/logs/'+args.run_id
    spec.update(run_id=args.run_id,display_name=args.name or args.run_id,description='Reusable matched comparison source training; fixed final checkpoint and separate clean/satellite evaluation.',parent_run_ids=[],status='PLANNED')
    spec['code'].update(checkout=str(ROOT),commit='recorded_at_release',cwd=base+'/releases/'+args.run_id)
    spec['checkpoint']['selection_rule']=f'fixed final epoch{args.epochs}; last.pt only; no best-by-test selection'
    spec['execution'].update(remote_run_root=runroot,remote_log_root=logs,launch_owner=args.owner,
        local_artifact_root='automation_reports/CV-SincNet/'+args.run_id,
        launch_command=f"{template['code']['environment']} tools/comparison.py launch --spec automation_reports/CV-SincNet/{args.run_id}/experiment.json --commit RELEASE_COMMIT")
    spec['final_evaluation']=dict(enabled=True,checkpoint='last.pt',views=['clean','practical_high','practical_mid','practical_low_urban'],
        separate_process=True,truth_last=True,selection_feedback=False,
        launcher='tools/comparison_final_eval.py',note='Register and queue evaluate --source-spec for the frozen source matrix; training itself never opens targets.')
    spec['final_evaluation']['spec_ref']='automation_reports/CV-SincNet/'+args.run_id+'-final/experiment.json'
    rows=[]
    for index, method in enumerate(methods):
        original=next(r for r in template['rows'] if r['row_id']==method+'-s392005')
        for seed in seeds:
            row=copy.deepcopy(original)
            cfg=json.loads((ROOT/original['config_ref']).read_text(encoding='utf-8'))
            sid=f'{method}-s{seed}'
            output=runroot+'/'+sid
            cfg.update(row_id=sid,gpu=gpus[index%len(gpus)],model_seed=seed)
            cfg['options'].update({'--seed':seed,'--epochs':args.epochs,'--batch_size':args.batch_size,'--output_dir':output})
            cfg['final_evaluation']=spec['final_evaluation']
            ref=f'configs/comparisons/{args.run_id}/{sid}.json'
            (ROOT/ref).write_text(json.dumps(cfg,indent=2),encoding='utf-8')
            row.update(row_id=sid,gpu=cfg['gpu'],config_ref=ref,output_root=output,
                resolved_config_ref=output+'/resolved_config.json',log_path=logs+'/'+sid+'.log',epochs=args.epochs,
                budget_ref=ref,command=f"{spec['code']['environment']} tools/run_practical_baseline.py --config {ref}")
            row['seeds']['model']=seed
            row['purpose']='historical_seed_replication' if seed==392005 else 'predeclared_model_seed'
            rows.append(row)
    spec['rows']=rows
    spec['notes'].append('Final clean/satellite evaluation is mandatory by default, separately registered; no live target validation.')
    draft=cfg_dir/'registration.json'
    draft.write_text(json.dumps(spec,ensure_ascii=False,indent=2),encoding='utf-8')
    subprocess.run([sys.executable,str(ROOT/'tools/experiment_registry.py'),'new','--spec',str(draft)],cwd=ROOT,check=True)
    prepare_evaluation(ROOT/'automation_reports/CV-SincNet'/args.run_id/'experiment.json',args.run_id+'-final')
    print(str(ROOT/'automation_reports/CV-SincNet'/args.run_id/'experiment.json'))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    sub=p.add_subparsers(dest='action',required=True)
    sub.add_parser('list')
    q=sub.add_parser('prepare')
    q.add_argument('--run-id',required=True)
    q.add_argument('--name')
    q.add_argument('--methods',default='all')
    q.add_argument('--seeds',default='392005,2026092701,2026092702,2026092703,2026092704')
    q.add_argument('--epochs',type=int,default=200)
    q.add_argument('--batch-size',type=int,default=128)
    q.add_argument('--gpus',default='0,1,2,3,4,5,6,7')
    q.add_argument('--owner',default='manual/comparison-launcher')
    q=sub.add_parser('launch')
    q.add_argument('--spec',type=Path,required=True)
    q.add_argument('--commit',required=True)
    q=sub.add_parser('evaluate')
    q.add_argument('--source-spec',type=Path,required=True)
    q.add_argument('--run-id',required=True)
    q=sub.add_parser('package')
    q.add_argument('--spec',type=Path,required=True)
    q.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.action=='list':
        print('\n'.join(METHODS))
    elif a.action=='prepare':
        prepare(a)
    elif a.action=='launch':
        spec=json.loads(a.spec.read_text(encoding='utf-8'))
        ref=spec.get('final_evaluation',{}).get('spec_ref')
        if spec.get('final_evaluation',{}).get('enabled',True) and (not ref or not (ROOT/ref).is_file()):
            raise ValueError('Register and package final evaluation before source launch')
        subprocess.run([sys.executable,str(ROOT/'tools/launch_practical_baselines.py'),'--spec',str(a.spec),
                        '--commit',a.commit,'--detach'],cwd=ROOT,check=True)
        if spec.get('final_evaluation',{}).get('enabled',True):
            if not ref:
                raise ValueError('Use evaluate to register final evaluation for this legacy source spec')
            subprocess.run([sys.executable,str(ROOT/'tools/comparison_final_eval.py'),'launch','--spec',ref,
                            '--commit',a.commit],cwd=ROOT,check=True)
    elif a.action=='evaluate':
        print(prepare_evaluation(a.source_spec,a.run_id))
    elif a.action=='package':
        if a.output.exists():
            raise FileExistsError(a.output)
        spec=json.loads(a.spec.read_text(encoding='utf-8'))
        selected=['baselines','code','tools','configs','automation_reports/CV-SincNet/'+spec['run_id'],
                  'automation_reports/CV-SincNet/'+SOURCE]
        final_ref=spec.get('final_evaluation',{}).get('spec_ref')
        if final_ref:
            selected.append(str(Path(final_ref).parent).replace('\\','/'))
        subprocess.run(['git','diff','--exit-code','HEAD','--',*selected],cwd=ROOT,check=True)
        a.output.parent.mkdir(parents=True,exist_ok=True)
        subprocess.run(['git','archive','--format=tar.gz','--output='+str(a.output.resolve()),'HEAD',*selected],cwd=ROOT,check=True)
        manifest=dict(commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
                      sha256=hashlib.sha256(a.output.read_bytes()).hexdigest(),bytes=a.output.stat().st_size)
        a.output.with_suffix('.manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
        print(json.dumps(manifest))


if __name__=='__main__':
    main()
