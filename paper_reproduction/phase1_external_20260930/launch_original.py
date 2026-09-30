"""Prepare pinned-source input layouts and launch the registered source-only pilot.

Run on N607 using its Python; no target scoring, checkpoint reuse or overwrite.
"""
import argparse
import json
import os
import pickle
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

RX = ['1-1','1-19','2-1','2-19','3-19','7-7','7-14','8-8','14-7','18-2','19-2','20-1']
SOURCE = RX[3:]
TARGET = RX[:3]


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')


def prepare(dataset, root):
    if root.exists():
        raise FileExistsError(f'Input export preserved: {root}')
    with open(dataset, 'rb') as handle:
        source = pickle.load(handle)
    assert len(source['tx_list']) == 6 and set(source['rx_list']) == set(RX)
    assert len(source['capture_date_list']) == 4
    raw_index = source['equalized_list'].index(0)
    eq_index = source['equalized_list'].index(1)
    root.mkdir(parents=True)
    selected = np.random.RandomState(300).choice(1000, size=200, replace=False)
    ids = {'schema':'wisig-original-physical-rows-v1','dataset':str(dataset),
           'tx_list':source['tx_list'],'rx_author_order':RX,'source':SOURCE,'target':TARGET,
           'capture_date_list':source['capture_date_list'],
           'physical_id_definition':'ManySig:TX:RX:date:packet_row; equalization is a view of the same row',
           'asknet_wavemlp':{'representation':'non_equalized','packet_rows':list(range(100)),
                            'source_days':[1,2],'validation_fraction':0.3,'split_seed':2023},
           'difl':{'representation':'equalized','packet_rows':selected.tolist(),
                   'source_days':[1],'validation_fraction':0.2,'split_seed':32,'selection_seed':300},
           'source_training_only':True,'target_scoring_authorized_in_this_launcher':False}
    ask_ids = []
    difl_ids = []
    for rx in RX:
        r = source['rx_list'].index(rx)
        for day in range(4):
            packets = [np.asarray(source['data'][t][r][day][raw_index][:100], dtype=np.float32) for t in range(6)]
            for block in packets:
                assert block.shape == (100,256,2) and np.isfinite(block).all()
                assert np.all(np.sum(block**2, axis=(1,2)) > 0)
            path = root/'author_layout/ManySig/non_equalized'/f'date{day+1}'/f'rx_{rx}_data.pkl'
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open('wb') as handle:
                pickle.dump({'data':packets},handle, protocol=4)
            if rx in SOURCE and day in (0,1):
                ask_ids.extend([f'ManySig:{tx}:{rx}:{day+1}:{i}' for tx in source['tx_list'] for i in range(100)])
        eq = np.concatenate([source['data'][t][r][0][eq_index][selected] for t in range(6)]).astype(np.float32)
        assert eq.shape == (1200,256,2) and np.isfinite(eq).all()
        (root/'difl').mkdir(exist_ok=True)
        np.save(root/'difl'/f'rx_{rx}.npy',eq)
        if rx in SOURCE:
            rows = np.asarray([f'ManySig:{tx}:{rx}:1:{i}' for tx in source['tx_list'] for i in selected])
            train, val = train_test_split(rows, test_size=0.2, random_state=32, shuffle=True)
            assert set(train).isdisjoint(val)
            difl_ids.append({'receiver':rx,'train':train.tolist(),'validation':val.tolist()})
    np.save(root/'difl/labels.npy', np.repeat(np.arange(1,7),200))
    # Matches native RX/day/TX concatenation order, not the compact-pickle RX order.
    train, val = train_test_split(np.asarray(ask_ids),test_size=0.3,random_state=2023)
    assert set(train).isdisjoint(val)
    ids['asknet_wavemlp']['train_ids'] = train.tolist()
    ids['asknet_wavemlp']['validation_ids'] = val.tolist()
    ids['difl']['receiver_splits'] = difl_ids
    write_json(root/'physical_roles.json',ids)
    print(json.dumps({'prepared':str(root),'ASK_Wave_train':len(train),'ASK_Wave_validation':len(val),
                      'DIFL_source_receivers':len(SOURCE),'source_target_disjoint':set(SOURCE).isdisjoint(TARGET)}),flush=True)


def commands(release, data, run, logs):
    py = sys.executable
    shared = ['--dataset_name','ManySig','--exp','CRD','--train_date','1','2','--test_round','0',
              '--seed','2023','--epochs','200','--batch_size','128','--code_state','only_train',
              '--data-root',str(data/'author_layout')]
    rows = {}
    for method,gpu in [('asknet',0),('wavemlp',1)]:
        cwd = release/'sources'/method/('WiSig' if method=='asknet' else '')
        argv = [py,'-u','main.py',*shared,'--gpu',str(gpu),'--output-root',str(run/method),'--log-root',str(logs/method)]
        rows[method] = {'cwd':str(cwd),'gpu':gpu,'commands':[argv]}
    files = [str(data/'difl'/f'rx_{rx}.npy') for rx in SOURCE]
    common = ['--source-files',*files,'--label-path',str(data/'difl/labels.npy'),
              '--epochs','100','--batch-size','32','--lr','0.001','--seed','300','--split-seed','32','--device','cuda:0']
    teacher = [py,'-u','pretrain.py',*common,'--output-dir',str(run/'difl/teacher'),'--log-dir',str(logs/'difl/teacher')]
    student = [py,'-u','train_DIFEX1.py',*common,'--output-dir',str(run/'difl/student'),'--log-dir',str(logs/'difl/student'),
               '--code-state','only_train','--teacher-checkpoint',str(run/'difl/teacher/pretrain.pth'),'--lam','0.05','--beta','0.01']
    rows['difl'] = {'cwd':str(release/'sources/difl'),'gpu':2,'commands':[teacher,student]}
    return rows


def worker(plan, method):
    payload = json.loads(plan.read_text(encoding='utf-8'))
    row = payload['rows'][method]
    env = dict(os.environ,OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',PYTHONUNBUFFERED='1')
    if method == 'difl':
        env['CUDA_VISIBLE_DEVICES'] = str(row['gpu'])
    # Native ASK/Wave CLI owns its --gpu setting; do not pre-mask its ordinal.
    for index, argv in enumerate(row['commands']):
        print(json.dumps({'method':method,'step':index,'argv':argv,'cwd':row['cwd']},ensure_ascii=False),flush=True)
        child = subprocess.Popen(argv,cwd=row['cwd'],env=env)
        write_json(Path(payload['run_root'])/method/'active_process.json',
                   {'pid':child.pid,'worker_pid':os.getpid(),'step':index,'argv':argv,'cwd':row['cwd'],'gpu':row['gpu']})
        result = child.wait()
        write_json(Path(payload['run_root'])/method/f'step_{index}_exit.json',{'exit_code':result,'finished_at':time.time()})
        if result:
            raise SystemExit(result)
    write_json(Path(payload['run_root'])/method/'source_training_complete.json',
               {'status':'SOURCE_TRAINING_COMPLETE_AWAITING_SOURCE_REVIEW','target_scored':False,'finished_at':time.time()})


def start(release, data, run, logs):
    if run.exists() or logs.exists():
        raise FileExistsError('Run/log output already exists; reconcile instead of launching twice')
    if not (data/'physical_roles.json').exists():
        raise FileNotFoundError('Prepared physical roles missing')
    run.mkdir(parents=True)
    logs.mkdir(parents=True)
    plan = run/'launch_plan.json'
    payload = {'run_root':str(run),'log_root':str(logs),'launch_owner':'root-phase1-three-baselines-20260930',
               'rows':commands(release,data,run,logs),'source_only':True}
    write_json(plan,payload)
    workers = {}
    for method in payload['rows']:
        path = logs/f'{method}.stdout.log'
        with path.open('xb') as handle:
            proc = subprocess.Popen([sys.executable,'-u',str(Path(__file__).resolve()),'worker','--plan',str(plan),'--method',method],
                                    stdin=subprocess.DEVNULL,stdout=handle,stderr=subprocess.STDOUT,start_new_session=True,cwd=release)
        workers[method] = {'worker_pid':proc.pid,'stdout':str(path),'gpu':payload['rows'][method]['gpu']}
    write_json(run/'workers.json',workers)
    print(json.dumps(workers,indent=2),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['prepare','start','worker','plan'])
    p.add_argument('--dataset',type=Path)
    p.add_argument('--data-root',type=Path)
    p.add_argument('--release',type=Path)
    p.add_argument('--run-root',type=Path)
    p.add_argument('--log-root',type=Path)
    p.add_argument('--plan',type=Path)
    p.add_argument('--method',choices=['asknet','wavemlp','difl'])
    args=p.parse_args()
    if args.action=='prepare': prepare(args.dataset,args.data_root)
    elif args.action=='start': start(args.release,args.data_root,args.run_root,args.log_root)
    elif args.action=='worker': worker(args.plan,args.method)
    else: print(json.dumps(commands(args.release,args.data_root,args.run_root,args.log_root),indent=2))


if __name__=='__main__': main()
