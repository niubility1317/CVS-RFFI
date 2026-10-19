"""Frozen-model inference costs, queued strictly after the comparison matrix."""
import argparse
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

from .data import read, write
from .driver import atomic, process_alive
from .plan import SEEDS, make_plan

COST_RUN = '20261019-phase1-external-dg-concat-pl-cost-m96-r03'
BATCHES = (1, 128)
WARMUP = 100
REPEATS = 1000


def tasks(suite=None):
    suite = suite or make_plan()
    selected = []
    for row in suite['rows']:
        if row['group'] not in ('F0','F0-pl','ManyRx','generic','strongce','strongce-pl'):
            continue
        version = row.get('route', row['variant'])
        if row['family'].startswith('resnet'):
            version = row['group'] + ':' + version
        selected.append(dict(cost_id=row['row_id'],origin=row['training_regime'],
            row_id=row['row_id'],family=row['family'],variant=version,
            model_seed=row['model_seed'],protocol_id=row['protocol_id'],
            num_classes=10 if row['protocol_id']=='ManyRx' else 6))
    if len(selected)!=96 or len({r['cost_id'] for r in selected})!=96:
        raise ValueError('Expected exactly 96 frozen cost rows')
    return selected

def policy(rt):
    if (rt['allowed_gpus'] != [2, 3] or rt['batches'] != list(BATCHES) or
        rt['warmup'] != WARMUP or rt['repeats'] != REPEATS):
        raise ValueError('Unregistered resource or measurement budget')
    forbidden = ('truth', 'target_truth', 'dataset', 'datasets', 'views_root', 'target_inputs')
    if any(rt.get(k) for k in forbidden):
        raise ValueError('Cost process rejects target or raw dataset inputs')


def load_task(rt, task, device):
    import torch
    from . import source
    handles=read(Path(rt['suite_run_root'])/'data_handles.json')
    expected=read(handles[task['protocol_id']]['source_contract'])
    model,meta=source.frozen_model(str(Path(rt['suite_run_root'])/task['row_id']/'source'),expected,device)
    if (meta['row_id']!=task['row_id'] or meta['model_seed']!=task['model_seed'] or
        meta['satellite_train_augmentation'] is not True or
        meta['default_head_joint_training'] is not True or meta['separate_head_training'] is not False):
        raise ValueError('Wrong frozen concat/PL checkpoint')
    forward=lambda x:source.forward(model,meta,x)
    model.eval()
    with torch.no_grad(): output=forward(torch.zeros(1,2,256,device=device))
    if output.shape!=(1,task['num_classes']) or not torch.isfinite(output).all():
        raise ValueError('Default-head inference differs')
    return model,forward

def parameter_cost(model, family):
    modules = [model.fed, model.ec] if family == 'riei' else (
        [model.encoder, model.tx_classifier] if family == 'drift' else [model])
    parameters = {id(p): p for module in modules for p in module.parameters()}
    buffers = {id(p): p for module in modules for p in module.buffers()}
    return dict(registered_parameters=sum(p.numel() for p in model.parameters()),
        identity_parameters=sum(p.numel() for p in parameters.values()),
        identity_parameter_bytes=sum(p.numel() * p.element_size() for p in parameters.values()),
        identity_buffer_bytes=sum(p.numel() * p.element_size() for p in buffers.values()))


def mac_profile(model, forward, x):
    """Conv dot products plus profiler mm/bmm/addmm; excludes elementwise work."""
    import torch
    convolutions = []; hooks = []
    def count(module, args, output):
        convolutions.append(output.numel() * module.weight[0].numel())
    try:
        for module in model.modules():
            if isinstance(module, (torch.nn.Conv1d, torch.nn.Conv2d, torch.nn.Conv3d)):
                hooks.append(module.register_forward_hook(count))
        with torch.no_grad(), torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU],
                record_shapes=True, with_flops=True) as profiler:
            forward(x)
        matrix_flops = {event.key: int(event.flops or 0) for event in profiler.key_averages()
            if event.key in ('aten::mm', 'aten::bmm', 'aten::addmm')}
    finally:
        for hook in hooks:
            hook.remove()
    total = sum(convolutions) + sum(matrix_flops.values()) / 2
    return dict(batch=int(x.shape[0]), conv_macs=sum(convolutions),
        matrix_flops=matrix_flops, macs=total, twice_macs=2 * total,
        definition='Conv1d/2d/3d dot products + mm/bmm/addmm FLOPs/2; excludes BN, activation, pooling, softmax and other elementwise work')


def gpu_pids(gpu):
    value = subprocess.check_output(['nvidia-smi', '-i', str(gpu),
        '--query-compute-apps=pid', '--format=csv,noheader,nounits'], text=True)
    return {int(x.strip()) for x in value.splitlines() if x.strip()}


def environment(rt, gpu=''):
    return dict(os.environ, PYTHONPATH=os.pathsep.join([rt['release_root'], rt['numpy_deps']]),
        CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2')


def worker(c):
    import torch
    rt, task = c['runtime'], c['task']; policy(rt)
    if task not in tasks():
        raise ValueError('Unregistered cost row')
    root = Path(rt['run_root']) / task['cost_id']; root.mkdir(parents=True, exist_ok=False)
    gpu = c['physical_gpu']
    if gpu not in (2, 3) or os.environ.get('CUDA_VISIBLE_DEVICES') != str(gpu):
        raise ValueError('Unexpected physical GPU')
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    device = torch.device('cuda:0')
    model, forward = load_task(rt, task, device)
    original = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    parameters = parameter_cost(model, task['family'])
    macs = mac_profile(model, forward, torch.zeros(1, 2, 256, device=device))
    measurements = []
    with torch.no_grad():
        for batch in BATCHES:
            x = torch.zeros(batch, 2, 256, device=device, dtype=torch.float32)
            for _ in range(WARMUP):
                forward(x)
            torch.cuda.synchronize()
            if gpu_pids(gpu) - {os.getpid()}:
                raise RuntimeError('TIMING_CONTENDED_BEFORE_MEASUREMENT')
            torch.cuda.reset_peak_memory_stats()
            values = []
            for index in range(REPEATS):
                torch.cuda.synchronize(); started = time.perf_counter_ns()
                output = forward(x)
                torch.cuda.synchronize(); values.append((time.perf_counter_ns() - started) / 1e6)
            if gpu_pids(gpu) - {os.getpid()}:
                raise RuntimeError('TIMING_CONTENDED_AFTER_MEASUREMENT')
            if output.shape != (batch, task['num_classes']) or not torch.isfinite(output).all():
                raise ValueError('Nonfinite or incompatible inference output')
            ordered = sorted(values); mean = statistics.mean(values)
            measurements.append(dict(batch=batch, warmup=WARMUP, repeats=REPEATS,
                latency_ms=values, median_ms=statistics.median(values),
                p95_ms=ordered[int(.95 * (len(ordered) - 1))], mean_ms=mean,
                examples_per_second=batch * 1000 / mean,
                peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                includes='synchronized model call with resident synthetic IQ; excludes data transfer and preprocessing'))
    if any(not torch.equal(original[k], v.detach().cpu()) for k, v in model.state_dict().items()):
        raise ValueError('Frozen parameter or buffer changed during cost measurement')
    result = dict(status='ANALYZED', task=task, physical_gpu=gpu, torch=torch.__version__,
        cuda=torch.version.cuda, cudnn=torch.backends.cudnn.version(),
        device_name=torch.cuda.get_device_name(0), precision='FP32, TF32 disabled',
        query_access=False, parameter_or_buffer_updates=False,
        parameter_cost=parameters, mac_profile=macs, measurements=measurements)
    write(root / 'result.json', result)
    print('COST_ANALYZED', task['cost_id'], flush=True)


def smoke(rt):
    import torch
    policy(rt); torch.set_num_threads(2)
    task = next(t for t in tasks()
        if t['origin'] == 'supervised' and t['protocol_id'] == 'F0' and t['family'] == 'ibrff' and t['variant'] == 'full'
        and t['model_seed'] == SEEDS[0])
    model, forward = load_task(rt, task, torch.device('cpu'))
    profile = mac_profile(model, forward, torch.zeros(1, 2, 256))
    if profile['macs'] <= 0:
        raise ValueError('No MACs measured')
    write(Path(rt['run_root']) / 'smoke.json', dict(status='PASS', task=task,
        mac_profile=profile, query_access=False, trained=False))


def dependency(rt):
    if process_alive(rt['suite_pid'],rt['suite_release_root']): return 'LIVE'
    path=Path(rt['suite_run_root'])/'completion.json'
    if not path.exists(): return 'FAILED'
    value=read(path)
    return 'DONE' if (value['status']=='ANALYZED' and value['complete'] is True and
        value['new_training']==219 and value['supervised_training']==192 and
        value['pl_training']==27 and value['frozen_eval']==0) else 'FAILED'

def summarize(results):
    groups = {}
    for result in results:
        task = result['task']
        for measure in result['measurements']:
            key = (task['origin'], task['protocol_id'], task['family'], task['variant'], measure['batch'])
            groups.setdefault(key, []).append((task['model_seed'], measure, result))
    records = []
    for key, values in sorted(groups.items()):
        if sorted(x[0] for x in values) != list(SEEDS):
            raise ValueError('Cost group does not contain exactly three model seeds')
        item = dict(zip(('origin', 'protocol_id', 'family', 'variant', 'batch'), key))
        item.update(seeds=[x[0] for x in values], actual_seeds=3)
        for field in ('median_ms', 'p95_ms', 'examples_per_second', 'peak_allocated_bytes'):
            data = [x[1][field] for x in values]
            item[field + '_mean'] = statistics.mean(data)
            item[field + '_sample_sd'] = statistics.stdev(data)
        counts = {x[2]['parameter_cost']['identity_parameters'] for x in values}
        macs = {x[2]['mac_profile']['macs'] for x in values}
        if len(counts) != 1 or len(macs) != 1:
            raise ValueError('Cost graph differs across model seeds')
        item.update(identity_parameters=counts.pop(), batch1_macs=macs.pop())
        records.append(item)
    return records


def dispatch(rt):
    policy(rt); root = Path(rt['run_root']); root.mkdir(parents=True, exist_ok=False)
    logs = Path(rt['log_root']); logs.mkdir(parents=True, exist_ok=False)
    state = dict(status='STARTING', pid=os.getpid(), cwd=os.getcwd(), rows={},
        allowed_gpus=[2, 3], query_access=False)
    atomic(root / 'state.json', state)
    while True:
        verdict = dependency(rt)
        if verdict == 'DONE':
            break
        state['status'] = 'WAITING_MATRIX' if verdict == 'LIVE' else 'FAILED_DEPENDENCY'
        atomic(root / 'state.json', state)
        if verdict == 'FAILED':
            return
        time.sleep(10)
    smoke(rt)
    selected = tasks(); gpu = None
    for task in selected:
        while gpu is None or gpu_pids(gpu):
            if gpu is None:
                gpu = next((g for g in (2, 3) if not gpu_pids(g)), None)
                if gpu is not None:
                    break
            state.update(status='WAITING_IDLE_GPU', physical_gpu=gpu)
            atomic(root / 'state.json', state); time.sleep(10)
        configuration = dict(runtime=rt, task=task, physical_gpu=gpu)
        path = root / (task['cost_id'] + '.json'); write(path, configuration)
        logpath = logs / (task['cost_id'] + '.log')
        command = [rt['python'], '-u', '-m', 'experiments.phase1_external_dg_aug.cost',
            'worker', '--config', str(path)]
        with logpath.open('x') as log:
            child = subprocess.Popen(command, cwd=rt['release_root'], env=environment(rt, gpu),
                stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
            state['rows'][task['cost_id']] = dict(status='RUNNING', pid=child.pid,
                gpu=gpu, cwd=rt['release_root'], argv=command, log=str(logpath))
            state['status'] = 'RUNNING'; atomic(root / 'state.json', state)
            code = child.wait()
        result = root / task['cost_id'] / 'result.json'
        if code or not result.exists() or read(result)['status'] != 'ANALYZED':
            state['rows'][task['cost_id']]['status'] = 'FAILED'
            state['status'] = 'PARTIAL'; atomic(root / 'state.json', state)
            return
        state['rows'][task['cost_id']]['status'] = 'ANALYZED'
        atomic(root / 'state.json', state)
    results = [read(root / t['cost_id'] / 'result.json') for t in selected]
    summary = summarize(results)
    write(root / 'summary.json', dict(status='ANALYZED', actual_rows=len(results),
        physical_gpu=gpu, summary=summary, results=results, target_access=False))
    lines = ['# 冻结模型统一推理成本', '',
        '96个既有checkpoint，32个配置×三种子；同一物理GPU、FP32、TF32关闭，batch1和128各预热100、同步计时1000次。',
        '', '| 协议 | 方法 | 版本 | batch | 身份参数 | batch1 MACs | 延迟中位数均值±SD（ms） | 各seed P95均值（ms） | 吞吐均值（包/s） |',
        '|---|---|---|---:|---:|---:|---:|---:|---:|']
    for item in summary:
        lines.append('| %s/%s | %s | %s | %d | %d | %.0f | %.4f±%.4f | %.4f | %.2f |' % (
            item['origin'], item['protocol_id'], item['family'], item['variant'], item['batch'],
            item['identity_parameters'], item['batch1_macs'], item['median_ms_mean'],
            item['median_ms_sample_sd'], item['p95_ms_mean'], item['examples_per_second_mean']))
    lines += ['', '延迟仅含驻留GPU的合成2×256 IQ模型调用，不含数据传输及预处理；三seed的计时重复不视为独立训练试验。',
        'MAC口径为Conv点积及mm/bmm/addmm的FLOPs/2，不含BN、激活、池化、softmax和其他逐元素工作；twice_macs仅为该乘加口径的两倍。',
        '峰值显存包含所加载模型中已注册的训练辅助参数；身份推理参数另排除不执行的辅助头。原始计时、环境及显存见summary.json。',
        '不读取目标数据，不重新拟合分类头，不改变任何识别结果；全部原实验性能及负结果保持。']
    (root / 'report.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    state['status'] = 'ANALYZED'; atomic(root / 'state.json', state)
    write(root / 'completion.json', dict(status='ANALYZED', rows=96, complete=True,
        upstream_training_total=219, upstream_frozen_eval=0, additional_training=0))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('action', choices=('dispatch', 'worker'))
    parser.add_argument('--config', required=True); args = parser.parse_args()
    globals()[args.action](read(args.config))
