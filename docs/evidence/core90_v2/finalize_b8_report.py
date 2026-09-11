"""Validate the complete fixed-budget benchmark and update its evidence report."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
data = json.loads((ROOT / 'b8_benchmark_cuda_deterministic.json').read_text(encoding='utf-8'))
rows = data['rows']
assert 'failure' not in data, data.get('failure')
assert len(rows) == 9
assert {(r['stage'], r['implementation']) for r in rows} == {
    (s, i) for s in (1, 80, 131) for i in ('reference', 'head_grad_only', 'graph_reuse')
}
assert data['source_kind'] == 'REAL_SOURCE_FIXED_CONTEXT'
assert data['checkpoint'] == 'scratch_only'
assert data['budgets']['warmup'] == 20 and data['budgets']['steps'] == 100
assert data['budgets']['repeats'] == 3 and data['budgets']['batch_size'] == 90
assert data['budgets']['telemetry_interval'] == 0
assert data['determinism']['algorithms'] and data['determinism']['cudnn_deterministic']
assert not data['determinism']['cudnn_benchmark']
assert data['determinism']['cublas_workspace_config'] == ':4096:8'
for row in rows:
    assert row['equivalence'] == 'PASSED_ATOL_1e-6_RTOL_1e-5'
    assert row['timed_endpoint_equivalence'] == (
        'REFERENCE_BASELINE' if row['implementation'] == 'reference' else 'PASSED_ATOL_1e-6_RTOL_1e-5'
    )
    assert len(row['seconds_per_step']) == len(row['peaks']) == len(row['counts']) == 3
    assert row['origin_terms']['fishr'] > 0
    for count in row['counts']:
        expected_fwd = (2 if row['stage'] == 131 else 1) * (1 if row['implementation'] == 'graph_reuse' else 2)
        assert count['model_forward'] == expected_fwd * 100
        assert count['field_backward'] == 200 and count['telemetry_backward'] == 0

table = [
    '|阶段|实现|秒/步中位数|3次范围|allocated峰值MiB|增量峰值MiB|reserved峰值MiB|模型前向/步|头前向/步|',
    '|---|---|---:|---|---:|---:|---:|---:|---:|',
]
for row in rows:
    peak = lambda key: max(p[key] for p in row['peaks']) / 2**20
    table.append(
        f"|E{row['stage']}|{row['implementation']}|{row['median']:.6f}|"
        f"{row['range'][0]:.6f}–{row['range'][1]:.6f}|{peak('allocated'):.3f}|"
        f"{peak('incremental_allocated'):.3f}|{peak('reserved'):.3f}|"
        f"{row['counts'][0]['model_forward']/100:g}|{row['counts'][0]['head_forward']/100:g}|"
    )
comparisons = []
for stage in (1, 80, 131):
    selected = {r['implementation']: r for r in rows if r['stage'] == stage}
    reference = selected['reference']['median']
    comparisons.append(
        f"E{stage}：D1观察耗时下降{100*(1-selected['head_grad_only']['median']/reference):.2f}%，"
        f"graph观察耗时下降{100*(1-selected['graph_reuse']['median']/reference):.2f}%。"
    )
summary = '''# B8 V2本地实现与验证

最终状态：VERIFIED。确定性CUDA真实source固定上下文benchmark已完成E1/E80/E131×reference/D1/graph全部9组，保留每组20步warmup＋100步计时×3次原预算（2700个计时更新，含warmup共3240个更新）。每阶段计时前单步等价检查通过；D1与graph共18个120步重复端点均通过原atol=1e-6、rtol=1e-5的模型、optimizer、RNG、mask、prototype全状态比较。端点检查在计时区域外进行。该结论不替代正式训练、跨数据泛化或长期收敛验收。

支持范围：本轮CORE90数值与成本验收为FP32，scratch-only、合法source。graph_reuse与启用GradScaler或CPU/CUDA autocast组合显式拒绝，不静默采用未经验证的AMP路径；拒绝检查在benchmark启动后补入，FP32数值路径不变。本轮也不声称D1完整CORE90 AMP数值等价已验收。未访问target、远端或旧checkpoint，未运行正式训练。

## 原预算CUDA成本

设备为RTX5070Ti，PyTorch2.10.0+cu128；启用deterministic algorithms、cudnn.deterministic，禁用cudnn.benchmark，CUBLAS_WORKSPACE_CONFIG=:4096:8；matmul TF32关闭、cudnn TF32开启。batch=90，同一合法WiSig source batch、阶段、scratch初态和RNG，每个repeat重置。source角色计数L_s=6300、U_s=56700、V=27000。以下为CUDA同步后的每步墙钟中位数及全部3次范围，telemetry_interval=0。

'''+ '\n'.join(table) + '''

''' + '\n\n'.join(comparisons) + '''

墙钟归因保持UNKNOWN_SHARED_LOAD：设备运行桌面和其他应用，重复顺序固定，期间有未计时CPU检查；观察耗时差不能直接归因为纯算法提速。allocated为本进程PyTorch分配峰值，incremental扣除相应warmup后的baseline；reserved受跨组缓存历史影响，不能等同活跃tensor需求或全系统GPU显存。原始JSON保留每个repeat的baseline、峰值、计数与耗时，未用CPU smoke代替本表。

每步field_evaluations均为2。独立CPU API拦截另证实9种阶段/实现组合各调用torch.autograd.grad两次，Tensor.backward零次、create_graph=True零次、显式额外HVP零次；reference目标参数张量数185→185，D1/graph为4→185。FISHR沿用解析logit-gradient-variance proxy，未在objective内部另调用autograd.grad。这里统计API调用，不能解释成引擎节点数或CUDA backward kernel数；详见[b8_autograd_call_counts.md](b8_autograd_call_counts.md)。graph减少完整模型前向，仍保留完整目标及两次场求导。

所有9组FISHR为正；自然伪标签选择为0且prototype空状态，不能用该benchmark声称活跃U/prototype的实际成本。E131保留强视图前向路径；有效mask与非零prototype数值覆盖来自独立受控单测。E1/E80/E131是同初态的固定阶段上下文，不是连续训练轨迹。低频分项telemetry额外reverse成本单列，主表未启用；CPU telemetry smoke只证明计量路径可运行，不给出稳定GPU遥测开销估计。

非确定性原始两行结果保留为PARTIAL，不并入本表。同实现负对照首次出现后端数值漂移，具体算子UNKNOWN；确定性ref/ref、ref/D1、ref/graph在E1和E131各8步全状态逐位检查通过，随后完成本表原预算。详细首差和停止证据见[b8_cuda_determinism_diagnosis.md](b8_cuda_determinism_diagnosis.md)。

正式证据：[完整9组JSON](b8_benchmark_cuda_deterministic.json)、[逐步确定性诊断](b8_cuda_determinism_diagnosis.md)、[独立API成本](b8_autograd_call_counts.json)。

```text
C:/Users/lh594/.conda/envs/ssr-gpu/python.exe -X utf8 code/scripts/benchmark_core90_b8.py --device cuda --wisig-pkl E:/type10-7/local_artifacts/cvs_publication_inputs_20260713/ManySig.pkl --implementation all --stage all --warmup 20 --steps 100 --repeats 3 --batch-size 90 --output local_artifacts/core90_v2/b8_benchmark_cuda_deterministic.json
```

## 实现与原始CPU功能验证记录

下面保留实现说明和既有CPU验证，CUDA最终状态与成本以上文为准。

'''
report_path = ROOT / 'b8_report.md'
old = report_path.read_text(encoding='utf-8')
start = old.index('实现文件：')
end = old.index('真实预算命令（尚未由本子任务执行）：') if '真实预算命令（尚未由本子任务执行）：' in old else len(old)
report_path.write_text(summary + old[start:end], encoding='utf-8')
readback = report_path.read_text(encoding='utf-8')
assert readback.startswith('# B8 V2本地实现与验证') and '全部9组' in readback
assert '\ufffd' not in readback
print(json.dumps({'status': 'VERIFIED', 'groups': len(rows), 'timed_updates': 2700,
                  'total_updates_including_warmup': 3240, 'report': str(report_path),
                  'observed_comparisons': comparisons}, ensure_ascii=False, indent=2))
