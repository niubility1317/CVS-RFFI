"""Materialize the final acceptance tables only from completed local evidence."""
from pathlib import Path
import csv
import json
import shutil

ROOT=Path(__file__).resolve().parents[2]
ART=ROOT/'local_artifacts/core90_v2'
DOC=ROOT/'docs/CORE90_GAME_V2_IMPLEMENTATION_ACCEPTANCE_20260911.md'
EVIDENCE=ROOT/'docs/evidence/core90_v2'


def main():
    report=json.loads((ART/'b8_benchmark_cuda_deterministic.json').read_text(encoding='utf-8'))
    assert 'failure' not in report, 'benchmark has a recorded failure'
    assert (report['budgets']['warmup'],report['budgets']['steps'],report['budgets']['repeats'])==(20,100,3)
    rows=report['rows']
    assert {(r['stage'],r['implementation']) for r in rows}=={
        (e,impl) for e in (1,80,131) for impl in ('reference','head_grad_only','graph_reuse')}
    assert len(rows)==9
    for row in rows:
        assert len(row['seconds_per_step'])==len(row['counts'])==len(row['peaks'])==3
        assert row['equivalence']=='PASSED_ATOL_1e-6_RTOL_1e-5'
        assert row['timed_endpoint_equivalence']==('REFERENCE_BASELINE' if row['implementation']=='reference'
                else 'PASSED_ATOL_1e-6_RTOL_1e-5')
        assert all(c['field_backward']==200 for c in row['counts'])
        assert all(c['model_forward']==(100 if row['implementation']=='graph_reuse' else 200)*
                   (2 if row['stage']==131 else 1) for c in row['counts'])
    EVIDENCE.mkdir(parents=True,exist_ok=True)
    # Preserve both unsuccessful and successful development records; never copy weights.
    copied=[]
    for source in sorted(ART.rglob('*')):
        if not source.is_file() or source.suffix not in ('.json','.md','.xml','.py'):
            continue
        if source.name=='code_delivery_paths.json':continue
        target=EVIDENCE/source.relative_to(ART)
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(source,target)
        assert source.read_bytes()==target.read_bytes()
        copied.append(target.relative_to(ROOT).as_posix())
    selected=ROOT/'configs/core90_game_v2_selected_probe_20260911.json'
    shutil.copyfile(ART/'source_probe_development/selected_probe_config.json',selected)
    lines=['## B8最终有界验收与成本','',
        f"确定性CUDA后端下，E1与E131的reference/reference、reference/head_grad_only、reference/graph_reuse各8步全部逐位一致。下表使用同一真实source固定batch={report['budgets']['batch_size']}，按原预算完成每组预热20步、计时100步×3；各repeat的120步终点模型、optimizer、RNG、mask及prototype在计时区域外按预定atol=1e-6、rtol=1e-5核对通过。没有失败后放宽容差。",
        '', '|阶段|实现|中位秒/步|范围|峰值allocated MiB|峰值reserved MiB|每100步模型forward|求解器场backward|',
        '|---|---|---|---|---|---|---|---|']
    cost=[]
    for r in rows:
        peak=max(p['allocated'] for p in r['peaks'])/2**20
        reserved=max(p['reserved'] for p in r['peaks'])/2**20
        counts=r['counts'][0]
        lines.append(f"|E{r['stage']}|{r['implementation']}|{r['median']:.6f}|{r['range'][0]:.6f}—{r['range'][1]:.6f}|{peak:.2f}|{reserved:.2f}|{counts['model_forward']}|{counts['field_backward']}|")
        cost.append(dict(stage=r['stage'],implementation=r['implementation'],median_seconds=r['median'],
            minimum_seconds=r['range'][0],maximum_seconds=r['range'][1],peak_allocated_mib=peak,peak_reserved_mib=reserved,
            model_forwards_per_100_steps=counts['model_forward'],solver_field_backward_per_100_steps=counts['field_backward']))
    with (EVIDENCE/'benchmark_summary.csv').open('w',encoding='utf-8',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(cost[0]));writer.writeheader();writer.writerows(cost)
    lines.extend(['',
        'GPU仍有桌面共享负载，纯算法墙钟加速归因为UNKNOWN；表中为实际测得的分布，不承诺固定加速百分比。模型forward包含E131的U强视图，但计时中自然伪标签选择为0且prototype为空；非零有效U/mask及prototype覆盖来自独立功能测试，不能由该计时推断所有分支已激活。独立CPU真实source的9组API调用审计每步autograd.grad=2、create_graph=True=0、额外显式HVP=0；当前既有FISHR实际为解析logit-gradient proxy，没有内部autograd.grad调用。API次数不等于底层kernel数量。benchmark未启用低频遥测，遥测功能另有验证。',
        '', '[完整确定性计时与终点核对](evidence/core90_v2/b8_benchmark_cuda_deterministic.json)；[最初非确定性首差定位](evidence/core90_v2/b8_cuda_determinism_diagnosis.md)；[CPU及B8实现验证](evidence/core90_v2/b8_report.md)。',
        '', '## 最终设计追踪', '', '|ID|本轮状态|实际闭合范围|', '|---|---|---|'])
    descriptions=[
        'scratch、source契约及版本化恢复；没有目标推理或新正式训练',
        '恢复失败/预算不足与可靠低gap分离，缺失不填0',
        '360条固定训练目标经验gap，完整head模式/尺度回放；外推范围受限',
        '跨TX交换两折、15域/5RX的linear/MLP读出独立记录',
        '90容器方向参考及H1两侧45容器，拒绝前32条截断',
        '完整fit轨迹、尺度、优化器冷重置、固定终点及停止原因',
        '坏lag或方向迁移失败不授权CORRECT，不进入健康稀疏化',
        '独立clock/定标及一次观测一次事件，恢复与拒绝记账',
        '合成阶段/mask边界和真实source14对无动作完整状态一致',
        '关闭全部head贡献后的零对抗退化与非头状态检查',
        'CPU与确定性CUDA的reference/D1/graph有界等价及终点核对',
        '裁剪、AdamW位移、角色遥测及完整固定预算计时；共享墙钟归因未知',
        '连续实际policy、真实mask/seed/计数与仅有效变化重置',
        'fit-only同head三视图身份风险、独立课程clock和滞回',
        '实际曝光文件生成/消费/信道契约检查；真实donor实验尚未执行',
        '18行因子配置、3个强普通source候选及一次source选择工具，均未派发',
        '独立donor、state_checked请求/接受/拒绝与预算；真实严格对照待donor',
        '固定预测先闭合后truth的TX/RX弱项、混淆和条件鲁棒性工具',
        'B4/B7新增修复训练暂缓；缺历史统计不能制造根因',
        'H1/J1复杂度扩展暂缓，保留共同测量修复与条件近似',
        '本地集成与Git交付；远端新release、E200和新固定评分不在本轮执行范围']
    for i,desc in enumerate(descriptions,1):
        lines.append(f"|R{i:02d}|{'deferred' if i in (19,20) else 'verified_local'}|{desc}|")
    lines.extend(['', '计数：verified_local=19、deferred=2、rejected=0、blocked=0。verified_local只表示本轮实现与有界验收范围；不包含表中明确尚未运行的正式实验。', '',
        '193个唯一测试用例均有通过记录：初次188项中187通过、1个旧V1 fixture失败；修复后相关2项通过，新增阶段失效1项、确定性配置2项和graph AMP拒绝1项通过。该统计是去重后的分阶段验证记录，并非声称一次命令193项全绿。graph_reuse仅验收FP32，配置和求解器入口均拒绝AMP/CPU或CUDA autocast组合。', '',
        '[原始全集成记录](evidence/core90_v2/pytest_integration.xml)；[去重验证记录](evidence/core90_v2/verification_record.json)；[source候选与迁移失败](evidence/core90_v2/source_probe_development_guarded/development.json)；[真实source负对照](evidence/core90_v2/source_negative_controls/acceptance.json)；[控制审查](evidence/core90_v2/control_review.md)；[能力审查](evidence/core90_v2/capability_review.md)。'])
    marker='<!-- FINAL_ACCEPTANCE_TABLES -->'
    text=DOC.read_text(encoding='utf-8').split(marker)[0].rstrip()
    DOC.write_text(text+'\n\n'+marker+'\n\n'+'\n'.join(lines)+'\n',encoding='utf-8')
    manifest=ROOT/'configs/core90_game_v2_planned_20260911/matrix.json'
    value=json.loads(manifest.read_text(encoding='utf-8'))
    value['b8_implementation_acceptance']='VERIFIED_LOCAL_DETERMINISTIC_FIXED_CONTEXT_AND_TIMED_ENDPOINTS'
    value['acceptance_scope']='Local functional acceptance; formal E200 and performance confirmation not launched'
    manifest.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(status='LOCAL_EVIDENCE_TABLES_WRITTEN',benchmark_groups=len(rows),
                          evidence_files=len(copied),git_delivery='PENDING_FINAL_COMMIT')))


if __name__=='__main__':main()
