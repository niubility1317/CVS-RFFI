"""Build traceability and delivery summaries from observed acceptance artifacts."""
from pathlib import Path
import hashlib
import json
import xml.etree.ElementTree as ET

ROOT=Path(__file__).resolve().parents[1]

REQUIREMENTS=[
('1.2','Isolated copy, pinned runtime imports and baseline identity','ir_types.py','test_ir_contract.py'),
('1.1/7','Frozen native joint source roles/configuration; launch disabled','ir_experiments.py','test_ir_experiment_manifest.py'),
('2','Object identity layout, aliases, frozen and zero-LR coordinates','ir_types.py; ir_metric.py','test_ir_contract.py; test_ir_metric.py'),
('2','Separate weighted L/U CE and phi ownership','ir_objective.py','test_ir_contract.py; test_ir_native_joint.py'),
('3.2','Raw h0 and real virtual AdamW predictor','ir_solver.py','test_ir_metric.py; test_ir_reference.py'),
('3.3','Frozen second-moment R, actual group/state/clip, no duplicate clipping','ir_metric.py','test_ir_metric.py'),
('3.4','Matrix-free weighted CE GGN; detached backbone and VJP cotangent','ir_head.py','test_ir_curvature.py'),
('3.5','Two-step whitened CG and honest truncated/failure statuses','ir_cg.py','test_ir_cg.py'),
('4','Native per-call dropout RNG/mask replay including zero activations','ir_head.py','test_ir_head.py'),
('5.1','Three-forward/two-full-backward reference; formal update from origin','ir_solver.py','test_ir_reference.py; test_ir_native_joint.py'),
('5.2','Cached native graph and pure leaf replacement without live mutation','ir_objective.py; objective.py; dr_objective.py','test_ir_packet.py; test_ir_cached.py'),
('5.2','kappa=0 direct original EG and packet equivalence','ir_solver.py','test_ir_reference.py; test_ir_packet.py'),
('1.1/3.6','U direct encoder domain gradient zero with U identity retained; no extra LEO domain CE','dr_objective.py','test_ir_packet.py; test_ir_native_joint.py; test_ir_head.py'),
('3.1/6','Origin-used normalizer sequence, frozen route/teacher and origin buffers/RNG','ir_solver.py','test_ir_native_joint.py; test_ir_cached.py'),
('6','Outer model/optimizer/ctx/EMA/prototype/normalizer/history transaction','ir_solver.py; game_tracking/state.py','test_ir_transaction.py'),
('6','Same-step numerical fallback; finite scalar/gradient/state checks; structural abort','ir_solver.py','test_ir_transaction.py'),
('7','E21 fixed response, no TR ramp or DRIC cadence','response_config.py','test_ir_reference.py'),
('1.2/7','Strict FP32 AdamW schema and optimizer variants','response_config.py; ir_solver.py','test_ir_metric.py; test_ir_reference.py'),
('8.1','BR previous accepted final raw psi field, current h0 and real mixed clipping','br_history.py; ir_solver.py','test_ir_br_history.py'),
('8.2','BR periodic/discrete/activity refresh and age-one history','br_history.py','test_ir_br_history.py'),
('8.3','R1=IR; kappa0=BR-EG; serialized history and restored stochastic state','br_history.py; resume.py','test_ir_br_resume.py; test_ir_native_resume.py'),
('9','Executed flags, actual model/derivative counts, norms, clip and displacement telemetry','ir_solver.py; ir_telemetry.py','test_ir_reporting.py; acceptance/*/profiling.json'),
('3.5/9.1','Quadratic energy and full/applied nonlinear residual','ir_cg.py; ir_telemetry.py','test_ir_analytic_examples.py; test_ir_native_diagnostics.py'),
('9.2','Isolated ten-step training-head recovery and disjoint fixed V monitor','ir_telemetry.py','test_ir_native_diagnostics.py'),
('9.3','Same-origin EG/IR AdamW displacement and common-clip counterfactual','ir_solver.py; ir_telemetry.py','test_ir_diagnostics.py; test_ir_native_joint.py'),
('9.2/9.4','Source TX/RX slice metrics, signed gains and null missing values','ir_telemetry.py','test_ir_diagnostics.py; test_ir_reporting.py'),
('11','CPU full development acceptance','scripts/check_ir_acceptance.py','acceptance/cpu/result.json'),
('11','Actual CUDA full development acceptance, unchanged declared tolerance','scripts/check_ir_acceptance.py','acceptance/cuda/result.json'),
('10.1','SIM/EG/IR/G0/OR/encoder-off and BR matrices, distinct seed roles','ir_experiments.py','test_ir_experiment_manifest.py'),
('10.2','Separate unfrozen source convergence preparation, historical budget unchanged','ir_experiments.py','test_ir_experiment_manifest.py'),
('Plan Tasks1/9','Real CLI validation, no accidental launch, legacy config compatibility','scripts/train_response_games.py','test_ir_legacy_config.py'),
('Plan extra examples','Rotation matrix, nonmonotonic task risk, quadratic and asymmetric LU examples','ir_cg.py','test_ir_analytic_examples.py'),
('10.3','Measured bounded local profiling and history memory with explicit scope','scripts/profile_ir_acceptance.py','acceptance/cpu/profiling.json; acceptance/cuda/profiling.json'),
]


def main():
    results={device:json.loads((ROOT/f'acceptance/{device}/result.json').read_text(encoding='utf-8')) for device in ('cpu','cuda')}
    if any(r['status']!='VERIFIED' for r in results.values()):raise RuntimeError('development acceptance incomplete')
    entries=[]
    for i,(section,requirement,files,verification) in enumerate(REQUIREMENTS,1):
        entries.append(dict(id=f'R{i:02}',source=section,requirement=requirement,files=files,status='verified',verification=verification,notes='Synthetic development evidence; no scientific training claim'))
    deferred=[('10.1B','Historical early/mid/late checkpoint probes','No compliant checkpoint ancestry/data manifest provided or loaded; random initialization fixtures used.'),
              ('10.1D','Three-seed real joint training and ablation performance','Future real-data training was not authorized or launched.'),
              ('10.2/10.3','Complete convergence and full-cost efficiency confirmation','Continuation LR/safety cap manifest is intentionally unfrozen; no source-convergence run.'),
              ('10.3','Frozen target prediction and independent scoring','Requires future frozen candidate/checkpoint; no target data or truth accessed.')]
    for i,(section,title,reason) in enumerate(deferred,1):
        entries.append(dict(id=f'D{i:02}',source=section,requirement=title,files='configs/; docs/experiment_protocol.md',status='deferred',verification='not executed',notes=reason))
    header='# IR-EG需求追溯与交付状态\n\n基准：`acf0a6407c4a2cd114851e9f60d0cd346e4b2c77`。33项开发要求已验证，4项需要未来真实实验的要求延期；无未解决实现阻塞。\n\n'
    header+='| ID | Source section | Requirement | Target files | Status | Verification | Notes |\n|---|---|---|---|---|---|---|\n'
    for r in entries:header+='| '+' | '.join(str(r[k]) for k in ('id','source','requirement','files','status','verification','notes'))+' |\n'
    header+='\n## 实施决定与边界\n\n'
    header+='- 当前用户请求授权实施；附件的旧“仅规划”状态不是当前任务指令。未启动远端训练。\n'
    header+='- 托管worktree工具因会话根目录不是Git而不可用，改用原生Git从精确基线创建隔离分支；历史证据未编辑。\n'
    header+='- 同一原生CUDA模型的reference/reference也出现非确定性误差。所有新包比较臂统一确定性cuDNN策略，未放宽容差；这不是全局确定性保证。\n'
    header+='- V监测固定前256条合法记录，切片只描述该面板；恢复头固定重放已登记Dropout掩码。\n'
    header+='- Native端到端checkpoint入口测试跨E1/E2；活跃BR历史的真实序列化续训由独立受限联合fixture验证，不声称跑过E21之后完整真实数据epoch。\n'
    header+='- IR严格实现设计中的有限步局部仿射近似；BR按设计是单独近似算法。最高剩余风险是实际数据上机制收益、稳定性和全成本尚未验证。\n'
    (ROOT/'docs/traceability.md').write_text(header,encoding='utf-8')
    (ROOT/'acceptance/traceability.json').write_text(json.dumps(entries,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    legacy=ET.parse(ROOT/'acceptance/legacy_regression.xml').getroot()
    counts={k:sum(int(s.attrib.get(k,0)) for s in legacy.findall('testsuite')) for k in ('tests','failures','errors','skipped')}
    if counts['failures'] or counts['errors']:raise RuntimeError('legacy regression failed')
    report='# IR-EG / BR-IR-EG开发交付\n\n已完成独立研究执行包，未替换默认基线或启动远端训练。\n\n'
    report+='实现覆盖原生完整EG、IR参考版/缓存版、BR历史预测、事务与恢复、源侧诊断、实验配置和登记。\n\n'
    report+='| 验证 | 测试数 | 失败 |\n|---|---:|---:|\n'
    for device,result in results.items():report+=f'| {device.upper()} IR验收 | {result["counts"]["tests"]} | 0 |\n'
    report+=f'| 旧版相关回归 | {counts["tests"]} | 0 |\n\n'
    report+='实际命令：\n\n```text\npython code/scripts/check_ir_acceptance.py --device cpu --output acceptance/cpu\npython code/scripts/check_ir_acceptance.py --device cuda --output acceptance/cuda\npython -m pytest tests/test_acceptance.py tests/test_native_joint.py tests/test_response_games.py tests/test_response_reaudit.py -q -o addopts= --junitxml=acceptance/legacy_regression.xml\npython code/scripts/profile_ir_acceptance.py --device cpu --output acceptance/cpu\npython code/scripts/profile_ir_acceptance.py --device cuda --output acceptance/cuda\n```\n\n'
    report+='两次验收失败均已保留：`acceptance/cuda_initial_failure/`记录原生非确定性卷积；`acceptance/cuda_resume_device_failure/`记录测试反序列化导致AdamW步数计数器设备变化。后者改用现有生产CPU加载入口，生产逻辑本来就保留正确设备。所有修复按原容差验证。\n\n'
    report+='独立审查提出的冻结参数、导入来源、非有限损失和旧配置兼容性问题均已修复并有针对性测试。无性能晋升结论。\n\n'
    report+='合成profiling只报告本地实际调用、用时和内存；包含求解器snapshot、CG及外部提交，排除真实数据/教师/评估。冷启动和并行干扰的早期计时另存，不用于比较。当前测量不能证明完整训练提速。\n\n'
    report+='24行计划配置均`launch=false`。尚未完成真实checkpoint探针、三seed训练、真实数据消融、source收敛成本及目标预测评分。逐项范围见[需求追溯](traceability.md)。\n'
    (ROOT/'docs/DELIVERY.md').write_text(report,encoding='utf-8')
    hashes={p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT/'code').rglob('*.py')}
    (ROOT/'acceptance/final_code_hashes.json').write_text(json.dumps(hashes,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(traceability=dict(verified=33,deferred=4,rejected=0,blocked=0),cpu=results['cpu']['counts'],cuda=results['cuda']['counts'],legacy=counts)))


if __name__=='__main__':main()
