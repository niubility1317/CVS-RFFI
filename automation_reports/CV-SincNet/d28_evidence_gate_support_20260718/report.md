# D28逐样本证据门support-only实验

## 启动前记录

- experiment ID：`d28_evidence_gate_20260718/support_screen_v1`；日期：2026-07-18；operator：Codex；状态：`LOCAL_IMPLEMENTATION_IN_PROGRESS`。
- 目标：在D27-B逐新类静态安全bias上增加极小逐样本old/new证据门，优先同时提升旧类遗忘保护、新类总体与弱类floor，不扩展288D表征、原型数量或optimizer step。
- 数据工作点：receiver `20-1`、开发seed `713101`、K=10、6个old+5个seen-new、3个LEO_weak场景×5个held-rank fold；直接复用同一sealed enrollment-only support，query不打开。
- D27诊断依据：D27-B held old/new/H为67.22%/47.33%/52.82%，新support LOO约77.67%而held new仅47.33%，且75/75个逐类bias选择停在安全cap，说明静态bias不能表达逐样本证据强弱。

## 方法锁

- 基础头固定为D27-B：同一唯一接收IQ的`z160+FFT96+RF32`拼接为288D；Stage2-B 15步，Stage2-C 10步；旧weight、shared diagonal与D27逐新类bias均不由gate更新。
- 对每个support score row独立提取`E5=[n1-o1,n1-n2,o1-o2,dN-dO,(n1-dN)-(o1-dO)]`，其中`dG`为组内top-3均值；不读取query角色、truth、quota、batch class count、顺序或全局assignment。
- support标签只用于按shot-rank构造5-fold gate层cross-fit，闭式class-balanced ridge在预注册`lambda={0.1,1,10}`中选择；gate增加0 optimizer step。
- 冻结gate对每个样本计算`Delta=clip(alpha*q,-delta,delta)`并等量加到所有new列，old列bitwise不变、new-new排序不变，随后对全部注册类一次argmax。
- K=1不伪造cross-fit/LOO，gate必须禁用并退化为D27-B。
- fail closed：fold缺组、非有限、条件数>1e6、权重范数>8、有效特征方差不足或OOF old/new/floor门失败均回退D27-B。

## 资源与协议边界

- 预计D27-B 2,016活动参数、25step、约31KB状态、3,456MAC/query；gate增加6个ridge系数和10个标准化标量、少于512B数值状态，score校正约6MAC/query并含5次减法、5次除法和new列加法，总step仍25。
- 每个物理support只对应一个已经叠加的LEO_weak IQ观测；FFT96/RF32/z160只是该固定IQ的确定性数学表征，不生成额外support行，不改变K。
- `query_opened=false`；query等同测试集，只能由封存预测后的独立scorer使用。当前仍是development support-only筛选，不允许正式性能或部署声明。

## N607计划

- 远端根：`/home/szu2070436088/2510044040/CV-SincNet`；Python：`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`；GPU在启动前按live inventory分配。
- output：`/home/szu2070436088/2510044040/CV-SincNet/runs/d28_evidence_gate_20260718/output/support_screen_v1`。
- log：`/home/szu2070436088/2510044040/CV-SincNet/logs/d28_evidence_gate_20260718/support_screen_v1.log`。
- 启动前要求：本地`ssr-gpu`测试、Git diff/SHA闭包、直连preflight、GPU/process inventory、远端SHA、`py_compile`、`bash -n`及output不存在门全部PASS。

## 本地实现与验证

- 新增独立核心`stage2_support_evidence_gate.py`；runner新增`d28_v1_evidence_gate`固定6候选：Z0、B3、C0、D28-A无gate的D27-B、D28-B E5/delta1、D28-C E5/delta2，共90行。
- OOF不是只拟合old/new二元标签：每个lambda都把OOF校正实际作用于D27全类score，再检查old总体、每个旧类、new总体、new class floor均不下降且至少一个新类指标严格提升；无安全lambda、退化方差、病态矩阵或权重超限均封存为disabled passthrough。
- `ssr-gpu`相邻73项测试PASS；覆盖K1精确透传、K=5合法交叉拟合、逐行独立、old列bitwise不变、new-new顺序不变、协议API、资源上限、D28真实runner fold及D25/D26/D27历史回归。`py_compile`与`git diff --check`PASS。

## 完成后补充

- 待记录N607候选联合表、逐场景/逐类floor、完整loss、gate OOF与启用/回退原因、资源Pareto、artifact哈希、独立审计和下一轮决定。
