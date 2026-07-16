# D11极轻量support-only可训练低秩adapter追踪

| ID | Source section | Requirement | Target files | Status | Verification | Notes |
|---|---|---|---|---|---|---|
| D11-01 | 项目.md§7.1/7.1.1 | 只读取sealed LEO_weak support；无clean/source/query | D11 module、runner、audit | verified | v5 audit/COMMIT | CLI仅有before/after enrollment-only；query/truth/scorer opened均false |
| D11-02 | 主任务 | `z'=norm(z+U diag(g)V^Tz)`，首轮rank8，参数≤12k | D11 module | verified | 4616参数 | d=288、rank8；backbone冻结 |
| D11-03 | 主任务 | ≤20epoch，class-balanced proto/supcon与强identity正则 | D11 trainer | verified | v6共1038行完整日志 | 12epoch；每行含全部loss、seed、rank、超参、support/LOO/floor，另含joint fold summary |
| D11-04 | 主任务 | K10物理sample leave-two-out预登记选参 | D11 selector | verified | 2候选×3场景×5fold | 三场景联合统一锁参后统一refit |
| D11-05 | 主任务 | 总体、逐类、floor非退化门 | D11 selector/audit | implemented-not-passed | v6 joint NOT_SELECTED | 三场景old逐类非退化均false；new/D10联合门也未全过 |
| D11-06 | 主任务 | K1/K5复用结构/超参数且只能读取各自严格support | D11 package API/tests | code-verified-real-deferred | 10/10 tests | v6未切K10 prefix；等待独立strict K1/K5 sealed包 |
| D11-07 | 主任务 | after优先旧类抗遗忘，冻结before adapter，仅追加新类prototype | D11 state lifecycle | verified-negative | joint L2O | full state位级冻结；joint held评估仍测得旧类遗忘8.33/6.67/15.00pp，故NO-GO |
| D11-08 | 主任务 | 最多3个固定received-IQ view，view不增加K | D11 operators/audit | verified | base单view | formal runner私有factory从actual received IQ逐sample提取并核对IQ SHA；无public callback factory |
| D11-09 | 主任务 | 资源审计：参数、epoch、状态、MAC、forward/view数 | D11 resource audit | verified-partial | v6 measured resource | 4616参数、12epoch、31136B、4616 adapter MAC/view、1 forward/sample；峰值CUDA allocated 58,253,312B；support-row latency/Pareto代码已修但按停止令不重跑、不新增性能声明 |
| D11-10 | 主任务 | D8b strict K10 support-only真实训练日志，不打开query | D11 runner/artifacts/report | verified-not-selected | v6 COMMIT | COMMIT SHA `d9f0a0afc15d5d8e554cae01e6c0a5663ecfbcca7a303136c5d9d56fe3ec58f2` |
| D11-11 | AGENTS.md/Git | 独立文件、本地ssr-gpu验证、报告与Git交接 | tests/report | verified-uncommitted | pytest 10/10 | 共享脏树中仅新增4个D11文件，交由主agent选择性提交 |

## v6 joint结论

统一候选为`d11_rank8_floor_seek`，但三场景联合门全部失败。clear/low/rain的After-old overall/floor分别为`0.6667/0.1000`、`0.6667/0.2000`、`0.6333/0.1000`，相对同fold Before-adapter旧类遗忘分别为`8.33pp`、`6.67pp`、`15.00pp`；After-new overall/floor分别为`0.5200/0.1000`、`0.4600/0.2000`、`0.6000/0.4000`，`H_old_new`分别为`0.5819`、`0.5025`、`0.6141`。旧类逐类非退化门三场景均为false，rain的新类总体/floor也低于D10，因此不得生成query候选。

锁定负结果证据位于`E:\type10-7\automation_reports\CV-SincNet\d8_second_block_dev_20260717_020200\d11_trainable_lowrank_rank8_v6_joint`。其COMMIT绑定的是当时已运行代码SHA。之后仅继续完成核心API硬化：ordinary Mapping拒绝、actual IQ SHA逐sample核验、无public callback factory、runtime/code/checkpoint/state绑定、state bytes-backed readonly与content SHA重算门、>256KiB拒绝、单行预测门；这些修复已通过10/10测试，但按停止指令未重跑真实性能，因此不改变v6 NO-GO结论。v7只有显式`FAILURE.json`且无候选COMMIT；v1仍标记`UNIFIED_HYPERPARAMETER_DRIFT_NOT_SELECTED`。
