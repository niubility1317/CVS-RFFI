# ADV3B02-CSIL/MoPC-HR CVS接口优化实验v1

- 实验ID：`adv3b02_csil_mopc_cvs_adapter_opt_20260724_v1`
- 日期：2026-07-24
- 状态：`LOCAL_VERIFIED`
- 目标：修复外部对比方法接入CVS K-shot与单阶段注册时的执行失配，同时保持CSIL和MoPC-HR官方核心方法不变。
- 比较基线：`adv3b02_official_newcount_scale_20260724_v7`的同receiver、seed、K、physical ID和LEO场景结果。

## 可行性摘要

1. 严格官方方法名、代码路径和v7结果保持不可变。
2. 新增显式CVS adapter方法，不把优化结果称为严格官方仓库复现。
3. CSIL先执行官方60%切分；若切分后的训练集未覆盖本阶段全部新类，则显式切换为CVS small-K adapter的全support训练。
4. MoPC-HR仅在当前增量训练行不足batch16时缩小effective batch。
5. MoPC-HR只缩小真实新类batch；每个optimizer step的protoAug伪样本数始终固定为官方16。
6. 不保留普通尾批，不改变epoch、SGD、学习率、momentum、weight decay或loss。
7. CSIL继续使用zero-bias、channel separation、官方逐块mask、EWC和KD，增量阶段继续冻结backbone。
8. MoPC-HR继续使用`CE+protoAug+逐参数非平方L2 HR`，KD只记录不反传。
9. MoPC-HR继续使用raw-dot MPC、`α=0.97`及classifier-logit query判决。
10. 新增MoPC顺序增量adapter：在25个新类上按封存类序、官方`interval_nums=5`执行5个阶段。
11. 每阶段纠正后的prototype只进入后续阶段protoAug，不直接替换query分类器。
12. 新类support/query继续使用固定LEO弱信道IQ；旧类和base/source权限沿用外部对比方法边界。
13. query在全部阶段模型锁定后才打开，不参与batch、早停、校准或选择。
14. 所有类别使用同一batch与阶段规则，不按TX/class ID调参。
15. 复用v7已验证缓存及官方base26/base31，不改变数据字节、physical ID或split。
16. MoPC顺序版本标记为`ORDERED_ARRIVAL_DIAGNOSTIC / SEQUENTIAL_CVS_ADAPTER`，不声称与单阶段new25等价。
17. 旧v7代码路径、结果和artifact只读保留；新实现使用新method、schema/version和不可覆盖run ID。
18. CSIL官方mask修复与small-K adapter分别写入机制receipt，禁止把新旧差异单因归因于batch。
19. 正式发布前必须完成定向测试、负向测试、真实checkpoint no-query smoke和独立`P0=0,P1=0`复审。

## 冻结矩阵

| 方法 | profile | 新类数 | K | receiver×seed | cells | 场景行 |
|---|---|---:|---|---:|---:|---:|
| CSIL-CVS adapter | small-K nonzero batch | 1、3 | 1、5、10、20 | 5×5 | 200 | 600 |
| MoPC-HR-CVS adapter | single-stage small-K nonzero batch | 25 | 1、5、10、20 | 5×5 | 100 | 300 |
| MoPC-HR-CVS sequential5 | 固定类序的5个顺序到达阶段+small-K nonzero batch | 25 | 1、5、10、20 | 5×5 | 100 | 300 |
| 合计 | — | — | — | — | 400 | 1200 |

## 核心方法锁

| 方法 | 不得改变 |
|---|---|
| CSIL | zero-bias公式与`normMag=5`、3epoch、batch请求值20、SGDM衰减、momentum0.9、L2更新0.05、EWC=1、KD=0.2、旧块mask、增量backbone冻结 |
| MoPC-HR | 20epoch、batch请求值16、SGD lr0.01、momentum0.9、wd=2e-4、noise std0.05、logits/2、逐参数非平方L2 HR、KD不入总loss、raw-dot MPC、α=0.97、classifier-logit判决 |

## 验收与停止规则

- 技术验收：400/400 cells、1200/1200场景行、prediction/score/receipt闭环，失败0。
- 机制验收：所有adapter cell的`optimizer_steps>0`；sequential5记录5个阶段，stage2起使用上一阶段纠正prototype。
- MoPC batch验收：`effective_batch_size<16`时，每step的`proto_aug_count`仍为16；定向测试锁定`real_batch_size=5/proto_aug_count=16`。
- CSIL覆盖验收：每个stage的每个新类均有训练样本；官方切分缺类时必须记录`full_support_class_coverage_adapter=true`。
- MoPC顺序验收：每个seed的类序在query前封存并写入receipt；报告stage-position与order sensitivity，不把顺序版本解释为单阶段等价优化。
- 性能报告：逐row比较旧类、新类、H、forgetting和min-old，不以单项最大值晋级。
- 停止仅限P0协议/安全错误，或两个不同row在prediction前出现同一确定性异常指纹；不得因性能低停止。
- 当前本地文件、验证命令、Git commit、N607同步路径、PID/GPU/log和最终结果在后续阶段补录。

## 设计追溯

| ID | 来源 | 要求 | 目标文件 | 状态 | 验证 | 备注 |
|---|---|---|---|---|---|---|
| T01 | 用户要求 | 审计CVS逻辑且不违反官方核心 | `adv3b02_official_repo_ci.py`、本报告 | verified | 官方源码逐项表 | strict与v2 adapter隔离 |
| T02 | v7根因 | CSIL小K不能零优化步且每个新类都有训练覆盖 | `adv3b02_official_repo_ci.py`、单测 | verified | new1/new3覆盖与步数测试 | 缺类时全support；仅adapter生效 |
| T03 | v7根因 | MoPC小K不能零优化步且protoAug固定16 | 同上 | verified | N=5步数与5/16测试 | 仅真实batch缩小；仅adapter生效 |
| T04 | 官方MoPC trainer | `interval_nums=5`顺序执行并传递纠正prototype | 同上、predictor | verified | 5阶段状态、类序封存测试 | `ORDERED_ARRIVAL_DIAGNOSTIC`；query仍用classifier |
| T05 | 项目边界 | 新类support/query保持LEO，query训练行数0 | predictor、计划、负测 | verified | no-query smoke | 外部对比方法豁免其余Stage2限制 |
| T06 | 报告真实性 | 区分base全训练与CSIL增量backbone冻结 | predictor receipt、单测 | verified | receipt字段断言 | 修复统一`backbone_frozen=false`误报 |
| T07 | 发布规则 | 完整矩阵、不可覆盖路径、独立复审、Git commit | plan/runner/report | pending | dry-run+复审 | N607由唯一运行代理负责 |

## 独立冻结前复审

- 初审结论：`P0=0,P1=3`，暂不冻结。
- `P1-1`已修订：CSIL不再采用“至少保留1条”的规则；官方切分缺少任一新类时，adapter使用全部本阶段support，保证逐类训练覆盖。
- `P1-2`已修订：MoPC顺序版本仅作为固定顺序到达诊断，不能与单阶段new25作等价因果结论。
- `P1-3`已纳入报告边界：v7的CSIL-new20使用`expanded_new_coordinates`，属于cardinality-extension adapter，不再称为strict official锚点。
- `P1-4`已确认：当前Python移植错误冻结了fingerprint的old-old块；官方`CSIL.m:81-83,222`要求old-old和new-new可写、两个cross-block冻结。本轮将先修复这一官方核心偏差，并新增逐块mask与旧fingerprint实际更新测试。
- 历史v7的CSIL结果降级为`OLD_FINGERPRINT_FROZEN_IMPLEMENTATION_DIAGNOSTIC`；新实验不能把变化仅归因于small-K adapter。
- 新代码隔离：使用新method名与`cvs.phase2.adv3b02_official_corefix_adapter.v2`机制schema；新run、state、prediction和receipt路径均不可覆盖v7。
- 终审结论：`P0=0,P1=0`，设计冻结；T01-T07仍待实现与发布验证。
- MoPC single-stage new25通常不会触发small-K batch适配，仅作为instrumentation parity/同语义重跑，不宣称batch修复收益。

## 本地实现与验证

| 文件 | 作用 |
|---|---|
| `paper_reproduction/cvs_aligned/adv3b02_official_repo_ci.py` | 新v2 adapter、CSIL官方mask纠正、逐类覆盖、MoPC固定16条protoAug、sequential5哈希链 |
| `paper_reproduction/scripts/run_adv3b02_paper_full_ci_truth_free_predictor.py` | v2 receipt、claim边界、正确记录增量backbone冻结状态 |
| `paper_reproduction/scripts/build_adv3b02_paper_full_ci_plan.py` | 接受新adapter方法 |
| `paper_reproduction/scripts/run_adv3b02_paper_full_ci_plan.py` | 运行器接受新adapter方法 |
| `tests/test_adv3b02_official_repo_ci.py` | 官方mask、逐类覆盖、5/16 batch、两阶段prototype传递测试 |
| `paper_reproduction/configs/cvs_official_*_cvs_adapter_*_split_20260724.json` | 冻结CSIL new1/3与MoPC new25注册集合 |

- 环境：`C:\Users\lh594\.conda\envs\ssr-gpu\python.exe`。
- 定向测试：`14 passed`。
- 定向+计划+相邻集成：`53 passed`。
- `python -m py_compile`：通过。
- `git diff --check`：通过。
- 真实checkpoint no-query smoke：`PASS`；严格加载`missing=0/unexpected=0/skipped=0`，CSIL new3为3步且训练类`[6,7,8]`，mask为old-old/new-new可写、cross冻结；MoPC new5为20步、`real_batch=5/protoAug=16`；`query_rows_opened=0`。
- 真实checkpoint：`E:\type10-7\automation_reports\CV-SincNet\qknnv42_strict_dual125_20260714_183556\artifacts\best_joint_safe_ssdg.pth`。
- 首轮实现复审：`P0=0,P1=3`；修复strict方法名隔离、诊断status和plan方法×新类fail-closed。
- 第二轮实现复审：`P0=0,P1=1`；修复strict MoPC内部resource schema误标v2。
- 最终独立发布复审：`P0=0,P1=0 / APPROVE`。
- 终审后最终代码真实checkpoint no-query smoke再次`PASS`：两方法resource均为v2，CSIL 3步覆盖3类，MoPC 20步且5/16，query打开0行。
