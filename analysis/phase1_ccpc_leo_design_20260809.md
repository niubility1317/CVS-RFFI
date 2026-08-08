# P1-CCPC-LEO冻结设计与可追溯记录

版本：2026-08-09
状态：`LOCAL_VERIFIED_NOT_N607_LANDED`
候选：`P1-CCPC-LEO`

## 1.边界与唯一机制

本轮以每折既有GeoSat-C C checkpoint为共同起点，进行固定40epoch continuation。C臂不启用CCPC；G臂只额外加入非对称class-conditional paired contrastive（CCPC）项。训练仅读取source-known TX及其同physical clean/单次LEO view；known-validation和proxy TX只作为分区receipt，均不进入loader、loss、校准或epoch选择。不存在unknown阈值、reject head、RX/domain条件、GRL、MMD、CORAL、扫参或多机制叠加。

对同一顺序的clean/LEO行，令`z_l`为LEO身份表征、`z_c`为clean身份表征、`y`为TX标签。`z_c`先detach并L2归一化；每个LEO anchor以同TX clean行作为正例、batch全部TX clean行作为分母：

\[
L_{CCPC}=-\frac{1}{N}\sum_i\frac{1}{|P_i|}\sum_{j\in P_i}\log\frac{\exp(\langle \hat z^l_i,sg[\hat z^c_j]\rangle/T)}{\sum_k\exp(\langle \hat z^l_i,sg[\hat z^c_k]\rangle/T)},\quad T=0.12.
\]

G总损失为原GeoSat-C loss加`0.02*L_CCPC`。CCPC只从`z_l`回传；clean分支仍只由原GeoSat-C loss更新。标签只通过相等关系产生正例掩码，因此对类别置换等价。

## 2.冻结C/G矩阵与资源映射

| fold | train TX | known-validation TX | proxy TX | C/G | GPU |
|---|---|---|---|---|---|
| F1 | 20-15,20-19,6-15,8-20 | 14-7 | 14-10 | C/G | 0/1 |
| F2 | 14-10,20-19,6-15,8-20 | 20-15 | 14-7 | C/G | 2/3 |
| F3 | 14-10,14-7,6-15,8-20 | 20-19 | 20-15 | C/G | 4/5 |
| F4 | 14-10,14-7,20-15,8-20 | 6-15 | 20-19 | C/G | 6/7 |
| F5 | 14-10,14-7,20-15,20-19 | 8-20 | 6-15 | C/G | 5/0 |
| F6 | 14-7,20-15,20-19,6-15 | 14-10 | 8-20 | C/G | 3/2 |

所有行固定相同seed、sampler、输入checkpoint、40epoch和`final_only`；launcher只打印或执行这12个已冻结任务。

## 3.可追溯矩阵

| ID | 冻结要求 | 目标文件 | 状态 | 验证 | 说明 |
|---|---|---|---|---|---|
| CCPC-01 | 独立、非对称paired contrastive；clean detach、LEO梯度、同TX正例、全TX分母、T=.12 | `code/cvsrffi/phase1_ccpc_leo.py` | verified | 单元数值/梯度测试 | 不接收RX/domain标签。 |
| CCPC-02 | 严格lambda=.02、40epoch continuation、无head/threshold/sweep及禁用loss组合 | `code/cvsrffi/phase1_ccpc_leo.py`,`code/SSDG/train_ssdg.py` | verified | CLI负测 | C路径不计算或相加CCPC零项。 |
| CCPC-03 | 同physical clean/LEO行序绑定、少类/无正样本/nonfinite fail-closed | `code/cvsrffi/phase1_ccpc_leo.py`,`code/SSDG/train_ssdg.py` | verified | 单元负测 | 不以RX或domain metadata构建损失。 |
| CCPC-04 | CONFIG、epoch、terminal receipt记录冻结字段和运行计数 | `code/SSDG/train_ssdg.py` | verified | focused pytest/代码审计 | 包含proxy_rows=0与held_rows=0。 |
| CCPC-05 | 六折C/G、12任务、固定GPU映射、共享checkpoint/seed/sampler/40E、dry-run | `code/scripts/launch_phase1_ccpc_leo12_20260809.sh` | verified | bash -n与DRY_RUN | 不访问N607。 |
| CCPC-06 | 正负样本、梯度、标签置换、禁用项、C不变和launcher闭环 | `code/tests/test_phase1_ccpc_leo.py` | verified | ssr-gpu focused pytest | 覆盖P0/P1风险。 |
| CCPC-07 | 默认GeoSat-C根路径必须指向已冻结的`phase1_loto_clsgeo12_20260808_v1` | `code/scripts/launch_phase1_ccpc_leo12_20260809.sh`,`code/tests/test_phase1_ccpc_leo.py` | verified | launcher字符串断言及DRY_RUN | 环境变量覆盖仍可用。 |
| CCPC-08 | C/G均只warm-start模型权重，严格拒绝missing/unexpected keys；新建AdamW/AMP状态 | `code/cvsrffi/phase1_ccpc_leo.py`,`code/SSDG/train_ssdg.py`,`code/tests/test_phase1_ccpc_leo.py` | verified | 严格load正反例和receipt测试 | 不是optimizer/RNG resume。 |
| CCPC-09 | 禁止teacher checkpoint及三项teacher distillation权重 | `code/cvsrffi/phase1_ccpc_leo.py`,`code/scripts/launch_phase1_ccpc_leo12_20260809.sh`,`code/tests/test_phase1_ccpc_leo.py` | verified | CLI负测与launcher检查 | 不改变原GeoSat-C损失。 |
| CCPC-10 | 有限的零CCPC-LEO特征梯度仅记录，不作为逐batch技术停止；None或非有限仍fail-closed；终态须证明至少一个非零批且无非有限批 | `code/cvsrffi/phase1_ccpc_leo.py`,`code/SSDG/train_ssdg.py`,`code/tests/test_phase1_ccpc_leo.py` | verified | `py_compile`及focused pytest（15 passed） | 仅修改监控/receipt语义；不改loss、lambda、T、数据或AMP。 |
| CCPC-11 | CCPC梯度审计的None或非有限异常在传播前，向当前candidate output_dir原子落盘数据无关的失败receipt；非有限计数随receipt持久化 | `code/cvsrffi/phase1_ccpc_leo.py`,`code/SSDG/train_ssdg.py`,`code/tests/test_phase1_ccpc_leo.py` | verified | 实际临时output_dir持久化测试 | 只记录schema、candidate/run、聚合CCPC receipt和固定错误指纹；不记录raw或批数据。 |
| CCPC-12 | failure receipt的mkstemp/write/fsync/replace/unlink异常不得遮蔽原始CCPC梯度异常 | `code/SSDG/train_ssdg.py`,`code/tests/test_phase1_ccpc_leo.py` | verified | 模拟writer失败及原异常身份测试 | 仅输出固定诊断marker和writer异常类型；随后重抛原始CCPCLEORuntimeError。 |
| CCPC-13 | GradScaler安全审计：在`scale(total_loss).backward()`前以`autograd.grad(lambda*L_CCPC,z_l)`取得未缩放、CCPC专属梯度；None/非有限仍fail-closed，有限零继续记录 | `code/cvsrffi/phase1_ccpc_leo.py`,`code/SSDG/train_ssdg.py`,`code/tests/test_phase1_ccpc_leo.py` | verified | `py_compile`；focused pytest 21 passed | 不再以retained intermediate的scaled `.grad`裁决；主total loss、AMP和optimizer流程不变。 |
| CCPC-14 | 聚合记录parameter-grad finite与optimizer step；新增仅技术健康的6折G-only、15epoch、GPU0-5 one-shot launcher | `code/cvsrffi/phase1_ccpc_leo.py`,`code/SSDG/train_ssdg.py`,`code/scripts/launch_phase1_ccpc_leo_gradient_audit6_20260809.sh`,`code/tests/test_phase1_ccpc_leo.py` | verified | `py_compile`、focused pytest 21 passed、`bash -n`、DRY_RUN=1为6条 | 同checkpoint/seed/T/lambda/AMP/data；无C/G性能比较、无postfreeze和性能读取。 |
| CCPC-15 | `gradient_audit_only`跳过heldout evaluator，固定技术审计receipt；终态强制技术专用、不可晋级、无性能结论；参数梯度/step只须各至少一次有效观察 | `code/cvsrffi/phase1_ccpc_leo.py`,`code/SSDG/train_ssdg.py`,`code/tests/test_phase1_ccpc_leo.py` | verified | `py_compile`、focused pytest 24 passed、`bash -n`、DRY_RUN=1为6条 | 保留GradScaler合法skip；不因参数nonfinite计数单独否决已完成的raw-CCPC审计。 |

## 4.失败边界

P0：分区泄漏、成对行数/顺序不一致、batch少于两类、任一anchor无同TX clean正例、非有限输入或损失、clean未detach、未缩放CCPC专属LEO梯度为None或非有限，均立即fail-closed；该梯度必须在`GradScaler.scale(total_loss).backward()`之前用`autograd.grad(lambda*L_CCPC,z_l,retain_graph=True,create_graph=False,allow_unused=True)`获得。不得将`scaler.unscale_(optimizer)`之后仍保持缩放的non-parameter retained `.grad`作为判据。梯度审计异常先best-effort原子写入数据无关的`ccpc_failure_receipt.json`，写入自身失败只输出固定诊断marker和writer异常类型，绝不遮蔽原异常。有限的零LEO梯度是可记录的合法驻点，不单独终止；终态必须证明至少一个非零raw-CCPC梯度批、至少一个finite parameter-gradient批和至少一次optimizer step。允许GradScaler合法skip产生parameter-nonfinite/step-not-applied计数，但不允许它替代前述三项正证据。`gradient_audit_only`固定写入`SKIPPED_TECHNICAL_AUDIT/TECHNICAL_ONLY/NO_PERFORMANCE_RESULT`，不调用heldout evaluator；terminal manifest及CCPC terminal receipt均标记`technical_only=true`、`promotion_ready=false`和无性能结论。P1：完整12任务后任一冻结结果门失败则标记`REJECT_CCPC_LEO_NO_RETRY`；不得借改变lambda、温度、采样、阈值、head或proxy/held反馈修补。中途不因性能早停。

## 5.本地验证

已在`ssr-gpu`串行通过`py_compile`、`pytest -q code/tests/test_phase1_ccpc_leo.py`（24 passed）、`bash -n`及gradient-audit launcher的`--dry-run`（6条）。CCPC-10额外验证有限零梯度驻点可继续、None/非有限仍fail-closed；CCPC-11验证train侧写入的`ccpc_failure_receipt.json`完整可解析、含nonfinite聚合计数，且None使用独立固定错误指纹；CCPC-12验证writer异常只产生固定诊断marker，原CCPC异常身份与文本保持不变。CCPC-13验证有限的未缩放CCPC梯度仍可被审计为通过，而模拟scaled intermediate溢出不会改变该裁决；CCPC-14验证6折G-only、15epoch、GPU0-5一任务一卡、固定AMP/seed/T/lambda/GeoSat-C checkpoint的dry-run闭环；CCPC-15验证heldout evaluator在技术审计路径不可达、冻结receipt固定、terminal不可晋级，以及含GradScaler合法skip的parameter/step合同。C/G只严格装载同一GeoSat-C模型权重，随后新建AdamW和AMP状态；receipt记录baseline路径、SHA、checkpoint epoch/role和未恢复optimizer/RNG。这只证明本地实现闭环，不构成N607落地、性能或晋级结论。
