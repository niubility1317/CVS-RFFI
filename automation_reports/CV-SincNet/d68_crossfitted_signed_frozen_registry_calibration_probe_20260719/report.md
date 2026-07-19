# D68交叉拟合有向冻结registry标定探针

## 1.执行前登记

- 实验ID：`d68_crossfitted_signed_frozen_registry_calibration_probe_20260719`；operator：Codex；状态：`PREREGISTERED_IMPLEMENTATION_PENDING`。
- 目标：把D65冻结Stage2-B决策几何的低遗忘信号转化为旧/新全注册类可比的单一affine head，同时避免D67连续堆叠的支持代理错配。
- 当前联合最强仍为D62：B/A/N/H/F/J=92.78/82.22/84.67/82.62/10.56/26.67，min-B/A/N=80.00/53.33/73.33，混淆23/8/15。
- D65信号为A86.11、F6.11、min-A70.00，但N59.33、H67.12；D67连续堆叠为A82.78、N83.33、H82.16、F10.00，未晋级。
- cell固定receiver`20-1`、seed`713101`、K10/new5、三场景×五outer fold、实际K8；复用同一`VALIDATED_ONCE/p2_min_v1`D18 enrollment-only support，不因方法变化重验数据。
- 根目录`E:\type10-7`非Git；代码、追踪和本报告镜像进入`E:\type10-7\github_publish\CVS-RFFI-repo`。D67最终证据提交为`9365099e`，其余工作树改动均不属于D68。

## 2.实证根因

D67完整日志显示，D62仿射行在before90行、final165行中的正类均值始终高于负类均值，方向反转数为0。D65则在before90个旧类行中有12个反转，final165行中有19个反转，其中12个旧类、7个新类。D67标准化使用`abs(mean_pos-mean_neg)`确定gap尺度，却保留原始行方向；因此D65 final支持风险4.139319，是D62的0.532406的7.78倍，闭式`alpha`只能压到均值2.906%。

D68检验一个可证伪假设：D65的主要失败之一不是冻结几何本身，而是冻结/追加行之间存在符号和尺度不一致；用相同support-only公式先校正每个匿名行的方向，再统一标定，可能保留D65旧类稳定性并恢复新类竞争力。

## 3.唯一机制锁

对每个stage、每个已注册匿名类`c`，按physical rank执行leave-one-rank-out交叉拟合。K8时为8折：每折held一个rank/类、train七个rank/类；每个support physical row恰好held一次，held不得参与对应D65 expert、方向或标定统计的训练。

每折在train support构造D65冻结Stage2-B covariance/Stage2-C append-only expert，并对held support产生原始score。聚合全部inner-held score后，对每个类计算：

```text
delta_cv,c = mean_positive_cv,c - mean_negative_cv,c
orientation_c = +1, if delta_cv,c >= 0; otherwise -1
```

方向只由交叉拟合held score确定；没有方向阈值、置信门或class名单。随后在full support构造一个D65 expert，并对其full-support原始score计算：

```text
center_c = (mean_positive_full,c + mean_negative_full,c) / 2
within_c = sqrt((var_positive_full,c + var_negative_full,c) / 2)
gap_c = abs(mean_positive_full,c - mean_negative_full,c) / 2
scale_c = max(within_c, gap_c, float32_eps)
h_c(x) = orientation_c * (g65,c(x) - center_c) / scale_c
```

全部`h_c`删除类公共affine项后编译为一个全注册类head。before/final、旧/新类均用同一公式；Stage2-C只沿用D65合法生命周期中的冻结covariance和追加行，不在query读取注册阶段或角色。K1因无法交叉拟合而精确回退D62；K≥2使用leave-one-rank-out，不设K专属参数。

## 4.与历史路线的区别及ground边界

- 不同于D67：不混合D62/D65，不求`alpha`，不把support平方风险当作连续专家权重；D67整条连续堆叠路线保持关闭。
- 不同于D65：最终用于argmax的不是原始冻结行，而是交叉拟合方向锁＋全support统一尺度的有向行。
- 不同于D62：没有TP/FP离散行替换、Fisher residual或atomic gate。
- 不使用旧/新角色offset、class ID规则、scene/receiver分支、outer-held/query拟合、threshold/temperature/ridge/fold扫描。
- D68不读取地面组件。当前D22 manifest明确为`formal_phase2_eligible=false`、`UNVERIFIED_UNDER_CURRENT_PROTOCOL`；把它加入候选会使方法无法满足最新正式目标。D66已真实读取84个ground int8 cell，但只得到A+1.11pp、N−1.33pp及floor交换，不能把“已读取”误写成“已有效利用”。

## 5.判门、停止条件与完整报告要求

- 相对D62，总体B/A/N/H/J、三项全局class floor、三场景同类指标、遗忘和三类混淆不得交换伤害，并至少严格改善A、F、J或任一floor。
- INT8相对matched FP32的before/final support与outer argmax变化、margin sign flip必须为0；全部分数有限。
- leave-one-rank-out必须exact-once且held/train交集0；`orientation∈{-1,+1}`，类置换等变；最终只保留单一affine state，query额外MAC/state为0。
- 若支持内有向D65风险仍显著高于D62，或真实outer不满足无交换门，状态即`COMPLETED_DIAGNOSTIC_NEGATIVE_NOT_PROMOTABLE`，停止方向标定路线；不扫描方向阈值、scale、fold、温度或按角色修补。
- 即使通过也先运行第二development seed，不直接启动125。
- 真实105行完成后必须报告7候选、3场景、11类、15fold、方向反转/稳定性、support风险、量化、20epoch训练、资源、artifact、D62/D65/D66/D67同排对照和目标缺口，不得只报告缺陷。

## 6.待实施与验证

新增独立D68数学core、probe、专项测试和摘要，不修改D62/D65/D67历史实现或artifact。测试至少覆盖leave-one-rank-out exact-once、符号解析例、类置换、K1 D62回退、D65 lifecycle、共同affine中心化、INT8编译等价、禁止分支和资源闭包。

本轮先本地实现与验证，不访问N607。代码验证后提交、建立干净worktree、复跑D42–D68完整链，再补精确105行命令和输出目录。

## 7.R1真实运行前生命周期修订

首版实现和D42–D68全链325/325通过后，在真实运行前复核发现：若Stage2-C用11类full support重新标定6个旧行，会改变旧行的center/scale/orientation并破坏D65最有价值的“注册后旧行冻结”性质。R1因此在任何真实性能计算前修订为：

1. Stage2-B完成6个旧行的交叉拟合方向锁和full-old support统一标定，删除旧类共同affine项后冻结全部旧行字节、方向和共同项。
2. Stage2-C仍对11类执行leave-one-rank-out，以同一匿名公式为5个新行确定方向和full-support尺度；只把新行减去Stage2-B冻结的同一个共同affine项后追加。
3. 6个旧行在Stage2-C输出中必须FP32逐bit不变；新行与旧行均为有向标准分数，最终仍是一个全注册类affine head。query没有old/new角色输入、分支、offset或quota。

此修订替代第3节中“final阶段重新编译全部`h_c`”的含义；full support统计在Stage2-C只决定新追加行，旧行只做只读诊断。它不新增超参数，也不改变cell、数据、判门或停止条件。专项测试必须新增旧行bitwise冻结断言，然后重跑完整链。

## 8.实现与本地验证

- `code/cvsrffi/stage2_d68_signed_calibration.py`：对称support验证、leave-one-rank-out exact-once、行标准化、class-balanced风险、方向解析解和单affine编译。
- `code/scripts/probe_d68_crossfitted_signed_frozen_registry_calibration.py`：D65生命周期、8折inner expert、Stage2-B共同affine与旧行冻结、Stage2-C有向新行追加、资源与runner闭包。
- `tests/test_stage2_d68_signed_calibration.py`与`tests/test_probe_d68_crossfitted_signed_frozen_registry_calibration.py`：10项专项，覆盖partition、解析翻转、风险下降、类置换、FP32编译、K1 D62回退、旧行bitwise冻结、禁止分支和source closure。
- 初版专项10项中唯一失败来自合成样本精确并列时FP64/FP32 tie-break不同；中心化分数误差仍满足阈值。测试修正为严格检查中心化误差，只对非并列样本要求argmax一致；真实INT8/FP32零变化门未放宽。
- R1专项10/10通过；显式激活`ssr-gpu`后的D42–D68完整链325/325通过，用时81.1s。pytest exit0后仍有既知Windows`pytest-current`临时链接清理权限告警，不属于测试失败。

当前仅有合成/代码验证，没有outer性能结论。下一步提交实现、建立干净worktree并复跑完整链，然后登记精确真实105行命令。
