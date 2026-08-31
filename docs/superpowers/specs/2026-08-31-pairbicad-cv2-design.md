# ADV3B02-PairBiCAD-CV2设计

## 1.目标与边界

本设计将现有`ADV3B02-PairBiCAD-P1`升级为收敛驱动的Phase1 source-only域泛化候选。它只使用ManySig源接收机、day1/day2/day3、既有`L_s/U_s/V_cal/V_select=0.07/0.63/0.15/0.15`和`concat_sat_ce_only+LEO_WEAK`数据路径；训练、选模和重跑不得访问Phase2、目标接收机support/query或truth。

本设计不把报告提出的全部机制一次开启。方法按`CV2-Core→CV2-BiAdv→CV2-Pair→CV2-TailGuard→SWAD`分层，必须通过同row source-only门槛才能进入下一阶段。

## 2.冻结基座

- 保留ADV3B02双骨干、共享Sinc/HF、RCN、CosFace、160维`z_id`和160维`z_dom`。
- 保留P1的receiver/day/channel因素化域表示、24维`z_int`和shared-stem gradient firewall，默认`alpha_shared=0.05`。
- 每个物理样本形成同裁剪、同基础处理、同Clean参考尺度的Clean/LEO对；48个物理样本拼接为96个网络输入，只执行一次backbone前向。
- 普通batch为16L+32U。每4步安排一次结构化batch；4接收机LORO fold使用24L+24U，最终5接收机重训使用30L+18U。
- 有标签样本参与Clean/LEO TX CE；无标签样本只参与合法元数据环境监督和已启用的无标签自监督目标。

## 3.候选层级

### 3.1 CV2-Core

`CV2-Core`在P1数值目标不变的前提下加入覆盖周期、平台学习率、候选特定收敛判断和SWAD能力。固定update只作为技术安全字段，不是科学完成定义。

### 3.2 CV2-BiAdv

- labeled class-conditional receiver/day/channel adversary只使用真实TX标签。
- `z_dom`的`[z_r,z_d,z_c]`接受低剂量TX adversary，`z_int`不接受该对抗。
- 判别器使用detached特征更新；编码器更新时冻结判别器参数但保留输入梯度，不增加第二次IQ backbone前向。
- 初始`LR_D/LR_encoder=1.5`。
- labeled conditional adversarial ratio目标为`[0.10,0.20]`，`z_dom` TX adversarial ratio目标为`[0.03,0.08]`。
- 当对抗梯度与TX梯度冲突时，只在identity最后一个block、fusion和projection执行任务保护投影。

### 3.3 CV2-Pair

第一版仅增加128维projector上的低权重pair identity hinge：容忍半径0.05、初始权重0.02、梯度占TX梯度不超过5%。VICReg、pair-delta和soft-U CDAN不进入第一正式矩阵。

### 3.4 CV2-TailGuard

结构化batch上计算class×receiver×view margin风险，默认`lambda_rex=0.02`、`lambda_cvar=0.05`、CVaR tail 20%、困难组采样上限30%。它必须作为独立候选，不得与Pair或BiAdv同时首次引入。

## 4.收敛控制

### 4.1覆盖周期

- `C_U=已访问U样本数/|U_s|`，U采样器按receiver×day无放回循环。
- `C_L`按TX×receiver×day最少有效暴露定义。
- 每次记录updates、backbone forwards、`C_L/C_U`、唯一覆盖率、分组暴露、GPU-hours和time-to-convergence。

### 4.2评估与综合分数

每`0.5 C_U`评估一次，相邻评估至少500 updates。`V_cal`用于状态转换、学习率和平台检测；`V_select`只在候选checkpoint形成后稀疏选模。

综合分数使用0—1尺度：

`S_DG=HM(clean_bal, leo_scene_floor_bal, receiver_floor)-0.10*receiver_std-0.05*negative_margin_rate`。

### 4.3科学停止

候选实际启用的最后一个核心机制至少训练3个`C_U`，且同时满足：最近6次`S_DG`斜率绝对值小于0.15pp/评估；最近6次无超过0.30pp的新最佳；学习率至少降低2次；`D_logit<0.01`；`D_theta<1e-3`；对抗候选的梯度比连续4次处于目标区间；margin 10%分位数不下降。

学习率为coverage warmup+`ReduceLROnPlateau(factor=0.3, patience=3, min_lr=1e-6)`。技术上限为12个U覆盖周期或24小时，未科学收敛时标记`NOT_CONVERGED_SAFETY_STOP`。

### 4.4 SWAD

进入平台后形成SWAD窗口；窗口checkpoint必须在最佳`S_DG`的0.50pp以内且Clean/LEO/receiver floor均无超过0.50pp回退。最终只在`final/EMA/SWAD`候选形成后稀疏使用`V_select`。

## 5.开发与最终重训

- 方法开发使用source-LORO：每个fold用4个源接收机训练，第5个源接收机只作source held-out评估。
- 方法、参数和停止规则冻结后，使用全部5个源接收机从头重训最终Phase1模型。
- 最终模型冻结前不得访问目标接收机。冻结后可进行一次严格零适配目标测试，但结果不得反馈选模、调参、重训或重跑。

## 6.正式快速矩阵

第一轮发布24行、seed392002、fold1/fold8：

- B0：同协议ADV3B02控制。
- B1：PairBiCAD-P1固定U6500控制。
- B2：B1+coverage convergence+ReduceLROnPlateau+无提前冻结。
- B3：B2+SWAD。
- D0：阶段0最佳的静态别名配置。
- D1：D0+detached双时间尺度+labeled CDAN。
- D2：D1+`z_dom` TX adversary+conditional cross-covariance。
- D3：D2+动态GRL+局部任务梯度保护。
- T0：阶段1最佳的静态别名配置。
- T1：T0+低权重pair identity hinge。
- T2：T0+Margin-REx/CVaR。
- T3：T0+pair hinge+Margin-REx/CVaR。

为避免运行中根据结果改变后续行，24行发布前必须把B/D/T候选冻结为明确配置；`D0/T0`不是运行时读取冠军的动态别名。每张GPU最多两个本run训练进程，共8张GPU、16并发槽位，dispatcher按不可覆盖row root排队。

## 7.晋级门槛

主线候选相对同row控制必须满足：`S_DG`提高至少0.50pp、LEO mean提高、LEO类别floor不下降、Clean下降不超过0.50pp、无fold/seed超过3pp的H坍缩。

TailGuard专门候选必须满足：LEO类别floor提高至少1.00pp、H下降不超过0.30pp、Clean下降不超过0.50pp、receiver floor不下降。资源只在性能位于0.30pp等价区间时排序。

## 8.延期和排除

延期：VICReg、pair-delta、soft-U CDAN/EMA teacher、robust class reference、sparse XDC、receiver-front-end augmentation、hard-LEO mining和第三视图。

排除：HCF counterfactual transport默认化、rank-4 common-specific、content-conditioned LODO、普通IQ MixUp、完整Fishr和第一轮FastTrust伪标签。

## 9.验收

- 新增函数必须经历TDD RED/GREEN。
- 聚焦协议负测必须证明target/Phase2/support/query/truth全部fail closed。
- 一次真实历史ADV3B02 checkpoint无query smoke必须完成新鲜optimizer step、严格checkpoint恢复以及Clean和三种`leo_*_weak`评估。
- 24行矩阵必须显式记录候选、fold、seed、GPU、收敛配置、停止状态和四场景预期artifact。
- N607启动后核对PID/CWD/cmdline/run root、GPU映射和日志增长；低性能不得停止实验。
