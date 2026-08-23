# ERBT-IDR M2.7 B3频域共识否决实现追踪

日期：2026-08-23

基线：去RF32 D92 E0（`M24-D1-COMPILE-PARITY`）

性能分支：M2.5 B3（`M25-B3-G0-STABLE-DUAL-PROTOTYPE-RESIDUAL`）

## 设计边界

M2.7不再构造独立类别残差。候选先计算B0与B3；只有当B3改变当前query的argmax时，频域表征才执行二元共识判断。共识通过则保留原始B3分数，共识失败则整行精确回退B0。频域表征不能改变B3残差方向或放大B3幅度。

目标域状态只由当前row的合法target support估计；query表征只参与逐样本推理，不更新中心、阈值、门控或持久状态。RF32始终不读取。

## 需求追踪

|ID|来源|可验证要求|实现目标|状态|验证|
|---|---|---|---|---|---|
|M27-R01|M2.6正式复盘|B0保持去RF32主基线且同row预测不变|`stage2_m24_row_executor.py`与M2.7runner|LOCAL_VERIFIED|B0编译等价检查、四臂同row集成测试、M2.5–M2.7相邻回归|
|M27-R02|M2.5完整125正证据|B3实现和强度选择保持原样|M2.7状态包装现有`fit_m25_anchored_residual()`|LOCAL_VERIFIED|独立B3与M2.7内嵌B3逐分数精确相等|
|M27-R03|M2.6失败机制|新表征不得生成独立类别logit，只能保留或否决B3翻转|`stage2_m27_spectral_veto.py`|LOCAL_VERIFIED|输出逐行只能精确等于B0或B3|
|M27-R04|目标域稳健中心建议|旧类target support估计类共享中心，类中心去共享偏移后零均值|`TargetCenteredCompetitionModel`|LOCAL_VERIFIED|类平衡、顺序/标签置换、零均值、单离群污染测试|
|M27-R05|FFT改进建议|MGD96作为第一共识表征，不能直接加入query类别分数|`stage2_m27_spectral_veto.py`|LOCAL_VERIFIED|V1共识保留/否决及truth-unopenedrow集成测试|
|M27-R06|复相位表征建议|从同一固定received IQ提取32维相位/倒谱描述，不增加K|`stage2_m27_phase_side_cache.py`|LOCAL_VERIFIED|32维有限性、确定性、增益/全局相位旋转不变性|
|M27-R07|`p2_min_v1`|side cache不得包含query truth/role，必须绑定原feature cache的capsule/split/tokens|`stage2_m27_phase_side_cache.py`及builder|LOCAL_VERIFIED|缓存只读发布、base manifest/capsule/split/token错配负测|
|M27-R08|query边界|query不能更新中心、可靠度、阈值或门控状态|M2.7不可变状态|LOCAL_VERIFIED|分批与顺序不变、收据`query_state_update=false`|
|M27-R09|安全幅度|M2.7每行分数必须精确取B0或B3，最大变化不超过B3|M2.7score路径|LOCAL_VERIFIED|逐行来源allowlist和精确数组选择测试|
|M27-R10|四状态报告规则|输出使用`DA0_REG0/DA1_REG0/DA0_REG1/DA1_REG1`|row executor、scorer、汇总器|LOCAL_VERIFIED|V1/V2四状态prediction闭合与truth-last接口测试|
|M27-R11|最小实验流程|首轮仅B0、B3、B3＋MGD否决、B3＋复相位否决，沿用4个paired identity|M2.7runner与预登记报告|LOCAL_VERIFIED|screen=4 identity/16行，full125=125 identity/500行冻结测试|
|M27-R12|晋级规则|仅当候选相对B0 `DeltaH>=0.002`、help>harm、两类floor下降均不超过0.005，且相对B3 `DeltaH>=0.0002`时运行完整125|M2.7汇总器|LOCAL_VERIFIED|合成summary同时验证B0与B3双阈值|
|M27-R13|完整125边界|`K=1`无法做support留一可靠性校准时不得失败或伪造门控证据|M2.7可靠性状态与score路径|LOCAL_VERIFIED|`K=1`显式标记`K1_EXACT_B0`，query不读取表征并逐行精确回退B0|
|M27-R14|truth-last完整性|scorer不得相信matrix自身缩小的receiver/seed/condition声明|M2.7scorer|LOCAL_VERIFIED|按冻结`matrix_kind`强制screen=4 identity/16行、full125=125 identity/500行及逐臂等量|

## 首轮矩阵

- receiver：`3-19`、`8-8`
- method seed：`7282101`
- 条件：`K=5/new=20`、`K=10/new=5`
- arm：B0、B3、B3＋MGD共识否决、B3＋复相位32维共识否决
- paired input identity：4
- method row：16
- 三场景单位：48

完整125不是首轮前置条件。首轮未达到M27-R12时，以负结果闭环并停止扩矩阵。

## 本地验证

- 聚焦M2.7测试：17项通过，覆盖单离群稳健中心、退化`K=1`完整row精确B0回退和部分矩阵拒绝。
- M2.5–M2.7相邻回归：50项通过。
- M2.7模块、builder、runner、scorer、summarizer与row executor的`py_compile`通过。
- `git diff --check`通过。
- 独立审查首次`P0=0、P1=1`，定位scorer接受自声明部分矩阵；修复后唯一一次定点复审为`P0=0、P1=0、READY`。
- N607直连只读预检确认正式feature/scoring/checkpoint输入存在，screen release/run/log目标路径不存在；prediction固定CPU执行。
- 当前证据状态仅为`LOCAL_VERIFIED`；N607真实checkpoint无query smoke、4-identity prediction、truth-last评分和screen裁决尚未执行。
