# WISER-RF v2/P3-Primary架构与实验设计

日期：2026-08-31

## 1. 决策与目标

本轮将WISER-RF阶段A从“冻结源头CE、目标原型和VSW驱动，P3只在训练后验收”重构为“可微old-only D92直接驱动完整identity表示更新”。正式优化目标是P3，而P1/P2只作为辅助约束与诊断。

本轮同时解决两个实验可信度问题：

1. 结果必须报告适应前后query绝对指标和百分点变化，不能只报告gate或support代理量；
2. 候选不能凭单receiver、单seed或单场景收益晋级，pilot通过后必须进入多receiver、多seed、大query的Target25确认。

推荐路线是“完整实现、分级放量”：实现N2～N6全部机制，先运行最小因果pilot；只有pilot通过才运行Target25，Target25通过后才扩展全部K10历史切片和阶段B。低性能只形成科学负结果，不得触发技术停进程。

## 2. 协议与权限边界

- `protocol_schema=p2_min_v1`，只复用匹配的`VALIDATED_ONCE/capsule_id/split_id`。
- 阶段A训练只读取旧类target support的固定接收IQ、标签和允许的support token。
- query在模型、D92状态、插值系数、阶段选择和所有超参数冻结后才由predictor只读打开。
- predictor不得读取query truth、角色、类别配额、全局类别计数或scorer输出；训练审计固定`query_rows_used=0`。
- Phase2不得读取source/clean样本、样本级源embedding或BatchNorm运行统计。N4只读取与checkpoint共同封存的int8域×类聚合中心。
- 当前5,363B源摘要足以验证共享域流形假设。低秩源类内协方差、源半径与identity–FFT源交叉协方差属于后续摘要升级，不作为本轮pilot前置条件，也不得从Phase2数据反推。
- 阶段B继续与阶段A分离。只有阶段A跨场景确认通过后，才默认冻结`phi_D`并训练注册专用`phi_R`。

## 3. 候选矩阵

|arm|方法|因果问题|正式性|
|---|---|---|---|
|N0|冻结ADV3B02+精确old-only D92|绝对基线|正式|
|N1|旧WISER A|复现旧损失参照|正式对照|
|N2|P3-Primary|D92原生梯度是否改善P3|正式|
|N3|N2+每类风险/floor约束|是否保护low-elev与最差类|正式|
|N4|N3+共享域流形锚|共享域状态是否优于旧VSW层级|正式|
|N5|N4+P3主导辅助梯度投影|辅助目标冲突是否是主因|正式|
|N6|N5+identity–FFT互补与能量约束|联合D92几何是否进一步稳定|正式|

模型反演C/ABC、当前classwise VSW、ASAM、SWA和大规模权重网格从本轮主矩阵移除。它们不会阻塞N2～N6实现和实验。

## 4. 可微cross-fitted old-only D92

### 4.1 五折构造

对每类K=10的support使用固定seed构造5个互补折。每折8条fit、2条held-out，并保证每个物理样本恰好作为held-out一次：

$$
S_c=F_{c,r}\cup V_{c,r},\qquad |F_{c,r}|=8,\quad |V_{c,r}|=2.
$$

折分仅由support physical ID和预登记seed决定，不能读取query或truth sidecar。

### 4.2 与正式D92数值同构

新模块`stage2_wiser_p3.py`提供纯PyTorch可微D92。它与`exact_d92_fit`统一：

- identity160与FFT96分块归一化；
- 单模态零安全、双模态同时退化拒绝；
- block scale；
- 类均值、old-only共享协方差、shrinkage和正则化；
- equal prior；
- Cholesky求解与仿射logits。

FFT96由固定support IQ计算，不参与梯度；梯度经过identity fit样本、共享协方差、D92判别行和held-out identity表示。五类测试输入的logits最大绝对误差必须满足：

$$
\max|g_{\mathrm{torch}}-g_{\mathrm{exact}}|<10^{-4}.
$$

五类输入为正常双模态、零identity、零FFT、极小范数和高条件数。任何语义近似都必须显式标为诊断，不能进入正式N2～N6。

### 4.3 P3主损失

每折held-out交叉熵为：

$$
L_r^{P3}=\operatorname{CE}[\Psi_r(\theta)(X_{V_r}(\theta)),Y_{V_r}],
$$

全折主损失为：

$$
L_{P3}=\frac{1}{5}\sum_{r=1}^{5}L_r^{P3}.
$$

每次优化使用完整support和全部cross-fit episode，不使用随机mini-batch估计P3方向。

## 5. 类别风险与floor保护

冻结模型先在同一support折分上计算每类基线风险`L_c,0`。当前模型每类风险为`L_c(theta)`，违规量为：

$$
v_c=[L_c(\theta)-L_{c,0}-\epsilon_c]_+.
$$

N3及以后使用：

$$
L_{core}=\frac1C\sum_cL_c+\sum_c\lambda_cv_c+\frac\rho2\sum_cv_c^2,
$$

$$
L_{floor}=\tau\log\sum_c\exp(L_c/\tau),
$$

$$
L_{primary}=L_{core}+\beta L_{floor}.
$$

拉格朗日乘子只由support OOF风险更新并保持非负。配置记录`epsilon_c/rho/beta/tau`和每阶段最终乘子；不得使用query结果调整。

## 6. 共享域流形锚

N4在冻结初始特征上计算六个target-old类中心，并从当前26×6×160的量化源域类别中心求一个跨类别共享的域权重：

$$
w_t^*=\arg\min_{w\in\Delta^{25}}\sum_c\rho(\|\mu_c^t-\sum_dw_dp_{d,c}^s\|_2^2)+\lambda_w\|w\|_2^2.
$$

`w_t`用投影单纯形优化求解一次，随后冻结。每类源锚为：

$$
a_c^s=\sum_dw_{t,d}p_{d,c}^s.
$$

训练损失为六类target中心到对应共享域锚的稳健距离。所有类别必须共享同一个`w_t`；禁止恢复旧classwise sliced-Wasserstein行为。

## 7. P3主导的辅助梯度投影

N5使用`torch.autograd.grad`分别取得主梯度`g0`和source-head、target-prototype、domain-manifold辅助梯度。若辅助梯度与主梯度冲突，则移除冲突分量：

$$
\widetilde g_j=g_j-\frac{\min(0,g_j^Tg_0)}{\|g_0\|_2^2+\varepsilon}g_0.
$$

最终梯度为：

$$
g=g_0+\sum_j\alpha_j\widetilde g_j.
$$

投影按当前已解冻参数的同一扁平向量执行；缺失梯度按零处理。每步记录各辅助目标与P3的整体夹角，每个阶段额外记录网络块级夹角摘要。

## 8. identity–FFT互补性与激活安全

N6在support上计算类内中心化identity–FFT交叉协方差：

$$
C_{zf}^{w}=\frac1N\sum_i(z_i-\mu_{y_i}^z)(f_i-\mu_{y_i}^f)^T.
$$

只惩罚相对冻结模型显著增加的冗余：

$$
L_{dup}=[\|C_{zf}^{w}\|_F^2-\|C_{zf,0}^{w}\|_F^2-\epsilon_{zf}]_+^2.
$$

同时增加identity预归一化能量下界：

$$
L_{energy}=\frac1N\sum_i[\tau_z-\|z_{i,pre}\|_2]_+^2.
$$

正式候选必须满足support`zero_identity_count=0`。精确D92继续保留单模态零安全路径，但该路径只保证运行时可评分，不能让发生identity坍缩的适配候选晋级。

诊断至少输出：identity/FFT block trace、联合协方差条件数、类内交叉协方差Frobenius范数、前5个canonical correlation、identity-only/FFT-only/joint OOF风险、预归一化identity范数Q01及按类zero-id计数。

## 9. P3驱动渐进解冻与support-only回滚

### Stage 1

开放`t3`、time projection、fusion和identity projection；频率分支保持冻结。每100～200步运行完整support OOF P3诊断。

### Stage 2

只有Stage 1相对冻结模型满足以下support-only条件才进入Stage 2。从同一个Stage 1 checkpoint分出三个临时分支：time-only增加`t2`，frequency-only增加`f3/f2`，time+frequency同时增加三者。分支只用support OOF指标按“P3 BA、P3 floor、联合协方差条件数”的固定字典序选择；未选分支不进入query prediction。频率路径只有自身support OOF P3满足条件才保留。

### Stage 3

只有Stage 2继续改善P3时才开放`t1`及对应有效路径。Sinc始终冻结。

阶段继续条件为：OOF P3 BA提高、OOF floor不下降、zero-id为0、联合协方差条件数不超过冻结基线2倍。阶段最大步数、检查间隔和学习率在run预登记报告中冻结。

每个阶段完成后使用support-only插值：

$$
\theta(\alpha)=\theta_0+\alpha(\theta-\theta_0),\qquad \alpha\in\{1,0.75,0.5,0.25,0\}.
$$

选择满足support约束的最大`alpha`；若均不满足则使用`alpha=0`。query不得参与阶段进入、checkpoint选择或插值。

## 10. 实验放量设计

### 10.1 最小因果pilot

- 历史outer：`rx_3_19__seed_713102__k_10__new_5`；
- 场景：`leo_clear_weak/leo_low_elev_weak/leo_rain_weak`；
- arms：N0～N6；
- support：每类K=10旧类物理样本；
- query：每场景完整旧类query包，不采样、不筛选、不用类别配额；
- 所有arm和场景prediction完整后，独立scorer一次性连接truth。

pilot晋级门槛：

$$
\operatorname{Median}_{scene}\Delta BA_{P3}\ge3\text{pp},
$$

$$
\min_{scene}\Delta BA_{P3}\ge-0.5\text{pp},
$$

$$
\operatorname{Median}_{scene}\Delta Floor_{P3}\ge0,
$$

且low-elev floor不下降、zero-id为0、联合协方差条件数不超过2倍、至少两个场景正向翻转多于负向翻转、P1/P2任一场景下降不超过2pp。

### 10.2 Target25大query确认

pilot冠军与N0在固定`K10/new5`切片上运行5个receiver×5个seed×3个场景，共25个outer/75个scene unit。若历史包保持每场景Q120，则每个arm共9,000条query预测；正式报告以实际package计数为准。

Target25确认门槛：

- 75个配对scene unit的P3 BA中位提升至少3pp；
- 三个场景家族各自的配对中位提升均不低于0；
- 75个单元的10%分位提升不低于-2pp；
- 全体与low-elev的P3 floor配对中位数均不下降；
- 至少4/5 receiver和4/5 seed的聚合P3 BA变化为正；
- zero-id、协方差条件数及prediction翻转门槛继续满足。

### 10.3 K10扩展与阶段B

Target25通过后，扩展三个K10切片`new5/new10/new20`，覆盖75个outer/225个scene unit；若仍为Q120/scene，则每个arm共27,000条旧类query预测。K1/K5不套用K10五折实现，除非另行预登记兼容折分或冻结回退。

只有K10扩展仍通过，才进入阶段B。阶段B默认冻结`phi_D`，训练注册专用`phi_R`，并报告`DA0_REG0/DA1_REG0/DA0_REG1/DA1_REG1`四状态。

## 11. 结果数据与报告格式

独立scorer对每个outer、arm和scene输出：

- query行数与每类行数；
- Accuracy、balanced accuracy、floor、NLL和每类Recall绝对值；
- `DA1_REG0-DA0_REG0`的Accuracy/BA/floor/NLL变化，其中准确率类指标使用百分点；
- P1/P2/P3同row绝对值和变化；
- B0→候选的help/harm/unchanged预测翻转数；
- zero-id、block trace、交叉协方差、CCA、联合协方差条件数；
- 训练时延、prediction时延、峰值VRAM/RSS、更新参数数与最终状态字节。

汇总表必须保留receiver、seed、K、new-count和scene，不拼接不同row的局部最优。Target25及以上额外报告mean、median、worst-scenario、10%分位、receiver/seed正向覆盖率和配对bootstrap置信区间。

## 12. 错误处理与停止规则

以下属于系统技术失败：数据权限/query泄漏、错误split/receiver/seed/K/scene、不可覆盖run root冲突、可微/精确D92数值同构失败、非有限loss/gradient、双模态退化、prediction不完整、scorer连接错误、进程归属不清或确定性重复异常。

低query性能、未过科学门槛、缺少非必要报告字段或某个辅助机制无收益不属于技术失败。此类情况保留全部prediction和评分结果，标记科学未晋级，并停止后续放量而不终止其他健康任务。

## 13. 测试与发布

测试先于实现，至少覆盖：

1. 五类输入的可微/精确D92logits同构；
2. 五折每类8/2且每条support恰好held-out一次；
3. 类别风险、soft floor和拉格朗日更新；
4. 共享域权重非负、和为1、所有类别共享；
5. 冲突梯度投影后与P3主梯度内积非负；
6. zero-id、energy和identity–FFT冗余诊断；
7. P3驱动阶段进入与`alpha=0`冻结回退；
8. query包在训练与选择期间不可达；
9. scorer输出绝对query指标、百分点变化和跨场景汇总；
10. Target25矩阵固定5receiver×5seed×3scene且不可覆盖。

本地聚焦测试通过后，执行一次真实ADV3B02 checkpoint无query smoke和一次独立P0/P1审查。随后以一个Git提交、一个release归档、一次本地/远端SHA比对和一次远端编译进入N607。每个run使用不可覆盖run ID；启动后只做一次PID/CWD/cmdline/GPU/log增长绑定检查。

## 14. 实现边界

预计新增或修改：

- `code/cvsrffi/stage2_wiser_p3.py`：可微D92、cross-fit、风险/floor、域流形、梯度投影与互补诊断；
- `code/cvsrffi/stage2_wiser_runner.py`：N2～N6训练、P3阶段控制和support-only插值；
- `code/cvsrffi/stage2_wiser_pilot.py`：N0～N6及新晋级门槛；
- `code/cvsrffi/stage2_wiser_scoring.py`：绝对query指标、delta和跨场景统计；
- `code/cvsrffi/stage2_wiser_target25.py`：固定Target25/K10扩展矩阵；
- `code/scripts/run_stage2_wiser_pilot.py`及新Target25入口；
- `configs/wiser_rf_p3_primary_20260831.json`；
- 对应聚焦测试、实验报告和追踪表。

本设计不声称低秩源类内协方差摘要已经实现，也不提前授权完整Target125或阶段B。第一轮严格设计同构的范围是N2～N6、三场景pilot、通过后Target25和K10扩展。
