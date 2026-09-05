# ADV3B02-ECRS-V1全面实现、实验数据与训练日记报告

## 摘要

本报告回答三个相互独立的问题：设计稿究竟落地了什么；R1–R8本次八卡直训实际执行了什么；现有运行证据能支持什么结论。结论不是“八组全部成功”：截至2026-09-05 23:40（Asia/Hong_Kong）的冻结快照，R1–R6已经完成200轮训练、clean和三类LEO最终评测及source-only诊断产物；R7为194/200且仍在运行；R8在E106因CUDA AMP与`binary_cross_entropy`不兼容而确定性中止，没有最终性能结果。

R1–R6的clean总体准确率为78.37%–79.43%，clean严格unseen-day/unseen-RX为75.38%–76.06%，主分数为76.94–77.71。三类LEO总体准确率均值为50.88%–52.87%，相对clean仍下降26.40–28.03个百分点。描述性最好clean/主分数为R3（79.43%/77.71），最好LEO均值为R4（52.87%）；但本次按用户覆盖从不同随机初始化分别训练、没有共享R0，因此相邻rung差值不能解释为单一机制的严格因果增益。

运行后代码审计还发现一个必须保留的接线缺口：schedule把R6标为启用same-TX跨receiver损失，但`compute_ecrs_paired_losses()`把该计算放在`resp_cls`外层条件内；R6的`resp_cls=false`，所以R6的same-TX项实际为零。R7起`resp_cls=true`，same-TX才与response CE、different-TX排序同时执行。因而R6并不是设计表所称的“R5＋same-TX”，R6→R7也不是单机制递进。该发现已回写设计追溯表，未在本次报告任务中擅自修复或重跑。

## 1.证据范围与状态

- run_id：`phase1_adv3b02_ecrs_v1_manysig_src5_s392005_e200_direct8_20260902_r2`
- 运行代码commit：`1fb9fe05d9dcaba5cd21e8fed16270d0745e2e72`
- 分支：`codex/adv3b02-ecrs-v1-parity-fix-20260901`
- 模式：`DIRECT_FROM_SCRATCH=1`，R1–R8分别从头训练，0个共享R0，0个`--init_checkpoint`
- 日志冻结快照：2026-09-05 23:40，共24个stdout/CSV/JSONL文件、9980795字节
- 结构化epoch：1500条；R1–R6各200，R7为194，R8为106
- 诊断产物：R1–R6各1个`ecrs_v1_diagnostics.pt`，每个98条记录、25088个clean/LEO双视图样本
- 最高整体状态：`PARTIAL_CLOSURE`；不能将整个R1–R8矩阵标为`ARTIFACTS_COMPLETE`
- 晋级状态：`NO_PROMOTION_DECISION`；本报告不运行Phase2、不访问注册query truth、不根据target结果回调模型

|rung|快照epoch|结构化日志|最终四场景|诊断artifact|状态|
|---|---:|---|---|---|---|
|R1|200/200|200 CSV＝200 JSONL|clean＋3 LEO完成|98条|`ARTIFACTS_COMPLETE`|
|R2|200/200|200＝200|完成|98条|`ARTIFACTS_COMPLETE`|
|R3|200/200|200＝200|完成|98条|`ARTIFACTS_COMPLETE`|
|R4|200/200|200＝200|完成|98条|`ARTIFACTS_COMPLETE`|
|R5|200/200|200＝200|完成|98条|`ARTIFACTS_COMPLETE`|
|R6|200/200|200＝200|完成|98条|`ARTIFACTS_COMPLETE`，但R6机制标签存在接线缺口|
|R7|194/200|194＝194|未产生|未产生|`SNAPSHOT_INCOMPLETE_RUNNING`|
|R8|106/200|106＝106|未产生|未产生|`STOPPED_EARLY_SYSTEMIC_TECHNICAL_FAILURE`|

这里的“完成”只表示该row的训练、最终评测和诊断产物闭合，不表示方法达到性能目标或获准晋级。

## 2.数据协议与实验权限

### 2.1冻结数据角色

- 数据集：`ManySig.pkl`，`equalized=1`
- 随机种子：392005
- source receiver：`[1,3,4,6,8]`
- source day：`[1,2,3]`
- source物理池：90000
- `L_s=6300`（0.07，有TX标签）
- `U_s=56700`（0.63，不向训练暴露TX真值）
- `V=27000`（0.30，只读source validation）
- target receiver：`[0,2,5,7,9,10,11]`
- target day：`[0,1,2,3]`
- target TX：`[0,1,2,3,4,5]`
- 每个target场景：168000样本，其中seen-day/unseen-RX为126000，unseen-day/unseen-RX为42000

训练器底层WiSig装载器会先显示9000/81000的临时train/val索引；实际ECRS/Meta-SSL数据角色随后按同一物理池重分为6300/56700/27000。报告采用后者，不能把底层临时索引误写成最终训练角色。

### 2.2星地训练与评测

|epoch|训练视图|采样概率|卫星辅助CE|
|---:|---|---:|---:|
|1–40|`leo_clear_weak`|0.30|关闭|
|41–79|`leo_low_elev_weak`、`leo_rain_weak`|0.60|关闭|
|80–90|同上|0.60|0.68|
|91–200|三种`leo_*_weak`|0.80|0.68|

`concat_sat_ce_only=true`，`lambda_sat_cls=0.68`，`lambda_sat_cons=0`。日志中的`meta_ssl_enabled=true`表示数据路由存在；`meta_ssl_loss_enabled=false`且TX/prototype/domain/adversarial权重均为零，所以本run没有实际启用Meta-SSL学习损失。

最终评测在训练门控E200后执行clean、`leo_clear_weak`、`leo_low_elev_weak`、`leo_rain_weak`。这些都是Phase1闭集TX识别；不是已注册旧类/新类竞争、unknown拒识、DA/REG四状态或Phase2结果。

## 3.端到端实现路径

ECRS没有替换成熟的ADV3B02。原主干继续输出160维单位化身份表示`z_id_raw`和CosFace分类头；旁路只做局部响应系统辨识：

```text
单条IQ x
 ├─ADV3B02主干──────────────────────────────→ z_id_raw(160D)
 └─NuisanceEstimator→解析规范化→ContentEstimator→固定响应字典Φ(28列)
                         →复数加权岭回归→8锚点响应面→z_resp(64D)
                                                    ↓
                          R1–R7:ρ=0；R8:质量门控0≤ρ≤0.25
                                                    ↓
                  normalize(z_id_raw＋ρ·P(z_resp))→z_id_fused(160D)
```

推理只需要一条IQ，不需要同时提供clean伴随视图。clean/LEO同步双视图只在训练自监督和诊断中使用。

### 3.1保守nuisance估计与解析规范化

`NuisanceEstimator`只有一个2→8的一维卷积、GELU、全局池化和3维线性头，头部零初始化。输出经过有界变换，只估计：

- 归一化CFO：`tanh(raw_0)×0.05`周期/采样
- 公共相位：`tanh(raw_1)×π`
- 标量log-gain：`tanh(raw_2)×2`

对复IQ `x[n]`，解析规范化为

`x_can[n]=x[n]·exp{-j(2π f_hat n+phi_hat)}·exp{-g_hat}`。

逆算子在cycle路径重新施加增益和相位。V1没有自由RX-IQ校正器或高容量FIR，也不声称恢复真实发射波形；其目标只是移除最保守、可解释的全局扰动。

### 3.2低容量内容估计

`ContentEstimator`用逐通道5点平滑卷积与原输入加权混合：混合系数为可学习sigmoid标量，初值0.5。残差功率经过有界标量映射得到逐采样置信度`w_n`。每4个采样位置中的1个被mask，再次估计内容，用于masked reconstruction。身份分类梯度与局部辨识分支隔离：`detach_identification_for_identity=true`，因此响应辨识器不会被主身份分类器任意塑形。

### 3.3固定28列复响应基

所有响应基都先以每包幅度95%分位归一化，避免绝对增益改变基尺度。28列按物理含义分为：

|块|列|维度|含义|
|---|---|---:|---|
|PA|0–7|8|当前/延迟激励乘幅度基|
|IQ|8–15|8|共轭当前/延迟激励乘幅度基|
|cross|16–19|4|当前激励与延迟幅度基交叉|
|slew|20–27|8|一阶差分/二阶差分乘当前幅度基|

R1使用`fixed_mp`：有效奇次为1/3/5阶，第四槽置零以保持统一28维接口。R2–R8使用`fixed_spline`：4个RBF中心`[0.15,0.45,0.75,1.05]`，宽度0.30。两者都不是自由MLP或可学习基；learnable low-rank basis留到R9之后且本run关闭。

### 3.4复数加权岭回归与nuisance正交化

每包在关闭autocast的`complex64`区域内独立求解。内容置信度先夹在`[0.05,1]`，再除以包内均值。4列nuisance字典为`s_hat`、线性趋势×`s_hat`、`j·s_hat`和一拍延迟`s_hat`。响应字典先对该nuisance子空间做加权投影消除，再与nuisance字典联合回归。

令`D=[N,Φ_perp]`、权重矩阵`W=diag(w)`，求解：

`theta_hat=(DᴴWD+Λ)^{-1}DᴴWy`。

实现没有显式调用`torch.inverse`。先Cholesky；失败后把岭参数扩大10倍再Cholesky；仍失败才把加权设计与正则项拼成增广系统并用`lstsq`。指纹系数`c_fp`只取联合解中28维`Φ_perp`部分，nuisance系数`gamma`单独保存。

相对岭基准为`alpha=0.01×trace(ΦᴴWΦ)/K`。R5以后分块可辨识性收缩生效：

- PA块按8个幅度锚点覆盖率`q_PA`
- IQ块按非圆度`q_IQ`
- memory块按有效样本数和时延相关性`q_mem`
- DAC/slew块按差分能量`q_DAC`

每块正则近似按`alpha/q_block`放大，`q`下界0.05；可辨识性差时只增强正则，不删除整个响应辨识器。

### 3.5响应面、协方差与64维表示

28维系数在8个固定实轴幅度锚点`[0.15,0.30,0.45,0.60,0.75,0.90,1.05,1.20]`上求值。岭系统对角倒数形成系数协方差近似，并通过锚点设计传播为锚点方差。可靠度为覆盖率乘`exp(-variance/T)`；复锚点先乘可靠度平方根，再展开为16个实数，送入无bias的16→64正交初始化线性层并做单位L2归一化，得到`z_resp`。

因此系统比较的是标准激励网格上的响应曲面，而不是直接把可能受基坐标和病态条件影响的原始系数当身份特征。

### 3.6受限残差融合门控

门控只读7个已detach质量量：log-condition、有效秩/28、有效样本比例、幅度覆盖率、`log1p(NMSE)`、SNR/40和平均协方差的`log1p`。两层MLP输出sigmoid，再乘当前阶段上限：

`rho=active_rho_max·sigmoid(g(q))`，硬上限`rho_max=0.25`。

R1–R7的`active_rho_max=0`，所以最终身份严格等于原ADV3B02路径的单位化结果；R8在E91–200从0线性升到0.20，代码能力上限仍为0.25。响应向量先detach后投影到160维，最终只做受限残差：`normalize(z_id_raw+rho·P(z_resp))`。质量量没有直接拼接到身份表示，也不读取类别ID。

### 3.7输出、checkpoint与单视图推理

ECRS输出包括raw/response/fused身份表示、28维系数、协方差、锚点、可辨识性、ridge回退码、nuisance参数、内容置信度、拟合目标和权重。checkpoint bundle保存：固定基及状态、`M_ref`、锚点网格/设计、16→64编码器、归一化统计、64→160投影、gate、response原型/协方差和`rho_max`。feature schema固定为`ADV3B02:ECRS:z_fused:unit_l2:160:v1`。

## 4.损失、梯度路由与阶段

### 4.1损失定义与权重

|项|权重|作用|
|---|---:|---|
|canonical|0.10|clean/LEO规范化波形NMSE|
|content＋masked reconstruction|0.10|双视图内容一致性和遮挡恢复|
|cycle|0.10|解析规范化逆变换重构|
|split-fit|0.10|包内按幅度分层50/50交叉拟合|
|pair-cross|0.10|clean系数预测LEO、LEO系数预测clean|
|pair-surface＋pair-embedding|0.03|配对响应曲面/表示一致性|
|raw CE|0.30|保留原身份分支监督|
|response CE|0.15|响应表示辅助分类|
|same-TX cross-response|0.05|同TX跨receiver/day响应迁移|
|different-TX ranking|0.03|同receiver/day/view且覆盖率、SNR匹配的异TX间隔|
|gate calibration|0.10|奖励rescue、惩罚harm|

split-fit、pair-cross和surface约束覆盖有标签与无标签source样本；TX监督项只在标签mask有效时使用。不同TX负样本需匹配receiver、day、view，并限制激励直方图距离和SNR差。

### 4.2E200阶段

|阶段|epoch|设计开关|缩放|
|---|---:|---|---|
|Stage2|1–40|canonical、content、cycle|三者1.0|
|Stage3|41–90|保留canonical/content，增加split/pair|canonical 0.25、content 0.50、cycle 0、split/pair 1.0|
|Stage4|91–200|增加response分类、same/diff-TX和gate|canonical 0.25、content 0.50，其余按rung|

阶段配置先冻结整个ECRS分支，再按开关解冻nuisance estimator、content estimator、anchor encoder、response projection和gate；岭回归本身是解析层，没有自由回归参数。

### 4.3“设计rung”与“实际执行rung”

|rung|设计/launcher标签|实际执行审计|
|---|---|---|
|R1|fixed MP＋规范化/内容/岭回归|一致；Stage2含cycle|
|R2|R1的样条基版本|一致|
|R3|R2＋split-fit|一致|
|R4|R3＋pair-cross/surface|一致|
|R5|R4＋分块可辨识性收缩|一致；诊断中的IQ/memory块`q`开始低于1|
|R6|R5＋same-TX跨receiver|**不一致**；schedule为true，但训练装配因`resp_cls=false`没有计算same-TX，实际近似R5的独立随机重复|
|R7|R6＋response CE/different-TX|实际同时首次启用same-TX、response CE和different-TX；`rho=0`，gate数值不起作用|
|R8|R7＋受限残差gate|E91开始`rho>0`；E106首次有效gate BCE路径触发AMP异常并中止|

另一个静态细节是训练装配没有显式检查`gate_calibration`开关；只要`resp_cls=true`就会构造gate校准项。不过R7的`rho=0`使raw与fused预测相同，rescue/harm集合为空，损失为数值零。R8出现有效rescue/harm后才调用BCE，也正是在此处暴露AMP异常。

## 5.实验矩阵、启动与可归因边界

用户明确覆盖设计稿的共享收敛R0前置，launcher以`DIRECT_FROM_SCRATCH=1`跳过R0，并把R1–R8映射到GPU0–7。每个row具有独立随机初始化、独立optimizer状态和独立训练轨迹。该模式可回答“每种完整配置最终能跑到什么结果”，但不能消除初始化、数据顺序与共享GPU负载的混杂。

因此本报告采用两层结论：同一row的clean/LEO和拟合诊断可以直接描述；不同row之间只给描述性差值，不使用“新增机制带来X点提升”这一因果措辞。

## 6.训练日记

### 6.1三阶段轨迹

下表的`loss`为训练器总loss，不是单独ECRS子loss；`val`是source validation TX准确率。完整逐epoch值和里程碑差分见数据包。

|rung|Stage2：loss/val首→末|Stage3：loss/val首→末|Stage4：loss/val首→末|累计epoch耗时|跳过反向批次|
|---|---|---|---|---:|---:|
|R1|20.322→6.726；24.43%→96.03%|7.059→9.383；96.20%→97.57%|10.299→9.089；97.65%→97.87%|21.95小时|31|
|R2|20.325→6.656；25.79%→95.97%|7.030→9.355；95.89%→97.66%|10.410→8.896；97.79%→97.94%|21.28小时|29|
|R3|20.323→6.755；25.35%→96.10%|7.254→9.334；96.50%→97.60%|10.382→8.981；97.59%→97.94%|44.24小时|30|
|R4|20.323→6.686；25.60%→95.97%|7.447→9.389；96.04%→97.69%|10.488→8.855；97.81%→97.95%|47.88小时|31|
|R5|20.323→6.690；25.31%→95.96%|7.372→9.319；96.11%→97.51%|10.441→8.948；97.67%→97.93%|47.70小时|30|
|R6|20.325→6.730；23.73%→96.26%|7.491→9.453；96.37%→97.74%|10.301→9.139；97.80%→97.96%|49.75小时|29|
|R7|20.325→6.671；24.63%→95.86%|7.529→9.411；96.21%→97.53%|12.473→10.984；97.76%→97.90%（至E194）|83.88小时|30|
|R8|20.324→6.714；24.96%→96.09%|7.402→9.419；96.21%→97.24%|12.409→12.315；97.60%→97.45%（至E106）|26.34小时|10|

E41和E91的loss跃升与训练目标及LEO课程切换同步，不能直接解释为优化发散。R7耗时显著高于R1–R6，来自新增response分类/匹配计算及共享GPU负载的共同影响；GPU即时占用包含外部任务，不能把上表当作独占硬件基准。

### 6.2数值稳定性

R1–R8累计220次保护性`unsafe backward/step skipped`，分别为31、29、30、31、30、29、30、10。R1–R7的epoch序列连续、checkpoint持续更新，故这些是非有限梯度保护记录而不是run终止。它们仍说明训练数值余量有限，后续若做正式复现实验，应把跳过比例和发生阶段作为稳定性指标，而不是只汇报最终准确率。

当前CSV/JSONL schema持久化了总loss、训练/验证准确率、耗时、跳过批次、卫星/Meta-SSL开关等，但没有持久化runtime已计算的`train_ecrs_loss`、方向NMSE、condition、effective rank、coverage、ridge fallback、rho、gate rescue/harm等逐epoch字段。R1–R6的终轮诊断artifact弥补了部分机制证据，但不能恢复完整200轮ECRS子loss曲线。这是日志闭合缺口，不能在报告中虚构缺失曲线。

### 6.3R8技术失败

R8在E106进入有效门控校准时，`response_gate_calibration_loss()`对`probability[active]`调用`torch.nn.functional.binary_cross_entropy`。PyTorch在CUDA autocast区域明确拒绝该算子并抛出：

`RuntimeError: torch.nn.functional.binary_cross_entropy and torch.nn.BCELoss are unsafe to autocast.`

这是确定性实现兼容性错误，不是低性能、OOM、数据泄漏或checkpoint丢失。R8保留`best.pth`和`latest.pth`，但没有E200、final clean或三类LEO结果，所以R8在所有性能表中必须为N/A。本任务只负责分析报告，没有修改loss、重启或热补丁授权。

## 7.R1–R6最终性能

### 7.1E200当前checkpoint

每个场景均为168000个target样本。LEO列给总体/严格unseen-day-unseen-RX准确率。

|rung|clean总体|clean严格UDU|主分数|clear|low-elev|rain|三LEO总体均值|clean→LEO下降|
|---|---:|---:|---:|---:|---:|---:|---:|---:|
|R1|78.83%|76.06%|77.45|51.62%/49.26%|50.41%/48.48%|50.60%/48.88%|50.88%|27.95点|
|R2|79.29%|75.68%|77.49|52.02%/49.45%|50.84%/48.35%|51.10%/49.27%|51.32%|27.97点|
|R3|**79.43%**|75.99%|**77.71**|52.07%/49.48%|50.99%/48.62%|51.15%/49.13%|51.40%|28.03点|
|R4|79.27%|75.75%|77.51|**53.57%/50.78%**|**52.44%/49.97%**|**52.60%/50.56%**|**52.87%**|**26.40点**|
|R5|78.50%|75.38%|76.94|52.37%/49.61%|51.33%/48.74%|51.54%/49.43%|51.75%|26.75点|
|R6|78.37%|75.57%|76.97|51.60%/49.26%|50.50%/48.63%|50.84%/49.22%|50.98%|27.39点|

主分数为clean总体与clean严格UDU的等权组合。R3只在clean/主分数上描述性最好；R4在三类LEO上整体最好。R5/R6没有延续R4的LEO结果，且R6根本未实际启用其标签中的same-TX loss，因此不能据此否定same-TX机制本身。

### 7.2相邻row描述性变化

- R2−R1：clean＋0.46点，主分数＋0.04，三LEO均值＋0.44。
- R3−R2：clean＋0.14点，主分数＋0.22，三LEO均值＋0.08。
- R4−R3：clean−0.16点，主分数−0.20，三LEO均值＋1.47。
- R5−R4：clean−0.77点，主分数−0.57，三LEO均值−1.12。
- R6−R5：clean−0.13点，主分数＋0.03，三LEO均值−0.77。

这些差值包含独立初始化混杂。尤其R5→R6近似两个独立随机训练，而非“加入same-TX”的消融。

### 7.3source-val selected-best复评

|rung|selected-best clean|selected-best三LEO均值|E200 clean|E200三LEO均值|
|---|---:|---:|---:|---:|
|R1|78.80%|50.25%|78.83%|50.88%|
|R2|78.61%|50.76%|79.29%|51.32%|
|R3|78.75%|51.04%|79.43%|51.40%|
|R4|78.75%|52.36%|79.27%|52.87%|
|R5|77.69%|50.96%|78.50%|51.75%|
|R6|78.90%|50.94%|78.37%|50.98%|

source-val最佳checkpoint并不稳定对应target最佳；除R6 clean外，E200通常更好。target结果不能回流重新选择checkpoint，本表只用于审计选择偏差。

### 7.4receiver异质性

clean按target receiver分解的范围如下；完整14个receiver×day-scope值见`receiver_results.csv`。

|rung|seen-day范围|unseen-day范围|全局receiver floor|
|---|---:|---:|---:|
|R1|65.31%–92.60%|67.28%–86.05%|65.31%|
|R2|66.46%–93.63%|67.87%–85.17%|66.46%|
|R3|67.91%–93.92%|67.98%–85.18%|67.91%|
|R4|64.34%–93.39%|66.95%–85.47%|64.34%|
|R5|67.11%–93.02%|67.17%–85.53%|67.11%|
|R6|66.71%–92.62%|66.98%–84.08%|66.71%|

R4虽然LEO总体最好，却有最低clean receiver floor（64.34%）。因此仅按总体LEO均值挑选R4会隐藏receiver公平性代价；当前单seed、非共享初始化证据不足以做晋级。

## 8.响应辨识与诊断结果

### 8.1终轮响应拟合

每个rung统计98个终轮source-only诊断记录。方向NMSE是“用clean拟合的系数预测LEO响应”和反方向的加权复NMSE；它不是分类准确率。

|rung|clean→LEO NMSE均值/中位/P95|LEO→clean NMSE均值/中位/P95|平均系数协方差|锚点绝对值均值|
|---|---|---|---:|---:|
|R1|0.710/0.740/0.784|0.664/0.683/0.710|0.006648|0.1516|
|R2|0.699/0.742/0.765|0.655/0.682/0.711|0.000813|0.2715|
|R3|0.079/0.085/0.095|0.066/0.068/0.071|0.001793|0.5054|
|R4|0.061/0.065/0.073|0.051/0.052/0.055|0.003488|0.5553|
|R5|0.051/0.054/0.062|0.042/0.043/0.045|0.003090|0.4565|
|R6|**0.050/0.052/0.059**|**0.040/0.042/0.043**|0.003506|0.4263|

R2替换样条基后方向NMSE仍约0.66–0.70；R3启用split-fit后骤降约一个数量级；R4的配对cross/surface进一步下降；R5/R6继续下降。这个机制链证明后续约束确实让响应系数更能跨clean/LEO方向预测，但分类性能并未同步单调提升，说明“拟合一致性更好”不是“开放环境身份识别必然更好”的充分条件。

### 8.2分块可辨识性

R1–R4关闭identifiability shrinkage，artifact按实现返回全1。R5/R6启用后：PA均值约0.998–0.999，DAC约0.995，IQ均值0.309/0.295，memory均值0.573/0.558；IQ的P05降到0.231/0.199，memory的P05约0.435/0.432。系统主要在IQ与memory块增强正则，说明这些数据包对共轭失衡和时延记忆的辨识条件明显弱于PA覆盖和slew能量。

### 8.3source-only泄漏probe

为避免clean/LEO成对样本跨训练/测试泄漏，本报告按`physical_sample_id`的SHA1固定80/20分组切分，同一物理样本双视图始终在同一侧。仅用训练侧均值/标准差标准化，再用最近质心分类。TX多数类基线17.61%，receiver基线20.50%，view基线50%。

|rung|`z_resp→TX`|`z_resp→RX`|`z_resp→view`|`c_fp→TX`|`c_fp→RX`|`c_fp→view`|
|---|---:|---:|---:|---:|---:|---:|
|R1|27.91%|27.74%|61.40%|34.73%|47.25%|65.44%|
|R2|29.40%|30.01%|62.24%|36.87%|46.95%|69.50%|
|R3|24.76%|30.81%|67.20%|30.05%|46.51%|66.77%|
|R4|22.17%|33.17%|68.45%|30.05%|45.51%|66.56%|
|R5|29.32%|31.73%|59.72%|38.77%|46.21%|62.44%|
|R6|26.25%|37.62%|63.90%|37.44%|46.64%|66.24%|

`z_resp`含有高于多数类基线的TX信息，但也保留可测receiver/view信息；原始系数`c_fp`的receiver可预测率约45%–47%，远高于20.5%基线。因而现有证据不支持“响应表示已经receiver/view不变”的强声明。首批次quality-only TX负对照测试样本仅46个，各row准确率15.22%–23.91%，低于该小测试集30.43%的多数类基线；它没有显示质量量单独承载稳定TX身份，但样本太少，不能作为强否证。

Probe是训练独立的source-only描述性诊断，不是已注册target classifier，也不参与模型选择。

## 9.负对照与产物闭合

每个R1–R6诊断artifact记录：同步crop和稳定`physical_sample_id/pair_id`；clean→LEO、LEO→clean方向误差；`z_resp`、内容摘要、原始系数、nuisance系数；锚点响应、协方差、四块可辨识性；首记录的激励打乱、残差打乱、quality-only、raw/whitened coefficient、anchor surface和pair-id打乱。98/98记录均`source_only=true`且`synchronized_crop=true`。

固定基控制项明确列出`fixed_mp`、`fixed_spline`、`learned_lowrank_deferred_R9`和`free_mlp_forbidden_in_v1`。这证明后续可学习基只作为deferred标签存在，没有在本run暗中启用。

## 10.设计追溯结论

ECRS-01至ECRS-11、ECRS-13、ECRS-14、ECRS-16至ECRS-18、ECRS-20至ECRS-22及ECRS-24具有本地或运行证据；ECRS-12、ECRS-15、ECRS-19因R6接线缺口降为`partial`。R9–R11及learnable basis、Fisher门控、反事实响应移植、response prototype Phase2注册保持`deferred`。没有把这些后续机制冒充为V1已运行能力。

因此最准确的结论是：网络主体、数据配对、固定基、加权岭、锚点表示、checkpoint和R1–R5路径与V1设计高度一致；R6–R8的训练装配尚未完全闭合，R8还存在AMP运行错误。不能继续写成“22项全部verified且完整八阶消融成功”。

## 11.科学结论与下一步边界

现有数据支持：

1.固定局部系统辨识路径能稳定完成R1–R6训练，clean约79%，三类弱LEO约51%–53%。
2.split-fit和pair-cross/surface显著降低跨视图响应预测NMSE；该改善是明确的机制证据。
3.R4取得最佳LEO总体结果，但receiver floor最低；总体鲁棒性与最弱receiver存在权衡。
4.更低响应NMSE没有带来单调更高分类准确率；拟合目标与身份判别目标仍有错位。
5.响应表示仍可预测receiver/view，不能宣称已经完成环境不变分解。

现有数据不支持：

- 不支持相邻rung的严格单机制因果增益。
- 不支持R6 same-TX机制的有效性判断，因为它没有实际执行。
- 不支持R7/R8最终性能判断。
- 不支持多seed稳定性、统计显著性或默认方案晋级。
- 不支持Phase2注册、unknown拒识或旧/新类竞争结论。

如果后续获得明确修复与重跑授权，最小科学闭环应先修正R6的条件接线、把gate BCE移出autocast或改为等价安全形式，并增加逐epoch ECRS字段持久化；然后使用matched初始化重跑至少R5/R6/R7/R8。该建议不构成本次已执行动作。

## 12.可复算数据与文件索引

- 设计追溯：`docs/CVS_PHASE1_ADV3B02_ECRS_V1_TRACE_20260901.md`
- 本run持续报告：`docs/experiments/phase1_adv3b02_ecrs_v1_manysig_src5_s392005_e200_direct8_20260902_r2_report.md`
- 本报告：`docs/experiments/phase1_adv3b02_ecrs_v1_manysig_src5_s392005_e200_direct8_20260902_r2_comprehensive_report_20260905.md`
- 数据包：`docs/experiments/results/phase1_adv3b02_ecrs_v1_manysig_src5_s392005_e200_direct8_20260902_r2/`
- 复算脚本：`tools/analyze_adv3b02_ecrs_v1_run.py`

数据包中的`epoch_metrics_full.csv`保留1500条逐epoch结构化数据；`training_diary.csv`给出关键epoch和阶段切换；`evaluations.csv`、`receiver_results.csv`给出所有已闭合checkpoint/场景/receiver结果；`diagnostics_summary.csv`和`probe_results.csv`给出机制诊断；`anomalies.csv`记录R7快照截断、保护性跳过和R8异常。任何引用R7最终结果的后续版本都必须使用新快照重新生成，而不能在本报告中外推。
