# ECRS-V1R/V2实施与配置审计

本次任务：按[落地计划](../plans/2026-09-06-adv3b02-ecrs-v1r-v2-implementation.md)完成优化实现，并检查训练配置、机制、参数和直接正确性问题。

基线：`262d84ffb644e8863dc35ee9af951ea1b2e0bb20`；工作分支：`codex/adv3b02-ecrs-v1r-v2-20260906`。本次进行本地实现、CUDA功能检查和配置验证，不由代码实施推导N607训练启动或性能晋级。

## 实施记录

- 已核对本地`ssr-gpu`：Python=`C:/Users/lh594/.conda/envs/ssr-gpu/python.exe`，PyTorch=`2.10.0+cu128`，CUDA可用。
- W1/W2与训练集成：主Agent负责`train.py`、`model_dual_cvsincnet.py`、`schedule.py`、`losses.py`及集成测试。
- W3/W4：独立子任务负责新增V2物理响应模块与数值测试。
- W5/W6辅助：独立子任务负责新增训练loss、连续采样/学习率/状态辅助模块与单元测试。
- W7/W8/W9：主Agent整合部署、日志、launcher、配置审计及最终文档。

## 执行决策

- 保留V1历史实现接口；修复语义以显式V1R/V2版本启用，旧V1对照仍可识别。
- 参考对齐无法由文本确认：先检查本地数据元信息；若不能证明公共参考对齐，则按计划采用明确标注的受约束估计参考，不伪称协议辅助已经验证。
- 动态收益gate与compile保持计划中的条件性后置；完整固定融合路线需实现并检查。
- 采用现有计划作为32项追溯主表；最终在本文逐项回填实现、验证或有理由的未启用状态。按项目最小流程仅做一次最终独立直接正确性审查，不增加重复审查门槛。

## 实施结果

V1R/V2主训练路线、源域筛选、固定融合、连续U采样、独立S-LR/S-LEO/S-BATCH开关及epoch边界恢复已经接入。保留显式V1历史路线。未启动N607实验，未产生真实数据识别收益，也未宣称晋级。

本轮依照计划W1/W5使用既有`code/train.py` ECRS路线，基线标识为`matched_ECRS_V1_train_py`。历史完整`ADV3B02_CORE90_SOFT_E200`运行于另一`SSDG/train_ssdg.py`路线，含额外训练机制。本轮保持ECRS父实现做匹配比较，不等同于已完整复现SSDG-Core90；启动配置显式记录此差别，不把其不支持的参数塞入当前入口。

### 机制、参数与实际执行

|部分|实现及默认参数|检查边界|
|---|---|---|
|V1R|独立64维响应分类头；响应身份梯度不进入物理特征生成；保留旧28维物理课程|旧物理目标与新响应CE分开，避免重复响应CE和raw CE|
|V2参考|默认`estimated_reference`；显式公共波形可配置为`protocol_reference`|模块内部映射为`public_reference`；提供文件不等于已证实WiSig对齐|
|估计器|8个复响应系数；实增益/相位/CFO nuisance；统一实Schur岭估计，两个正则默认0.01|同一估计器用于响应及诊断；固定缓存与动态估计分开|
|锚点|B3采用real8；B4及后续采用complex24显式历史；encoder输出64维|公开输入尺度和外推标志；不把残差评分叫SNR|
|基础损失|响应CE=0.15；跨RX=0.05；单向U EMA=0.03；EMA decay=0.99|默认跨RX/U关闭，B5-X/U/XU分别开启；三项独立记录configured/executed/valid_count/raw/weighted|
|U路径|按RX/day连续无放回循环；学生LEO→EMA clean|U接口不接收TX；仅运行响应分支，不运行双骨干；跨epoch不重置队列|
|固定融合|rho=0.05，cap≤0.25；零初始化切向投影；独立融合头；融合CE=1|默认第91轮开启；第一步投影可更新；默认只更新投影和融合头，raw/encoder与融合CE隔离|
|S-LR|独立有效step时钟，5轮等效warmup；冻结/无梯度/失败step不推进|保留AdamW动量；raw维持原epoch余弦，新增组不会被epoch调度覆盖|
|S-LEO|CE权重从首轮线性爬升，40轮达0.68|独立对照，默认Core90仍从第80轮启用LEO CE|
|S-BATCH|6TX×全部source RX×4包，目标120L；U按实际L数配对|缺格点不伪补；同步loader不预取，保存TX/RX/day队列；格点计数写日志|
|验证|clean+三种weak LEO，raw/response/fused及共头比较|200轮48个固定检查点；逐RX/day与rescue/harm；固定选择路径，不按结果挑头|
|恢复|model/optimizer/scaler/epoch调度、EMA、有效时钟、U/L队列、数据generator及RNG|只声明epoch边界恢复；拒绝同run改变训练开关、课程或核心参数|

### 发现并修复的错误

1. 响应分类开关、raw CE掩码以及跨包目标曾耦合在同一个条件分支；已分开，使各目标按自身条件执行。
2. AMP下gate概率BCE不安全；使用logit形式BCE，旧概率入口转换为logit。
3. 新模块可能扰动公共初始化RNG；构造新模块使用独立RNG上下文。compute-only已验证raw梯度、优化器更新、BN和RNG与基线一致。
4. V2参数真实注册名与属性名不同，按名字分组容易将encoder分到物理组；按实际encoder参数身份分组。
5. 融合损失配置键曾写成`fusion`，而运行时读取`fused_ce`；已修正并以非默认0.2权重做真实入口验证。
6. 源域筛选仍先构建target数据，导致无target时失败；现在直接构建source L/U/V，训练期间与结束后均不访问target评估。
7. V2检查点被写成V1 schema，且平均工具对非Tensor extra_state报错；已按实际bundle版本保存，并深拷贝非Tensor状态。
8. 纯计算模式的额外clean前向、旧concat模式元数据错位、恢复时参数悄然改变：分别增加旁路、明确拒绝和恢复配置比对。
9. B2/V1真实入口的诊断保存中，复数与实数view共享typed storage使`torch.save`失败；CPU归档张量现在独立clone。物理模块rho上限也已与CLI的0.25统一。

### 验证证据

最终本地回归：`test_ecrs*.py`共121项全部通过，包含6条真实训练入口检查；另5项历史ECRS launcher和1项Meta-SSL CLI兼容检查通过，合计127项。旧Meta-SSL测试首轮遇到Windows子进程编码不一致，统一继承`PYTHONUTF8=1`后通过，未以修改训练逻辑掩盖环境问题。现有AMP弃用提示不影响本次结果。`git diff --check`通过。

- 物理数值：Schur/联合参考解、系数恢复、固定缓存等价、cross-fit评估侧替换负测、零能量/非有限输入及CUDA AMP。
- 模型集成：V1/V1R/V2 identity-only路径；V1R/V2公共初始化一致；compute-only优化器一步逐位一致；正rho零投影首步非零梯度；checkpoint严格重载一致。
- 真实训练入口：使用明确标注的合成ManySig格式，target格点为空，运行1轮1step；响应CE、U一致性及融合CE均实际执行，EMA和有效LR计数推进，投影实际改变。无融合行源域选模成功保存best；融合行未过门槛可合法报告未选中。
- 采样与状态：连续U唯一覆盖、缺格点计数、同步L loader恢复；失败step不更新EMA/时钟。
- 桥接：28/28/8维输出、CE梯度只更新encoder/头、旧物理组件冻结；三个桥接的公共encoder/头初始化逐位相同；评估侧替换不改变fit系数、锚点与尺度。公共参考文件移走后，检查点仍从保存的波形buffer严格恢复。
- CUDA分段profile仅是合成分支检查：RTX5070Ti，B=2、T=64、FP32、warmup=1、3次采样。physical/cross-fit/encoder/分支推理/分支前后向的p50约23.55/78.60/0.278/25.19/27.50ms，训练峰值allocated约22.20MB。这里的分支推理不含ADV3B02骨干，不能当部署延迟或真实训练预算。

## 32项计划追溯

原计划是实施前快照，以下为本次实际状态。`已接入`表示代码和对应功能验证，不表示真实数据性能已验证。

|项|当前状态|证据或剩余边界|
|---|---|---|
|T01–T06|已接入|版本/开关/梯度/独立头/zero-effect及旧路线兼容测试|
|T07–T08|估计参考已接入；协议参考未验证|本机常用3个数据路径未发现ManySig.pkl；不把元数据路径当对齐证据|
|T09–T12|已接入|8复系数、Schur、缓存、fit-first cross-fit与边界保护|
|T13|锚点已接入；真实probe待数据|real8/complex24与历史输入检查；尚无真实TX/RX/view可分性结论|
|T14|数值敏感性已接入；统计覆盖待噪声模型|不能把敏感性矩阵称为已校准置信区间|
|T15|已接入|质量标量、能量分母、外推和残差评分；SNR未知|
|T16–T17|已接入|三目标独立与label_mask/U隔离|
|T18|单视图推理已接入；增强前向机制诊断未全量执行|identity-only无需增强真值；cross-fit诊断可独立调用|
|T19–T23|已接入|L/U采样、时钟、课程、融合及恢复测试|
|T24|按计划后置|尚无真实B6互补证据，不实现动态gate冒充收益校准|
|T25–T27|已接入适用部分|V2批量估计、向量化合法pair、响应专用U与裁剪推理；未宣称旧V1全部重复前向已等价优化|
|T28|按计划后置|compile明确拒绝，未证明收益|
|T29|主矩阵与桥接配置已接入|B0/B1/B2-V1/B2/B3a/B3b/B3c/B3/B4/B5-X/U/XU/B6/S-LR/S-LEO/S-BATCH；每个桥接的解释范围见下文|
|T30|评估/预测/profile接口已接入；真实结果待实验|逐物理ID导出不混入truth；合成profile不能替代真实部署资源|
|T31|epoch边界恢复已接入|新run目录独占；不覆盖历史R8，不声称mid-epoch恢复|
|T32|保留判定边界|没有source互补证据就不晋级；不引入自由基/null/FastTrust模块|

## 使用与未完成边界

`code/scripts/launch_phase1_adv3b02_ecrs_v1r_v2.py`默认dry-run，输出展开配置和命令；`--execute`才在当前主机串行执行。真实训练需提供ManySig.pkl和新的run目录。该入口不负责N607发布，也不授权远端启动。

已生成[15行展开训练配置](ECRS_V1R_V2_TRAINING_CONFIG_20260906.json)，含200轮、48个source验证时点和逐行命令，`executed=false`。其中数据路径是待提供的本机路径，`data_file_verified_available=false`；没有据此创建或启动训练run。

逐step证据为`ecrs_training.jsonl`，源域三路径为`ecrs_source_validation.jsonl`，筛选结论为`ecrs_source_screen_result.json`；检查点区分最新恢复包与实际选中best。选中模型另在完整source V上导出逐物理样本预测及`ecrs_final_source_evaluation.json`，不沿用快检batch上限。source内固定融合门槛不是独立B0比较或target确认，最终科学晋级仍未评估。合成profile原始证据见[ECRS_V2_SYNTHETIC_PROFILE_20260906.json](ECRS_V2_SYNTHETIC_PROFILE_20260906.json)。

桥接采用共同的实三维nuisance、W=I、column-RMS岭估计、real8读出和冻结物理规则。B3a复用原ContentEstimator/NuisanceEstimator/Canonicalizer及旧28维ResponseBasis，相对B2是统一估计/读出/冻结组合；B3b改用受约束估计参考；B3c转到compact8字典。B3a不称“仅替换求解器”，B3c不称“仅改变维数”，因为它们还涉及对应尺度与冗余约定。B3与B3c数值配置相同，保留B3名称供整包比较。

当前不能宣称32项全部科学验证完成：真实参考对齐、probe与统计覆盖、真实数据训练及完整部署资源尚未完成；动态gate/compile是计划明确后置项。已交付主训练实现及其直接正确性验证，不以合成数据代替研究结论。
