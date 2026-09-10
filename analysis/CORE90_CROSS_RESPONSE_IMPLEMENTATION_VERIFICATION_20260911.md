# CORE90交叉响应实现与验证

本次交付实现已批准的交叉响应方案，并围绕“配置存在但机制不工作”做了数学、梯度路由、实际训练入口、状态续跑和评分边界验证。实现位于独立工作树；未启动N607正式实验，未加载历史CORE90权重。本文的`VERIFIED`指明确列出的代码行为，不代表E200性能提升、干净目标泛化或多seed晋级。

## 实现入口与数据边界

- [主配置](../code/configs/phase1_core90_cross_response_v1.json)：E200、label130+pseudo70、L_s/U_s/V=.07/.63/.30、scratch、final_only；保留历史CORE90/B02有效损失和LEO日程。
- [训练入口](../code/SSDG/train_ssdg.py)：配置在建数据和模型前解析；[历史损失适配](../code/cvsrffi/cross_response/baseline_compat.py)局部绑定，不改变其他方法。
- [矩阵工具](../code/scripts/core90_cross_response_matrix.py)：生成全部10个变体的argv，要求显式源/目标RX和day，不负责启动。U1—U4共用数据seed、完整块、角色和更新预算；U5相对U4_bilinear只启用反馈。
- [运行时](../code/cvsrffi/cross_response/integration.py)：辅助头独立于部署模型；读取真实接收IQ计算固定统计，源训练拟合尺度后冻结。源验证用于梯度开放和供体置换诊断，目标指标不参与选择。

输入checkpoint门拒绝外部初始化和不明继承。唯一续训入口为本方法自己保存的`cross_response_epoch_resume`，检查变体、配置、baseline参数、物理记录/角色和实际输入文件身份。相同索引布局的另一数据文件、同路径替换内容、final模型冒充续训状态均不接受。输入文件绑定用于checkpoint继承契约，不重验VALIDATED_ONCE数据，不增加receipt审批流程。

## 逐项追踪

|ID|已实现的要求|验证证据与适用边界|
|---|---|---|
|CR01|双重中心化、交互诊断、补角恒等式|`test_cross_response_math.py`含加性混合反例；不以身份交互为0宣布解耦|
|CR02|P×Q×K真实完整块及物理去重|`test_cross_response_sampling.py`拒绝缺角、重复/冲突物理记录；同记录多view不增加独立覆盖|
|CR03|供体TX/RX与查询双向排除|角色集合负测；配置至少3×3以形成独立2×2查询；主配置4×4×2|
|CR04|共享响应读出，无TX/RX查表|读出形状、供体置换及真实主干响应梯度验证|
|CR05|固定FFT统计、源训练尺度|解析输入、零能量、固定输入长度和冻结尺度测试；实际运行读取归一化前received IQ|
|CR06|自相关、IQ矩、事件统计可选|`test_cross_response_runtime_contract.py`验证真实runtime目标和梯度；事件需真实事件ID及固定窗口，缺失直接拒绝|
|CR07|加性/共享低秩双线性预测|数学与梯度测试；实际CUDA分别跑U4_additive/U4_bilinear|
|CR08|观测/预测交互与可选重加权|双重中心化及交互权重梯度测试；默认interaction_weight=0，属于明确关闭项|
|CR09|身份交互诊断及独立Ux|Ux不混合响应/决策任务；真实CUDA观察到身份交互梯度|
|CR10|真实分类头的无margin决策保护|正且预测正确的其他RX记录生成稳健参照；参照stop-gradient；单侧满足约束时零梯度|
|CR11|仅新增响应和决策两项主损失|逐项权重/开关、loss与梯度测试；Ux为单列反证对照|
|CR12|warmup与受控身份尾部梯度|按参数对象身份排除共享stem；观察到域/辅助/开放后身份梯度，共享前端响应贡献为0；AMP在autograd前缩放|
|CR13|训练辅助可移除，部署无额外输入|辅助ModuleDict单独保存，部署只加载model；输出回归见验证汇总|
|CR14|稀疏矩形及定向迁移覆盖|反向迁移分别计数，失败步不增加成功覆盖；记录真实独立物理样本|
|CR15|可靠性、有限K菜单和失配处理|多候选/高噪声反馈测试；K=2/3预算测试。主配置菜单仅[2]，不宣称默认运行发生增K|
|CR16|历史反馈、探索混合、成本与resume|无候选模型前向；成功步才提交历史；U5确定性续训状态逐值相同。单候选短测不能证明采样分布适应|
|CR17|条件/事件隔离，BN及MixStyle污染防护|同day/condition构块，真实事件隔离；查询扰动通过两层实际MixStyle后不改供体输出；clean/LEO不混用供体|
|CR18|U0—U5、Ux、梯度对照及配对联动|10种真实parser和CUDA入口；联动分析使用实际U4_additive/U4_bilinear名称分别计算，单seed不生成显著性结论|
|CR19|固定源验证与供体更换/打乱|固定随机性和eval状态，换供体TX、打乱RX、常数参照误差；恢复所有RNG和逐模块training状态|
|CR20|身份结果、成本和最终truth-last|四场景固定预测后独立评分；完整性负测和本地合成证据见验证汇总。正式E200结果尚不存在|

## 防止静默失活

`cross_response_activation.json`逐run记录实际成功step、有效块、响应/决策/交互梯度次数、身份开放次数、共享/前端梯度、角色轮换、供体读出方差、决策有效率、特征尺度、概率/K、显存、时间及覆盖/历史数量。未知配置、启用任务但权重为0、不可达到的源门槛和非对照任务的零梯度cap直接拒绝。

时间字段`auxiliary_step_seconds`从交叉batch接入到优化提交，包含主训练前后向，不是“新增辅助loss独占耗时”；显存为运行期CUDA分配峰值。真实额外成本必须通过匹配预算的U0/U1/U4/U5整体时间比较，不能直接把该字段解释成独立辅助开销。

有效块为0、未观察到要求的梯度、角色未轮换、U5历史未更新时，报告`ACTIVATION_INCOMPLETE`及具体缺项；最终完整评分也不能把该run改称机制成功。已应用的optimizer step和有效机制块分别计数：NaN反馈不进入成功覆盖/激活计数，失败步不提交采样反馈。U0明确无辅助参数；head_only和permanent_detach按其设计接受身份响应路径关闭。

正式默认门为源验证至少16个有效块、响应/常数误差≤0.95、连续3次达标。局部CUDA测试为了覆盖开放分支，将测试门改为1个块、1次、宽松误差阈值，且缩短训练/提前历史损失日程；这些覆盖记录保存在合成结果中，不回写正式配置。正式训练如果不达标，门保持关闭并如实报告未激活，不能为获取非零梯度而降低门槛。

## 基线、数值与证据限制

[基线审计](CORE90_CROSS_RESPONSE_BASELINE_AUDIT.md)列出历史三西格玛半径、原型memory梯度语义等漂移和恢复证据。历史快照没有独立完整保存模型及公共数据文件，因此这里只证明本次匹配基线和已捕获损失语义，不能声称历史全训练轨迹逐位复现。历史target选模及污染checkpoint均未继承。

实际短测使用合成WiSig物理记录、真实CORE90主干、真实训练循环和本机CUDA，覆盖label/pseudo、clean/LEO拼接和全部10个开关组合。普通短测省略耗时的最终heldout；其训练终态必须诚实标记`HELDOUT_EVAL_INCOMPLETE`，不接受旧的无关P0/P1门失败作为成功替代。四场景预测评分另做完整合成验证。

确定性续训测试对模型、EMA、optimizer、scaler、原型、伪标签状态、辅助头、尺度、门、sampler/coverage/history和Python/NumPy/CPU/CUDA/LEO RNG使用零容差。原CUDA benchmark模式出现约1e-8差异，测试固定确定性后消失。耗时和显存遥测不要求相同，生产环境跨硬件/内核也不承诺逐位一致。

AMP另做真实前后向验证。初次U0与U4各16步均失败，异常定位至历史proxy-unknown的半精度几何链路；在本方法适配层保留FP32损失代数后，两者均为16批次、15次有效更新，scaler自然65536→32768，首个溢出step未计成功。U4_bilinear完成60个有效块，辅助/域梯度各15步、身份尾部11步、决策12步，共享/身份前部响应梯度为0，最终参数均有限。另有单位测试直接证明：小梯度经过fp16中间激活时，必须在autograd之前缩放；在梯度赋值时才乘scale无法恢复已经下溢的值。

本次不能回答正式clean/LEO性能是否提高、弱接收机是否改善或统计上是否协同。`Δ_joint=A_U4−A_U2−A_U3+A_U1`需同指标、同场景、配对seed；单seed仅描述，不用样本数冒充训练seed。Phase1辅助拟合更好不自动证明Phase2或真实卫星泛化。

## 复现与后续正式运行

本机已验证解释器为`C:/Users/lh594/.conda/envs/ssr-gpu/python.exe`。在工作树运行`python -m pytest code/tests/test_cross_response_*.py`时应由shell展开实际文件列表；Windows推荐使用pytest传入`code/tests`并限定`-k cross_response`，或显式列出文件。实际CUDA训练回归在`CORE90_SYNTHETIC_INTEGRATION=1`时启用，也可直接调用[合成验证脚本](../code/scripts/verify_core90_cross_response.py)。生成目录必须是新目录，保留已有证据。

正式运行仍需明确实际PKL、源/目标物理RX与day、seed和授权计算预算；使用矩阵工具生成并检查argv，再按项目最小流程执行。本次交付没有自动启动正式GPU矩阵或N607任务。

## 最终验证汇总

|检查|结果|可检查证据|
|---|---|---|
|数学、采样、配置、源验证、梯度、评分及严格checkpoint回归|84项通过；13项重计算测试在普通pytest中按设计跳过，已通过下列独立脚本执行实际CUDA/部署检查|[JUnit结果](core90_cross_response_verification_20260911/unit_results.xml)|
|最终版本全部10种变体、真实CUDA训练入口|10/10通过，每项2epoch/2次实际更新；需要联合梯度的变体均观察到开放后身份梯度；对照路径按定义关闭|[完整矩阵](core90_cross_response_verification_20260911/matrix.json)、[实际日志目录](core90_cross_response_verification_20260911/runtime_logs)|
|真实AMP更新|U0与U4_bilinear各15/16次有效更新；保留首步非有限和自然scaler回退证据|[AMP结果](core90_cross_response_verification_20260911/amp.json)、[原始故障定位](core90_cross_response_verification_20260911/amp_diagnosis/original_u0_anomaly.log)|
|U0关闭包的完整训练对照|模型、EMA、optimizer/scaler、原型、RNG及121项baseline损失/梯度指标零容差一致|[U0回归](core90_cross_response_verification_20260911/u0_off.json)|
|U5续训与连续训练|完整训练/辅助/反馈/随机状态零容差相同；四类错误续训拒绝|[续训结果](core90_cross_response_verification_20260911/resume.json)|
|部署移除辅助|实际checkpoint严格加载，仅model参数；3个样本全部4类logits逐值一致|[部署结果](core90_cross_response_verification_20260911/deployment.json)|
|真实模型+真实卫星增强+独立评分|8条独立合成记录、4场景、32行完整预测先固定，随后评分|[预测清单](core90_cross_response_verification_20260911/final_predictions/prediction_manifest.json)、[独立评分](core90_cross_response_verification_20260911/final_predictions/independent_scores.json)|
|修改范围|独立分支`codex/core90-cross-response-20260911`；未动主承载面的其他暂存/未暂存修改|Git提交及远端OID核对结果在交付回复中给出|

普通短测的正式heldout被显式省略，因此其终态为`HELDOUT_EVAL_INCOMPLETE`；这是测试范围声明。仅“构造器为空”的U0对照会返回旧路径的`NON_PROMOTABLE_P0_DISABLED`，脚本只在该对照里接受并标注此状态，不能替实际新方法掩盖失败。

证据目录约0.5MB，保存最终配置/计数、真实日志、失败定位、四场景预测及评分。完整本地合成checkpoint与较大fixture留在各`analysis/synthetic_cross_response_*`目录，不作为正式训练产物、不上传合成二进制。所有机制可达/数值检查已完成；正式性能实验与科学晋级仍未进行。
