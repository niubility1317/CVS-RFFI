# 全部源 RX：L6300 到独立 V27000 的共享统计确认

状态：ANALYZED／VERIFIED，仅完成源统计；实际执行提交 f6eee0d13b61edcbd5e85c50bca1b1177a2e675c。全局 SSE 降幅 79.860766%，18/18 折和 5/5 RX 改善，独立完整标量复算误差≤8.89e-16。无 IQ 算子或识别成绩，完整联合目标 ACTIVE。见 [完整源分析](source_analysis.md)、[聚合结果](source_result.json)、[独立复算](independent_recount.json) 和 [执行终态](execution_verification.json)。

以下为实际执行前固定的范围和判据。唯一问题是原固定窗口的circularity RX对比，能否由合法L估计并迁移到物理ID互斥的原V，而非识别每个RX的唯一代码。旧4RX circularity按原规则FAIL且固定候选保持关闭；本行不将其改判或实现镜像增强。

输入：同一既有VALIDATED_ONCE源缓存与source_contract，split392005、L6300／U56700隐藏TX／V27000、全部5源RX[1,3,4,6,8]、day1/2/3、6TX、EQ1/RMS/IQ256/25Msps。L每TX/RX/day70包，V每格300包，90格全部保留。L/V/U-ID权限和物理角色不变；U容器、IQ、TX真值不访问。载体内未使用源特征随缓存反序列化，但不参与任何统计。V只读、不再拆分角色、不更新任何模型、EMA、归一化、原型或教师。

方法：原观察器window80:160、lag20、epsilon1e-6及既有参考CFO补偿完全不改，不把CFO作为身份特征。先对每TX/day的5RX单元均值减去该5RX公共均值，得到RX对比。18个TX/day组合逐一留出：其余5TX×2day×全部5RX的50个L单元，估计五个RX对比均值；在该TX/day的全部5RX×300个V包对应五单元上读出。这里是无参数网络的统计交叉拟合，始终保留全部5RX；没有留出训练域，没有Ldev或新分类头。V对比的居中仅用于有合法标签的源分析，不是可部署处理或未来IQ算子。

固定判据：V对比平方误差相对零对比基线改善至少20%；18折至少15折正收益；每个RX的误差不大于其零对比基线（数值SSE容差1e-20，零基线且零误差不惩罚；全局零基线不证明关联）。全部44个五RX错误指派只作描述，不再要求唯一RX指派。阈值在本行真实源读出前固定，是源诊断要求，不是识别准确率或通用显著性阈值。

任一固定条件失败即停止从本全5RX L到V标量依据推导增强，不换window、坐标、RX、阈值或物理样本救跑。通过也只证明独立物理源样本上的统计关联，不识别纯RX硬件、不证明标签保持或Ag、As、DG收益，不自动实现IQ增强、扩种子或访问target。L与V物理ID不交，但同源TX/RX/day采集条件共享，不声称独立硬件/采集日或因果分离。

本行不构造模型/分类器、不加载CP、不前向NN/教师、不做梯度或optimizer更新、不调用星地信道、不读target。F＋跨RX同TX／SIG_TIME三路／统一Sinc／joint160／NoExplicitCFO／默认cosine头联合训练与原U约定保持，后续模型预算仍E30；0epoch仅指本源统计行，不冒充完成E30或联合目标。

本地源角色/配置14个反例、共享RX与零RX对比正例、TX-only/跨day或TX反向/L-V反向四类负构造通过，V变化不改变L拟合预测。独立读取器在全部33300合成元数据与166500合成标量上复算PASS，误差1.55e-15，并拒绝CP/Utruth/target三个非法状态；0真实IQ/CP/NN/更新。工程构造不是本行真实结果。

发布：唯一owner codex/root/F-sharedRX-source-20261018；普通N607既有CVS-RFFI Python，CPU/OMP/MKL/OPENBLAS各2，CUDA_VISIBLE_DEVICES空，GPU none。CWD PROJECT/releases/cvs_F_sharedRX_sourceLtoV_20261018_r01；命令 `CVS-RFFI/bin/python -u -m experiments.cvs_rx_shared_source18.run`。本地发布入口 `F:/App/miniconda3/python.exe -X utf8 -m experiments.cvs_rx_shared_source18.publish`，CWD本分支工作树；不可覆盖root PROJECT/runs/20261018-diagnostic-F-sharedRX-sourceLtoV-manysig-m1-r01，日志release/source_shared_RX.stdout.log。实际代码身份为一次Gitcommit，写release_commit.txt、registration与resolved_config；无设计/报告/成员hash或其他签名。

仅执行既有最小流程：一次限定实际代码/权限/可运行性的P0/P1审查；一次普通账户/源路径/空间预检、一份发布archive及一次本远端SHA比较、一次remotecompile、启动后PID/CWD/argv/CPU/CUDA/log或终态核实。本G0不消费模型/CP，真实CP smoke不适用；launcher首步14反例/标量控制smoke通过后直接继续，不生成审批令牌。缺输入、权限/handle/cell错误、非有限、输出冲突或异常保留failure、停止本行，无自动重试，不停止/热改健康其他任务。性能不作为技术中止理由。

实际预算33300唯一源IQ统计观察，生成166500个标量描述，0训练IQ变换／NN／CP／teacher／梯度／optimizer／target；父训练成本不消费也不重复计入本行，既有缓存创建成本为历史。保存全部L/V逐包标量及元数据于原私有输出，独立recount不导入observer/core/判定函数，从全部保存标量重建180单元、18折、5RX及44错误对照，再报告同row结果和实际成本。独立reader不再读IQ，也不宣称重做观察器数值变换。

预期：active/checks/resolved/result/source_ids/observations/completion或failure、stdout、独立recount、全量源结果报告/登记/Git。原始IQ、逐包ID/描述、私有运行路径/日志不进Git；仅小型聚合与代码/配置。没有身份预测，独立targettruth评分不适用；本行没有训练，故不追加正式七视图。完整目标ACTIVE，仍缺有效IQ Ag、四格交互与跨seed联合证据。

唯一独立P0/P1审查PASS，未发现直接影响下一真实源运行的缺陷；核对V不进入L拟合、全部5RX和角色ID/90cells、独立recount及发布依赖闭合。审查者未访问真实IQ/CP/Utruth/target或远程。源V曾用于本项目健康检查，本行物理互斥不等于全历史未见盲测。
