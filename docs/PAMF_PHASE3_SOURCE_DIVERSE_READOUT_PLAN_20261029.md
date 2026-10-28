# Phase3 多设备源辅助暴露与域均衡拒识

2026-10-29。候选 `20261029-phase3-source-diverse-readout-s2026092701-r01`。当前完整六类源跨分支版本 MLP 未知宏拒识85.7460%，源各RX仍有明显差异，未达95%。本候选增加源辅助设备覆盖，并按源RX/day匹配已知与辅助未知两组的训练质量；不继续调高known风险预算。

新数据已由`20261029-phase3-source-oe-diverse-data-r01`完成VERIFIED/VALIDATED_ONCE，6134唯一source记录、21台辅助设备，保留原3台novel和正式4unknown。新训练集含原4台的1920条及额外17台4214条；跨候选可能复用部分原训练记录，不能称6134条全是独立新增观测。源L6300／V27000和novel3150物理角色保持。

复用当前完整六类F、默认头、旧L-only几何和`head_aligned_branch_summary_v1`100维描述符。只为新OE6134前向冻结F，首步两条合法L IQ真实checkpoint smoke，共6136新F前向；旧L/V/novel的特征直接复用，不重复导出。先核对新数据最小capsule/split/count及public字段，N607路径由给定diverse_data_root拼接；Windows旧public_root/truth_path只是来源，不能驱动远端读取。传输只需complete和public3文件，辅助TX truth/builder私有名单不交给训练器。

新OE在RX18-2和3-19合占64.1180%，而原known L每个RX/day域均420条。每个role分别给每域1/15质量，即每样本`w=1/(15*n_domain)`、每role权重和1。目标为`.5 sum(w_known*softplus(-g_known)) + .5 sum(w_OE*softplus(g_OE))`。不将RX/day输入部署分数；域元数据只用于source训练配重。

描述符normalizer也按域加权，但固定known/OE统计先验为0.6/0.4，来自旧L6300/OE4200的混合比例，防止数据数改变这一先验。在旧数据域均匀时，它严格退化为旧unweighted normalizer。使用weighted population mean/var，std≤1e-8置1；不保存逐样本源特征／TX／域权重作部署输入。两个新读出仍为linear101/MLP1633，同seed2026092701、AdamW .001/.0001、各200CPU全批量更新，无V早停。

本版同时改变源辅助设备覆盖和域配重，不能把结果归因于仅样本数增加或仅加权。F／默认身份头／源几何不更新；新状态必须记录`source_role_domain_balanced_v1`及原head-aligned descriptor schema，避免与旧100维状态混用。known V只在新状态冻结后以固定4%宏FRR、1.5pp正确拒绝损失校准单全局最大阈值，总体和每源RX均约束；阈值标记落盘后才读取novel公开特征。全部30150×7预测固定后独立旧scorer连接未知TX。

仍只允许两个OE候选推进，源unknown宏拒识总体及各RX≥95%、known宏误拒≤5%、较同输入闭集宏损失≤2pp，双过优先linear。五静态控制不触发新方法推进。novel3为重复source开发，known是同批校准经验风险，不宣称独立新确认；即使源过门也不等价跨RX／星地／无标签适应／匿名注册成功。正式RX24 U/Q始终不读。

流水线为export（最多一条新增GPU worker）→fitCPU→predictCPU→独立score。每卡可新增2的用户资源授权保持，按实际显存挑卡，不停止任何他人任务。新不可覆盖release/run，实际code/config提交、一次本候选P0/P1、一次N607 preflight／归档传输比较／编译、真实模型smoke及PID/CWD/argv/GPU/log读回留同名最小登记。当前实现中，没有本候选实际性能。

发布前状态：4新模块／对应4测试共34项测试和3个subtests通过，无warning；一次独立P0/P1审查PASS并独立同34项通过。N607一次preflight PASS，父源run ANALYZED_SOURCE_ONLY、原F权重与缓存在位，新release/run未存在，各卡可用显存约19–24GB。后续实际容量选择只新增一条导出worker，所有其他进程保持。新数据归档只含complete及public3文件，旧E盘路径不驱动取数；状态READY_FOR_RELEASE，尚未真实前向或训练。
