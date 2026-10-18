# WiSig 处理前后 IQ 配对可行性审计（2026-10-18）

结论：当前已核实的代码不能支持“eq0/eq1 相同行号就是同一物理包”。不按行号配对，不将 eq0 静默加入现有 L/U/V，不启动该训练候选。本结论只关闭当前数据上的索引配对前提，不宣称两种观测没有训练价值。

## 范围与执行起点

- 工作分支：`codex/phase1-urban-sigtime-20261014`。
- 起点 commit：`603db5759325c78c97bd1fd3b97f38e8cd3d9208`。
- 延续 E30、完整五个源 RX、L_s/U_s/V=6300/56700/27000、eq1、256 点/25 MHz、F/SIG_TIME_X3 与默认 cosine 头的当前契约。未更改数据协议、模型或测试信道。
- 本次仅读取本地代码、已有登记/协议和公开官方处理代码。未打开 ManySig/WiSig pickle、源 IQ、角色缓存、checkpoint 或目标数据。
- 新训练 0；模型前向 0；额外模型 seed 0；正式预测/评分 0；远端实验启动 0。没有可报告的新增准确率或机制曲线。

## 已核实的官方处理路径

1. [equalize_dataset.m](https://github.com/WiSig-dataset/wisig-process-raw/blob/master/matlab_equalization/equalize_dataset.m) 顺序读取输入包；只有均衡函数返回非空结果时才向输出列表追加。失败包不占输出槽。因此处理后列表是成功包的压缩序列。
2. [equalize_channel.m](https://github.com/WiSig-dataset/wisig-process-raw/blob/master/matlab_equalization/equalize_channel.m) 可在包检测失败或剩余长度不足时返回空结果；其路径包括重采样、检测起点裁剪、频偏估计及 MMSE 处理。这不是现有 eq1 上可凭简单逆滤波恢复的原始观测。
3. [build_dataset_multirx.ipynb](https://github.com/WiSig-dataset/wisig-process-raw/blob/master/python_pkl_creation/build_dataset_multirx.ipynb) 的 process_save_rx 遍历当前 packet_log，按当前顺序提取 IQ。它最终保存 data 和 node_list；该构造没有保存原输入包序号映射。此次通过官方 raw 文件取得 HTTP 200 并解析代码单元；网页渲染/搜索服务取得原始 notebook 失败，不把渲染失败当成文件不存在。
4. [data_utilities.py](https://github.com/WiSig-dataset/wisig-examples/blob/master/data_utilities.py) 的 create_dataset_impl 分别读取两类输入，独立取各自的前 max_sig 条记录放入 eq 轴。检查到的 compact 构建逻辑没有传递跨 eq 原包映射。

一个确定的反例足以否定行号规则：输入包为 A、B、C，若 A 处理失败，输出对应 B、C；两边第 0 行分别属于 A、B。相同 TX/RX/day、相同条数上限、相同行号均不能排除此情况。这里是控制流反例，不是对真实包失败率的测量。

## 本地消费路径

- `experiments/adv3b02_xuc/code/dataset_wisig.py`：WiSigIndex 含 tx_i/rx_i/day_i/eq_i/sig_i；_build_sample_index 独立遍历各 eq 单元。equalized='both' 表示收集两类记录，不建立物理配对。
- `code/dataset_wisig.py` 中相关索引规则相同。双目录存在不提供配对证据；本次没有启动 launcher，因此没有新运行模块 __file__ 可报告。
- `experiments/adv3b02_xuc/code/cvsrffi/game_tracking/data.py:11` 的 opaque_id 输入包含 eq_i 和 sig_i。因此 eq0 和 eq1 的本地 ID 不同；删除 eq 字段只会掩盖映射缺失，不能证明是同一物理包。
- 当前角色缓存检查其实际 ID 与登记角色完全匹配。未发现可以把 eq0 原包追溯到当前 eq1 角色的映射。本结论限于已检查的生成/消费路径，不声称磁盘上所有历史归档都不存在映射。

## 科学与权限判断

[PROJECT_PROTOCOL](../docs/PROJECT_PROTOCOL.md) 要求衍生视图追溯原始物理记录，并保持互斥角色分离。若未经配对就把 eq0 第 k 行作为 eq1 第 k 行的 L_s 双视图，可能实际引入 U_s/V 或其他物理记录。已获星地增强授权不消除这一对应关系要求。

本次不新增审批、签名、哈希或数据重验证流程。当前 eq1 数据与已完成实验继续复用；没有任何现有实验因本次审计失效。也不请求通过授权放宽来绕过错误配对。

当前候选状态：`INDEX_PAIRING_PREMISE_REJECTED_NO_TRAINING`。已有真实包映射且能证明角色归属时才重新评估；不能用波形最近邻、相同 TX 或人为重编号代替物理对应证据。此处未编写未接入训练的配对模块或额外诊断分类头。

## 与联合目标的关系

本次没有产生 clean 或 urban 增益。最近一次新增实际训练仍是 HardLCE45 的 H01、model seed 2026092701：clean 85.497024%、mid urban 62.870833%、low urban 51.318452%。这些是原实验冻结后独立 truth-last 的历史结果，不能归因于本次审计。其 source 判据失败，已关闭该训练配方；不依据这些目标分数晋级或继续调参。

low urban >60%、clean 净收益、suburban 保持及跨 seed 重现仍未完成。本审计排除了一个可能造成角色混用的配对办法；它没有证明新的 IQ 泛化增强有效。

证据关联：

- [上一轮真实源域 Tukey 诊断](../automation_reports/CV-SincNet/20261018-diagnostic-FcrossRX-nativeTaper-L6300-s2026092701-m2-r01/source_analysis.md)
- [最近实际训练与完整负结果](../automation_reports/CV-SincNet/20261018-phase1-FcrossRX-hardLCE-e30-s2026092701-m2-r01/final_report.md)

本轮验证仅检查报告引用的本地实现位置、既有结果来源和文档格式；没有将只读审计包装成实验完成。
