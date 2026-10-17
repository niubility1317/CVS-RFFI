# 校准E30之后的两个入口核查

状态：VERIFIED；没有新增训练候选、源数据实验或正式测试。上一轮校准刷新已完整E30／七视图评分并关闭，本轮没有改变其结果、预算、父、seed或健康旧任务。

## 共享传播入口撤回

本轮先提出按同批不同TX共享H、噪声及接收残差逐包独立的组织问题。随后读取了完整历史报告 [PROP_GROUP](E:/type10-7/automation_reports/CV-SincNet/20261017-phase1-FcrossRX-propgroup-sourceRX6-e20-s2026092701-m4-r01/report.md)，确认同一物理机制早已实现：共享geometry/probs/state/z/ray_hz/ray_coef/f_doppler，总频率由共享轨道Doppler＋个人RX CFO重算，noise／phase／sync／tracking私有。该固定路线已按增量条件失败关闭。

原组键是step/RX/day/eq；改为按TX出现次数分组仍属于改变分组键重试同一机制，不能换名作为新方向。因此撤回该入口，未实现新信道算子、登记或启动训练，新增IQ／CP／NN／更新均0。历史源RX6留出契约只用于说明旧路线事实，不作为当前全源L6300训练参照，也不重开训练域留出。

## 顺序前向与BatchNorm解释被否定

当前完整源ordinary/As RDrop目标的确分多次前向执行，但不能据此推断存在BatchNorm状态偏置。实际构造现有F＋CROSS_RX的900722参数模型，随机初始化、原默认cos头／joint160，沿一次train-mode无梯度前向记录实际归一化模块：GroupNorm10次、LayerNorm4次、活动BatchNorm0、变化持久缓冲0。未读取checkpoint或真实数据。

同4个合成IQ在eval下按4包整体和2＋2拆分输入，默认头logit最大差1.4305114746e−6，joint160最大差9.5367431641e−7，均通过预先写定atol=rtol=1e−5。实际12包NN前向（4 train-mode＋8 eval），0反向／优化器更新／teacher／CP／真实IQ／源或目标访问。因此关闭“顺序前向改变BatchNorm状态导致As代价”这一具体解释，不以真concat改写为由启动性能训练。

该结果不证明训练目标整体batch无关：原SupCon含当前L批描述统计和跨样本正负对，Dropout的随机实现也会随batch组织变化。这里只排除活动BN运行状态问题，不从合成等价宣称源或目标识别收益。

首次直接文件入口因项目模块路径缺失，在模型导入前失败，0NN；随后沿模块入口完成。既有Torch/NumPy桥接警告保留，未安装或更改环境，代码未使用ndarray桥接。执行原始代码、stdout/stderr及结果在local_artifacts/F_batch_axis_probe_20261017；公开重现入口 [verify_F_batch_axis_20261017.py](verify_F_batch_axis_20261017.py)只把结果写入用户指定的新输出文件，主体与已执行检查一致；本次未为输出入口重排重复12包NN。

本地实际执行命令（CWD当前工作树）：

    E:/type10-7/local_envs/ssr-gpu/python.exe -X utf8 -m local_artifacts.F_batch_axis_probe_20261017.probe

公开可重现入口（需选择不存在的输出路径）：

    E:/type10-7/local_envs/ssr-gpu/python.exe -X utf8 -m analysis.verify_F_batch_axis_20261017 --out <new-output.json>

## 当前科学状态

两项核查都改变了下一步行动：不重试共享传播分组，不发布基于BN解释的训练。没有新Ag、As身份收益或联合证明，当前无晋级候选。F＋跨RX同TX、完整五源RX/L6300/U56700隐藏TX/V27000、默认头联合训练、E30预算与原U保持；完整研究目标ACTIVE。下一项仍须建立不同的源物理／观测依据，不能以工程检查、源健康或目标微小差替代识别收益。
