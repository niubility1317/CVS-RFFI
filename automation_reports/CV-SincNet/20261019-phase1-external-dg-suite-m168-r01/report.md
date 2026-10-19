# Phase1非PAMF完整对比预登记

162个新增训练、6个历史冻结模型补测，接续原24行；不训练或评估PAMF。三seed2026092701/02/03，实际矩阵在experiment.json，运行配置由同一plan.py生成matrix.json。详见docs/PHASE1_EXTERNAL_DG_EXECUTION_PLAN_20261019.md。

## 数据与方法

17个源数据契约；F1/F2各5源→7目标RX，chrono源D0/D1/D2、D3未见；ManyRx为10TX、16源→16目标RX。ManyRx实际L/U/V=6533/58828/28023，目标123021包。源数量及标签量控制、默认头、预算和全部seed角色逐row列明。

RIEI/DRIFT按用户E100，恒定LR无需移动milestone；IB两阶段各最多200及原早停；SDG200；ResNet通用DG5000更新、源数量10000更新、标签量E200。完整方法与自身CE匹配停止规则，实际成本单列，不声称严格等GPU时间。所有新训练从零，U标签隐藏不用；IB仅继承自身同契约stage1。历史6模型只加载既有联合训练默认头，不拟合新头。

原F0源缓存和七视图复用。新物理ID的六practical由原7669a6de发布内的冻结引擎生成，原有168000 ID观测直接复制；不使用当前工作区不同版本的引擎。新wrapper一次验证物理源目标互斥，后续复用VALIDATED_ONCE，不追加hash链。

## 生命周期与资源

代码提交8c50d20ecf62d9273a9c5c07f12a04ad7da33685已推送并独立核对远端OID。新队列启动VERIFIED，当前PREPARE_DATA，未开始新增GPU训练；实际CWD、命令、解释器和私有输入输出路径引用本地runtime.json。

源角色和观测构建在CPU完成；数学库2线程、最多4个生成worker。真实L_s checkpoint无query smoke后直接继续。历史checkpoint使用原torch2.1解释器，其余方法使用已验证作者torch1.11环境，无共享安装。

仅物理GPU0/1，每卡一个本任务worker；原dispatcher必须实际结束且24行终结后才开始新GPU行，其他owner占用时等待。绝不借用2至7或停止健康任务。输出root不可覆盖、一个launch owner；同指纹两行技术失败暂停未启动行，健康行完成。低分不停止、不调参、不改变后续顺序。

每row冻结后自动预测clean与六practical；chrono两个任务全完成后独立CPU scorer才读truth。历史补测只补六practical，旧clean保留并在最终报告独立复算后引用。逐RX/date/TX及actual seed数保留，TX报告recall/precision；Macro-F1为全部注册类别口径。

## 验证和完成

发布前聚焦负测、一次独立P0/P1审查；归档只做一次传输SHA比较、一次remote compile。post-launch以实际PID/CWD/argv/GPU/log增长判定VERIFIED，提交回执单独不足为证。

最终审计读取原24行及新增168行实际预测、源checkpoint及预算、选模和评分，复算总体CM/Accuracy/F1、检查目标覆盖及actual seed。只有186个训练实例及6个补测均闭合才完成整体目标。所有负结果与失败产物保留；三seed不冒充独立接收机总体显著性。

## 当前证据

本地24项聚焦检查全部通过；输入120组与不可变数据代码逐元素一致。独立P0/P1审查PASS，无P0/P1；已核对原生和ResNet依赖。非阻断P2为技术失败指纹可能合并不同外层异常，保留记录，不新增发布门槛。ManyRx分别报告等权RX主指标和总体包准确率，新增统计检查通过。原IB-RFF六行已独立复算完成，原SDG健康训练。所有目标性能不反馈到此固定矩阵。

## 发布读回

一次N607普通账户preflight、一次归档传输SHA比较和一次远端编译通过。独立读回dispatcher及CPU子进程实际PID／CWD／argv匹配，CUDA可见设备为空；prepare_data.log从618字节增长至1120字节，16个新源缓存已产出。当前只证明CPU准备队列健康启动，不代表新观测、smoke或168行实验完成。证据：local_artifacts/phase1_external_dg_20261019/post_launch_verified.json。

原24行队列同次读回为8行ANALYZED、2行RUNNING，dispatcher继续存活。新GPU队列仍等待父任务实际终结，随后仅允许物理GPU0/1并等待其他owner释放。不停止、重启或修改健康原任务。
