# CORE90交叉响应V2矩阵启动

状态：LOCAL_VERIFIED。日期2026-09-11。用户授权“启动实验矩阵，数据配置和之前一致”。本次唯一launch owner为主Agent；不停止其他健康任务。

## 范围与机制

启动8个可执行对照：U0、U1、U1_mask_off、U3、Ux、Ux_normalized、head_only、permanent_detach。V2全部16个注册行均保留在配置中；另外8行U2、U4_additive、U4_bilinear、U5、U3_delta、U3_delta_pairs、U4_decomposed、U5_reliable记为DEFERRED_SOURCE_PARAMETERS_UNFROZEN，没有启动。

缺少真实源三条件门的4个联合行无法达到身份响应激活验收，不能以域/辅助头有梯度替代；delta、关键类对、分解响应和可靠反馈还缺冻结源产物。启动时明确采用先可执行对照的方案，不伪造阈值，不把合成测试参数移入正式配置，不降低终态activation要求。

|行|实际机制|预登记对照|
|---|---|---|
|U0|原始loader，无交叉响应|普通训练参考|
|U1|V2完整块及角色轮换，无附加损失|块组织参考|
|U1_mask_off|相同块组织，原MixStyle mask|与U1隔离mask效应|
|U3|V2向量化决策校准，原始delta=0语义|与U1对照决策项|
|Ux|原始身份交互loss|与U1对照|
|Ux_normalized|归一化身份交互loss|与Ux对照尺度处理|
|head_only|隔离的辅助头optimizer/scaler，主训练不承受响应梯度|与U1检查主轨迹/资源|
|permanent_detach|响应更新域/辅助头，身份永久detach；保留本行决策项|描述域响应与决策组合，不作联合身份响应声明|

## 同前数据与训练配置

复用上一批core90_cross_response_s392005_20260911_r1数据配置：ManySig.pkl、equalized=1；source RX=1,3,4,6,8，source day=1,2,3；target RX=0,2,5,7,9,10,11，heldout day=0，named seen/unseen day合并保持原规则。模型seed和data_seed均392005，L_s/U_s/V=0.07/0.63/0.30，前次实际数量6300/56700/27000。输入不变，复用原数据验证结论；launcher的本次源smoke核对当前代码/随机checkpoint严格加载和源路径，不增加数据builder许可。

每行E200=label130+pseudo70、batch128、steps/epoch49；固定final_only，从零初始化，不读取任何baseline/resume/外部teacher checkpoint，EMA只来自本run随机student。source V单一且只读，U_s TX真值隐藏。完整baseline_args与前次逐字段比对，仅V2实现和声明的附加机制变化。最终clean与三LEO预测完整固定后独立连接truth评分，结果不反馈调参/重训/选择。该矩阵是单seed描述性对照，不作独立确认或晋级声明。

## 运行与资源

项目=/home/szu2070436088/2510044040/CV-SincNet；数据=Dataset_WigSig/ManySig.pkl；Python=/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python。使用本地已验证Git代码生成新的不可变release；实际代码commit及精确argv存publication.json。

run ID=core90_cross_response_v2_s392005_20260911_r1。输出runs/<run-id>，日志logs/<run-id>，dispatcher日志logs/<run-id>.dispatcher.log；所有输出必须新建。训练CWD固定release/code，不落入项目根旧代码。

18:44普通用户direct preflight通过。8张RTX3090，4个已有compute进程，显存余18–24GB/卡，RAM可用约486GB，磁盘余7.1TB。dispatcher总compute数上限2/GPU，最多新增2/GPU，启动尚未注册CUDA的所属进程计入容量与6500MiB预留；保留其他进程，资源变化则排队。

## 最小验证和停止规则

修复基线d2368f2c包含189通过/12环境跳过及真实FP32/AMP合成入口证据，本次仅针对dispatcher选行/容量和已选行做聚焦验证。独立P0/P1启动审查发现缺门联合行无法闭合，已以预登记延期处理，未改变activation语义；审查结论和定点验证随后补充。

launcher先真实源随机checkpoint保存/严格加载smoke，PASS后立即派发；没有额外smoke审批。源码归档只做一次本地/远端SHA比较和一次远端compile。启动后独立核对PID/PPID/CWD/完整argv/CUDA UUID/log增长。

低性能、负收益或未达到科学门槛不触发停机。协议/错split/错checkout、输出碰撞、不能执行或合法prediction不能闭合为技术失败；两个相同预测前异常停止后续派发，已健康运行行保留。无自动重启、不覆盖旧输出、不回退checkpoint；任何技术修复先定位后本地验证并使用新run ID。

最终每行必须有E200日志、final_ssdg.pth、实际activation、四场景prediction_manifest和独立scores；启动不等于实验完成。

独立审查已完成：8行build_matrix与runtime配置均通过，无新增P0/P1；未放宽激活或truth-last闭合要求。本次dispatcher/matrix/integration聚焦验证及数据配置逐字段比对见validation.json。
