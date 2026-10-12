# E20 旧门控＋cos：历史 ADV3B02 划分、同 seed392002


- run_id：`20261012-phase1-e20-historical-adv3b02-manysig-s392002-r02`
- group_id：`cvs-historical-adv3b02-network-comparison`；类别：`cvs`；阶段：`Phase1`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：ANALYZED / VERIFIED（训练、预测、独立评分与分析完成）。experiment.json 保留发布前登记，生命周期以 events.jsonl 和完成证据为准。

## 目的与对照

当前纯CE网络从零训练20轮，严格使用历史ADV3B02物理划分，自动三个clean测试集合

历史候选为 `ADV3B02_CORE90_SOFT_E200`，2026-07-01 queue 的 seed 为392002；本轮同 seed392002，单行 `E20-s392002`。当前身份网络为 reference_response，保留内部门控、参考响应结构与 scale30 cosine 分类头；AdamW lr2e-4至1e-6，weight_decay1e-4，完整20轮cosine日程，batch128，每轮222次更新，共4440次。纯CE、label_smoothing=0；无U迭代、伪标签门控、EMA、额外loss或星地训练增强。固定 final20，不用V或target选择轮次。

历史数据配置：ManySig equalized1，全部6个TX、源RX0至6、源day0/1；每个TX/RX/day/eq内sig_i排序，前100条L、随后700条U、最后200条V，L/U/V=8400/58800/16800。历史训练池标注率12.5%，这是本次用户指定的历史例外，不能标成当前10%标注协议。采用专用局部split实现，仅对本run生效，不修改当前协议的通用splitter。输入尺寸256，中心裁剪/补零，逐packet复数RMS归一化，center=False，25MHz。

数据契约改变，当前E20及历史ADV3B02 checkpoint均不作为初始化；从零训练、祖先为空。launcher首先严格重载本row刚保存的真实scratch checkpoint并完成无query前向smoke，PASS后继续。源角色由源专用preflight逐物理ID固定；源训练不构造目标loader。

测试范围由用户补充固定为clean，不追加卫星视图。三个互斥集合共204000条：unseen_day_seen_rx=84000（RX0至6/day2,3），seen_day_unseen_rx=60000（RX7至11/day0,1），unseen_day_unseen_rx=60000（RX7至11/day2,3）。源冻结后独立builder输出IQ/opaque ID capsule和单独truth；predict进程只读取前者。全部204000条预测、log概率和完成标记固定后，单独score进程连接truth，报告每集合accuracy、Macro-F1、NLL、混淆矩阵及RX/day/TX分层。

与历史ADV3B02 E200比较时，差异同时包含架构、损失、星地增强及训练预算；结果是同数据/同模型seed的实际性能比较，不是架构单因素消融。历史基准已暴露，不声称新独立盲测；无目标反馈调参、候选重排、选择性重跑。

## 数据、seed与模型来源

实际数据契约、权限例外、完整seed角色、checkpoint来源和选择规则见experiment.json。
逐行配置通过config_ref/resolved_config_ref定位；待补项必须在对应生命周期补齐。

## 执行与存储

命令、环境、CWD、commit、launch owner、输出和日志路径见experiment.json。
实际PID/GPU、读取时间、remote readback、失败或替代关系在此追加，并用record命令记录证据指针。

代码/config身份为本次Git提交，immutable release中 `release_commit.txt` 和远端登记写入同一OID。执行命令：`python -m experiments.cvs_adv3b02_e20.publish --output local_artifacts/cvs_e20_historical_adv3b02_20261012_r02`。远端CWD为该release；普通N607账户，现有CVS-RFFI Python，无安装或管理员操作。唯一owner为本run登记owner。按全部owner计数选择GPU，每卡最多3个进程且至少12000MB空闲；不停止、重启或修改其他任务。每个stage单独进程，顺序train→build→predict→score。

技术停止规则：本row协议越界、错误物理ID/seed/checkout、输出碰撞、非有限loss/gradient、无法完成预测或评分、stage非零退出即记录failure并停止后续stage，保留全部产物，不自动重试，不因低性能停机。readback核对dispatcher和stage的PID/CWD/argv/GPU及日志增长；通信不明先查原run，禁止重复提交。

本地聚焦检查：PASS，历史角色数、sig_i边界、U标签隐藏（远端数据preflight）、source/query配置负测、重叠物理ID负测、真实网络CE梯度、222更新循环、辅助对象关闭、metric算例。独立 P0/P1 审查已通过；无报告审查、签名或额外seal门槛。

独立 P0/P1 审查：PASS。已核对14源域与独立222更新预算、source-only及U隐藏、无继承、204000个源/测试互斥物理ID、预测完成后单独truth-last scorer、所有owner计入GPU容量、不可覆盖输出与发布依赖。未发现阻止本次合法run的P0/P1；不追加完整复审。

## 结果与覆盖

全部预登记 clean 集合完成，结果见下方“完整 clean 结果”。源训练完成标记、204000 条预测完成标记、独立评分完成标记及远端 completion.json 相互对应，无缺项。

## 交接

记录已完成、当前run/commit、证据路径、阻塞与下一步；恢复先查原run，不重复启动。

技术恢复：r01在训练前因Windows登记远端路径反斜杠与Linux config不同而停止；preflight已PASS、没有训练或预测。r02统一使用POSIX路径文本并完成本地回归。保留原r01目录，不重启或覆盖；seed392002、历史角色、E20/4440预算和clean三集合不变。仅该路径问题做一次定点复审。

路径问题唯一一次定点复审：PASS。SOURCE/CAPSULE/TRUTH/output_root及登记row config全部使用同一POSIX表示；dispatcher比较字段没有跨平台差异。无新增P0/P1，不做完整复审。

## 完整 clean 结果（VERIFIED）

执行提交：`046c0ce3345ec4434b968b79b8994979494b0406`。seed392002、历史物理划分、scratch E20和4440次成功更新均已核实。训练、builder、204000条预测和独立评分全部完成，无缺项。

|测试集合|样本数|历史 ADV3B02 E200 (%)|当前 E20 (%)|差值（百分点）|当前 Macro-F1 (%)|
|---|---:|---:|---:|---:|---:|
|unseen_day_seen_rx|84000|93.4917|92.8202|-0.6714|92.8382|
|seen_day_unseen_rx|60000|87.4483|91.2200|+3.7717|91.1434|
|unseen_day_unseen_rx|60000|84.8900|87.6067|+2.7167|87.6467|

本轮训练耗时567.10秒；clean纯推理耗时5.62秒。实际日志成功更新4440次；本轮只测clean，不作卫星部署结果声明。
历史参考来自原run的metrics_epoch.jsonl第200轮，不使用best_joint_safe挑出的最佳分数；原checkpoint元数据已核实seed392002、RX/day、equalized1和10/70/20。历史训练启用了joint_safe及测试监测，存在目标驱动保护/选模背景，因此只作已暴露历史参考，不作为独立干净泛化证明。
当前与历史同时改变架构、损失、半监督/信道增强和训练成本；单seed差值不能单独证明网络结构贡献或三seed稳定收益。

完整RX/day/TX分层、混淆矩阵和NLL保存在本run的独立scores/metrics.json及本地final_results.json；原始checkpoint、IQ、prediction和训练日志保留远端，不进入Git。
