# A1结合ECRS及教师、采样和有界域混淆的八行实验

状态：LOCAL_VERIFIED，待Git固定与远端启动。用户授权在A1加速版中加入ECRS相关机制、结合其他机制，并明确每卡两个训练进程。所有新增实验从头训练，不使用其他checkpoint。

## 科学范围与预登记矩阵

实际设计及完整追溯见[适配说明](../../../docs/A1_ECRS_CROSS_RX_ADAPTATION_20260909.md)。已读取引用任务“制定ADV3B02-ECRS落地计划”及原规范。本轮移植的是跨RX判别数学，直接更新A1预测所用`z_id`，不是完整物理响应/固定融合模块；没有已证实准确率收益的声明。

|行|GPU|相对X2的变化|
|---|---:|---|
|X0_A1_RUNTIME|0|关闭跨RX，作为新匹配控制|
|X1_CROSS_RX_CLEAN|1|只用clean进行跨RX约束|
|X2_CROSS_RX_VIEWS|2|clean+实际LEO视图跨RX约束|
|X3_WARM_EMA|3|加EMA启动期平均|
|X4_COVERAGE_KL|4|加归一化后共识覆盖率加权|
|X5_BALANCED_L|5|加4TX×4RX/day×8的L平衡采样|
|X6_BOUNDED_DOMAIN|6|加有界域混淆及梯度隔离|
|X7_COMBINED|7|预登记组合X3/X4/X5/X6；不作单因素归因|

每卡新增一个进程，保留GPU0 LoRa、GPU1—3 R3、GPU4—7先前A1-V2。最大每卡两个；没有停止、改版续训或覆盖旧run。并发影响单run速度，需同时记录GPU共享负载，不能把共卡时间差直接归因于算法。

所有行M/lite_d/dual、160维、seed392005、E200、L128/U256、final-only；`from_scratch=true/a1_scratch_only=true`、baseline/teacher路径空。EMA由本轮随机学生在内存中建立。源数据契约不变：ManySig equalized=1，source RX=1,3,4,6,8/day=1,2,3；L/U/V=6300/56700/27000。单一V只读；U TX真值不参与新目标或采样。

跨RX权重0.05、margin0.2来自ECRS提案，不由target选择。正例同TX异RX，负例异TX同RX/day/view；按合法triplet、再按anchor平均。空集合断图，日志明确configured/executed/valid_count。已有LEO学生前向没执行时，只用clean，不制造副本或额外主干前向。新约束从E1开始；LEO部分遵循现有课程，实际调用再计数。

X5/X7保持batch128但改变L采样轨迹；采样器直接检查各TX/domain cell非空以及实际batch等于预登记值，避免空批循环。replacement会重复抽样，不能声称同样本顺序或无放回覆盖。X6/X7中L权重是`cur_w[adv]×(0.5×disc+0.5×confusion)`，U分别是`0.16×disc+0.16×confusion`。只有身份混淆KL有界，判别CE和域骨干CE仍无此上界；域头Dropout会改变随机数轨迹。

## 版本、位置与启动

- 分支：`codex/a1-fast-v2-20260909`；本轮实现基于已发布`89758ce1`，实际新commit在落地后追加。
- 项目：`/home/szu2070436088/2510044040/CV-SincNet`。
- 环境：`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`，Torch2.1+cu121。
- CWD：新建不可变release。
- 启动命令：`python -u <release>/code/scripts/run_a1_ecrs_cross_rx.py --project-root /home/szu2070436088/2510044040/CV-SincNet --run-id a1_ecrs_cross_rx_s392005_20260909_r1 --detach`。
- outputs：`<project>/runs/a1_ecrs_cross_rx_s392005_20260909_r1/<row>`。
- logs：`<project>/logs/a1_ecrs_cross_rx_s392005_20260909_r1`。
- 完整argv和PID：根`effective_matrix.json/pipeline_state.json`、每行`launch_config.json/process.json`。
- 唯一launch owner：本任务主Agent及其单一dispatcher。

## 直接验证与已知边界

本机`ssr-gpu`完成35项不同聚焦检查：crossRX数值/梯度/空集/metadata/LEO20项、八行配置及dispatcher6项、原V2矩阵1项、有界域混淆7项、平衡sampler1项。CPU/CUDA、AMP边界、字面triplet梯度、零权重、已有AdamW动量下空集不更新均通过。

`check_a1_ecrs_cross_rx.py`验证8行模型/heads/CPU RNG初始化完全一致；真实lite_d主干下8种scope/applied/AMP组合均有有限非零身份梯度，无domain骨干梯度。实际X7训练入口使用合成source数据经过真实source loader，4次成功optimizer更新、5次有效跨RX调用，每次128个合法anchor；未读checkpoint或target观测。目标loader builder替换为空映射，测试不包含最终target评估，不是正式训练结果。

第一次本地检查因原数据builder不接受“零target测试集合”失败，失败层是合成测试脚手架。保留该输出，修正为显式空target builder后PASS；未改变正式数据或训练权限。独立P0/P1审查PASS。远端由dispatcher先进行同样的fresh/source-only执行检查，再启动八行。

`--detach`通过实际Popen参数测试，使用独立session、release CWD和专属日志；重复日志拒绝二次dispatch。远端是否真正启动仍以独立PID/CWD/argv/GPU及日志读回为准。

## 完成、评分与停止规则

每行E200后只加载本行`final_ssdg.pth`，复用已验证opaque目标输入包。prediction完成后由独立scorer连接truth，clean及三个LEO weak场景完整才标记ARTIFACTS_COMPLETE。source日志记录新损失raw/weighted、合法anchor、实际LEO数量、身份梯度诊断与成功更新次数。单seed结果仅描述同row性能、弱RX、耗时、显存和稳定性，不晋级默认、不从target回流调参/重跑/候选重排。

低性能不停止。仅所属run发生协议越界、错split/stage/release、输出碰撞、无法执行、prediction或scorer无法合法闭合等技术故障时按预登记边界处理；保留所有产物，禁止影响其他任务。

## N607启动及首轮核验：RUNNING/VERIFIED

代码commit`def66f05eb87601c48441b156204a6263027e3eb`已push并独立核对远端OID。release=`releases/a1_ecrs_def66f05`，本地/远端归档SHA256一致：`0bd44828c2ecf8e9e8429b64cf498c6e53cc0be8a0d70a2970bde9caa4c64eb6`。远端编译及完整fresh/source-only执行检查PASS，实际组合入口完成4次成功更新后按测试约定退出，无checkpoint/target输入。

dispatcher PID4187454。独立读取`/proc`核对新八行PID、父PID、实际argv、CWD及`CUDA_VISIBLE_DEVICES`，与预登记一致；八卡实际compute PID数各为2，总16个，旧8项均保留。

|行|GPU|训练PID|首轮状态|跨RX weighted loss|身份梯度探针范数|
|---|---:|---:|---|---:|---:|
|X0_A1_RUNTIME|0|4187605|E1完成|关闭|N/A|
|X1_CROSS_RX_CLEAN|1|4187610|E1完成|0.01007015|0.00047777|
|X2_CROSS_RX_VIEWS|2|4187615|E1完成|0.01007003|0.00047778|
|X3_WARM_EMA|3|4187620|E1完成|0.01006976|0.00047777|
|X4_COVERAGE_KL|4|4187625|E1完成|0.01006988|0.00047778|
|X5_BALANCED_L|5|4187693|E1完成|0.01005605|0.00036892|
|X6_BOUNDED_DOMAIN|6|4187761|E1完成|0.01004057|0.00047778|
|X7_COMBINED|7|4187834|E1完成|0.01006067|0.00036891|

上述为服务器2026-09-09 02:10左右读回。八行均`init=scratch`，实际L/U/V=6300/56700/27000；首轮成功更新率99.5495%，loss非有限跳步为0，gradient跳步比例0.45045%，首批AMP异常诊断已保留。不宣称“全程无跳步”。此后健康训练继续，不因初期指标停止或改参。

X5/X7实际检查到90个非空L cell、6TX、15RX/day、batch128，首轮每批128个合法anchor；其余启用行平均约127.941个。X3/X7首轮EMA有效decay均值约0.971053，其他行0.99，启动平均实际改变教师更新。成功步计数在epoch日志中经过均值归约，110.5不是epoch末的更新次数，不作错误解读。

E1尚未执行单独LEO学生前向，所以所有行`leo_count=0`符合固定课程。LEO视图约束的正式激活证据须等待该路径实际启用；当前只证明clean跨RX已进入身份梯度，不把配置开关等同于后续课程完成。coverage蒸馏的实际非零贡献同样须在其原DAOT课程展开后分析。

首轮log均从启动输出增长为完整epoch记录。完整证据：`analysis/a1_ecrs_remote_startup.json`、`analysis/a1_ecrs_remote_first_epoch.json`；包含原始实际参数、进程、GPU、运行检查和机制字段。最高剩余科学风险仍是准确率及综合资源收益未证实。追溯9项verified、1项deferred（物理响应/融合）、0项blocked；这是有明确范围的ECRS判别适配，非完整物理响应设计复现。
