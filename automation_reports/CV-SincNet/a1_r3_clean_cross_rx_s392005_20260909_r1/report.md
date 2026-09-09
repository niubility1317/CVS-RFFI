# R3参考版＋clean跨RX身份约束

## 目标与既有组合核查

用户于2026-09-09明确要求发布“R3参考版＋clean跨RX约束”。本轮只发布一个组合候选，不扩大到新的机制矩阵。此前source机制讨论指出，R3控制解码器的同物理样本clean/LEO一致性与同TX不同RX的判别几何有互补可能；性能是否改善仍未知。

发布前只读检查N607当前19份A1相关`runs/*/*/launch_config.json`，未发现同时配置`use_a1_r3=true`和`a1_ecrs_cross_rx_weight>0`的实验：

|已有组|R3结构及辅助目标|跨RX身份约束|是否本次组合|
|---|---|---|---|
|R3_REFERENCE/R3_FAST_SEQUENTIAL|结构开启，aux_scale=1|0|否|
|R3_STRUCTURE_CONTROL|结构开启，aux_scale=0|0|否|
|X1_CROSS_RX_CLEAN|关闭|clean，0.05|否|
|X2—X7|关闭|clean_leo，0.05|否；X7也不含R3|
|A1续训/V2/旧CORE90依赖链|关闭|0|否|

核查范围与逐行读回：`analysis/a1_r3_cross_rx_existing_audit.json`。这不宣称穷尽所有其他历史项目。

## 冻结方法和输入权限

run_id：`a1_r3_clean_cross_rx_s392005_20260909_r1`。
row：`R3_REFERENCE_CLEAN_CROSS_RX`，GPU1，seed392005，E200，L128/U256，最多每GPU两个训练进程。GPU1发布前空闲，磁盘可用7.1TB。

学生随机初始化，EMA来自本轮学生。baseline/resume/teacher历史权重均不加载，保留scratch guard与无锚点RC4。不是R3或X1现有checkpoint续训，不隐含继承额外训练预算。

严格继承`run_a1_scratch_no_checkpoint.scratch_matrix()`的R3_REFERENCE：M/lite_d/dual、160维最终身份表征、TX六类、输入256点；DAOT使用reference的legacy执行，R3 aux_scale=1。RC4身份项从E21开始，source V更新E1/E21/E41/E91/E161；R3原self/swap/shared日程不变。

唯一方法增量为：

`L_total = L_A1 + L_R3_original_schedule + 0.05 * L_cross_rx(z_f_id_clean)`

跨RX目标：同TX不同RX为正例，不同TX但同RX/day/view为负例；余弦距离、margin=0.2，全部合法triplet按anchor平均。只使用L_s TX标签，只对clean表示计算，不使用U_s真值，不增加LEO配对，不改变采样器、EMA或domain目标。接线为现有`out_l["z_id"]`，开启R3时即最终分类使用的`z_f_id`。复用已存在主干前向，不增加推理分支。

ManySig equalized；split=`tx_rx_day_1_7_2`、source RX=1,3,4,6,8/day=1,2,3；保留原target RX=0,2,5,7,9,10,11/day=0,1,2,3作为禁止训练集合。L/U/V=0.07/0.63/0.30，实际预期6300/56700/27000；实际物理ID按已验证划分复用，不重复builder验证。源域V只评估，不参与反传或持久状态更新。

## 评价范围

本次组合在原target结果已被查看后提出。依`项目.md`第4.2、4.4节，不将该target用于新组合调参或独立确认。本轮`final_evaluation=source_only`；训练内target评估保持关闭，launcher也不要求target manifest/truth文件，E200后只检查本行最终checkpoint，不调用target predictor/scorer。最终状态为`SOURCE_TRAINED_PENDING_ANALYSIS`，不能写成四场景测试闭合或正式干净泛化确认。

本轮可提供完整source训练/validation、机制激活、耗时、显存和数值稳定性证据。没有匹配、未查看的独立确认数据时不作新组合target性能声明。单seed不晋级默认。此前R3结果是历史参照，不替代共同训练条件下的消融。

## 实现与验证

现有训练入口已按独立条件累加R3和跨RX损失；不改骨干或loss数学。新增单行launcher和组合执行检查，共用dispatcher增加可选source_only收尾，原矩阵默认truth_last行为保持。

检查覆盖：真实parser相对R3_REFERENCE仅cross_rx_weight及输出身份变化；从零权限与target评估关闭；最终z_id与fingerprint.z_f_id为同一张量；跨RX梯度进入fingerprint及身份主干，不进入域主干/nuisance/decoder；R3 E1/E41/E65/E92奇偶更新/E161在FP32/AMP与跨RX共同反传；合成source真实train入口四次成功更新，R3的L_s/U_s目标及clean跨RX同时执行。执行检查不构成真实数据泛化证据。

聚焦测试和一次独立P0/P1审查结果在发布记录追加。

## 执行和产物

环境：N607 `/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`；项目根`/home/szu2070436088/2510044040/CV-SincNet`；CWD为本地Git提交生成的不可变release。

命令：`python -u code/scripts/run_a1_r3_cross_rx.py --project-root /home/szu2070436088/2510044040/CV-SincNet --run-id a1_r3_clean_cross_rx_s392005_20260909_r1 --detach`。

输出：`runs/<run-id>/R3_REFERENCE_CLEAN_CROSS_RX/`；日志：`logs/<run-id>/`。拒绝已有run/log root，detach日志独占创建，唯一launch owner为本任务主Agent。启动前dispatcher执行一次组合CUDA检查，PASS后自动训练，不设置额外审批。

技术停止：无法执行、参数/数据角色越权、输出碰撞、确定性异常/OOM或无法有效更新属于技术失败，保留现场，不自动重试；零星AMP梯度跳步沿用既有保护，不按低性能停止。无权修改其他任务。所有checkpoint、日志与输出保留。

预期产物：完整E200 CSV/stdout、final_ssdg.pth、source V指标、资源汇总、完成记录、source原型导出。新target prediction/score不属于本次source研发的预期产物。

## 发布状态

本地聚焦40项测试通过。组合CUDA执行检查PASS：12种阶段/精度组合，最终身份输出与z_f_id同一张量；跨RX独立梯度进入身份主干及fingerprint；实际source train入口4次成功AdamW更新，R3的L_s/U_s均执行，checkpoint_loads=0、target_inputs=0。证据：`analysis/a1_r3_cross_rx_local_execution.json`。

另用历史真实N607 R3_REFERENCE launch argv做全量parser对比：除run/output/candidate身份外，仅`a1_ecrs_cross_rx_weight`从0变为0.05。证据：`analysis/a1_r3_cross_rx_actual_reference_diff.json`。本地比较最初暴露的是Windows/Posix路径表示差异，改用PurePosixPath完成比较，不涉及方法配置修改。

独立P0/P1审查PASS，无阻断项。审查确认source_only不调用外部预测/评分，训练器内部最终target评估由既有`muse_external_final_eval=true`委托给launcher，也不会绕过此边界。非阻断事项：通用前序状态列表尚未包含SOURCE_TRAINED_PENDING_ANALYSIS，本次没有after_run或后继依赖，不影响执行。

## N607发布与首轮读回（2026-09-09 15:15）

状态：`RUNNING`；发布与启动核验：`VERIFIED`。代码commit=`8df3de21a69e01842d0c96d50975d715ec44eef4`，分支`codex/a1-fast-v2-20260909`，push后远端OID与本地一致。

release=`/home/szu2070436088/2510044040/CV-SincNet/releases/a1_r3_cross_rx_8df3de21`；归档35,990,967字节，SHA256=`404cd20b32175e965605e44e421f8d9be056e6fb390ac22eb4eb832ba384755e`。本地/远端一次比较一致，四个受影响入口远端编译通过。远端组合CUDA检查PASS，12种阶段/精度、真实source入口4次成功更新均通过。

dispatcher PID=327909；训练PID=328173，GPU1。实际argv与process.json一致，CWD为上述release，PPID等于dispatcher，CUDA_VISIBLE_DEVICES=1。训练日志确认`init=scratch`及L/U/V=6300/56700/27000；pipeline读回`final_evaluation=source_only`。未触碰其他run。

|首轮证据|值|
|---|---:|
|完成epoch|1/200|
|epoch耗时|120.25秒|
|有效optimizer更新|221/222|
|非有限loss跳步|0|
|R3 L_s/U_s加权辅助loss|28.448744/28.448279|
|跨RX原始/加权loss|0.201328/0.010066|
|跨RX有效anchor均值|127.864865|
|跨RX身份梯度范数（首batch探针）|0.002577071|
|跨RX clean/LEO样本数|128/0|

首batch的`id_backbone.sinc.low_hz_`出现一次非有限梯度，loss有限，既有AMP保护跳过更新并将scale降至32768，后续221步成功；异常包保留。不能把有效运行写为零数值异常。跨RX成功步日志110.5是累计计数的epoch均值，不是最终成功步数。

启动日志从6374字节增长到12058字节，进程仍存活。新组合R3 E1仅self阶段有效，swap/shared按原日程后续开启；本次首轮证据不提前声称它们已在正式数据上激活。无E200结果、target预测或性能改进结论。

证据：`analysis/a1_r3_cross_rx_release.json`、`analysis/a1_r3_cross_rx_remote_startup.json`、`analysis/a1_r3_cross_rx_remote_progress_e1.json`。本地合成夹具与完整检查日志保留于analysis/a1_r3_cross_rx_local_execution/，不纳入正式数据结果。
