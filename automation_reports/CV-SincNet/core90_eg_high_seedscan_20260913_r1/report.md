# 完整EG与高学习率组合seed扫描

用户明确授权“将两者组合，发布seed扫描实验”。本次冻结6行：完整EG、lr=0.0004，adv=0及0.35各配seed392005/392006/392007。与上一轮V2 E/F对应seed仅修改lr、run_name与output_dir；不增加其他超参搜索或训练方法。唯一launch owner=/root。

## 科学权限与输入

本组合是在研究者查看既有目标基准结果后提出，标记SOURCE_ONLY_FOLLOWUP_AFTER_BENCHMARK_EXPOSURE，不声称独立盲测确认。训练代码不读取target预测或truth，训练结束仅生成source V四场景预测评分；本次不自动执行新的target评分，不用性能决定停止或改配。后续目标结果只能按已接触基准的探索复测解释。

六行均from_scratch=true、baseline_ckpt=""、game_resume=""，无外部teacher或EMA继承，teacher从本run fresh model创建。E200 final_only，不复用任何历史checkpoint。启动smoke只保存/读回当场随机初始化模型，使用真实source L_s/U_s，非历史权重或query。

source数据契约沿用V2：ManySig.pkl，RX1/3/4/6/8、day1/2/3、六TX，L_s/U_s/V=6300/56700/27000；split_seed392002，物理角色与schema不变，复用已有验证。AdamW、FP32、确定性开启、固定课程、control=off、game_no_audit=true、130+70=200epochs；game_solver=extragradient、reference实现。更新预算与对应V2 E/F匹配，但与普通更新比较须报告额外梯度成本，不能称等计算量。

## 发布与运行

代码以本地Git提交固定并push读回后发布。配置目录code/configs/core90_eg_high_seedscan_20260913_r1；6份持久化JSON通过实际parser核对，4项调度容量测试PASS。独立P0/P1限定审查按最小流程执行。首次本地校验缺少parser必需的--output_dir，已修正校验调用，实际配置/训练argv未受影响；没有远端失败启动。

N607普通用户szu2070436088，Python=/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python，RTX3090。CWD/release=/home/szu2070436088/2510044040/CV-SincNet/releases/core90_eg_high_seedscan_20260913_r1；输出=同项目runs/core90_eg_high_seedscan_20260913_r1/<row>；日志及queue_status.json=同项目logs/core90_eg_high_seedscan_20260913_r1。

命令：`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python -u code/scripts/dispatch_core90_game.py --matrix code/configs/core90_eg_high_seedscan_20260913_r1/matrix.json --release /home/szu2070436088/2510044040/CV-SincNet/releases/core90_eg_high_seedscan_20260913_r1 --status /home/szu2070436088/2510044040/CV-SincNet/logs/core90_eg_high_seedscan_20260913_r1/queue_status.json --max-processes-per-gpu 1`。

以全服务器现有GPU计算进程为容量，每卡最多1个计算进程，用空卡均匀铺开，等待其他任务自然结束，不干预任何健康进程。preflight时GPU0/1/4有任务，预计GPU2/3/5/6/7先各1行，第6行排队；实际以启动后UUID/PID读回为准。worker OMP/MKL线程各4。

launcher先执行真实source fresh-checkpoint smoke及E1/E80/E131的九次solver更新，覆盖真实增强器与既有确定性cumsum兼容路径。PASS后自动进入训练，不需要二次许可。

技术规则：来源/输入权限错误、输出碰撞或smoke失败停止发布；两个pre-prediction训练失败停止新派发，保留pending及已启动健康任务。无自动重试，不因低分停止。新run路径排他创建，全部旧checkpoint、日志和输出保留。

预期每行：resolved_config.json、backend_configuration.json、全epoch/actions日志、final_ssdg.pth、source_final_eval四场景预测与scores、completion.json。仅exit0且SOURCE_ARTIFACTS_COMPLETE记row完成。报告clean/LEO Accuracy、Macro-F1、RX/TX floor、三seed均值与SD、对旧E/F配对变化和实际训练成本；不自动选模或科学晋级。

当前状态：LOCAL_VERIFIED，实际发布commit、GPU/PID及日志增长在启动后追加。

## 实际发布后态（2026-09-13 11:01 CST）

状态：RUNNING / VERIFIED，5行训练、1行排队、0失败。发布commit=2d472c23807b5930de4c26b03441a69d1af3803f，GitHub OID独立读回一致；归档SHA256=4b270902b80e3c8bda473d99b7f6e73d27c4bed145d27abf4193b5972c70aaeb，一次本地/远端比对一致，远端编译PASS。独立P0/P1审查通过，发现的非阻断旧矩阵元数据已在发布前修正。

调度PID3521900，真实source fresh-checkpoint smoke及九次阶段/solver更新全部PASS，无query访问。已启动worker均核实CWD、argv、PPID、GPU UUID、resolved_config中的lr=0.0004、extragradient、scratch及空继承路径；日志显示adamw_isolated_predictor_raw_gradient_extragradient。GPU0/1/4原有任务继续。

|模型|seed|GPU|PID|
|---|---|---|---|
|ADV0|392005|2|3522040|
|ADV0|392006|3|3522041|
|ADV0|392007|5|3522042|
|ADV035|392005|6|3522043|
|ADV035|392006|7|3522044|
|ADV035|392007|待空卡|排队|

首个读回每行29—30次动作，第二次读回增至40—42次，确认持续增长，见launch_snapshot.json及growth_snapshot.json。这里只确认训练已启动，尚无E200最终成绩；第6行由调度器等待空卡后自动启动。
