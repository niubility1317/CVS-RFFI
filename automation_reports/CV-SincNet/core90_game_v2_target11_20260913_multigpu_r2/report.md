# V2剩余11模型七卡并行测试

状态：LOCAL_VERIFIED，尚未启动。用户要求将11模型均匀分配GPU，替代单卡run core90_game_v2_target11_20260912_r1。旧进程3244844在核实PID/CWD/argv后定点停止，全部partial产物保留；不删除、覆盖或合并旧预测，不改权重、不训练、不依据测试分数选择。

沿用code/configs/core90_game_v2_target11_20260912_r1.json的完整11行final E200及原预登记数据契约：D两seed、E/F各三seed、三个strong LR对照。scratch_only/final_only/target_contact=false及source completion检查保持。训练release=5bd1665631b15b1ed97fae0f6b0ed57f25b68ec9。目标RX0/2/5/7/9/10/11、day0/1/2/3、168000样本/场景，四场景672000/模型，共7392000预测。batch256、augmentation_seed392002和全部数据/模型设置不变；previously_exposed_benchmark_recheck，无拟合、选模或新盲测声明。

新增--devices，11模型按清单顺序轮询7卡，物理GPU顺序5/0/1/2/3/6/7，负载2/2/2/2/1/1/1。GPU4其他任务保留。单进程7个GPU线程；主线程仍在物理GPU5生成每批唯一增强视图，显式同步后复制到各卡，组内串行、组间并行。每线程进入inference_mode；所有预测按原清单顺序写出。全部11模型四场景预测关闭、计数和buffer不变检查后才建立truth并统一artifact评分。

启动source checkpoint smoke额外比较同卡串行与并发推理的预测及置信度完全一致，不用query验证。聚焦测试覆盖均匀分配、线程inference_mode、输入不变及非有限输出拒绝，保留原7项协议测试。一次独立P0/P1审查。新目录全量推理，原部分预测不作为完成数据。

环境：N607普通用户szu2070436088，Python=/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python，PyTorch2.1.0+cu121，唯一launch owner=/root。

CWD=/home/szu2070436088/2510044040/CV-SincNet/releases/core90_game_v2_target11_20260913_multigpu_r2；输出=同项目runs/core90_game_v2_target11_20260913_multigpu_r2；日志=同项目logs/core90_game_v2_target11_20260913_multigpu_r2/eval.stdout.log。

命令：`CUDA_VISIBLE_DEVICES=5,0,1,2,3,6,7 python -u code/scripts/core90_game_target_test.py --input-manifest code/configs/core90_game_v2_target11_20260912_r1.json --output-dir /home/szu2070436088/2510044040/CV-SincNet/runs/core90_game_v2_target11_20260913_multigpu_r2 --device cuda:0 --devices cuda:0 cuda:1 cuda:2 cuda:3 cuda:4 cuda:5 cuda:6`。实际发布用UUID列表固定映射，OMP_NUM_THREADS=1、MKL_NUM_THREADS=1以避免7线程CPU过度订阅。

失败规则：来源/契约错误、非有限logits、串并行smoke不一致、buffer变化、预测不完整或输出已存在立即失败并保留产物，不盲目重试，不以低分停机。预期frozen_manifest含model_devices、11份预测、predictions_complete、11份target_scores及complete.json；完整闭合后才报告分数。
