# V2剩余11模型目标测试预登记

状态：LOCAL_VERIFIED，尚未启动。用户明确授权补测尚无测试集成绩的11个模型；不重复已完成target10。

- 固定矩阵：D×392006/392007，E/F×392005/392006/392007，STRONG_SOURCE_LR_LOW/BASE/HIGH×392005。全部使用core90_game_v2_20260911_r3的final E200，无重新训练或选模。
- 来源：训练release 5bd1665631b15b1ed97fae0f6b0ed57f25b68ec9；沿用已完成21模型source核验。加载前逐模型要求completion、E200、scratch_only、final_only、target_contact=false、对应seed及一致source数据契约，禁止不明继承或替换checkpoint。
- 科学口径：previously_exposed_benchmark_recheck，6类闭集、逐样本argmax、无support适配或拟合。研究者历史目标集接触已披露；不称全新盲测，不以本次分数回流调参、选择或重跑。
- 数据与前10模型相同：ManySig.pkl，target RX=0/2/5/7/9/10/11，day=0/1/2/3，source RX=1/3/4/6/8；复用已有数据验证。out_len=256，center crop、normalize、equalized=1，batch=256，augmentation_seed=392002。
- 场景：clean、leo_clear_weak、leo_low_elev_weak、leo_rain_weak。每场景168000样本，每模型672000预测，11模型共7392000条。每batch共享同一增强视图。
- 先执行全部11模型source checkpoint smoke，再冻结推理。全部11模型四场景prediction关闭并验证计数与buffer不变后，才写truth并由独立artifact scorer统一评分。预期frozen_manifest.json、predictions_complete.json、target_truth.jsonl、各模型target_scores.json和complete.json。
- 既有runner保持不变。一次独立P0/P1审查通过：strong ID及异lr可加载、契约一致、同视图和truth-last顺序成立。新manifest由21矩阵减已测10自动生成并断言11行。
- 环境：N607普通账户szu2070436088，Python=/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python，PyTorch2.1.0+cu121。唯一launch owner=/root。
- CWD=/home/szu2070436088/2510044040/CV-SincNet/releases/core90_game_v2_target11_20260912_r1；输出=/home/szu2070436088/2510044040/CV-SincNet/runs/core90_game_v2_target11_20260912_r1；日志=同项目logs/core90_game_v2_target11_20260912_r1/eval.stdout.log。
- GPU预定5，UUID=GPU-ef5f59f2-e03e-fe6e-ea6b-3e03afc9b331；发布前再次检查空闲，不干预其他进程，每GPU最多2个计算进程。
- 命令：`CUDA_VISIBLE_DEVICES=GPU-ef5f59f2-e03e-fe6e-ea6b-3e03afc9b331 /home/szu2070436088/.conda/envs/CVS-RFFI/bin/python -u code/scripts/core90_game_target_test.py --input-manifest code/configs/core90_game_v2_target11_20260912_r1.json --output-dir /home/szu2070436088/2510044040/CV-SincNet/runs/core90_game_v2_target11_20260912_r1 --device cuda:0`。
- 技术停止：来源/契约不符、输出已存在、维度错误、非有限logits、预测不完整或buffer变化立即报错并保留产物，不自动重试；低分不停止。

实际发布commit、归档校验及启动后态在本报告追加。

## 启动后态（2026-09-13 00:02 CST）

状态：RUNNING / VERIFIED，测试尚未完成。7项聚焦测试通过；发布commit=e7ba210359f5e19852cc6658eb4a47b3b71a26a9，远端分支OID独立读回一致。归档SHA256=0e79c74c28aba570aedb8ea37591c60baccb29d6fe9843de7ac415a993d43820，本地/远端一次比对一致，远端编译PASS。

PID=3244844，PPID=1，CWD/argv与预登记一致，实际GPU5 UUID绑定正确（708MiB）。日志出现SOURCE_CHECKPOINT_SMOKE_PASS及clean预测进度；frozen_manifest核实11模型、168000样本/场景。11份prediction文件连续两次读回均增长，尚无complete.json。未干预其他任务。结束后runner自动执行完整预测检查及独立评分；此启动记录不代表测试成绩。

## 用户要求调整GPU分配

2026-09-13 00:30 CST按用户要求停止单卡PID3244844（独立核实退出），所有partial预测保留，未生成最终评分。替代run为core90_game_v2_target11_20260913_multigpu_r2，11模型均匀放置七卡并行推理，见相邻run报告。原RUNNING记录为历史状态。
