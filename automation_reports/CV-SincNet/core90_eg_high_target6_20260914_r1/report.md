# 完整EG＋高学习率六模型目标测试

用户明确授权“进行测试集评估”。固定source run core90_eg_high_seedscan_20260913_r1六模型final E200：adv0/.35各seed392005/392006/392007，lr=0.0004。无新增训练、适配、选模或选择性重跑。训练release=2d472c23807b5930de4c26b03441a69d1af3803f。

科学口径：这组组合在研究者查看历史目标基准后提出，目标复测是previously_exposed_benchmark_recheck，不是独立盲测确认或自动晋级。训练模型来源仍须逐个检查SOURCE_ARTIFACTS_COMPLETE、E200、scratch_only、final_only、target_contact=false、seed及完整source角色契约一致。无外部checkpoint继承，不以研究者历史接触替代模型训练接触核查。

测试契约与上一轮V2一致：ManySig equalized=1、out_len256、center crop、normalize；target RX0/2/5/7/9/10/11、day0/1/2/3、六类，source RX互斥；复用既有物理ID和验证。每场景168000，clean及leo_clear_weak/leo_low_elev_weak/leo_rain_weak，672000/模型，总4032000预测。batch256、augmentation_seed392002。

沿用已验证多GPU runner，不改推理代码。物理GPU5/1/2/3/7，模型轮询分配2/1/1/1/1；物理GPU0/4/6现有任务保留。增强仍在物理GPU5生成一次后同步复制到各worker，所有模型同批同视图。单进程线程并发，线程各自inference_mode，主线程统一写预测。首先真实source checkpoint smoke，核对串并行输出一致；全部六模型四场景预测关闭、计数完整且buffer不变之后才连接truth统一artifact评分。

唯一launch owner=/root。N607普通用户szu2070436088，Python=/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python，PyTorch2.1.0+cu121。CWD/release=/home/szu2070436088/2510044040/CV-SincNet/releases/core90_eg_high_target6_20260914_r1；输出=同项目runs/core90_eg_high_target6_20260914_r1；日志=同项目logs/core90_eg_high_target6_20260914_r1/eval.stdout.log。

命令：`CUDA_VISIBLE_DEVICES=5,1,2,3,7 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -u code/scripts/core90_game_target_test.py --input-manifest code/configs/core90_eg_high_target6_20260914_r1.json --output-dir /home/szu2070436088/2510044040/CV-SincNet/runs/core90_eg_high_target6_20260914_r1 --device cuda:0 --devices cuda:0 cuda:1 cuda:2 cuda:3 cuda:4`。实际发布以UUID绑定各卡。

本地清单核对六行完整；复用九项目标协议/线程测试，独立P0/P1限定审查。发布commit先push并核对OID，唯一归档一次SHA比较及远端compile。技术失败：来源/契约错误、非有限或维度错误、输出已存在、smoke不一致、buffer变化或预测不完整直接报错并保留产物；不自动重试，不以低分停止。

预期frozen_manifest.json、六份predictions、predictions_complete.json、target_truth.jsonl、六份target_scores.json及complete.json。完整闭合后才报告分数。当前LOCAL_VERIFIED，启动后态随后追加。

## 实际启动后态

RUNNING / VERIFIED。发布commit=950a67e86db82514968774fde3842282b169e2a5，GitHub独立OID读回一致；此前source报告的未推送提交同步完成。Git默认代理路径TLS失败，单次调用禁用代理并使用openssl后恢复，未修改全局配置。

归档SHA256=386f49a843bec8e661067a814a5c549e0dd347fb5174783081069165519591e5，本地/远端一次比对一致，远端编译PASS；九项聚焦测试和独立P0/P1审查通过。源训练六份completion均为SOURCE_ARTIFACTS_COMPLETE、E200、9800步、target_evaluated=false。

PID373822、PPID1，CWD/argv绑定本release；nvidia-smi核实物理GPU1/2/3/5/7，其他GPU0/4/6的原PID仍在。六checkpoint均通过来源检查及串并行source smoke，frozen_manifest确认6模型/5设备/168000每场景。首次六文件各约9700条，第二次全部增长，尚无complete。全量预测结束后自动统一评分，本记录不代表已有最终成绩。
