# V2已完成10模型目标域测试

用户要求“要的是目标域测试集结果”。此前V2只完成source评分，本次固定当前已完成的A/B/C×392005/392006/392007及D×392005，共10个E200 final checkpoint，统一进行目标域测试。

## 测试预登记

- 科学口径：6类闭集、逐样本全部注册类argmax，无support适配、无梯度、无阈值拟合、无未知类或新类注册声明。既有目标集已被历史CORE90研究观察，本次为previously_exposed_benchmark_recheck，不称全新未见确认集；测试结果不用于本批训练调参或选择。
- checkpoint：来自core90_game_v2_20260911_r3，训练代码5bd1665631b15b1ed97fae0f6b0ed57f25b68ec9。每个加载前检查对应seed、E200、scratch_only、final_only、target_contact=false、source合同一致以及completion。训练接触字段不替代研究者目标接触历史说明。
- 数据：ManySig.pkl；target RX=0/2/5/7/9/10/11，与source RX=1/3/4/6/8互斥；day=0/1/2/3。每场景168000，四场景672000/模型，共6720000条预测。沿用既有测试IQ/物理ID/预处理，不改变数据契约，不新建数据验证链。
- 固定设置：out_len=256、center crop、normalize、equalized=1、batch=256、augmentation_seed=392002，clean/leo_clear_weak/leo_low_elev_weak/leo_rain_weak。多seed模型面对同一IQ及同一信道视图。
- 预测/评分：全部模型构建并冻结后先做source checkpoint smoke；所有10行四场景prediction关闭、完整性检查和buffer不变检查之后，才写target_truth并由独立artifact scorer评分。无跨样本配额、真实类数推断或全局重排。
- 本地修复：既有runner新增显式多seed输入manifest，保留旧15行接口；对应seed、重复模型、未完成checkpoint负测与原目标输入隔离测试共7项PASS。独立P0/P1审查通过。
- 环境：N607普通账户szu2070436088，PyTorch2.1.0+cu121，Python=/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python。确定性配置在CUDA模型构建前应用，使用已验证的cumsum兼容函数。
- 新release/CWD：/home/szu2070436088/2510044040/CV-SincNet/releases/core90_game_v2_target10_20260912_r1；输出root：同项目runs/core90_game_v2_target10_20260912_r1；日志：logs/core90_game_v2_target10_20260912_r1/eval.stdout.log。
- 唯一launch owner：本任务/root。使用一个空余GPU进程名额，启动前检查每卡总计算进程≤2，保留全部健康训练。
- 命令：`python -u code/scripts/core90_game_target_test.py --input-manifest code/configs/core90_game_v2_target10_20260912_r1.json --output-dir /home/szu2070436088/2510044040/CV-SincNet/runs/core90_game_v2_target10_20260912_r1 --device cuda:0`，由CUDA_VISIBLE_DEVICES固定实际GPU UUID。
- 预期输出：frozen_manifest.json、每模型672000条predictions、predictions_complete.json、target_truth.json、每模型target_scores.json、complete.json。全部score及complete闭合才报告结果。
- 技术停止：checkpoint/契约不符、错误维度/非有限logits、输出冲突、预测不完整、buffer改变直接报错，保留产物，不自动重试；不因性能低停止或干预训练。

当前：LOCAL_VERIFIED，目标测试尚未完成。

## 实际启动后态

发布commit：0f8d7a40f5b2c0e36592e68be0ba92d5dd0908d2，Git远端OID读回一致。归档SHA256=90b36ad1be61e002141c22875e1bbbd792ac71a32e5b958db84754621915c0c3，本地/远端一致；远端编译PASS。

运行PID=2493804，PPID=1，CWD为本次release，命令行对应冻结manifest。实际GPU UUID=GPU-56adac86-77cd-36c9-8770-dbf002650461；同卡原训练PID1812772继续运行，总计算进程2个。SOURCE_CHECKPOINT_SMOKE_PASS，10个模型进入同一批目标预测，clean输出持续增长。状态RUNNING，尚不报告分数。
