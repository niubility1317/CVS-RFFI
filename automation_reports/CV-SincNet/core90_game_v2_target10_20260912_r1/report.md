# V2已完成10模型目标域测试

当前状态：**TARGET_ARTIFACTS_COMPLETE / VERIFIED**。10个模型×4场景共6720000条预测及独立评分已完成；[目标域详细结果](target_report.md)、[总体指标CSV](target_summary.csv)、[逐RX/TX/day指标CSV](target_detailed_metrics.csv)、[完整评分证据](target_results.json)。下述预登记与启动记录保留为历史过程。

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
- 预期输出：frozen_manifest.json、每模型672000条predictions、predictions_complete.json、target_truth.jsonl、每模型target_scores.json、complete.json。全部score及complete闭合才报告结果。
- 技术停止：checkpoint/契约不符、错误维度/非有限logits、输出冲突、预测不完整、buffer改变直接报错，保留产物，不自动重试；不因性能低停止或干预训练。

预登记时状态：LOCAL_VERIFIED，目标测试尚未完成。

## 实际启动后态

发布commit：0f8d7a40f5b2c0e36592e68be0ba92d5dd0908d2，Git远端OID读回一致。归档SHA256=90b36ad1be61e002141c22875e1bbbd792ac71a32e5b958db84754621915c0c3，本地/远端一致；远端编译PASS。

运行PID=2493804，PPID=1，CWD为本次release，命令行对应冻结manifest。实际GPU UUID=GPU-56adac86-77cd-36c9-8770-dbf002650461；同卡原训练PID1812772继续运行，总计算进程2个。SOURCE_CHECKPOINT_SMOKE_PASS，10个模型进入同一批目标预测，clean输出持续增长。状态RUNNING，尚不报告分数。

## 完成结果与独立核验

2026-09-12目标评估完成，`complete.json`与10份target score均闭合；实际预测及评分用时142.72分钟。独立逐条遍历6720000条预测，检查唯一ID、场景、接收机/日期、标签范围和有限置信度；重新计算总体混淆矩阵、Accuracy、Macro-F1及Macro-Recall，最大绝对评分误差0。每模型四场景共672000条预测。评估PID已退出，未重启或调整训练任务。

- A的目标LEO Accuracy为60.146±1.632%，B为61.166±2.396%（三个seed的均值±样本标准差）。B−A均值为+1.021个百分点，但三个seed分别为+3.646、−1.069、+0.485个百分点，收益并不一致；clean均值下降1.224个百分点。
- C−A在全部三个seed、四种场景的预测类别和置信度完全一致，符合关闭对抗项的负对照结果。
- D的392005目标LEO Accuracy为59.219%，Macro-F1为58.308%；对同seed、同adv=0.35的B，分别下降4.532和5.256个百分点。四场景Accuracy均下降；当前单seed结果不支持B8-D1带来目标域收益，不能扩展为跨seed定论。
- 这是既有目标接收机基准上的6类闭集测试。目标RX与source RX互斥，但day不全部互斥，且目标集有历史研究接触；不称新的完全未见确认集，不回流训练调参、选择或重跑。
- 本批次仅包含冻结时已完成的10个E200模型，其余模型不在本次结果范围内。完整报告提供3360行分组指标、1440格混淆矩阵、28行同seed配对预测汇总及跨seed描述性统计。
