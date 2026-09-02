# ADV3B02-FCR R1-R8八卡并行实验v6预登记报告

## 状态

- run_id：`phase1_adv3b02_fcr_r1r8_s392002_20260902_v6`
- 当前状态：`ARTIFACTS_COMPLETE / ANALYZED_INTERNAL_EVAL`；独立truth-last评分因prediction ID不可复现而未闭合。
- protocol_scope：Phase1 source-only；R1-R8；seed392002；E200；无R0和旧ADV3B02基线
- branch：`codex/adv3b02-fcr-20260901`
- predecessor：v5在训练前因15域checkpoint与14域训练split被错误要求全模型形状一致而系统性失败，无性能结果，全部partial artifacts保留。

## 冻结矩阵、GPU与初始化

R1→GPU0、R2→GPU1、R3→GPU2、R4→GPU3、R5→GPU4、R6→GPU5、R7→GPU6、R8→GPU7；每卡只新增一个v6 row。R1-R8语义和v4/v5完全不变，训练split仍为day0/1×receiver0-6。

强制初始化：`phase1_adv3b03_core90seed_near3_day123_e200_20260830_r1/S392002_ADV3B03_MU10_ALPHA20_E200/final_ssdg.pth`。固定候选、seed392002、E200，拒绝旧FCR状态；显式使用checkpoint的`branch_ablation=no_dac`、`domain_branch_ablation=no_stats`。成熟`id_backbone.*`必须100%完整加载；允许4个15→14域分类输出层按当前split重建，36个FCR张量为新增初始化。

## 修复验证与审查

- RED复现：v5八行相同`locked mature base checkpoint is incomplete`指纹。
- GREEN：真实checkpoint在14域模型加载191个兼容旧张量，skipped仅4个域输出层，身份主干无缺失，FCR logits有限。
- Phase1-FCR聚焦测试97/97通过；Python编译和`git diff --check`通过。
- 独立P0/P1定点复审仅覆盖本兼容修复，结论`Ready`，无未闭合P0/P1。

## 路径、命令、停止规则与artifact

- release：`/home/szu2070436088/2510044040/CV-SincNet/releases/phase1_adv3b02_fcr_r1r8_s392002_20260902_v6`
- run：`/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_adv3b02_fcr_r1r8_s392002_20260902_v6`
- 日志：`/home/szu2070436088/2510044040/CV-SincNet/logs/phase1_adv3b02_fcr_r1r8_s392002_20260902_v6.Rk.launcher.out`
- 命令：`bash <release>/docs/automation_reports/CV-SincNet/phase1_adv3b02_fcr_r1r8_s392002_20260902_v6/launch_r1r8_remote.sh`

只在数据/query越界、错误row/seed/split/checkpoint、输出覆盖、错误checkout、不能启动、无prediction闭合、进程归属不明或至少两个row出现同一确定性pre-prediction异常时停止本run精确进程树并保留产物；低性能和既有GPU任务数量不触发停止。每行预期生成`best_joint.pth`、`fcr_diagnostics.json`、`fcr_predictions.json`、`train.log`、`status.txt`和四场景prediction。

## 发布与启动证据（2026-09-02 16:49 CST）

- Git提交：`ed4346cadb043d4d14716ef9c7e11b25e00b8faf`；已push，远端分支OID独立回读一致。
- release归档：`E:\type10-7\release_archives\phase1_adv3b02_fcr_r1r8_s392002_20260902_v6_ed4346ca.tar.gz`→`/home/szu2070436088/2510044040/CV-SincNet/release_archives/phase1_adv3b02_fcr_r1r8_s392002_20260902_v6_ed4346ca.tar.gz`；唯一一次传输SHA256一致：`61934122dbe91551febaa43f47b2d09b9231f2fe3eaa6870057e6fb13611c033`。
- 远端解包、`train.py`编译及两个Bash启动脚本语法检查通过。
- launcher PID：R1-R8分别为`142397,142398,142399,142400,142401,142402,142403,142404`；8个CWD均严格绑定v6 release，每个launcher仅有一个直属主训练进程。
- GPU绑定：R1→GPU0、R2→GPU1、R3→GPU2、R4→GPU3、R5→GPU4、R6→GPU5、R7→GPU6、R8→GPU7，8个主训练进程均已产生GPU显存占用。
- 8行真实checkpoint加载均为`loaded=191,skipped=4,missing=40,unexpected=0`；skipped仅为`dom_head.net.3.{weight,bias}`和`adv_head.net.3.{weight,bias}`，36个其余missing为新增FCR张量。
- 首轮训练进展：R1到E004，R2-R8到E003；8行均已生成`latest.pth`和`best_joint.pth`，未见`Traceback`、`RuntimeError`或OOM。未启用诊断字段中的`nan`不属于优化损失或系统技术故障。
- 每30分钟Luna只读监控已启用；监控不得因低性能干预实验。
- Luna独立首检：`HEALTHY / RUNNING`。R1到E012，R2-R6到E010，R7-R8到E009；8份日志持续增长，最新`train_loss`均有限，每行已有2个checkpoint。未发现Traceback、RuntimeError、CUDA错误、OOM或确定性失败；训练阶段尚无prediction符合预期。

## 完成与测试

- R1-R8全部完成E200，8/8行`status.txt=PREDICTIONS_READY`。完整日志中`Traceback/RuntimeError/CUDA error/out of memory/Killed=0`；1600条epoch指标中`train_loss`无NaN/Inf。
- 每行保留8个checkpoint、完整训练日志/指标、FCR诊断和prediction。每行100,800条prediction，clean及3种LEO_WEAK场景各25,200条；8行合计806,400条。
- 本地聚焦测试97/97通过，真实seed392002 checkpoint smoke通过；8行加载结果均为`loaded=191,skipped=4,missing=40,unexpected=0`。

## 机制和完整评测数据

R1=`self+eta`；R2增加`swap`；R3增加`shared`；R4增加`latent_cycle`；R5增加`need`；R6增加定向因子移植；R7增加物理有序解码器和`phys`；R8增加三轴干预和`factor`。本批无R0/旧ADV3B02基线，只允许R1-R8内部归因。

固定`best_joint.pth`完整评测样本为clean及每种卫星场景各204,000条；`strict_udu`为未见日期×未见接收机60,000条。下表均为准确率百分数。

|行|clean总体|clean strict|clear总体|clear strict|low总体|low strict|rain总体|rain strict|
|---|---:|---:|---:|---:|---:|---:|---:|---:|
|R1|93.11|88.50|81.52|74.76|78.68|71.99|78.56|72.07|
|R2|93.00|88.61|81.44|74.84|78.41|71.95|78.39|72.06|
|R3|93.02|88.25|81.12|74.39|77.88|71.39|77.96|71.59|
|R4|**93.13**|**88.77**|80.91|74.03|78.04|71.28|77.99|71.31|
|R5|92.68|88.35|78.73|72.33|75.50|69.24|75.33|69.28|
|R6|93.00|88.42|81.45|75.01|78.39|72.06|78.26|71.98|
|R7|92.90|88.28|**82.22**|**75.91**|**79.22**|**73.09**|**79.03**|**72.99**|
|R8|93.03|88.43|81.64|75.21|78.53|72.35|78.42|72.38|

正确数/204,000依次为clean、clear、low、rain：R1=`189945/166308/160499/160264`；R2=`189711/166142/159962/159922`；R3=`189759/165487/158878/159035`；R4=`189992/165055/159197/159099`；R5=`189064/160605/154023/153665`；R6=`189727/166155/159914/159658`；R7=`189521/167737/161603/161231`；R8=`189785/166552/160191/159977`。

## 训练末态与资源

|行|E200 loss|E200 train/val acc|训练时长|峰值显存|推理延迟|有效秩|Gram条件数|
|---|---:|---:|---:|---:|---:|---:|---:|
|R1|7.1633|91.46/98.68|3.55h|9758.6MB|0.212ms|2.737|20.35|
|R2|7.9571|91.58/98.70|3.99h|9757.8MB|0.259ms|2.749|20.46|
|R3|8.4711|90.49/98.67|3.97h|9757.8MB|0.184ms|2.741|19.96|
|R4|8.6369|91.37/98.69|3.95h|9757.7MB|0.264ms|2.702|19.87|
|R5|8.5522|92.17/98.69|4.06h|9758.6MB|0.231ms|2.661|21.13|
|R6|8.6490|91.37/98.65|4.12h|9758.7MB|0.267ms|2.707|19.80|
|R7|8.1392|91.41/98.69|4.43h|9760.1MB|0.257ms|2.193|118.98|
|R8|8.2465|91.27/98.69|4.36h|9759.4MB|0.264ms|2.194|110.45|

## 归因、缺陷与晋级边界

- R5相对R4是最明显退化点，3种卫星总体下降2.18/2.54/2.66pp；R6定向移植恢复2.72/2.89/2.93pp。
- R7相对R6使卫星总体提高0.77/0.83/0.77pp、strict提高0.90/1.03/1.01pp，代价是clean下降0.10pp；R7为卫星弱信道冠军。
- R8相对R7的三轴`factor`使clean提高0.13pp，但卫星总体下降0.58/0.69/0.61pp，故R8不晋级；clean目标下R4最佳。
- prediction的`sample_id`由dataset实例随机HMAC密钥生成，且未封存独立truth sidecar或稳定映射，因此806,400条prediction无法由新scorer可靠关联truth。本报告不伪造独立评分。表中结果是重新加载固定checkpoint后的完整内置有标签评测，数据有效，但truth-last工程闭环未完成。
- 当前仅支持单seed、R1-R8内部机制筛选，不能声称超过ADV3B02或提升为默认模型。下一次最小修复应只保留R7，补齐稳定token、隔离truth sidecar和独立scorer。
