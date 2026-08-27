# SF-TAPFT S15+首发实验报告

## 当前结论与状态

- run ID：`stage2_sf_tapft_s15plus5_rx20_1_s392002_20260827_r1`。
- 当前状态：`LOCAL_VERIFIED/PRE_REGISTERED`；尚未把远端启动等同于科学完成。
- 实现提交：`2eb30bdaa8ebdc4eff0bfbc10f395b9d1568bd4a`。
- 协议：`p2_min_v1`、`VALIDATED_ONCE`、K=10×6=60条support；适配、选择和bundle构建不读取query、query truth、clean/source样本。
- Phase1基础：ADV3B02 CORE90 checkpoint及其既有Phase1 bundle。
- capsule：`d18-enrollment-before-rx20-1-seed713101-k10-smoke-reuse`。
- split：`stage2b-rx20-1-seed713101-before-support-prefix`。

## 历史F0锚点

F0直接复用同一bundle、同一support和同一4折OOF的既有S15结果，不重复启动：

|候选|步数/时钟|BA|fold floor|NLL|full-support步数|wall-clock|最大RSS|GPU峰值|
|---|---|---:|---:|---:|---:|---:|---:|---:|
|F0/S15|300/ref4500,warmup0.05|86.1111%|77.7778%|0.514836|203|0:39:17.91|1,829,304KiB|690MiB|

长程S00同run锚点为BA=86.1111%、fold floor=77.7778%、NLL=0.436715、full-support选择327步、wall-clock=3:46:23。

## 首发矩阵

|row|候选|单一科学目的|训练配置|GPU|
|---|---|---|---|---:|
|F1|S15-SCHED300|验证短程时钟校准|300/ref300,warmup0.10|0|
|F2|F1+TEMP|验证OOF正标量温度能否修复NLL且保持argmax|F1+OOF temperature|1|
|F3|HEAD60+JOINT240+TEMP|验证冻结embedding的head预热|前60步head-only缓存，后240步head+norm|2|
|Q2A|S16-A mixed norm|结构轴：t3 weight+bias+t2 weight|4500/ref4500,warmup0.05|3|
|Q2B|S16-B mixed norm|结构轴：t3 weight+bias+早层norm weight|4500/ref4500,warmup0.05|4|

F1–F3使用预登记稀疏验证步：1、5、10、20、30、50、75、100、150、200、250、300。Q2A/Q2B保持S00逐步验证和4500步长程条件，使其只改变Norm集合。

## 已落地实现

- 300步独立cosine时钟，30步warmup并在第300步附近衰减至零。
- 仅在cross-fitted OOF logits上拟合单一正温度T；记录NLL前后、argmax不变性，并把T持久化到strict clean-single bundle。
- F3前60步复用冻结support embedding，不执行昂贵backbone训练forward/backward；后240步联合head+norm。
- 支持S16-A/B分层Norm规则，不再要求所有Norm共享相同weight/bias策略。
- LOO prototype从样本×类别Python双循环改为向量化等价计算。
- KD权重为0时不复制完整teacher。
- checkpoint snapshot仅保存head和许可参数，最终以完整不可变anchor重建，非许可state必须严格相等。
- 新增fold floor、ECE、per-class NLL、温度、验证步、backbone forward、snapshot字节、可训练/实际变化元素等artifact字段。

## 验证与独立审查

- 聚焦测试：76项通过。
- Python编译与row launcher CLI回读通过。
- 独立P0/P1审查发现温度bundle丢失和短refit派生两项P1；均已定点修复并复核闭合，无残留P0/P1。
- 全仓测试收集受既有Python3.10缺少`tomllib`以及`tests/`与`code/tests/`同名模块冲突阻断；该问题不属于本次改动，未作为额外发布门。
- 真实checkpoint无query smoke作为远端launcher第一步；PASS后立即进入正式row。

## 发布命令和路径

- 远端CWD：`/home/szu2070436088/2510044040/CV-SincNet/<release-checkout>`。
- 矩阵：`configs/stage2_sf_tapft_s15plus5_rx20_1_s392002_20260827.json`。
- row命令模板：`CUDA_VISIBLE_DEVICES=<gpu> /usr/bin/time -v python code/scripts/run_sf_tapft_slim_matrix_row.py --matrix <matrix> --row-id <row> --output-dir <run-root>/<row> --device cuda:0 --folds 4`。
- run root：`/home/szu2070436088/2510044040/CV-SincNet/runs/stage2_sf_tapft_s15plus5_rx20_1_s392002_20260827_r1`。
- 预期每行artifact：`selection.json`、`sf_tapft_clean_single_bundle.pt`、stdout/stderr、GNU time记录；矩阵级保留launch receipt和GPU采样。

## 停止规则

只允许因协议/query泄漏、错误checkpoint/capsule/split/K、错误checkout、输出碰撞、无法产生selection/bundle、同一确定性预prediction异常至少两行或进程归属不清而定点停止本run进程。不得因低性能停止；不得触碰无关任务。

## 晋级门槛

support OOF相对S00同run锚点：

- BA≥85.6111%；
- fold floor≥77.7778%；
- NLL≤0.466715；
- 理想NLL<0.50且优先≤0.4667。

F4暂缓：必须先取得严格绑定的长M02 cross-fitted teacher logits，不能用普通source teacher冒充OOFKD。Q1必须使用新的独立query；Q3仅在Q1通过后启动。
