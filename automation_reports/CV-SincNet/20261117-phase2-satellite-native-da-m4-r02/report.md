# Phase2 星地 ProtoNet 与 Two-stage 缺项接续

本队列仅执行原生 ProtoNet-CDA 和 Two-stage 的 A0/B，使用固定三池、七接收机、四模型种子，覆盖 MIX02–MIX08 的 K1、K10、K20。每方法 588 任务、3528 条预测；K5 和 MIX01 已评分行保留不重算。LS/MU/LU 沿用原 K5。星地 received 作为 support 和 query；对应方法/seed 的原合法源权重复用，不继承 clean 适应状态。

使用 20261101 的恢复评分索引重新清点：五个原生域适应方法各 4368 条已评分、3528 条缺失，每种方法 1933 个原评分任务均有预测与评分身份及文件，清点无错误。此前使用旧索引误报的 ProtoNet 和 Two-stage 缺失已更正。已评分结果固定，目标指标不用于选模型或调整参数。

新运行根 `runs/20261117-phase2-satellite-native-da-m4-r02`，预备目录同名，release `releases/20261117-phase2-satellite-native-da-m4-r02-release`。代码为包含本报告的 Git commit。`satellite_native_da prepare` 必须显式提供 `--profiles protonet_cda twostage --allowed-gpus 2 3 4` 和 r01 的 `--prior-config`；逐 K 任务及 GPU 集合必须与 r01 不重叠。每 GPU 最多 7 个本任务 worker、启动时空余至少 6000 MiB，并避开其他 owner。实际无空槽时等待，不停止或干预其他任务。

逐 worker 首步真实 checkpoint 的无 query source-only smoke；固定预测核对后自动独立逐任务 truth-last 评分。具体源 runtime、数据句柄、seed、运行命令和输出见 experiment.json 及 scheduler.json；系统技术故障停止派发、排空并保留产物，低分不停止或调参。此队列只代表两个原生域适应方法的缺项，其他注册和交叉组另行处理。
