# Phase1半监督Baseline固定协议（2026-07-13）

本文件是根目录`E:\type10-7\项目.md`中“5.1 Phase1半监督对比矩阵固定协议”的Git-backed镜像交接面。

## 实验矩阵

|路线|CVCNN-CE|RIEI-FD|DRIFT|无标签损失|
|---|---:|---:|---:|---|
|`pseudo_label`|1条|1条|1条|置信度门控硬伪标签CE|
|`augmentation_consistency`|1条|1条|1条|clean→satellite强视图soft KL一致性|

两条路线互斥，不在同一候选中联合启用。总计6条正式实验。

## 固定数据与星地协议

- 源域有标签训练：`0.1`。
- 源域无标签训练：`0.6`。
- 源域验证：`0.3`。
- 三部分互斥，均仅来自`R_s`；目标接收机域`R_t`不参与训练、阈值、伪标签或checkpoint选择。
- 两条路线均默认开启有标签clean+satellite双视图训练增强。
- 训练与正式测试场景：`leo_clear_weak,leo_low_elev_weak,leo_rain_weak`。
- 训练：`200`epoch，seed=`713101`。

## 无标签路线参数

- `pseudo_label`：`pseudo_start_epoch=1`、`pseudo_threshold=0.95`、`pseudo_margin=0.0`、`lambda_pseudo=1.0`。
- `augmentation_consistency`：clean预测经stop-gradient作为soft target，星地强增强视图作为student；`consistency_start_epoch=1`、`temperature=1.0`、`lambda_consistency=1.0`。
- 一致性路线不调用硬伪标签CE；伪标签路线不调用soft KL一致性损失。

## 正式checkpoint

正式checkpoint固定为`best_by_val.pt`：只在未取整source validation TX accuracy严格提高时覆盖。test、LEO test、伪标签精度/覆盖率、一致性损失和最终epoch均不得参与模型选择。

## 启动面

专用launcher：`scripts/launchers/run_phase1_ssl_baseline_matrix.sh`。它默认生成两条路线各3种方法，并把同一方法的两条路线放到同一GPU上；GPU0/1/2各不超过两条训练进程。
