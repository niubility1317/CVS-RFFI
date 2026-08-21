# Phase1 ADV3B02 FastTrust 16条实验矩阵预登记

## 当前状态

```text
run_id=phase1_adv3b02_fasttrust16_s392002_20260821
status=LOCAL_DESIGN_FROZEN_NOT_LAUNCHED
seed=392002
epochs=200
matrix_rows=16
gpu_count=8
rows_per_gpu=2
```

本报告发布矩阵和实现边界，不表示代码已经修改、同步或启动。正式启动只能使用完成TDD和真实checkpoint smoke后的Git提交；每个candidate使用不可覆盖输出目录。

## 共同协议与配置

- source角色：`L_s/U_s/V_cal/V_select=0.07/0.63/0.15/0.15`，物理样本两两不交，source/target receiver不相交。
- seed：全部为392002；训练200epoch；formal checkpoint为`final_ssdg.pth`。
- 历史初始化：除R0外均使用`ADV3B02_CORE90_SOFT_E200/best_joint_safe_ssdg.pth`，不叠加额外冻结teacher蒸馏。
- 星地增强：ADV3B02 CORE90同款clean+satellite拼接，`lambda_sat_cls=0.68`、`lambda_sat_cons=0`及三段LEO弱信道日程。
- U伪身份：只有`high∩temporal stable∩three-head agreement∩class-balanced cap`进入hard CE和U satellite CE；U_M只使用soft/candidate，U_L没有唯一身份梯度。
- 速度主配置：L batch128、U batch256、每epoch完整覆盖U_s、strong+nuisance/satellite拼接前向、S1跳过identity图、关闭分支零前向。
- 训练后必须严格执行clean、`leo_clear_weak`、`leo_low_elev_weak`、`leo_rain_weak`测试并保留逐场景JSON和日志。

## GPU矩阵

| GPU | slot | candidate | 初始化 | U batch | 唯一变量 |
|---:|:---:|---|---|---:|---|
| 0 | A | R0_SCRATCH_CONTROL_U256 | scratch | 256 | from-scratch控制 |
| 0 | B | R1_ADV_INIT_CONTROL_U256 | ADV3B02 | 256 | ADV初始化 |
| 1 | A | R2_FAST_HML_U256 | ADV3B02 | 256 | FastTrust H/M/L，无U proto/卫星身份 |
| 1 | B | R3_FAST_HML_UPROTO_U256 | ADV3B02 | 256 | 增加U prototype更新 |
| 2 | A | R4_FAST_FULL_U256 | ADV3B02 | 256 | FastTrust完整候选 |
| 2 | B | R4_NO_U_SAT_ID_U256 | ADV3B02 | 256 | 关闭U伪身份星地增强 |
| 3 | A | R4_FAST_FULL_U128 | ADV3B02 | 128 | U batch128 |
| 3 | B | R4_FAST_FULL_U384 | ADV3B02 | 384 | U batch384 |
| 4 | A | R4_NO_PROTO_EVIDENCE_U256 | ADV3B02 | 256 | prototype不参与第三路证据 |
| 4 | B | R4_NO_U_PROTO_UPDATE_U256 | ADV3B02 | 256 | prototype保留，只关闭U更新 |
| 5 | A | R4_NO_TEMPORAL_U256 | ADV3B02 | 256 | 关闭temporal gate |
| 5 | B | R4_NO_PRIOR_U256 | ADV3B02 | 256 | 关闭source prior alignment |
| 6 | A | R4_NUISANCE_DETACHED_U256 | ADV3B02 | 256 | nuisance对identity stop-gradient |
| 6 | B | R4_NO_NUISANCE_U256 | ADV3B02 | 256 | 关闭nuisance任务 |
| 7 | A | R4_NO_CROSSRX_U256 | ADV3B02 | 256 | 关闭cross-receiver alignment |
| 7 | B | R4_NO_CLASS_CAP_U256 | ADV3B02 | 256 | 关闭class-balanced acceptance cap |

两条实验将在同一GPU上并发，符合每卡最多两个训练进程的资源上限。并发wall time不用于跨GPU速度排名；速度分析使用samples/s、forward样本数、峰值显存和同卡配对结果。

## 预期输出

每条candidate目录必须包含：

```text
final_ssdg.pth
train.log
metrics_clean.json
metrics_leo_clear_weak.json
metrics_leo_low_elev_weak.json
metrics_leo_rain_weak.json
eval_clean.log
eval_leo_clear_weak.log
eval_leo_low_elev_weak.log
eval_leo_rain_weak.log
status.txt
```

bundle导出artifact独立记录；其失败不能删除checkpoint、覆盖训练状态或阻断四场景评测。

## 追踪表

| ID | 来源 | 要求 | 目标文件 | 状态 | 验证 | 备注 |
|---|---|---|---|---|---|---|
| FT-01 | 用户 | ADV3B02 CORE90同款星地增强 | train、launcher、tests | pending | 待边界测试 | L_s与U_H共享场景日程 |
| FT-02 | 用户 | U获得伪身份后加入星地增强 | train、MUSE loss、tests | pending | 待梯度/样本mask测试 | 仅U_H |
| FT-03 | 用户 | 优化伪标签精度 | muse_ssdg、train、tests | pending | 待H/M/L路由测试 | stable+三头一致+class cap |
| FT-04 | 用户 | 优化训练速度 | loader、train、launcher、benchmark | pending | 待吞吐与峰值显存 | 不降低U每epoch覆盖率 |
| FT-05 | 指导P0-4 | S1身份梯度严格为0 | train、tests、telemetry | pending | 待E1/E16/E17测试 | 禁止误导selected遥测 |
| FT-06 | 指导P0-1/P0-2 | class-complete V_cal/V_select与准确异常分类 | split、prototype、tests | pending | 待类别覆盖/错误类型测试 | 训练前P0 |
| FT-07 | 指导P0-3 | train/eval/export解耦 | launcher、tests | pending | 待失败注入测试 | eval不得被bundle失败阻断 |
| FT-08 | 指导稳定性 | warmup/cosine/tail LR与grad clip | train、launcher、tests | pending | 待边界测试 | `max_grad_norm=5` |
| FT-09 | 用户 | 每GPU两条实验 | matrix、launcher | verified | JSON统计16条、每GPU2条 | 尚未启动 |
| FT-10 | 项目规则 | final clean+三LEO自动评测 | launcher、tests | pending | 待fake+真实smoke | 四场景分文件 |

当前追踪计数：`verified=1`、`pending=9`、`deferred=0`、`rejected=0`、`blocked=0`。最高风险是U batch384与拼接student forward在两进程同卡时的峰值显存；launcher必须在启动前执行单batch非阻断microbenchmark并记录结果，若超显存则该行技术失败，不自动改变矩阵或覆盖输出。

## 计划启动形式

实现完成后的目标调用面为：

```text
RUN_ID=phase1_adv3b02_fasttrust16_s392002_20260821
MATRIX=configs/phase1_adv3b02_fasttrust16_s392002_20260821.json
bash code/scripts/launch_phase1_adv3b02_fasttrust16_20260821.sh
```

当前launcher尚未实现，因此本命令仅冻结接口，不可执行。矩阵进入`LANDED/RUNNING`前必须写入真实Git commit、N607 CWD、release路径、PID与GPU映射。
