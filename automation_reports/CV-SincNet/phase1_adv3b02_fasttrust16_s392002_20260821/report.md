# Phase1 ADV3B02 FastTrust 16条实验矩阵预登记

## 当前状态

```text
run_id=phase1_adv3b02_fasttrust16_s392002_20260821
status=LOCAL_VERIFIED_AWAITING_RELEASE
seed=392002
epochs=200
matrix_rows=16
gpu_count=8
rows_per_gpu=2
```

2026-08-21 12:38 CST按用户要求执行停止核验：N607直连预检通过；项目实验进程匹配为空，`nvidia-smi`计算进程为空，GPU0–7均为0%利用且各仅1MiB占用。因此没有可归属实验需要终止，未执行kill，既有checkpoint、日志和部分产物均保留。

本轮代码已在本地按TDD完成并通过真实checkpoint无query smoke；尚未同步或启动。正式启动只使用本轮提交固定的代码与配置；每个candidate使用不可覆盖输出目录。

## 本地实现与验证

- FastTrust严格路由已接入：hard必须同时满足high、temporal stable、三头一致与class-balanced cap；全批hard上限25%，全部身份样本上限50%。
- E1–E16在base domain/self/nuisance后提前返回，不执行融合、temporal observe或U prototype更新；E17以后才启用身份路由。
- U侧卫星身份CE仅消费严格U_H，伪标签detach，权重为`lambda_u(epoch)×0.68`，复用nuisance/satellite视图，不增加第三次student前向。
- L batch固定128，U batch独立为256；GPU3保留128/384配对。strong与nuisance/satellite通过拼接完成一次student调用，新增U samples/s、前向samples/s和CUDA峰值显存遥测。
- source validation改为按TX及receiver/day分层的`V_cal/V_select`互斥划分；双空间审计在导出边界检查类别覆盖、有限/非零特征、类间几何和类别顺序。
- 16条launcher从机器矩阵读取候选，每张卡最多并发两条；每条只执行一次严格联合评测，再拆分clean和三种LEO弱信道JSON/log。输出根已存在时拒绝覆盖。
- 聚焦联合回归：188项收集，186项通过、2项按既有条件跳过；退出码0。`bash -n`、`py_compile`、`git diff --check`和16条dry-run均通过；dry-run计数为row16、train16、联合eval16、分场景输出64，且没有创建输出目录。
- 真实checkpoint无query smoke：严格恢复ADV3B02 checkpoint，missing/unexpected均为0；CPU单batch前向、FastTrust hard身份、U satellite CE、U prototype更新、反向、grad clip和AdamW step均完成，`tx_logits=[2,6]`、`z_id=[2,160]`，hard/satellite/prototype计数均为2，query与target truth读取均为0。仅出现既有AMP弃用warning，不影响数值有限性或退出码。
- 独立P0/P1审查首轮发现4项阻断：可选export被旧终态误判、U尾批丢弃、prior可能翻转hard标签、M0未执行统一LR。四项均按TDD修复；唯一一次定点复审逐项GREEN，最终结论为GO。未扩大审查范围，未增加发布gate。

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
| FT-01 | 用户 | ADV3B02 CORE90同款星地增强 | train、launcher、tests | verified | 日程边界与launcher参数测试通过 | L_s与U_H共享场景日程 |
| FT-02 | 用户 | U获得伪身份后加入星地增强 | train、MUSE loss、tests | verified | 严格U_H mask、梯度与真实checkpoint smoke通过 | 仅U_H |
| FT-03 | 用户 | 优化伪标签精度 | muse_ssdg、train、tests | verified | H/M/L、稳定性、三头一致与cap测试通过 | stable+三头一致+class cap |
| FT-04 | 用户 | 优化训练速度 | loader、train、launcher、telemetry | verified | U batch、融合前向、LR与遥测测试通过 | 实际吞吐待运行后同卡分析 |
| FT-05 | 指导P0-4 | S1身份梯度严格为0 | train、tests、telemetry | verified | E1/E16/E17边界测试通过 | 首批分项梯度，其他batch不重复反向 |
| FT-06 | 指导P0-1/P0-2 | class-complete V_cal/V_select与准确异常分类 | split、prototype、tests | verified | 类别覆盖、互斥与五类错误测试通过 | 训练前失败关闭 |
| FT-07 | 指导P0-3 | train/eval/export解耦 | launcher、tests | verified | 失败注入测试通过 | export非必要失败不阻断eval |
| FT-08 | 指导稳定性 | warmup/cosine/tail LR与grad clip | train、launcher、tests | verified | E1/E5/E160/E161/E180/E181/E200测试通过 | `max_grad_norm=5` |
| FT-09 | 用户 | 每GPU两条实验 | matrix、launcher | verified | JSON统计16条、每GPU2条 | 尚未启动 |
| FT-10 | 项目规则 | final clean+三LEO自动评测 | launcher、tests | verified | 单次联合eval、四场景拆分与缺失失败注入通过 | 真实metrics待训练结束后生成 |
| FT-11 | 指导P0-5 | `z_id/feat_joint`双空间审计与identity feature contract | train、phase2_prototypes、tests | verified | finite/nonzero/coverage/geometry/contract测试通过 | export前失败关闭 |

当前追踪计数：`verified=11`、`pending=0`、`deferred=0`、`rejected=0`、`blocked=0`。当前没有性能结果；代码验证不能替代真实训练结论。主要运行期资源风险是U batch384与两进程同卡的峰值显存，launcher会保留每条技术失败和部分产物，不自动改变矩阵或覆盖输出。

## 计划启动形式

实现完成后的目标调用面为：

```text
RUN_ID=phase1_adv3b02_fasttrust16_s392002_20260821
MATRIX=configs/phase1_adv3b02_fasttrust16_s392002_20260821.json
bash code/scripts/launch_phase1_adv3b02_fasttrust16_20260821.sh
```

launcher已实现并完成本地dry-run；矩阵进入`LANDED/RUNNING`前仍需写入真实Git commit、N607 CWD、release路径、PID与GPU映射。
