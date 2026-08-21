# Phase1 ADV3B02 FastTrust 16条实验矩阵预登记

## 当前状态

```text
run_id=phase1_adv3b02_fasttrust16_s392002_20260821
status=STOPPED_EARLY_SYSTEMIC_TECHNICAL_FAILURE / NO_PERFORMANCE_RESULT
seed=392002
epochs=200
matrix_rows=16
gpu_count=8
rows_per_gpu=2
```

2026-08-21 12:38 CST按用户要求执行停止核验：N607直连预检通过；项目实验进程匹配为空，`nvidia-smi`计算进程为空，GPU0–7均为0%利用且各仅1MiB占用。因此没有可归属实验需要终止，未执行kill，既有checkpoint、日志和部分产物均保留。

本轮代码已在本地按TDD完成并通过真实checkpoint无query smoke，随后完成Git固定、N607发布和正式启动。正式运行使用不可覆盖run root，每个candidate使用独立输出目录。

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

## 发布与启动证据

- 实际代码与配置提交：`0fde2fa92fa8e93b077b954750888198aae67837`；自动push后独立读取远端`origin/work/cvs-active`，OID与本地`HEAD`一致。
- 单一release归档：本地`E:\type10-7\release_archives\phase1_fasttrust_cd3045d8.tar.gz`映射到远端`/home/szu2070436088/2510044040/CV-SincNet/releases/phase1_fasttrust_cd3045d8.tar.gz`；本地与远端SHA-256均为`efef54b0923a8eded085b6dbeddce81ff8551f20cd8e7d48e193c5ba715d957e`。归档后仅将提交`0fde2fa`修复的两个launcher文件补充同步到解压release，不创建第二个归档或额外校验门。
- 远端release目录：`/home/szu2070436088/2510044040/CV-SincNet/releases/phase1_fasttrust_cd3045d8`；远端生产Python编译、两个launcher语法检查和16条dry-run均通过，dry-run得到row16、train16、eval16、分场景输出64且未创建run输出。
- 正式启动时间约为2026-08-21 13:31 CST；dispatch PID为`266946`，PPID为1，启动后状态为`S`；dispatch CWD为`/home/szu2070436088`，cmdline绑定上述release launcher。
- dispatch日志：`/home/szu2070436088/2510044040/CV-SincNet/launcher_logs/phase1_adv3b02_fasttrust16_s392002_20260821.dispatch.log`；run root：`/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_adv3b02_fasttrust16_s392002_20260821`。
- 启动快照中GPU0–7均恰好有2个本run的Python训练进程：GPU0=`267011,267019`，GPU1=`267012,267015`，GPU2=`267018,267020`，GPU3=`267016,267022`，GPU4=`267024,267027`，GPU5=`267028,267031`，GPU6=`267033,267037`，GPU7=`267034,267035`。
- 启动约2分钟后的只读复查：16/16个训练进程存活，16/16个candidate日志非空且持续增长，错误指纹计数为0；15/16条已记录`[EPOCH-END] E001/200`，`R4_FAST_FULL_U128`仍在首轮初始化。该候选的U batch为128，每epoch步数多于U256/U384，因此首轮更慢属于预期速度差异，不能解释为崩溃。
- 当前尚无最终性能结果。每条训练完成E200并写出`final_ssdg.pth`后，launcher才会自动执行一次严格联合评测并拆分保存clean、`leo_clear_weak`、`leo_low_elev_weak`、`leo_rain_weak`结果；四场景未齐全前不会标记`ARTIFACTS_COMPLETE`。

## 发布后逻辑与负载复核

2026-08-21 13:42 CST按用户要求对实际发布面执行一次主Agent只读一致性复核，不新增独立审查gate，也不干预健康训练。复核范围限定为矩阵唯一性、同seed、每卡并发上限、batch参数传递、U_s完整覆盖、FastTrust身份路由、不可覆盖输出、训练后`final_ssdg.pth`严格恢复和clean+三LEO评测闭环。

- 矩阵逻辑检查为`VERIFIED`：16条candidate名称唯一、seed统一为392002、E200、GPU0–7各2条；每张卡的U batch总和均为512。
- GPU0、1、2、4、5、6、7均为`256+256`；GPU3为`128+384`。因此大batch与较小batch已经位于同一卡，且与其他GPU形成精确相同的batch总负载。
- FastTrust launcher、protocol、speed、MUSE训练集成和U侧星地身份增强聚焦回归退出码为0，共69项通过；提交后`git diff --check`通过且没有未提交的已跟踪修改。
- N607只读快照显示dispatch PID`266946`持续运行，16个训练进程仍为每卡2个，GPU利用率94%–98%，错误指纹计数为0；各候选已到E001–E004。GPU3总显存占用约5.99GB/24GB，没有出现大/小batch配对导致的显存风险。
- 用户本次文本写作“batch383”，但已发布并正在运行的候选是`R4_FAST_FULL_U384`。本轮不把健康运行中的384静默改成383，也不覆盖run root；若后续明确要求精确batch383，应创建新的不可覆盖矩阵，并用较小batch129与其同卡组成`383+129=512`，不能改写本run的实验身份。

复核结论：未发现会导致本轮训练跑错、数据角色越权、输出覆盖、每卡超过2个进程、`final_ssdg.pth`遗漏评测或四场景结果伪闭合的逻辑硬伤。当前矩阵保持`RUNNING`，不重复发布。

## 2026-08-21 15:21 CST全量训练健康复查

本次按监控请求完整解析16个candidate从E1到最新epoch的全部`metrics_epoch.jsonl`记录和完整`train.log`，并读取dispatch、16个GPU进程、GPU温度/显存/利用率和磁盘状态。该复查是运行中证据，不是E200性能结论。

- 系统执行面健康：dispatch PID`266946`存活；16个Python训练进程均存活且每卡2个；GPU利用率91%–99%，显存4.27–5.83GB/24GB，温度57–79°C；磁盘剩余7.3TB；Traceback、RuntimeError、OOM、Killed、`TRAIN_FAILED`和`EVAL_FAILED`均为0。
- 数据与日志结构未见异常：16条均严格使用`L/U/Vcal/Vselect=5880/52920/12600/12600`；16个JSONL均可完整解析、epoch从1连续到最新且没有缺号；核心loss、accuracy、gradient norm和epoch time序列中没有序列化NaN/Inf。训练期`TEST=nan(0/0)`、inactive direct-metric字段和`nonfinite_*_metric_count`来自尚未运行的test/关闭分支占位，不是数据损坏。
- 当前进度：R0/R1为E038，U256系列为E036左右，`R4_FAST_FULL_U384`为E047，`R4_FAST_FULL_U128`为E024；尚无`final_ssdg.pth`和正式clean/三LEO结果。
- 发现严重数值优化异常：`train/skipped_nonfinite_grad`是逐batch真实0/1跳步标志，`train/optimizer_step_applied`是epoch平均实际更新率。R0/R1控制组最近更新率分别为100.0%和99.7%；所有MUSE候选从E17起更新率均为0.0%，即U256每epoch约207个batch、U128约414个batch、U384约138个batch全部因非有限梯度跳过optimizer step。
- 异常在S1已经出现：MUSE候选E1–16平均更新率仅22.9%–53.9%，首个低于50%的epoch位于E5–E11；E17身份路由开放后共同降到0%。关闭U卫星身份、U prototype更新、prototype evidence、temporal、prior、nuisance、cross-RX或class cap均未恢复更新，故问题不属于单一消融项或数据切分，更可能位于所有MUSE候选共享的CUDA AMP/backward梯度路径。现有日志没有记录具体首个非有限参数，不能进一步宣称唯一根因。
- CPU单batch smoke之所以未发现，是因为它只证明CPU单步有限，未覆盖N607 CUDA AMP、多batch累计和E17后的身份路径。当前异常说明此前“无逻辑硬伤”的判断被真实GPU长程证据推翻。
- 若不干预，按最近5个完整epoch机械外推：U384约2026-08-21 23:00完成训练；控制/U256多数约2026-08-22 01:30–02:50；恒定双进程负载下U128约2026-08-22 14:00。考虑同卡伙伴先结束后U128可能加速，矩阵训练完成规划区间为2026-08-22 08:00–14:00，四场景评测闭合再预留约1–2小时。该ETA只表示进程何时跑完，不表示结果科学有效。

健康裁决：`SYSTEM_RUNNING_HEALTHY`，但`MUSE_OPTIMIZATION_UNHEALTHY`。即使继续到E200并自动完成clean和三LEO测试，当前MUSE权重在E17后没有更新，结果不得作为有效FastTrust性能证据。依据monitor-only边界，本次未停止、重启或修改任何远端进程；建议取得用户明确授权后停止本run并保留全部artifact，再对CUDA AMP下首个非有限梯度参数进行最小复现和修复。

## 2026-08-21 15:36–15:50 CST技术停止与根因定位

用户授权“查找定位分析问题，修复问题后重新发布实验”后，本run按预注册系统技术失败规则停止。停止前重新解析早期epoch，修正上一节“E17后才全部失败”的粗粒度表述：MUSE候选从E1已出现少量非有限梯度跳步，E3–E6跳步率持续上升，主候选在E7起更新率已降为0；E17仅是身份路由开启边界，不是故障首次发生点。R0/R1控制组同期维持约99%–100%更新率。

- 停止前精确绑定dispatch PID`266946`、release路径和run ID，共解析出233个run-owned进程（16个训练进程、launcher/worker及DataLoader后代）；233/233均由该run ID或专属release路径归属，没有混入无关任务。
- 仅对这棵进程树按叶节点到根节点发送SIGTERM；20秒边界内全部退出，不需要SIGKILL。独立回读为根PID不存在、run命令行匹配0、GPU0–7利用率0%且各1MiB、旧run root仍存在。
- 旧run目录`/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_adv3b02_fasttrust16_s392002_20260821`、全部日志、部分checkpoint和metrics均原位保留；未删除、移动或覆盖任何artifact。
- 最小数值复现确认根因在所有MUSE候选共享的本地分类头AMP路径：`local_prob`先以float32做softmax，随后又把概率转回float16；置信度升高时非目标类概率下溢为精确0。`NLL(log(clamp_min(1e-8)))`中的`1e-8`在float16同样下溢为0，导致标量loss仍有限时反向传播产生`0/0`NaN梯度。极端三类logit`[0,-20,-40]`复现中，forward loss有限，但输入梯度及本地头五组参数梯度全部非有限。
- 这一路径只存在于MUSE候选，且从E1即参与L_s本地头监督，因此同时解释了控制组正常、所有MUSE消融共同失败、前几轮间歇跳步和随后全批次跳步；它不依赖U伪标签、prototype、prior、temporal、nuisance或LEO身份分支。

当前裁决固定为`STOPPED_EARLY_SYSTEMIC_TECHNICAL_FAILURE / NO_PERFORMANCE_RESULT`。本run不得用于FastTrust性能结论，也不会继续执行final clean/三LEO测试；修复后的实验必须使用新的不可覆盖run ID和output root。

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
| FT-09 | 用户 | 每GPU两条实验 | matrix、launcher | verified | JSON统计16条且启动快照每GPU2条 | 16条均已启动 |
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

launcher曾按上述调用面完成正式启动；矩阵现已因确定性非有限梯度故障停止，最终状态为`STOPPED_EARLY_SYSTEMIC_TECHNICAL_FAILURE / NO_PERFORMANCE_RESULT`。本报告保留原Git commit、N607 CWD、release路径、dispatch PID、GPU映射、故障和停止证据；修复后只允许使用新的run ID重新发布。
