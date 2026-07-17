# D25-C3 288D拼接对角floor适配support-only实验

## 启动前记录

- experiment ID：`d25_c3_diag_floor_support_20260718/support_screen_v2`。
- 日期：2026-07-18；operator：Codex；状态：`DEVELOPMENT_SUPPORT_ONLY_COMPLETE_NEGATIVE`。
- 目标：在D25-C0的288维拼接基础上，只训练288个块内对角尺度，比较0-step与20-step新类suffix注册，集中改善旧类`14-10/20-19/6-15`和新类`cls_09f8/cls_f608`的support-held floor。
- 假设：B3的优势来自FFT96/RF32中可学习的维内身份尺度；保留固定块能量`5/9、1/3、1/9`并只学习块内尺度，可在不恢复辅助分支能量支配和dense head的情况下逼近其floor收益。
- 对比：Z0 identity-only、B3诊断、C0固定拼接、C3A 20+0、C3B 20+10、C3C强floor 15+15。
- 矩阵：6候选×3个LEO_weak场景×5个held-rank fold=90行；receiver `20-1`、开发seed `713101`、K=10，每fold每类fit8/held2，旧6类+seen-new5类。

## 协议与算法锁

- 复用已验证的D25 sealed enrollment-only LEO_weak support；每个物理IQ只有一个已叠加一次且仅一次LEO_weak信道的观测。
- `z160+FFT96+RF32`只形成一条288维拼接行；`support_view_count=1`、`support_row_multiplicity=1`、`derived_support_rows=0`。
- runner不得提供query/truth/role/quota/global assignment/scorer/source/clean入口；query rows/labels用于fit均为0。
- `0.07/0.63/0.30`只表示Phase1 L/U/V数据比例，不作为C3损失权重。
- C3 shared adapter严格为288个`gamma`；块内零和、`gamma∈[-0.35,0.35]`，变换后重新固定三块能量。
- C3A：Stage2-B 20步，Stage2-C 0步；C3B：20+10步，CE主导；C3C：15+15步，显式class-balanced CE+tail CVaR+hard-negative margin+prox。三者总epoch与optimizer step均不超过30，留在正式档，不使用150%探索放宽。
- Stage2-C只允许new prototype suffix更新；旧gamma、旧prototype、旧类顺序、旧prefix hash和旧raw score列逐字节冻结。旧raw score冻结不等于无遗忘，runner还必须记录注册前后fit-old support逐类预测/floor并对任何floor退化fail closed。
- int8地面锚不进入本轮梯度loss，避免违反`项目.md`禁止source-target分布损失；C1/C2直接中心/半径融合的负结果已经保留，不重复运行。

## 本地实现与验证

- Git仓库：`E:\type10-7\github_publish\CVS-RFFI-repo`；根目录`E:\type10-7`不是Git仓库，本报告保留根目录镜像。
- C3核心：`code/cvsrffi/stage2_multimodal_diag_floor_adapter.py`。
- 核心追溯：`analysis/d25_c3_diag_floor_adapter_traceability_20260718.md`。
- 核心提交：`780f6389`；runner/launcher提交：`d2fdb0af`。
- runner：`code/scripts/run_d25_support_only_concat.py --candidate-set c3_v1`；SHA256=`51f6a7ae52525f00a873f64cea357b2fb2a407fcbbca26cd8b9a02ca370fdbff`。
- C3核心SHA256=`0b3354f76b281710e2c538d2a1a81a7c00c258b91a7dbe8415ccde48c5d84df0`；launcher SHA256=`5eddb568487f83c8909d6d9a2f3f6f156654053f00ddd03908701b9dff30008f`。
- 本地验证：runner/core/tests `py_compile` PASS；历史D25默认分支+C3 focused+核心/D25相邻回归共41项PASS；launcher `bash -n` PASS；CLI help只有`candidate-set`命中，不含query/truth/quota/scorer入口。
- N607的NumPy2.2.5与Torch2.1.0组合暴露`torch.from_numpy` ABI异常；提交`b5da911`改为仅在小规模support适配时使用Python list桥接，逐query FP32 NumPy路径不变。v1因此在第一个C3 fold前终止，未形成性能矩阵且query未打开；v2使用修复后核心重新完整运行。
- 独立review发现并修复：重复Stage2-C绕过step、正式epoch漏门、old-score冻结误当无遗忘、逐query FP64、candidate lock基线漂移、full-K10终门缺失。现在fold正向路线还必须通过三个场景full-K10 old-support逐类非退化终门，否则自动回退C0。

## 预注册选择门

- 协议/资源全PASS：无query/clean/source、无派生行、总steps≤30、shared adapter=288、无dense query图。
- 相对C0，每场景每类pooled held准确率最多下降10pp；三个场景的pooled old floor和new floor都至少严格提升10pp。
- 定向弱类组旧类`14-10/20-19/6-15`和新类`cls_09f8/cls_f608`不得退化，并且old弱类组与new弱类组都出现严格改善。
- 平均`H_old_new`不低于C0，平均forgetting不高于C0；B3仅作为诊断距离，不具晋级资格。
- 排序优先最大化`min(old floor增量,new floor增量)`，再比较worst joint floor、H、forgetting，最后选择steps/状态更小者。

## N607计划

- 2026-07-18 01:16 CST直连只读preflight PASS；项目根目录可见，8张RTX3090均约10MiB占用，预检结束后本地`ssh.exe`和N607:22连接均为0。
- 远端根目录：`/home/szu2070436088/2510044040/CV-SincNet`。
- 计划GPU：启动前重新读取live inventory后选空闲卡，默认优先GPU0。
- 计划环境：`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`，启动前由launcher校验。
- 计划log：`/home/szu2070436088/2510044040/CV-SincNet/logs/d25_c3_diag_floor_20260718/support_screen_v1.log`。
- 计划output：`/home/szu2070436088/2510044040/CV-SincNet/runs/d25_c3_diag_floor_20260718/output/support_screen_v1`。
- 同步仅包含本地验证后的C3核心、独立runner和launcher；不会覆盖D25历史runner或删除任何既有artifact。
- 02:13 CST live inventory：`gpu_compute=[]`、`active_training_processes=[]`、`unknown_training_active=false`，允许选择GPU0。
- 精确同步映射：
  - `E:\type10-7\github_publish\CVS-RFFI-repo\code\cvsrffi\stage2_multimodal_diag_floor_adapter.py`→`N607:/home/szu2070436088/2510044040/CV-SincNet/code/cvsrffi/stage2_multimodal_diag_floor_adapter.py`；
  - `E:\type10-7\github_publish\CVS-RFFI-repo\code\scripts\run_d25_support_only_concat.py`→`N607:/home/szu2070436088/2510044040/CV-SincNet/code/scripts/run_d25_support_only_concat.py`；
  - `E:\type10-7\github_publish\CVS-RFFI-repo\code\scripts\launch_d25_c3_diag_floor_support_20260718.sh`→`N607:/home/szu2070436088/2510044040/CV-SincNet/code/scripts/launch_d25_c3_diag_floor_support_20260718.sh`。
- 远端同步后三文件SHA与本地完全一致；C3/D25/D24/CIAF/D19六成员closure校验、远端`py_compile`与launcher `bash -n`均PASS。
- 精确启动命令：`cd /home/szu2070436088/2510044040/CV-SincNet && D25_C3_GPU=0 bash code/scripts/launch_d25_c3_diag_floor_support_20260718.sh`。
- v1 PID：`3587960`；因上述NumPy/Torch ABI异常终止，保留原log作为实现故障证据，不作为性能证据。
- v2精确启动命令不变，PID：`3591057`；GPU0；log：`/home/szu2070436088/2510044040/CV-SincNet/logs/d25_c3_diag_floor_20260718/support_screen_v2.log`；output：`/home/szu2070436088/2510044040/CV-SincNet/runs/d25_c3_diag_floor_20260718/output/support_screen_v2`。
- v2耗时71.150秒并正常退出；每次SSH/SCP后均确认本地`ssh.exe`和到N607:22的ESTABLISHED连接为0。

## 预期产物

- `training_log.jsonl`：90个fold summary并嵌入每step完整loss/gradient/gamma/suffix hash trace。
- `selection.json`：pooled逐类门和联合排序。
- `support_audit.json`：单观测、单行、跨场景互斥、query/clean/source不可达证据。
- `resource_audit.json`：参数、步数、状态、MAC、适配/注册/单样本时延、RAM/VRAM。
- `geometry_audit.json`：逐类半径、pair gap和碰撞。
- `RECEIPT.json`：source closure、产物SHA、状态和声明边界。

本轮仍是开发support-only原子筛选，不是正式query性能；只有正向门通过后才重建joint checkpoint+int8+method lock并进入正式独立确认矩阵。

## 完成状态与完整性

- `RECEIPT.status=DEVELOPMENT_SUPPORT_ONLY_COMPLETE`，`query_opened=false`，`training_log_row_count=90`，正好等于6候选×3场景×5fold。
- 完整stdout为一行可解析JSON；完整结构化训练日志无Traceback、OOM、数值NaN/Inf或异常退出。所有C3 trace的`query_rows_used=0`，`old_prototype_intrusion=0`。
- 六件主产物齐全；另保存完整stdout。哈希如下：

| 产物 | 字节 | SHA256 |
|---|---:|---|
| `training_log.jsonl` | 4,207,180 | `3f31d36d18104cb6a6cf6c00ce35d185a406d2132f835ef1760475090b7caccb` |
| `selection.json` | 7,011 | `6de4b33b399c76c38588a0a6168a56fd0557664f5d91afa01fe743c9b0893b3e` |
| `support_audit.json` | 306,934 | `5e8ab470227a2ee246a61fbb9b51c8d50472822a99c9ffd272ae8190aff790d6` |
| `resource_audit.json` | 456,494 | `8404c392899a120b5d6ff76c11d64cc27f4c9c48072c8855191faa0b7594ba52` |
| `geometry_audit.json` | 323,464 | `952512c04fae186164a70a5206eb17239a51b1d40c3273f8b708f53a931c3e29` |
| `RECEIPT.json` | 2,376 | `c914e75501343509df51ae71a27b5ccde1d2259432e8db7d3c84a78db00d307b` |
| `support_screen_v2.log` | 2,364 | `20fb1fe0e46b74d716a55cdd9add212c57ea239e09e9eb3700f1d1bdce82fbaf` |

本地下载目录：`E:\type10-7\automation_reports\CV-SincNet\d25_c3_diag_floor_support_20260718\remote_output_v2`。

## 候选联合结果

这里的`fit_k_shot=8`是K=10 support内部leave-two-out后的训练量；held2只用于开发support选择，不是query。`H`按每个fold先计算再平均，未用边际均值重算。

| 候选 | 注册前old | 注册后old | seen-new | H | forgetting | fit-old非退化 | 结论 |
|---|---:|---:|---:|---:|---:|---:|---|
| Z0 | 71.11% | 48.33% | 52.67% | 48.97% | 22.78pp | N/A | identity control |
| B3 | 86.67% | 73.33% | 73.33% | 72.65% | 13.33pp | N/A | 最强诊断，但60 optimizer steps超正式50-step上限 |
| C0 | 71.67% | 50.56% | 54.00% | 50.35% | 21.11pp | N/A | 回退基线 |
| C3A | 72.22% | 55.00% | 54.67% | 53.16% | 17.22pp | 0/15 | H改善但floor门失败 |
| C3B | 72.22% | 64.44% | 41.33% | 48.72% | 7.78pp | 2/15 | 保护旧类但严重牺牲新类 |
| C3C | 72.78% | 60.56% | 49.33% | 53.38% | 12.22pp | 1/15 | H略升但新类floor未改善 |

`selection.json`在full-K10前已无C3候选通过初筛，因此`pre_full_k10_selected_candidate_id=selected_candidate_id=D25-C0-DIM-CONCAT`，`selected_positive_route=false`，不是full-K10异常回退。

## 逐场景结果

| 候选 | 场景 | 注册前old | 注册后old | seen-new | H | forgetting | pooled old floor | pooled new floor |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| B3 | clear | 90.00% | 75.00% | 82.00% | 77.51% | 15.00pp | 50.00% | 50.00% |
| B3 | low-elev | 81.67% | 70.00% | 72.00% | 70.25% | 11.67pp | 50.00% | 40.00% |
| B3 | rain | 88.33% | 75.00% | 66.00% | 70.19% | 13.33pp | 50.00% | 30.00% |
| C0 | clear | 73.33% | 50.00% | 56.00% | 51.78% | 23.33pp | 0.00% | 0.00% |
| C0 | low-elev | 66.67% | 46.67% | 54.00% | 46.40% | 20.00pp | 10.00% | 0.00% |
| C0 | rain | 75.00% | 55.00% | 52.00% | 52.87% | 20.00pp | 10.00% | 0.00% |
| C3A | clear | 75.00% | 56.67% | 56.00% | 55.70% | 18.33pp | 0.00% | 0.00% |
| C3A | low-elev | 68.33% | 53.33% | 58.00% | 52.05% | 15.00pp | 10.00% | 0.00% |
| C3A | rain | 73.33% | 55.00% | 50.00% | 51.74% | 18.33pp | 20.00% | 0.00% |
| C3B | clear | 75.00% | 65.00% | 46.00% | 52.21% | 10.00pp | 10.00% | 0.00% |
| C3B | low-elev | 68.33% | 61.67% | 46.00% | 51.21% | 6.67pp | 10.00% | 0.00% |
| C3B | rain | 73.33% | 66.67% | 32.00% | 42.73% | 6.67pp | 20.00% | 0.00% |
| C3C | clear | 75.00% | 60.00% | 54.00% | 56.14% | 15.00pp | 0.00% | 0.00% |
| C3C | low-elev | 70.00% | 60.00% | 54.00% | 55.79% | 10.00pp | 10.00% | 0.00% |
| C3C | rain | 73.33% | 61.67% | 40.00% | 48.22% | 11.67pp | 20.00% | 0.00% |

## 逐类floor诊断

15fold pooled准确率表明失败不是平均域偏移：

| 候选/阶段 | 14-10 | 14-7 | 20-15 | 20-19 | 6-15 | 8-20 | old floor |
|---|---:|---:|---:|---:|---:|---:|---:|
| C0注册前 | 46.67% | 83.33% | 96.67% | 33.33% | 80.00% | 90.00% | 33.33% |
| C0注册后 | 13.33% | 70.00% | 86.67% | 20.00% | 23.33% | 90.00% | 13.33% |
| C3A注册后 | 10.00% | 66.67% | 86.67% | 30.00% | 46.67% | 90.00% | 10.00% |
| C3B注册后 | 23.33% | 73.33% | 90.00% | 30.00% | 80.00% | 90.00% | 23.33% |
| C3C注册后 | 13.33% | 70.00% | 90.00% | 30.00% | 70.00% | 90.00% | 13.33% |
| B3注册后 | 70.00% | 66.67% | 90.00% | 63.33% | 60.00% | 90.00% | 60.00% |

| 候选 | cls_09f8 | cls_1c2a | cls_b8fb | cls_d3af | cls_f608 | new floor |
|---|---:|---:|---:|---:|---:|---:|
| C0 | 3.33% | 100.00% | 73.33% | 86.67% | 6.67% | 3.33% |
| C3A | 0.00% | 96.67% | 76.67% | 86.67% | 13.33% | 0.00% |
| C3B | 0.00% | 56.67% | 70.00% | 80.00% | 0.00% | 0.00% |
| C3C | 3.33% | 80.00% | 76.67% | 83.33% | 3.33% | 3.33% |
| B3 | 40.00% | 86.67% | 76.67% | 86.67% | 76.67% | 40.00% |

C3B确实把旧`6-15`从23.33%提高到80.00%，却把两个新弱类降到0。C3B/C的原型中心已消除cosine distance<0.05碰撞，但逐样本新类floor仍接近0，说明只拉开中心不能修复类内覆盖和新旧score尺度。

## 训练与资源审计

- 所有15个fold的Stage2-B loss均下降；C3A/B平均首末下降0.1433，C3C下降0.2079。
- C3B的Stage2-C有3/15个fold末值反升；C3C虽15/15下降，仍没有转化为new floor。C3A trace中的第21行是0-step注册哨兵，不是第21个optimizer step。
- 三条C3在全部15fold都触及`gamma_abs_max=0.35`，多数在约第10步触顶；继续堆epoch只会在裁剪边界上优化，不是优先方向。

| 候选 | 峰值训练参数 | epoch/step | 状态 | MAC/query | qKNN MAC比 | 适配+注册 | CPU FP32 head均值 | 判定 |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| C0 | 0 | 0/0 | 17,616B | 3,456 | 19.64% | 约16ms | 约0.105ms | 最快但性能低 |
| C3A | 288 | 20/20 | 14,796B | 3,456 | 19.64% | 约852ms | 约1.93ms | 资源PASS，性能门FAIL |
| C3B | 1,440 | 30/30 | 14,796B | 3,456 | 19.64% | 约891ms | 约1.93ms | 资源PASS，性能门FAIL |
| C3C | 1,440 | 30/30 | 14,796B | 3,456 | 19.64% | 约687ms | 约1.92ms | 资源PASS，性能门FAIL |
| B3 | 3,456 | 20epoch/60step | 14,618B | 3,456 | 19.64% | 诊断 | N/A | 性能最好，但step超限 |

C3的MAC和状态优于identity-only单qKNN，但当前NumPy逐样本实现的实测head latency约为C0的18倍，不能只按MAC声称更快；下一版需要向量化并保持逐样本决策语义。

## 结论与下一轮决策

本轮不晋级C3，也不把support-held结果表述为正式query性能。失败可分解为两点：共享对角尺度不足以同时修复旧类floor和新类类内覆盖；新类suffix优化虽然冻结旧raw score，新增类仍通过全注册类竞争造成遗忘。

下一轮直接进入研发与实验，不再扩展数据准备：

1. 把B3的高性能结构压缩成全批次≤30-step版本，保留单IQ的288维`z160+FFT96+RF32`拼接与class-specific head，但减少mini-batch更新和状态。
2. 增加仅由注册support选择的1标量new-group bias及逐旧类support非退化约束；推理仍对全部注册类一次argmax，不读取query角色。
3. 对旧类注册前floor不足，优先测试半径门控的int8旧锚切空间校正；不把地面锚作为source-target分布loss，不更新sealed int8组件。
4. 只有旧类保护通过后才测试对称双medoid；K=1自动退化为单medoid，避免额外数据或伪view。

触发下一轮的核心证据是B3诊断的old/new均73.33%、H72.65%、pooled old/new floor60%/40%，显著高于所有C3，但其60 optimizer steps不合正式上限。研发目标因此是压缩B3并修正注册score尺度，而不是继续扩大前处理。
