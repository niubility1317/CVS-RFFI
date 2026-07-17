# D25多表征拼接support-only原子筛选

## 启动前记录

- experiment ID：`d25_multimodal_concat_support_20260717/support_screen_v1`
- 日期：2026-07-17；operator：Codex
- 状态：`DEVELOPMENT_SUPPORT_ONLY_COMPLETE`；v4已在N607完整形成75行support-only矩阵，query始终未打开，未获得formal authority。
- 目标：在不打开query的前提下，使用同一密封LEO_weak enrollment-only support，对D25的288维分块拼接、不确定度ground-z融合和逐块半径评分执行15fold原子筛选。
- 假设：保留`z_id160+FFT96+RF32`完整288维，同时把块平方能量从D1的辅助分支94.12%支配修正为按维数比例`5/9、1/3、1/9`，可以保留多表征平均增益并改善旧类与新类floor稳定性。
- 对比：`Z0_SUPPORT_ONLY`、`B3_SINGLE_IQ_DIAG_FFTRF`、`D25_C0_DIM_CONCAT`、`D25_C1_UF_GROUNDZ`、`D25_C2_BLOCK_RADIUS`。
- 筛选规模：5候选×3个互斥LEO_weak场景×5个held-rank fold=75条support-fold结果。

## 协议边界

- 复用D22 v4的before/after `enrollment_only`密封support package、seal、formal policy、authorization、signed envelope、class binding和历史int8组件。
- 历史int8组件仍为`UNVERIFIED_UNDER_CURRENT_PROTOCOL`，仅用于用户授权的`PRE_FORMAL_SUPPORT_ONLY_INT8_SCREEN`；本实验不获得formal authority或正式性能声明资格。
- 每个物理support只有一个已叠加一次且仅一次LEO_weak信道的IQ观测。
- `z_id160`、`FFT96`、`RF32`是同一received-IQ的一组三个数学feature blocks，最终只形成一条288维行；`support_view_count=1`、`support_row_multiplicity=1`、`derived_support_rows=0`。
- runner CLI和运行时不接收query root、query seal、truth、role、真实query批次类别数、类别quota、global assignment或scorer输入。
- 不访问clean/source样本、sample-level source/full-precision feature或clean-derived信号。
- held support仅用于预登记leave-two-out确认，不是正式query；结果只能用于support-only方法锁。

## 本地版本与实现

- Git仓库：`E:\type10-7\github_publish\CVS-RFFI-repo`；根目录`E:\type10-7`不是Git仓库，本报告同步维护Git镜像。
- D25核心提交：`f349850d`。
- D25核心SHA256：`c8789679888bee15e9e3167dcdd576458494fd471f5f83b747836720657f75c7`。
- runner基线提交：`912e49c2`；v3 runner SHA256：`38f98b8022dd5f9b6b8d327226b2463fab125c9baf6efd7116b3fe72a96c780d`。
- v4 D25核心SHA256：`2c43008c1f14f6a6173c3680b3af8a8b4015dfde662b0d4fcfb11e74829dac1e`。
- D24依赖SHA256：`2ed2067c4636447f9e013bab2b99d6bc94e149ed5152907fc363b7e802bd2b86`；CIAF依赖SHA256：`f46c5007cb1c0279bf2b27169ad79989eba908f32658c5a4d7f819916381aeb1`。
- D19控制helper SHA256：`7e46db1e99ac40f4e9d7679dcb7f668553d928a0672a7bcf07022383949c8553`。
- 本地文件：`code/cvsrffi/stage2_multimodal_concat_fusion.py`、`code/scripts/run_d25_support_only_concat.py`、`tests/test_run_d25_support_only_concat.py`、`code/scripts/launch_d25_concat_support_screen_20260717.sh`。
- 本地环境：`C:\Users\lh594\.conda\envs\ssr-gpu\python.exe`。
- 当前验证：runner+D25+D24+D23共44项PASS；runner/测试`py_compile`通过；launcher `bash -n`通过；runner `--help`确认无query/truth/scorer CLI。

## 数据与筛选定义

- receiver：沿用D22开发receiver `20-1`。
- seed：沿用开发seed `713101`。
- K=10；每fold每类fit K=8、held K=2。
- 旧类6个；真实seen-new5个。
- 场景：`leo_clear_weak`、`leo_low_elev_weak`、`leo_rain_weak`，三者physical sample/token/received-IQ hash必须两两不交。
- `HELD_RANKS=((0,1),(2,3),(4,5),(6,7),(8,9))`。

## 晋升与停止条件

D25候选相对Z0必须同时满足：

- 15/15fold逐旧类after-old非劣；
- 15/15fold逐新类after-new非劣；
- before/after/new floor、H均非劣；
- forgetting不增加；
- Stage2-C后旧类score列、旧prefix、旧prototype/radius/count逐字节冻结；
- 最差after-old floor或最差joint floor严格改善。

`B3_SINGLE_IQ_DIAG_FFTRF`只作为历史机制的同run诊断对照，不可晋级。若全部D25候选失败，回退Z0并记录负证据，不打开query、不生成正式prediction artifact。

## 资源审计

对每个候选×场景分列：

- trainable parameters、epoch、optimizer steps；
- persistent state、fit scratch、Phase1 int8 logical/serialized bytes；
- backbone forward、FFT96 `O(T log T)`、RF32 `O(T)`及quantile/sort单列；
- concat/head MAC与identity-only单qKNN对照；
- enrollment适配时延、batch=1推理平均/P95、峰值RAM/VRAM；
- no dense query graph、query rows used for fit=0。

## N607计划

- 远端根目录：`/home/szu2070436088/2510044040/CV-SincNet`。
- Python：启动前重新确认；历史为`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`。
- GPU/PID：待preflight和live inventory后确定。
- 远端runner：`/home/szu2070436088/2510044040/CV-SincNet/code/scripts/run_d25_support_only_concat.py`。
- 远端log：`/home/szu2070436088/2510044040/CV-SincNet/logs/d25_multimodal_concat_20260717/support_screen_v4.log`。
- 远端output：`/home/szu2070436088/2510044040/CV-SincNet/runs/d25_multimodal_concat_20260717/output/support_screen_v4`。
- 计划启动命令：`D25_GPU=<preflight后选定GPU> bash code/scripts/launch_d25_concat_support_screen_20260717.sh`。
- 本地到远端同步映射：D25 runner、D25核心和launcher同步到相同repo相对路径；D19 helper仅校验既有远端SHA，不覆盖无关文件。
- 23:03 CST直连preflight通过；N607项目根目录、服务器时间和8张RTX3090可见。
- live inventory：无GPU compute、无active training process；GPU0～7均约10MiB空闲状态。本轮选择GPU0。
- v1曾同步并验证：runner `ea49bf78...8985`、核心`c8789679...75c7`、launcher `7f5bf005...d148`；既有D19 helper `7e46db1e...8553`未覆盖且哈希匹配。
- 远端`py_compile`与launcher `bash -n`通过；同步后本地无残留N607 SSH连接。
- 精确启动命令：`D25_GPU=0 bash code/scripts/launch_d25_concat_support_screen_20260717.sh`。
- PID在启动后补入。

## v1 support前失败与v2闭包修复

- v1 PID：`3508035`；进程已退出，GPU无残留任务。
- 精确错误：`ModuleNotFoundError: No module named 'cvsrffi.stage2_uncertainty_proto_fusion'`。
- 失败发生在Python import阶段，尚未执行manifest/materialization或打开support；output目录未形成，query始终不可达。因此v1不是性能负结果。
- 根因：首次source closure只锁runner、D25核心和D19 helper，漏同步D25核心的D24依赖。
- v2修复：candidate lock增加D24与CIAF SHA；launcher切换独立`support_screen_v2`日志/output/PID/pycache并校验五成员闭包，保留v1日志不覆盖。
- v2本地验证：runner/launcher语法通过；runner+D25+D24+CIAF共42项PASS。
- v2 launcher SHA256：`4179d1c7c398bd58736961c1a14ffc362d098a8f5ba95ddd1d471383a3d3524d`。

## v2 support-fold失败与v3类序修复

- v2 PID：`3510903`；已退出。
- v2越过import、manifest、signed authority和support materialization，在D25 C0首个fold触发`D25 registered class order drift`后fail closed。
- 根因：纯target旧类fit默认按类名排序，而正式manifest使用预注册class handle顺序；ground候选因int8组件自带registry而没有该歧义。
- v2已经打开合法LEO_weak support，但未打开query、未产生75行结果或可评价性能；output不得作为实验结果使用。
- v3修复：`fit_old_concat`新增显式`registered_classes`，无ground与ground路线都严格沿用manifest顺序；新增非字典序旧类回归。
- v3本地验证：runner、D25、D24、D23共45项PASS；launcher `bash -n`通过。
- v3 runner/core/launcher SHA256分别为`38f98b80...c780d`、`a950d663...c296`、`f351f0c7...89ea`；使用独立`support_screen_v3`路径保留v2证据。

## v3旧列冻结失败与v4分离点积修复

- v3 PID：`3513836`；已退出。
- v3越过类序检查并完成新类append，在`D25 old score columns changed after registration`硬断言处fail closed。
- 旧prototype、radius和old-prefix SHA未改变；根因是NumPy对注册前6列与注册后11列矩阵采用不同形状的点积kernel，旧列产生末位浮点差异。
- 协议要求bitwise冻结，因此不放宽为容差比较。v4让旧prefix与新suffix始终分别点积后拼接，使注册前后旧列使用完全相同的6列运算形状。
- v4新增6旧+5新随机几何bitwise回归；runner+D25+D24+D23共46项PASS。
- v4 core/launcher SHA256分别为`2c43008c...ac1e`、`cf9d48c1...57d2`；使用独立`support_screen_v4`路径。

## 三次修复回顾

- v1是传递依赖未闭合，v2是纯target类序未绑定manifest，v3是score计算形状导致bitwise冻结失败；三者都没有形成完整75行矩阵，均不得作为性能证据。
- 三次均保持LEO_weak-only、单物理样本单观测、无clean/source、无query/truth/role/quota/global assignment边界；没有为通过而放宽任何硬断言。
- v4不增加候选、不改support、K-shot或选择门，只修复旧列确定性计算路径。若v4仍出现同一score冻结问题，将停止重复启动并转为本地/远端最小复现，而不是继续盲目重跑。

## 预期产物

- `RECEIPT.json`
- `training_log.jsonl`
- `selection.json`
- `support_audit.json`
- `resource_audit.json`
- `geometry_audit.json`

不得输出原始IQ、样本级特征、FP32 prototype向量、query prediction、truth sidecar或score table。

## v4完成状态与证据边界

- v4 PID：`3516759`；GPU0；远端运行耗时`12.3914s`，进程已正常退出。
- 精确运行入口：`D25_GPU=0 bash code/scripts/launch_d25_concat_support_screen_20260717.sh`。
- 远端log：`/home/szu2070436088/2510044040/CV-SincNet/logs/d25_multimodal_concat_20260717/support_screen_v4.log`。
- 远端output：`/home/szu2070436088/2510044040/CV-SincNet/runs/d25_multimodal_concat_20260717/output/support_screen_v4`。
- 本地证据镜像：`E:\type10-7\automation_reports\CV-SincNet\d25_multimodal_concat_support_20260717\remote_output_v4`。
- `RECEIPT.json`确认5候选×3场景×5fold=75行；`selected_candidate=Z0_SUPPORT_ONLY`、`selected_positive_route=false`。
- `query_opened=false`、query rows/labels均为0；formal/performance claim均为false；历史int8组件仍为`UNVERIFIED_UNDER_CURRENT_PROTOCOL`。
- 完整stdout扫描只有最终JSON结果行，无`Traceback`、`RuntimeError`、OOM、`Killed`、warning、独立单词`NaN`或`Inf`。

本节结果全部是开发support内部held-rank诊断，不是正式query准确率，不得外推为5receiver×5seed独立确认性能。

## 75行联合结果

| 候选 | 机制 | 注册前旧类 | 注册后旧类 | seen-new | H(old,new) | 旧类遗忘 | 对Z0非劣fold | 最差旧/新类floor | 判定 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| `Z0_SUPPORT_ONLY` | z160 identity-only prototype | 71.11% | 48.33% | 52.67% | 48.97% | 22.78pp | 15/15 | 0%/0% | 回退基线 |
| `B3_SINGLE_IQ_DIAG_FFTRF` | 288维旧D1式诊断头 | 86.67% | 73.33% | 73.33% | 72.65% | 13.33pp | 7/15 | 0%/0% | 仅诊断，不可晋级 |
| `D25_C0_DIM_CONCAT` | 160+96+32按维数能量拼接 | 71.67% | 50.56% | 54.00% | 50.35% | 21.11pp | 15/15 | 0%/0% | 非劣但无严格floor改善 |
| `D25_C1_UF_GROUNDZ` | C0+不确定度ground-z旧类融合 | 65.56% | 50.00% | 42.00% | 43.81% | 15.56pp | 1/15 | 0%/0% | 拒绝直接中心融合 |
| `D25_C2_BLOCK_RADIUS` | C1+逐块半径似然 | 61.67% | 15.56% | 65.33% | 23.67% | 46.11pp | 0/15 | 0%/0% | 拒绝当前半径评分 |

C0满足15/15fold相对Z0逐类非劣，但因最差floor没有严格改善而按预注册规则不晋升。B3均值明显更高，说明同一received-IQ的FFT96/RF32确有可用身份信息；其60 optimizer steps超过正式50-step上限且旧D1辅助能量占比94.12%，因此只作为机制上界。

## 逐场景结果

| 候选 | LEO_weak场景 | 注册前旧类 | 注册后旧类 | seen-new | H | 遗忘 |
|---|---|---:|---:|---:|---:|---:|
| B3 | clear | 90.00% | 75.00% | 82.00% | 77.51% | 15.00pp |
| B3 | low_elev | 81.67% | 70.00% | 72.00% | 70.25% | 11.67pp |
| B3 | rain | 88.33% | 75.00% | 66.00% | 70.19% | 13.33pp |
| C0 | clear | 73.33% | 50.00% | 56.00% | 51.78% | 23.33pp |
| C0 | low_elev | 66.67% | 46.67% | 54.00% | 46.40% | 20.00pp |
| C0 | rain | 75.00% | 55.00% | 52.00% | 52.87% | 20.00pp |
| C1 | clear | 68.33% | 60.00% | 44.00% | 49.92% | 8.33pp |
| C1 | low_elev | 56.67% | 40.00% | 46.00% | 41.06% | 16.67pp |
| C1 | rain | 71.67% | 50.00% | 36.00% | 40.43% | 21.67pp |
| C2 | clear | 70.00% | 21.67% | 64.00% | 30.83% | 48.33pp |
| C2 | low_elev | 51.67% | 15.00% | 66.00% | 23.12% | 36.67pp |
| C2 | rain | 63.33% | 10.00% | 66.00% | 17.07% | 53.33pp |

## 逐类floor诊断

旧类class handle与TX绑定为：`cls_75aa→14-10`、`cls_8b02→14-7`、`cls_1f33→20-15`、`cls_f8df→20-19`、`cls_a53c→6-15`、`cls_33bb→8-20`。

| 候选 | 14-10 | 14-7 | 20-15 | 20-19 | 6-15 | 8-20 |
|---|---:|---:|---:|---:|---:|---:|
| B3注册后均值 | 70.00% | 66.67% | 90.00% | 63.33% | 60.00% | 90.00% |
| C0注册后均值 | 13.33% | 70.00% | 86.67% | 20.00% | 23.33% | 90.00% |
| C1注册后均值 | 50.00% | 33.33% | 83.33% | 16.67% | 26.67% | 90.00% |
| C2注册后均值 | 0.00% | 3.33% | 20.00% | 3.33% | 0.00% | 66.67% |

| 候选 | new `cls_09f8` | new `cls_1c2a` | new `cls_b8fb` | new `cls_d3af` | new `cls_f608` |
|---|---:|---:|---:|---:|---:|
| B3 | 40.00% | 86.67% | 76.67% | 86.67% | 76.67% |
| C0 | 3.33% | 100.00% | 73.33% | 86.67% | 6.67% |
| C1 | 10.00% | 83.33% | 53.33% | 53.33% | 10.00% |
| C2 | 43.33% | 70.00% | 76.67% | 66.67% | 70.00% |

主要floor不是平均域偏移，而是`14-10/20-19/6-15`旧类与`cls_09f8/cls_f608`新类的局部碰撞。后续优化必须显式优化support逐类floor和类间margin，不能只提升总体均值。

## 几何与半径根因

| 候选 | clear碰撞/55 | low_elev碰撞/55 | rain碰撞/55 | 最小gap范围 |
|---|---:|---:|---:|---:|
| C0 | 31 | 40 | 37 | -0.8772～-0.6763 |
| C1/C2 | 18 | 18 | 20 | -0.6763～-0.4541 |

ground-z融合减少了几何碰撞数，却没有改善联合分类。原因是C1/C2的融合旧类平均半径被压到clear/low/rain的`0.0195/0.0297/0.0320`，而新类半径仍为`0.2440/0.2732/0.3388`。C2直接使用独立半径似然后，半径尺度严重不对称，旧类距离惩罚被放大，新类普遍压过旧类，导致旧类遗忘升至46.11pp。结论是停止直接旧中心融合和未校准的独立半径评分；int8地面锚下一轮只作为旧类训练正则/冻结先验，不直接替换target prototype。

## 资源与Pareto

| 候选 | 可训练参数 | epoch/step | 持久状态 | head MAC/query | 相对qKNN MAC | K10 prototype状态比FP16单qKNN | 结论 |
|---|---:|---:|---:|---:|---:|---:|---|
| Z0 | 0 | 0/0 | 7,834B | 1,760 | 10.00% | 0.223× | 最轻回退 |
| B3 | 3,456 | 20/60 | 14,618B | 3,456 | 19.64% | 未作为正式状态基线 | 超50-step，仅诊断 |
| C0 | 0 | 0/0 | 17,616B | 3,456 | 19.64% | 0.500× | 拼接零训练基线 |
| C1 | 0 | 0/0 | 43,044B | 3,456 | 19.64% | 1.223× | 当前dense int8逻辑状态拖累 |
| C2 | 0 | 0/0 | 43,044B | 3,522 | 20.01% | 1.223× | 精度与状态均不取 |

qKNN参考为K10、11类、288维FP16逐样本状态35,200B和17,600 MAC/query。C0将状态减半，head MAC减少80.36%；C1/C2因当前历史int8组件逻辑状态25,428B而超过qKNN，后续必须使用中心+偏移+半径的联合密封紧凑表示后再审计。

| 场景 | backbone ms/physical sample | FFT96 ms/sample | RF32 ms/sample | FFT96+RF32额外开销 |
|---|---:|---:|---:|---:|
| clear冷启动 | 35.688 | 0.206 | 0.394 | 0.600ms |
| low_elev稳态 | 3.143 | 0.151 | 0.392 | 0.543ms |
| rain稳态 | 3.136 | 0.152 | 0.431 | 0.582ms |

clear含首次CUDA/模型冷启动；稳态额外FFT96+RF32为约`0.56ms/physical sample`。所有算子都在同一received-IQ上计算一次，不增加support行、物理样本或LEO信道视图。

## v4产物哈希

| 产物 | SHA256 |
|---|---|
| `training_log.jsonl` | `c756b0f887a1c134a148869c8f925831c6c57e708feae1236de0d2be37af2639` |
| `selection.json` | `05b969a2b8deefb153c98312b59e7482ef68d697640177621658333f3d45f364` |
| `support_audit.json` | `f3627ba193f35890158f556a9ea66a945c7c25567f3f7fe23c5a77ad3e1ee04d` |
| `resource_audit.json` | `31ece6ff1cd7ee313c81f24536490ae2c2680a8a4978be62dd9ddcafd5b482a7` |
| `geometry_audit.json` | `92b01f0eea74311d49807a4487c26d02d91c870115df221b1ef2ac47bb70d8fc` |
| `RECEIPT.json` | `e2119d21b38265f8cb64603c74216bd0f12f239208a1c0714725d6f86ebbe181` |
| `support_screen_v4.stdout.log` | `feb0c9a7094058ad0f3b8bffed0bdec5f41b5718cbdb61675cbc96247c466061` |

## 下一轮决策

保留288维拼接主线，不降维、不构造额外view。下一候选D25-C3采用极轻量block-diagonal适配：只训练288个对角尺度，块内中心化并裁剪，Stage2-B最多20个full-batch step；Stage2-C默认0-step闭式append，可选最多30步且只训练新类suffix，总步数不超过50。优化目标必须含逐类smooth worst-floor与类间margin；旧类prototype、旧score前缀和共享对角参数在注册阶段逐字节冻结。地面int8旧类锚只作为旧类稳定正则，不直接进入最终旧中心评分。正式query仍保持完全不可达，先完成新的support-only原子筛选；只有达到预注册floor门后才允许生成新的方法锁并进入正式独立确认矩阵。
