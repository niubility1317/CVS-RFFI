# D25多表征拼接support-only原子筛选

## 启动前记录

- experiment ID：`d25_multimodal_concat_support_20260717/support_screen_v1`
- 日期：2026-07-17；operator：Codex
- 状态：`V3_OLD_SCORE_FREEZE_FAILED_V4_LOCAL_VERIFIED`；query始终未打开，v4待同步启动。
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
