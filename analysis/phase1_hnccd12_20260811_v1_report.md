# Phase1 HNCCD 12臂训练实验报告

状态：`TRAINING_TELEMETRY_ANALYZED / POSTFREEZE_LOCAL_IMPLEMENTING / NO_POSTFREEZE_RESULT`

## 1.实验身份、目标与声明边界

- 实验ID：`phase1_hnccd12_20260811_v1`
- 日期：2026-08-11
- 操作方：Codex主控；N607唯一Runner（Terra/max接手）
- 实现commit：`b6afc5a3e19ae3146dd6afcfe8a90abff35f3cbb`
- 候选：P1-HNCCD（head-nullspace cross-covariance decorrelation）
- 对照：C为GeoSat-C `training_final_only`共同续训；G只在同一共同路径上增加固定`0.02L_HNCCD`
- 目标：检验source-L的单LEO表征中，exact-head行空间坐标与其正交残差在receiver×class条件下的线性交叉协方差，是否是影响已知类LEO稳定性和后冻结连续几何的可证伪因素
- 声明边界：本实验不训练unknown、不访问真实unknown、不构成FAR、注册授权、Phase2、Phase3或多卫星协同能力声明。进程落地、技术receipt或checkpoint生成均不等于性能结果

P1-RCMMC已经在完整42步中永久拒绝，本轮不得重命名、拼接、调参或复活RCMMC及更早机制。HNCCD是新的单一结构原语，不读取clean特征做辅助对齐，也不训练Gaussian proxy分数。

## 2.冻结机制

仅对source-known-train L中已有单LEO `feat_joint`计算：

```text
W=model.id_backbone.cls_head.head.weight∈R^(4×160)
L=chol(WW^T)
Q=W^T L^(-T)
u=T(z_leo), ||z||=0时T(z)=0
h=Q^T u
b=u-Qh
C_rc=(H-Hbar)^T(B-Bbar)/n_rc
L_HNCCD=(1/28)Σ_(r,c)||C_rc||_F²
L_G=L_base+0.02L_HNCCD
```

冻结合同：

- source receiver slot固定7个、local class固定4类，共28个cell；
- `n_rc<2`为可微零贡献，始终除28，不按occupied/active cell重归一；
- `W`必须有限、full-row-rank；AMP外FP32 Cholesky和triangular solve失败即fail-closed；
- 禁止`pinv`、epsilon、降秩或fallback；
- C辅助严格N/A/0；G只增加HNCCD；
- `0≤L_HNCCD≤1`，固定`lambda_hnccd=0.02`使原始辅助上界不超过0.02；
- 每个clear/low/rain scene首个正项批需raw-unscaled VJP：LEO `feat_joint`、shared encoder和exact `W`均finite/nonzero；clean与head bias均None或数值零；
- 公共`L_base→W`保持live；
- raw VJP后只允许一次正常AMP backward/unscale/step-update；finite和AMP-skip两支都在下一forward前释放本批所有连图root，禁止第二次backward/unscale/forward、`gc`或`empty_cache`。

## 3.权限、公平与资源

- 训练辅助只读`source_known_train L`及sealed source receiver slot；
- U零iterate/forward/loss/backward/optimizer；
- V、proxy、held、target、day、fold和score对训练、校准、选模、停止与状态均零反馈；
- C/G同一GeoSat-C warm-start、physical row/order、seed、sampler、40E、新AdamW、AMP、single-LEO、三scene和共同`L_base`；
- 无额外model/head forward、view、epoch、参数、EMA、prototype或跨批feature cache；
- 可微FP32只允许当前批`Q[d,4]`、`h[B,4]`、`b[B,d]`和逐cell `[4,d]`统计；禁止`B×d²`或`B×28×d²`；
- 每共同物理batch记录一条峰显存与step-time观测；terminal要求数量逐批闭合，观测只作资源证据，不得反选候选。

## 4.本地文件、SHA与验证

|文件|SHA256|用途|
|---|---|---|
|`analysis/phase1_hnccd_design_20260811.md`|`886a2784ad338d64ef91f2cb028925e37ddfdf7bc42bcf497823d72937f98ea0`|冻结设计与追踪|
|`code/cvsrffi/phase1_hnccd.py`|`7d2d80d69892a42a14cbd32cda82ba001652f04e0976dba765f6da430fee4a9b`|公式、绑定、VJP、AMP、receipt和terminal|
|`code/SSDG/train_ssdg.py`|`7d7674cb0782880e0efaf7cedf3f04b372e373709cc43d093fc41e3246328f2d`|共同forward、G loss、AMP、资源、图释放和终态接线|
|`code/tests/test_phase1_hnccd.py`|`25378f427a0f67037aa7b52a28985ba7bad63172c05fc168ccdd4d927164580f`|29项正负与集成测试|
|`code/scripts/launch_phase1_hnccd12_20260811.sh`|`889d998fbf7612d3c95ea0d6f0b88dc3d05c9e2bbd81c2a25c5c7fe753604c71`|冻结12臂launcher，Git mode100755|

官方Conda hook激活`ssr-gpu`后串行验证：

- `python -m py_compile`：core、trainer、test通过；
- HNCCD focused：29 passed；
- HNCCD+HSCF+RCMMC共享回归：64 passed；
- `train_ssdg.py --help`识别3个HNCCD CLI；
- launcher `bash -n`通过；
- launcher dry-run精确12行、C=6、G=6、40E=12、G `lambda=.02`=6；
- normal/overflow AMP测试均只有一次backward和一次unscale；
- 无GC saved-tensor/root释放测试通过；
- terminal资源负测覆盖空列表、少一项、布尔/负峰值、NaN/负step-time和selection反馈篡改；
- `git diff --check`通过；
- 独立actual-diff审查：`P0=0 / P1=0 / ALLOW`。

本地没有GeoSat-C 4类checkpoint原件；现有本地真实checkpoint均为6类head，禁止截取或重写后冒充严格smoke。唯一Runner必须在launch前对远端F1C真实4类checkpoint执行strict load、无query前向和HNCCD VJP smoke；失败则`launch=0`。

## 5.冻结12臂矩阵与GPU映射

|顺序|候选|fold|arm|GPU|source train TX|known-val TX|proxy TX|
|---:|---|---:|---|---:|---|---|---|
|1|F1C_HNCCD12|1|C|0|20-15,20-19,6-15,8-20|14-7|14-10|
|2|F5G_HNCCD12|5|G|0|14-10,14-7,20-15,20-19|8-20|6-15|
|3|F1G_HNCCD12|1|G|1|20-15,20-19,6-15,8-20|14-7|14-10|
|4|F5C_HNCCD12|5|C|1|14-10,14-7,20-15,20-19|8-20|6-15|
|5|F2C_HNCCD12|2|C|2|14-10,20-19,6-15,8-20|20-15|14-7|
|6|F6G_HNCCD12|6|G|2|14-7,20-15,20-19,6-15|14-10|8-20|
|7|F2G_HNCCD12|2|G|3|14-10,20-19,6-15,8-20|20-15|14-7|
|8|F6C_HNCCD12|6|C|3|14-7,20-15,20-19,6-15|14-10|8-20|
|9|F3C_HNCCD12|3|C|4|14-10,14-7,6-15,8-20|20-19|20-15|
|10|F3G_HNCCD12|3|G|5|14-10,14-7,6-15,8-20|20-19|20-15|
|11|F4C_HNCCD12|4|C|6|14-10,14-7,20-15,8-20|6-15|20-19|
|12|F4G_HNCCD12|4|G|7|14-10,14-7,20-15,8-20|6-15|20-19|

共同配置：`epochs=40`、`label_epochs=40`、`pseudo_epochs=0`、`seed=7281105`、`sat_view_seed=9281105`、`lr=0.0002`、`weight_decay=0.0001`、`batch_size=128`、`lambda_sat_cons=0.10`、`amp=true`、`max_grad_norm=5.0`、`checkpoint_selection=final_only`。每GPU最多2个训练进程。

## 6.N607发布预登记

- 仅普通账号`N607`，禁止管理员账号；
- Python：`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`；
- 项目根：`/home/szu2070436088/2510044040/CV-SincNet`；
- release：`/home/szu2070436088/2510044040/CV-SincNet/releases/phase1_hnccd12_20260811_v1_b6afc5a3`；
- CWD：上述release的`code`目录；
- run根：`/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_hnccd12_20260811_v1`；
- log根：`/home/szu2070436088/2510044040/CV-SincNet/logs/phase1_hnccd12_20260811_v1`；
- outer：`/home/szu2070436088/2510044040/CV-SincNet/logs/phase1_hnccd12_20260811_v1_launcher.out`；
- ManySig：`/home/szu2070436088/2510044040/CV-SincNet/Dataset_WigSig/ManySig.pkl`；
- warm-start根：`/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_loto_clsgeo12_20260808_v1`；
- retry：`NO`；
- 启动所有权：唯一Runner；主控不得重复启动。

冻结唯一命令：

```bash
cd /home/szu2070436088/2510044040/CV-SincNet/releases/phase1_hnccd12_20260811_v1_b6afc5a3/code && nohup env RUN_ID=phase1_hnccd12_20260811_v1 PROJECT_ROOT=/home/szu2070436088/2510044040/CV-SincNet CODE_ROOT=/home/szu2070436088/2510044040/CV-SincNet/releases/phase1_hnccd12_20260811_v1_b6afc5a3/code bash /home/szu2070436088/2510044040/CV-SincNet/releases/phase1_hnccd12_20260811_v1_b6afc5a3/code/scripts/launch_phase1_hnccd12_20260811.sh > /home/szu2070436088/2510044040/CV-SincNet/logs/phase1_hnccd12_20260811_v1_launcher.out 2>&1 < /dev/null &
```

唯一Runner启动前硬门：

1. direct `tools\n607_ssh_preflight.ps1`通过，普通账号、项目根和8张GPU可见；
2. 从commit生成完整LF、无prefix Git archive，核对成员数、`code/code=0`、五冻结文件SHA与launcher mode；
3. 远端release/run/log/outer必须均不存在；
4. ManySig SHA与6个GeoSat-C checkpoint SHA存在且记录；
5. 远端`py_compile`、`train_ssdg.py --help`、`bash -n`、dry-run12(C6/G6)通过；
6. F1C真实checkpoint strict load：missing=0、unexpected=0、exact `W[4,160]`、前向`z=[128,160]`且finite；
7. 使用真实模型、真实F1C checkpoint和仅source-L合成metadata做HNCCD no-query smoke：positive cell存在，LEO/encoder/W raw VJP finite/nonzero，clean/bias None-or-zero，query rows opened=0；
8. GPU并发记录后才允许唯一命令调用1次。SSH超时先清理绑定本地客户端，再只读确认是否landed；不得重发。

## 7.技术健康、停止与工件

预期每臂：

- `final_ssdg.pth`；
- config、resource、heldout、training completion、terminal status；
- `phase1_hnccd_terminal_receipt.json`；
- 完整arm日志。

G臂额外闭合：

- 1200个HNCCD batch及三scene共同coverage；
- 每scene至少一个正项批与一次raw VJP；
- optimizer attempts=effective steps+raw-finite AMP skips；
- raw/material nonfinite计数为0；
- terminal连续skip为0、effective step>0；
- 每共同batch一条资源观测；
- graph-release failure stage不得出现。

预注册技术停止：

- P0协议、权限、checkout/hash、输出覆盖或receiver/class/order绑定错误；
- launcher-wide确定性故障；
- 至少2个不同arm在产生final前出现相同标准化异常指纹；
- Cholesky/full-rank、OOM、CUDA、argparse、路径/权限、SIGSEGV、零final或terminal闭合失败。

停止只依据技术健康，不读取accuracy、loss、floor或proxy表现。停止前绑定本run的PID/CWD/cmdline，仅处理本run进程并保留部分工件。技术失败记为`STOPPED_EARLY_SYSTEMIC_TECHNICAL_FAILURE / NO_PERFORMANCE_RESULT`，不得重跑本ID。

## 8.完成后分析与Phase1条件

只有12/12臂技术闭合后，主控才读取同row性能。训练阶段先检查：

1. 无表征塌缩或HNCCD数值失稳；
2. known RX/class/day稳定；
3. 6fold C/G共同输入与checkpoint闭合；
4. G资源与AMP/图释放没有系统性异常；
5. 每fold输出真实`training_final_only` checkpoint。

训练技术闭合后才允许固定后冻结42步：

`12 clean export + 12 LEO/binding + 12 fixed400 proxy + 6 same-fold pair = 42`。

最终非补偿门保持：

- clean 6/6；
- LEO clear/low/rain共18/18四floor；
- 每fold三scene等权overall 6/6；
- global 18-cell overall；
- fixed400 proxy AUROC和u-gap逐fold严格正，6/6；
- F6重开F1–F5 raw原件；
- 任何门失败则永久拒绝HNCCD，不调参、不挑fold、不改名或拼接。

当前结果：

|矩阵|技术状态|性能状态|最终判定|
|---|---|---|---|
|F1C/F1G…F6C/F6G|ARTIFACTS_COMPLETE / TECHNICALLY_CLOSED / NON_PROMOTABLE_P0_DISABLED|NO_PERFORMANCE_RESULT|不作晋级或拒绝|

## 9.已知风险

- exact head在训练中可能接近秩亏；本轮不加epsilon或fallback，Cholesky失败按技术故障闭合；
- 去相关可能通过head行空间迁移或残差塌缩达到低loss；这不是能力，由clean/LEO/fold/global/proxy非补偿门证伪；
- raw VJP和AMP保留图会增加瞬时显存；批尾无GC显式释放及C/G资源receipt用于验证，不用于选模；
- HNCCD不保证proxy或真实unknown改善；
- 本地缺少严格4类GeoSat-C checkpoint，远端启动前真实smoke是不可跳过的唯一外部正确性门。

## 10.Runner预检与发布封存（2026-08-11）

状态仍为：`LOCAL_VERIFIED / P0=0 / P1=0 / N607_PRELAUNCH_PENDING / NO_PERFORMANCE_RESULT`。

- direct`tools\n607_ssh_preflight.ps1`通过；普通账号`szu2070436088`、项目根、8张RTX3090可见。当前GPU0–7均为0%利用率、约1MiB显存；本run无活动进程。每次短SSH结束后本地`ssh.exe=0`且N607/bridge TCP22均为0。
- 启动前远端release、run、log、outer均为`ABSENT`；ManySig SHA=`2b0a7a7488dd3650bcae7b1d80efbcffd1598aaa671ae6b0a0df2a24dc0f694f`。
- F1C/F2C/F3C/F4C/F5C/F6C warm-start checkpoint均存在，SHA分别为`4d515204f2cea62c5b82313a01b722b3b3d13a3e4fe647ff4b723b69e8a0c040`、`29c7d7ca31d80d90d7c0235fa234707b05866914dc0acdae5c44505af1bbd76d`、`39c6cdd65aade504efdea956db02cc5e762aee299a9e9319c07ed6fb839434b7`、`32d956f44f60844471ba2ef04526c5f40cad0f8bc8acb7249be6035aa85005e4`、`2b9381546878b19e7e8e2106a82b0d0a4672a3012ef79bd7f28eadfd03b75a9f`、`573ca9d039a8c854f9c0927b5b5c303ab8eeaf527ccd42cd0d764b81e630de6f`。
- F2C预注册SHA抄录少末尾`d`；六个checkpoint经只读`sha256sum`逐一复核后更正为上述64位值。这是报告事实修正，不是warm-start输入、checkpoint、矩阵或方法变更。
- 原始Git archive仅作本地证据：`phase1_hnccd12_20260811_v1_b6afc5a3.tar`，262615040B，SHA=`bdebe028a9dee9dcf8f1ff5644b45e0ac1b330413721617e4fb5dbe12eabf435`，4981成员，无prefix，`code/code`=0。其train工作树SHA=`7d7674cb0782880e0eFAF7CEDF3F04B372E373709CC43D093FC41E3246328F2D`对应CRLF；`git show b6afc5a3:code/SSDG/train_ssdg.py`与原始archive成员均为commit-LF SHA=`5b873c26bc00aa01edc4ad0caa9076f7d3ed63394f4abfc75a1e8d4c816e82c7`。
- 为满足远端发布LF/mode合同，在本run artifact目录由上述原始archive生成final artifact-only LF层（只归一化无NUL UTF-8文本EOL；不改科学内容）：`phase1_hnccd12_20260811_v1_b6afc5a3_final_lf.tar`，264089600B，SHA=`41d338266919d743c4971e99086e23cb097c64a501f1ac50f9623545904e460a`，4981成员，common prefix为空，`code/code`=0；4303个无NUL regular成员CR=0；目录模式0755、普通文件0644、launcher模式0755。五关键成员SHA闭合：design=`886a2784ad338d64ef91f2cb028925e37ddfdf7bc42bcf497823d72937f98ea0`、core=`7d2d80d69892a42a14cbd32cda82ba001652f04e0976dbA765f6da430fee4a9b`、train(commit-LF)=`5b873c26bc00aa01edc4ad0caa9076f7d3ed63394f4abfc75a1e8d4c816e82c7`、test=`25378f427a0f67037aa7b52a28985ba7bad63172c05fc168ccdd4d927164580f`、launcher=`889d998fbf7612d3c95ea0d6f0b88dc3d05c9e2bbd81c2a25c5c7fe753604c71`。
- archive QA manifest：`final_archive_manifest.json`（SHA=`e8bcc9bd6009c0417c7742aa50ac6c820a94ceee4130fe1dea4e8fad4930b69b`，2705B）。截至本节，SCP=0、launch=0；strict F1C checkpoint/HNCCD no-query smoke尚未执行。

## 11.Runner静态与严格smoke闭合（2026-08-11）

- 本运行唯一SCP已完成：final LF archive仅落地到预登记release；incoming已不存在。release经精确清理5个历史`__pycache__`目录和45个`.pyc`后复核为4981成员、`code/code=0`、text CR=0、目录0755、普通文件0644、launcher0755；五冻结成员SHA均闭合。清理对象均为已验证的生成字节码，不涉及科学源文件。
- 外置`PYTHONPYCACHEPREFIX`静态门通过：core/trainer/test的`py_compile`3/3通过，`train_ssdg.py --help`的3个HNCCD flag存在，`bash -n`通过，dry-run精确12行、C=6、G=6、40E=12、G `lambda_hnccd=0.02`=6，12项旧机制`enabled=false`均为12；检查结束release仍为pycache=0和4981成员。
- `ManySig.pkl`SHA=`2b0a7a7488dd3650bcae7b1d80efbcffd1598aaa671ae6b0a0df2a24dc0f694f`，F1C–F6C checkpoint均经只读SHA256复核；F2C抄录修正见第10节，不代表任何输入变更。
- 严格F1C no-query smoke通过：真实F1C checkpoint strict load为missing=0、unexpected=0，checkpoint domain head=14；exact `W=[4,160]`，真实CUDA及`leo_clear_weak`前向`z=[128,160]`且finite。仅source-L合成metadata产生28个positive cell；LEO `z`、shared encoder、exact `W`的raw VJP均finite/nonzero；clean和head bias均None-or-zero；`query_rows_opened=0`。smoke只使用临时外置字节码路径，未写release、run或log。
- 至本节唯一启动命令仍未调用：SCP=1、launch=0、retry=NO、NO_PERFORMANCE_RESULT。待新的direct预检、run/log/outer空路径、run PID=0和GPU并发复核后，唯一Runner才可调用第6节命令一次。

## 12.N607唯一启动与首波技术健康（2026-08-11）

- 最终direct预检通过：普通账号、项目根、GPU0–7可见；发射前release=4981成员、pycache=0、launcher SHA闭合，run/log/outer与incoming均为空，run PID=0、GPU计算进程=0。
- 第6节冻结命令只调用1次，retry=NO。提交连接在34秒后超时，但只读落地核验确认远端submit shell PID=`1652232`、launcher PID=`1652233`和12行`pids.tsv`均已生成；本地残留`ssh.exe` PID=`27392`已精确关闭后确认TCP22=0，未重发命令。

|候选|主PID|GPU|首波技术状态|
|---|---:|---:|---|
|F1C_HNCCD12|1652236|0|存活、绑定通过、日志增长|
|F5G_HNCCD12|1652238|0|存活、绑定通过、日志增长|
|F1G_HNCCD12|1652240|1|存活、绑定通过、日志增长|
|F5C_HNCCD12|1652242|1|存活、绑定通过、日志增长|
|F2C_HNCCD12|1652248|2|存活、绑定通过、日志增长|
|F6G_HNCCD12|1652250|2|存活、绑定通过、日志增长|
|F2G_HNCCD12|1652256|3|存活、绑定通过、日志增长|
|F6C_HNCCD12|1652261|3|存活、绑定通过、日志增长|
|F3C_HNCCD12|1652267|4|存活、绑定通过、日志增长|
|F3G_HNCCD12|1652269|5|存活、绑定通过、日志增长|
|F4C_HNCCD12|1652271|6|存活、绑定通过、日志增长|
|F4G_HNCCD12|1652273|7|存活、绑定通过、日志增长|

- 首波核验：12/12主PID存活且CWD均为release的`code`目录；命令行逐臂闭合`run_id`、输出目录、C/G开关与`lambda_hnccd`；12/12 arm日志非空并增长，12个GPU计算进程与映射一致。只扫描技术异常指纹，当前`Traceback/RuntimeError/OOM/Cholesky/full-rank/SIGSEGV/argparse/权限`标记均为0；`final_ssdg.pth`数量为0，仍为运行中而非性能结果。

## 13.终态技术闭合与小型工件回收（2026-08-11）

- 唯一启动已自然终态：12个主PID均已退出、GPU compute进程=0。本run未重启、未重传、未进行第二次SCP；累计`SCP=1`、`launch=1`、`retry=NO`。
- 12/12臂均有`final_ssdg.pth`、config、resource、heldout、completion、terminal status和HNCCD terminal receipt；六个warm-start SHA及每臂selected final checkpoint SHA均闭合。冻结顺序、GPU、40E和C/G lambda闭合。
- 12/12`hnccd_terminal_contract_passed=true`。C臂辅助为N/A/0；G臂均为1200batch、clear/low/rain各400、coverage/positive/raw VJP闭合、每共同batch资源观察1200。G的AMP按`attempts=effective steps+raw-finite skips`闭合；raw/material nonfinite=0、terminal consecutive skip=0、无persistent overflow。
- 12/12统一记录`exit_code=8`与`NON_PROMOTABLE_P0_DISABLED`，这是冻结trainer的formal promotion/P0未启用guard，不是训练异常：`final_guard_reason`为空、`phase1_v2_final_blocked=false`、HNCCD terminal contract通过。该guard同时保持`performance_result_available=false`和`promotion_ready=false`；本Runner未读取任何性能字段。
- `failure receipt=0`；完整arm/outer日志只按技术指纹扫描，`Traceback/RuntimeError/OOM/Cholesky/full-rank/CUDA/argparse/权限/SIGSEGV/graph-release`均为0。每个有界SSH结束后均确认本地`ssh.exe=0`、N607/bridge TCP22=0。
- 终态后只读卫生核验发现训练生成的5个`__pycache__`目录和45个`.pyc`；在12个run PID和GPU compute均为0后，Runner只删除这5个已逐路径验证且仅含`.pyc`的目录。release复核回4981成员、pycache=0、pyc=0，未改科学源文件。

|候选|GPU|final与checkpoint SHA|HNCCD/资源技术合同|终态guard|
|---|---:|---|---|---|
|F1C_HNCCD12|0|闭合|C aux N/A/0；共同资源1200|预期`NON_PROMOTABLE_P0_DISABLED`|
|F5G_HNCCD12|0|闭合|1200；3×400；VJP；AMP1200/1196/4；raw/material=0；terminal skip=0；资源1200|预期`NON_PROMOTABLE_P0_DISABLED`|
|F1G_HNCCD12|1|闭合|1200；3×400；VJP；AMP1200/1196/4；raw/material=0；terminal skip=0；资源1200|预期`NON_PROMOTABLE_P0_DISABLED`|
|F5C_HNCCD12|1|闭合|C aux N/A/0；共同资源1200|预期`NON_PROMOTABLE_P0_DISABLED`|
|F2C_HNCCD12|2|闭合|C aux N/A/0；共同资源1200|预期`NON_PROMOTABLE_P0_DISABLED`|
|F6G_HNCCD12|2|闭合|1200；3×400；VJP；AMP1200/1197/3；raw/material=0；terminal skip=0；资源1200|预期`NON_PROMOTABLE_P0_DISABLED`|
|F2G_HNCCD12|3|闭合|1200；3×400；VJP；AMP1200/1194/6；raw/material=0；terminal skip=0；资源1200|预期`NON_PROMOTABLE_P0_DISABLED`|
|F6C_HNCCD12|3|闭合|C aux N/A/0；共同资源1200|预期`NON_PROMOTABLE_P0_DISABLED`|
|F3C_HNCCD12|4|闭合|C aux N/A/0；共同资源1200|预期`NON_PROMOTABLE_P0_DISABLED`|
|F3G_HNCCD12|5|闭合|1200；3×400；VJP；AMP1200/1194/6；raw/material=0；terminal skip=0；资源1200|预期`NON_PROMOTABLE_P0_DISABLED`|
|F4C_HNCCD12|6|闭合|C aux N/A/0；共同资源1200|预期`NON_PROMOTABLE_P0_DISABLED`|
|F4G_HNCCD12|7|闭合|1200；3×400；VJP；AMP1200/1197/3；raw/material=0；terminal skip=0；资源1200|预期`NON_PROMOTABLE_P0_DISABLED`|

- 本地小型技术bundle：`phase1_hnccd12_20260811_v1_technical_bundle.tar.gz`，SHA256=`18dab2c4436d815c8ff33f7ced9aba12bde30ae916ae9ee6cce27a3ccc81f2fe`，6,874,571B，74个唯一成员。内容为pids、outer+12 arm日志及每臂config/resource/completion/status/HNCCD terminal receipt；未包含`pth/npz/pt/npy/metrics_epoch`，forbidden=0、unexpected path=0。相邻`phase1_hnccd12_20260811_v1_technical_bundle_manifest.json`（SHA256=`669ab8868d89f11db38d86d82bf67a3c3187b8e68a2ed83e3e7282024f0b21ae`，1,449B）记录传输、清单和release卫生回执。远端工件保留不动。
- 本run以`ARTIFACTS_COMPLETE / TECHNICALLY_CLOSED / NON_PROMOTABLE_P0_DISABLED / NO_PERFORMANCE_RESULT`交接；不作任何性能解释、晋级或拒绝。

## 14.主控训练telemetry同row复核（2026-08-11）

在12/12臂技术闭合后，主控只读解析完整arm日志中的40个source-known validation点。该telemetry不含sealed clean/LEO/fixed400 proxy后冻结结果，不能替代42步非补偿门，也不用于改变候选、lambda、fold、seed、checkpoint选择或停止规则。训练checkpoint仍严格为`final_only`；历史best只作为轨迹说明，不替代终态checkpoint。

|fold|C终态source-val|G终态source-val|G−C终态|C历史best|G历史best|G−C历史best|
|---:|---:|---:|---:|---:|---:|---:|
|F1|99.29%|99.30%|+0.01pp|99.33%|99.31%|−0.02pp|
|F2|99.24%|99.18%|−0.06pp|99.26%|99.27%|+0.01pp|
|F3|99.35%|99.31%|−0.04pp|99.37%|99.39%|+0.02pp|
|F4|99.26%|99.23%|−0.03pp|99.35%|99.33%|−0.02pp|
|F5|98.28%|98.10%|−0.18pp|98.28%|98.10%|−0.18pp|
|F6|97.53%|96.33%|−1.20pp|98.32%|98.39%|+0.07pp|

观察边界：F1–F4终态差异很小，F5轻微下降；F6终态G相对C下降1.20pp，但历史best并未下降，提示的是`final_only`终态稳定性风险而非可挑选checkpoint的许可。现有训练证据既不能晋级也不能永久拒绝HNCCD；按预注册继续实现并执行固定42步，由clean6/6、LEO18/18、fold/global overall及fixed400 proxy双门统一裁决。
