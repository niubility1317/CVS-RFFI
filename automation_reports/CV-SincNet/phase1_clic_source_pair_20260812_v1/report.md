# Phase1 CLIC source common／proxy／PAIR v1预注册报告

## 状态与目标

- 实验ID：`phase1_clic_source_pair_20260812_v1`。
- 当前状态：`STOPPED_EARLY_SYSTEMIC_TECHNICAL_FAILURE / FORMAL_LAUNCH=1 / NO_PERFORMANCE_RESULT`。
- 操作者：主控Codex；N607唯一runner：`Luna/max`。
- 目标：复用已完成且不可覆盖的v5训练、clean v2和source-LEO v4工件，为F1—F6×C／G生成12份common receipt、12份fixed400 proxy diagnostic，并为每个fold生成1份C／G source-only PAIR记录，共6份。
- 假设：完整重开checkpoint、terminal、clean、LEO NPZ／binding后，C／G同fold的训练物理顺序和received-IQ绑定一致；source-L拟合的Gaussian geometry与三场景tail policy可被无target输入地封存，为后续bundle及target盲态评分提供冻结source规则。

## 冻结输入与矩阵

- 训练根：`/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_clic12_20260812_v5`。
- clean根：`/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_clic_postfreeze_20260812_v2`。
- source-LEO根：`/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_clic_source_leo_20260812_v4`。
- 新输出根：`/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_clic_source_pair_20260812_v1`。
- 新日志根：`/home/szu2070436088/2510044040/CV-SincNet/logs/phase1_clic_source_pair_20260812_v1`。
- postfreeze matrix ID：`phase1_clic_postfreeze_20260812_v2`；训练run ID：`phase1_clic12_20260812_v5`。
- 场景顺序固定为`leo_clear_weak,leo_low_elev_weak,leo_rain_weak`。

|fold|C候选|G候选|source TX（固定顺序）|预期工件|
|---:|---|---|---|---|
|1|F1C_CLIC12|F1G_CLIC12|20-15,20-19,6-15,8-20|C／G common+proxy；F1 pair|
|2|F2C_CLIC12|F2G_CLIC12|14-10,20-19,6-15,8-20|C／G common+proxy；F2 pair|
|3|F3C_CLIC12|F3G_CLIC12|14-10,14-7,6-15,8-20|C／G common+proxy；F3 pair|
|4|F4C_CLIC12|F4G_CLIC12|14-10,14-7,20-15,8-20|C／G common+proxy；F4 pair|
|5|F5C_CLIC12|F5G_CLIC12|14-10,14-7,20-15,20-19|C／G common+proxy；F5 pair|
|6|F6C_CLIC12|F6G_CLIC12|14-7,20-15,20-19,6-15|C／G common+proxy；F6 pair|

## 科学与协议边界

- 仅source-L拟合geometry和三场景tail；source-V与fixed400 proxy仅连续评分。proxy、held validation、LEO、target、query均不得进入fit或threshold fit。
- 本run不读取target、query truth／role、target clean IQ或任何性能sidecar；`target_artifacts_present=false`。
- 同fold C／G必须重开同一物理顺序receipt，且LEO binding中的received-IQ SHA与physical-order SHA相同；不允许调用方注入几何、阈值或policy state。
- common receipt仅从实际checkpoint与terminal导出；proxy diagnostic仅从正式clean NPZ重算；PAIR再次逐原件重开并逐值复算，任何路径、SHA、row、scene、arm或fold漂移均fail-closed。
- 本run只产生source规则与技术证明。AUROC、u-gap等即使出现在proxy工件中，也不得在本阶段用于C／G选臂、调参、停止或性能结论。

## 入口、资源与运行合同

- launcher：`code/scripts/launch_phase1_clic_source_pair6_20260812.sh`。
- Python：`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`；工作目录为不可变release的`code`。
- launcher按6个fold并行启动CPU worker；每个worker内部顺序执行C common、C proxy、G common、G proxy、C／G pair。`CUDA_VISIBLE_DEVICES`为空，BLAS线程每worker固定为2，不占用GPU训练槽。
- 正式入口：`bash <release>/code/scripts/launch_phase1_clic_source_pair6_20260812.sh`，formal launch恰1次，retry=`NO`。
- 预期闭合：12／12`common_training_receipt.json`、12／12`proxy_diagnostic.json`、6／6`F*_C_vs_G_pair.json`、6行PID表、6份fold日志；run-owned进程及SSH连接全部退出。

## 本地验证与发布门

- 当前核心实现commit：`25ff2c3362f04fc3aae0f55c34c6da1c1050209e`。
- `ssr-gpu`下`py_compile`通过；common receipt与完整postfreeze联合回归=`125 passed`，仅3条既有Torch AMP弃用warning。
- launcher`bash -n`通过；dry-run=`30`行，其中common=`12`、proxy=`12`、pair=`6`，target／query／truth／role参数=`0`。
- launcher SHA256=`0039046E2C528EF2758127904728DF8BC37B8AC26269C422A712E5D492B14427`。
- 独立终审结论：`P0=0/P1=0/ALLOW`。路径注入攻击、v4 LEO binding兼容、同fold C／G绑定、CPU并发和source-only边界均通过；终审另以`ssr-gpu`运行22项common／PAIR／F6相关测试，全绿。
- `reopen_f6_raw_artifacts`不进入本入口：它需要F1—F5的G bundle，属于后续sealed60／F6聚合阶段；本run生成的6份PAIR是其必要前置原件。
- N607前先做直连preflight、资源与新run／log路径ABSENT检查；Git archive至多SCP一次；远端SHA、compile、help、bash语法、dry-run通过后唯一正式launch。

## 健康停止与结果边界

- 错误checkout／hash、输出覆盖、target／query访问、协议字段漂移或至少2个fold在pair前出现同一确定性异常时，只停止本run的确切进程树并保留所有工件。
- 不因AUROC、u-gap或任何性能数值停止。技术失败标记`STOPPED_EARLY_SYSTEMIC_TECHNICAL_FAILURE / NO_PERFORMANCE_RESULT`；完整工件只标记`ARTIFACTS_COMPLETE / NO_PERFORMANCE_RESULT`。
- 完成后回填commit／文件SHA、archive／release／SCP、PID／日志／工件计数与逐fold技术结论；性能分析属于后续独立阶段。

## v1唯一正式运行结果（2026-08-12）

- 唯一formal launcher invocation=`1`，outer PID=`2578494`；6个CPU fold worker按冻结顺序启动并写入`pids_source_pair6.tsv`（6行），未重试。
- 6/6 common receipt成功生成；随后每个fold在C proxy阶段、任何proxy/pair工件写出前，以同一确定性异常退出：`CLICPostfreezePairError: CLIC proxy clean source-V TX labels drifted`（`evaluate_phase1_clic_postfreeze_pair.py:1348`）。异常日志=6/6、Traceback=6/6。
- 工件计数：common=`6/12`（仅各fold C common先完成）、proxy=`0/12`、PAIR=`0/6`；6份fold日志和PID表保留。outer与6个worker已自然退出，未执行额外kill；GPU compute=0、run-owned PID=0、SSH/TCP22=0。
- 该共同异常满足“至少2个fold在pair前同一确定性异常”系统性技术停止规则。结论：`STOPPED_EARLY_SYSTEMIC_TECHNICAL_FAILURE / NO_PERFORMANCE_RESULT`；未读取AUROC、u-gap或任何性能值，不得将本run用于性能结论。后续修复必须新commit、新run ID，禁止重试或覆盖v1。

## v1发布与静态门检查点（2026-08-12）

- 冻结commit=`8fa919c1855806316088ad598aeb09cdc265563f`；Task7 dirty/untracked未打包、未stage。干净archive=`E:\type10-7\code\runner_tmp_phase1_clic_source_pair_20260812_v1_8fa919c1_git_archive.tar`，SHA256=`5C06A9B7316C61F23943B9B2D32D261FDC39EDFD5522C483F766F455FDD0C494`，bytes=`267079680`。
- SCP恰1次，远端archive SHA／bytes闭合；原子release=`/home/szu2070436088/2510044040/CV-SincNet/releases/phase1_clic_source_pair_20260812_v1_8fa919c1`。launcher SHA=`0039046E2C528EF2758127904728DF8BC37B8AC26269C422A712E5D492B14427`。
- 远端静态门通过：`evaluate_phase1_clic_postfreeze_pair.py`及依赖`py_compile`、common/proxy `--help`、launcher`bash -n`、dry-run精确30行（common=12、proxy=12、PAIR=6），输出无target/query/truth/role输入。一次只读脚本误假设不存在的`export_phase1_clic_pair.py`而退出，未写远端、未启动；随后已按release实际入口复验通过。
- v1正式launch仍为0；下一步在报告提交后，唯一调用`bash "$REL/code/scripts/launch_phase1_clic_source_pair6_20260812.sh"`，retry=NO。

## v1根因只读inventory（2026-08-12）

- clean v2 F1C/F1G/F2C `source_clean_proxy.npz`均为21120行，role计数固定为`labeled_fit=3920`、`source_validation_known=16800`、`proxy_unknown=400`。manifest分别声明F1 source=`20-15,20-19,6-15,8-20`、known=`14-7`、proxy=`14-10`；F2 source=`14-10,20-19,6-15,8-20`、known=`20-15`、proxy=`14-7`。
- 但`tx_ids`按role实际为：F1/F1G labeled四source各980、validation仍四source各4200、proxy=14-10×400；F2C labeled四source各980、validation仍四source各4200、proxy=14-7×400。即`source_validation_known`物理TX未保留manifest声明的held-validation TX，转为局部source class标签；该语义漂移在F1/F2及不同arm均一致，并非单臂随机损坏。
- 该字段不一致直接触发`evaluate_phase1_clic_postfreeze_pair.py:1348`的`CLIC proxy clean source-V TX labels drifted`，解释6/6 fold相同失败；inventory仅读取manifest/role/TX元数据，未读取features/logits/性能。修复需新的clean exporter或兼容映射与新run，禁止在v1上重试。
