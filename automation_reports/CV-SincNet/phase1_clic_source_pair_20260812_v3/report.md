# Phase1 CLIC源域common／proxy／PAIR v3预注册报告

## 状态与目标

- 实验ID：`phase1_clic_source_pair_20260812_v3`。
- 当前状态：`LOCAL_VERIFYING / FORMAL_LAUNCH=0 / NO_PERFORMANCE_RESULT`。
- 操作者：主控Codex；N607唯一runner：`Luna/max`。
- 目标：对训练v5、clean v4和source-LEO v4的F1—F6×C／G源域证据链生成12份common训练收据、12份fixed400 proxy诊断和6份不可变C／G PAIR工件，为后续C描述器、G部署包及同一目标LEO-weak包上的12臂预测提供唯一来源。
- 本阶段不读取或比较性能数值；成功只说明源域工件完整，不代表unknown拒识、target-known准确率或域泛化达标。

## 冻结输入与唯一变化

- 训练根：`runs/phase1_clic12_20260812_v5`。
- clean根：`runs/phase1_clic_postfreeze_20260812_v4`；其12份NPZ均已由production loader重开，单臂21120行，L／V／proxy=`3920／16800／400`，特征、logit有限，角色物理行互斥，source-only与零fit证据闭合。
- source-LEO根：`runs/phase1_clic_source_leo_20260812_v4`。
- 本轮相对PAIR v2只把clean来源与matrix identity推进至v4，并使用当前生产PAIR的逐原件SHA、C／G派生期TOCTOU复验和local4 class-order绑定；训练、TX折、三scene、proxy400、阈值和方法均不改变。
- 用户允许训练数据和测试数据配置相同即可，不要求复用同一个封存目标包；本阶段仍不读取目标数据。

## 入口、资源与输出

- launcher：`code/scripts/launch_phase1_clic_source_pair6_v3_20260812.sh`。
- 正式入口：release内launcher的唯一`nohup bash`调用，retry=`NO`。
- 运行环境：`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`，远端工作根`/home/szu2070436088/2510044040/CV-SincNet`。
- 输出根：`runs/phase1_clic_source_pair_20260812_v3`；日志根：`logs/phase1_clic_source_pair_20260812_v3`；二者启动前必须不存在且不可覆盖。
- 资源：6个CPU fold worker，每worker的BLAS线程固定为2，不占GPU。
- 预期工件：12份`common_training_receipt.json`、12份`proxy_diagnostic.json`、6份`F*_C_vs_G_pair.json`、6行PID表和6份日志。

## 协议与停止规则

- source-L是唯一geometry／tail拟合输入；source-V和fixed400 proxy仅连续评分且fit／threshold-fit均为0；source-LEO只用于冻结三scene tail。
- target／query／truth／role访问必须为0；PAIR只保存aggregate及完整性证据，不保存样本feature、logit或raw IQ。
- 同fold C／G必须闭合training common、clean、source-LEO、binding、physical-order及source class-order；PAIR写入前后逐原件SHA必须不变。
- 错误checkout／hash、覆盖风险、协议访问或至少2个fold出现相同确定性异常时，唯一runner停止本run并保留证据；不得按AUROC、u-gap或其他性能值停止。
- 只有12common＋12proxy＋6PAIR全部可重开且日志无技术异常，才标记`ARTIFACTS_COMPLETE / NO_PERFORMANCE_RESULT`。

## 发布前门与后续

- 本地验证已通过：PAIR／common／proxy聚焦回归`20／20`；launcher在WSL下`bash -n`通过；dry-run精确30行，其中common12、proxy12、pair6，target／query／truth／role参数命中0；`git diff --check`通过。
- launcher SHA-256：`3AFEB89D0424EA06916E815EEF6535AACF5869BB5FCCAF18E6E0B2A8DDA0D16B`。
- 独立首轮审查发现并阻止了发布前的matrix identity错配：launcher已传v4，但PAIR生产常量仍为v2。根因是clean链升级后权威常量未同步；当前最小修复仅将生产权威及CLIC测试夹具推进为`phase1_clic_postfreeze_20260812_v4`，不改变方法、数据或矩阵。
- 修复后PAIR聚焦回归`20／20`、postfreeze全量`135／135`均通过；仅有既有Torch AMP弃用warning，无失败。
- 待完成：新Git提交、独立P0／P1复审、N607 preflight、archive／SCP／远端静态门和唯一正式启动。
- PAIR完成后立即生成6个C predictor descriptor和6个G deployment bundle，做真实前向烟测；随后使用已封存target confirmation v2缓存生成validation、IQ-only package及12臂目标预测。
- 正式评分将同时报告target LEO-weak下的target-known细分、unknown显式拒识和三scene域泛化；缺少合法ADV3B02同配置基线时不得伪造非劣结论。

## v3落地与唯一formal launch

- 冻结commit：`a43b301339fac02b328badee134d0d6835d4c3ab`。工作树除既有未跟踪`conversation_index/`外未改动；commit包含PAIR生产矩阵绑定修复、聚焦测试和本报告预注册版本。
- `git archive`严格取该commit，未本地解包：`clean_commit_211031633.tar`，267530240bytes，SHA256=`2F99733DD35731D455DA9C5AE4CFCAEBFB30B41A75DE4FB1713B66BE040DB018`。SCP恰1次到`/home/szu2070436088/2510044040/CV-SincNet/clean_commit_pair_v3_211031633.tar`；远端bytes/SHA闭合；全新stage解压后原子改名为`releases/phase1_clic_source_pair_20260812_v3_a43b3013`。
- 远端release静态摘要：launcher物理SHA=`3afeb89d0424ea06916e815eef6535aacf5869bb5fccaf18e6e0b2a8dda0d16b`；PAIR entry物理SHA=`1d9a9e578a5edcf8126711400374dd398ad9c04818f2fe162ff5464b361d0ed2`。`py_compile`、PAIR`--help`、`bash -n`均PASS；dry-run恰30行（common12、proxy12、pair6），target/query/truth/role命中0。
- N607直连preflight通过：项目根可见，GPU0–7各0%/1MiB，无相关旧runner。v4run、log、outer和v3 release启动前均ABSENT。输入链只读确认：训练v5的12checkpoint/terminal、clean v4的12NPZ、source-LEO v4的12NPZ+12binding均由launcher核验。
- 唯一正式命令于2026-08-12T21:11:34执行：
  `nohup bash /home/szu2070436088/2510044040/CV-SincNet/releases/phase1_clic_source_pair_20260812_v3_a43b3013/code/scripts/launch_phase1_clic_source_pair6_v3_20260812.sh > /home/szu2070436088/2510044040/CV-SincNet/phase1_clic_source_pair_20260812_v3_outer.out 2>&1 &`
  - `FORMAL_INVOCATION=1`、`RETRY=NO`；outerPID=`2727219`。foldPID表为`2727223,2727224,2727225,2727227,2727229,2727230`，写入`logs/phase1_clic_source_pair_20260812_v3/pids_source_pair6.tsv`（834bytes，SHA256=`bd98f7673699fd8245c8776a42d1346be3e4dd95baae30d4097aa9b42f8bdb8e`）。launcher将每fold固定为6CPU worker、`CUDA_VISIBLE_DEVICES=""`、每worker BLAS线程2。

## 工件QA与最终状态

- 生产工件计数闭合：12`common_training_receipt.json`、12`proxy_diagnostic.json`、6`F*_C_vs_G_pair.json`、6日志、6PID行；PAIR阶段不产生NPZ。outer为0bytes属于正常无标准输出，技术异常扫描（Traceback/RuntimeError/CLICPostfreezePairError/FAILED/fatal/路径或权限错误）为0。
- 只读QA逐fold通过：PAIR schema=`cvs.phase1.clic_postfreeze_pair.v1`、matrix=`phase1_clic_postfreeze_20260812_v4`、training=`phase1_clic12_20260812_v5`、`source_only=true`、`target_artifacts_present=false`；C/G common同fold且scene/class-order/physical/source-split/common-batch SHA一致；raw 14项（checkpoint、terminal、clean、LEO、binding、common、proxy及对应SHA）逐项TOCTOU闭合；clean与LEO manifest/binding source-only闭合；single-LEO physical/received SHA绑定一致；proxy及source-V fit/threshold均为0。未提取或报告AUROC、u_gap等性能字段。

| fold/candidate | source TX class order | common/proxy/pair | pair artifact（bytes，SHA256） | log（bytes，SHA256） | technical verdict |
|---|---|---|---|---|---|
| F1 C/G | 20-15,20-19,6-15,8-20 | 2/2/1 | 300813；`c8f9bf399eacc86e6089889eaba39f461dd36fc1d9364143a04081a30ee92ced` | 302283；`3a89206aa257c0110275dfbbf70e51fd979fa89eeb3a7dd8948730e955a439b3` | PASS；性能=N/A |
| F2 C/G | 14-10,20-19,6-15,8-20 | 2/2/1 | 301512；`e5bfacc2e3fee6402f6d47aea31327d4fbaabacf1bc6dcd9c09ec97440a69f68` | 302982；`4ca776b36e60f77c8f74f1a72f3dcb6079a5e41a200d39fa464415525dbd121b` | PASS；性能=N/A |
| F3 C/G | 14-10,14-7,6-15,8-20 | 2/2/1 | 297225；`731f04b25c418f10244fef5f77a7c42b1c96d889e7523d183517698783783ca7` | 298695；`1f4154c670da458661438a931b819db649062fdd6368dde912ec9f466a3a39ba` | PASS；性能=N/A |
| F4 C/G | 14-10,14-7,20-15,8-20 | 2/2/1 | 299431；`15e88228f91e1bb0d30ab1bdd7e9636735049ec90d3680eb4a70654b786a4b92` | 300901；`5b3921f42b8387573a88a020f1381e0f05b7e9d7fdc684e527a90986ce92b116` | PASS；性能=N/A |
| F5 C/G | 14-10,14-7,20-15,20-19 | 2/2/1 | 300881；`b42582f85ba3439c58c2d602d015662c08eb193954b8cf374ea33fe1c4806b7d` | 302351；`09e61121314d4dfc6f7a35600796d721f94bc5751cfe7bf8c6764f59e8d007b7` | PASS；性能=N/A |
| F6 C/G | 14-7,20-15,20-19,6-15 | 2/2/1 | 300562；`091df2584c8930ea45e50e17deb1ad0a6655ceb4a1d3836fd9ad4dba057a7565` | 302032；`060f4bf9356c2c1707f918f87cebb7456c86b3a334cae256c9a3ecb18e9e05a7` | PASS；性能=N/A |

- 完成后短连接清零：outer及6fold worker均退出，GPU compute-app为空，本地SSH/SCP进程与TCP22连接清零；未重试、未读取性能、未删除或修改非本run数据。最终状态：`ARTIFACTS_COMPLETE / NO_PERFORMANCE_RESULT`。
