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
