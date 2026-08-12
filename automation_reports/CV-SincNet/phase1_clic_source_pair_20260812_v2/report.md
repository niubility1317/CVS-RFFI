# Phase1 CLIC source common／proxy／PAIR v2预注册报告

## 状态与目标

- 实验ID：`phase1_clic_source_pair_20260812_v2`。
- 当前状态：`ARTIFACTS_COMPLETE / FORMAL_LAUNCH=1 / NO_PERFORMANCE_RESULT`。
- 操作者：主控Codex；N607唯一runner：`Luna/max`。
- 目标：沿用v1完全相同的v5训练、clean v2、source-LEO v4、F1—F6×C／G矩阵和三scene规则，生成12份common、12份fixed400 proxy及6份C／G PAIR工件。

## v1技术失败与唯一修复

- v1已不可变封存为`STOPPED_EARLY_SYSTEMIC_TECHNICAL_FAILURE / NO_PERFORMANCE_RESULT`：6／6 common成功，6／6 fold在proxy阶段同一`CLIC proxy clean source-V TX labels drifted`退出，proxy=0、pair=0；未读取性能、未重试。
- 只读inventory证明clean v2的`source_validation_known`为当前local4 source partition内部30%验证切片：F1每个source TX各4200行，F2结构相同；外部单独held-validation TX只存在于checkpoint／manifest审计字段，从未materialize为该角色的feature row。
- v1错误地要求V行TX等于外部held TX。v2唯一语义修复为：L和V行均须完整覆盖当前local4；proxy行仍须精确等于单独proxy TX。V依旧仅连续评分，`fit_rows=0`、`threshold_fit_rows=0`；geometry、tail、阈值、模型、数据、seed、矩阵均不改变。

## v2发布与F1C proxy smoke检查点（2026-08-12）

- 冻结commit=`ee83be7fa4c91adb6edd777917b44ba9690cb54a`；Task7 dirty/untracked未进入archive、未stage。干净archive=`E:\type10-7\code\runner_tmp_phase1_clic_source_pair_20260812_v2_ee83be7f_git_archive.tar`，SHA256=`76B8947F4A4CDC4F20F27FBC41D5844FDCD56FEFDC4EA3BBD2C3704DEF4F8DE6`，bytes=`267089920`。
- SCP恰1次；远端SHA／bytes闭合并原子落地release=`/home/szu2070436088/2510044040/CV-SincNet/releases/phase1_clic_source_pair_20260812_v2_ee83be7f`。launcher SHA=`E1C8292D099A72FF9B46B6BB765AF4081DF642724F2EC151E349B25FA4003F58`。
- 远端静态门通过：PAIR evaluator及依赖py_compile、common/proxy help、launcher bash-n、dry-run精确30行（common12+proxy12+pair6），forbidden target/query/truth/role=0。
- F1C proxy writer技术烟测成功：writer exit=0，临时输出`.smoke_phase1_clic_source_pair_20260812_v2_F1C/proxy_diagnostic.json` schema=`cvs.phase1.clic_proxy_diagnostic.v1`，clean SHA=`42253b0fd58f1bdd08e331082a6967154509b87a678643666134a1a8555cca18`闭合；顶层与嵌套proxy/source-validation fit/threshold rows均0，`tail_policy_used=false`。一次校验脚本误读顶层字段后AssertionError，未重试writer；随后按正确嵌套字段复验SMOKE_PASS。正式run/log/outer仍ABSENT，launch=0，SSH/TCP22清零。
- 本轮发布边界只含PAIR、测试、launcher和本报告；deployment bundle与Task7未提交实现不进入本次archive，也不被本次source PAIR入口消费。bundle同义修复留给其独立集成提交，不能借本轮夹带。

## 冻结输入、输出与入口

- 训练根：`runs/phase1_clic12_20260812_v5`；clean根：`runs/phase1_clic_postfreeze_20260812_v2`；source-LEO根：`runs/phase1_clic_source_leo_20260812_v4`。
- 新输出／日志根：`runs/phase1_clic_source_pair_20260812_v2`与`logs/phase1_clic_source_pair_20260812_v2`，必须不存在且不可覆盖。
- launcher：`code/scripts/launch_phase1_clic_source_pair6_20260812.sh`；正式入口为release内该文件的`bash`调用。
- 执行仍为6个CPU fold worker并行；每fold内部C common→C proxy→G common→G proxy→C／G pair。BLAS线程每worker固定2，GPU不占用。
- dry-run应为30行：common12、proxy12、pair6；target／query／truth／role参数0。

## 协议、停止与完成条件

- source-L是唯一geometry／tail拟合输入；source-V和fixed400 proxy只评分且不定阈值；LEO只用于冻结三scene source tail。target／query／truth／role访问为0。
- 同fold C／G必须闭合training common receipt、received-IQ SHA和physical-order SHA；所有输出为aggregate-only，不保存样本feature／logit／raw IQ。
- 正式launch恰1次，retry=`NO`。错误checkout／hash、覆盖风险、协议访问或至少2个fold出现相同确定性异常时，仅停止确切run-owned进程并保留工件；不得按AUROC／u-gap等性能值停止。
- 预期：12／12 common、12／12 proxy、6／6 pair、6行PID表、6日志；成功仅标记`ARTIFACTS_COMPLETE / NO_PERFORMANCE_RESULT`。

## 发布前门

- TDD：真实结构fixture先使v1检查精确RED；修复后正例与“把外部held TX伪装成V行”负例均通过。
- `ssr-gpu`串行验证：common＋postfreeze共126／126通过，仅3条既有Torch AMP弃用warning；PAIR／测试`py_compile`通过；cached diff-check通过。
- launcher `bash -n`通过；dry-run精确30行，其中common12、proxy12、pair6，target／query／truth／role禁参数0。
- launcher SHA-256：`E1C8292D099A72FF9B46B6BB765AF4081DF642724F2EC151E349B25FA4003F58`。
- 独立终审：`P0=0/P1=0/ALLOW`。审查方独立确认local4正例通过，外部held伪装、缺任一local4类、旧v1 run-ID注入均fail-closed；PAIR／proxy专项8项通过。
- Git提交、archive／release／SCP／真实F1技术烟测、正式launch与工件闭合待回填。

## v2正式运行与工件闭合（2026-08-12）

- 唯一formal launcher invocation=`1`，outer PID=`2604466`，6个CPU fold worker写入`pids_source_pair6.tsv`（6行）；未重试，GPU未占用。
- 工件计数闭合：12/12 `common_training_receipt.json`、12/12 `proxy_diagnostic.json`、6/6 `F*_C_vs_G_pair.json`，6/6日志；worker与outer已退出，run-owned PID=0、GPU compute=0、SSH/TCP22=0。
- common receipts均`source_only=true`。proxy diagnostics schema均`cvs.phase1.clic_proxy_diagnostic.v1`，proxy/source-validation fit rows与threshold rows均为0，`tail_policy_used=false`。
- PAIR records schema均`cvs.phase1.clic_postfreeze_pair.v1`、`source_only=true`、`target_artifacts_present=false`，六fold same-fold/common binding闭合；同fold C/G raw/LEO binding由PAIR重开验证通过。日志技术异常标记（Traceback、TypeError、RuntimeError、CLICPostfreezePairError、ERROR、Exception）=0。
- F1C技术烟测writer exit=0，schema与clean SHA闭合；一次校验脚本误读顶层字段的AssertionError不影响writer，已用正确嵌套字段复验通过。全程未读取AUROC、u_gap或其他性能值；本run仅记`ARTIFACTS_COMPLETE / NO_PERFORMANCE_RESULT`，后续指标分析另行执行。
