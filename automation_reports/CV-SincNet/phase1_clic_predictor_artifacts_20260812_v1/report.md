# Phase1 CLIC C／G predictor artifacts v1预注册报告

## 状态与目标

- 实验ID：`phase1_clic_predictor_artifacts_20260812_v1`。
- 当前状态：`LOCAL_VERIFIED / REVIEW_ALLOW / FORMAL_LAUNCH=0 / NO_PERFORMANCE_RESULT`。
- 操作者：主控Codex；N607唯一runner：`Luna/max`。
- 目标：从训练v5、clean v4、source-LEO v4及PAIR v3不可变原件生成6个C predictor descriptor、6个C训练配置原件和6个G deployment bundle，为同一target confirmation配置上的12臂零适配预测提供可重开模型输入。
- 本阶段不打开target cache／IQ／truth，不读取或计算性能；成功只表示12个predictor工件技术闭合。

## 冻结输入与输出

- 训练：`runs/phase1_clic12_20260812_v5/F{1..6}{C,G}_CLIC12/{final_ssdg.pth,phase1_clic_terminal_receipt.json}`。
- clean：`runs/phase1_clic_postfreeze_20260812_v4/F{1..6}{C,G}_CLIC12/source_clean_proxy.npz`。
- source-LEO：`runs/phase1_clic_source_leo_20260812_v4/F{1..6}{C,G}_CLIC12/{source_leo.npz,source_leo.binding.json}`。
- PAIR：`runs/phase1_clic_source_pair_20260812_v3/F{1..6}_C_vs_G_pair.json`，已完成12common＋12proxy＋6PAIR QA，逐原件SHA、source-only、local4 class-order和零fit／threshold闭合。
- 输出：`runs/phase1_clic_predictor_artifacts_20260812_v1`；日志：`logs/phase1_clic_predictor_artifacts_20260812_v1`；启动前必须不存在且不可覆盖。
- 预期：6份`c_predictor_state.json`、6份`c_predictor_state.train_config.json`、6份`g_deployment_bundle.zip`、6份日志和6行PID表。

## 入口、资源与协议

- launcher：`code/scripts/launch_phase1_clic_predictor_artifacts12_v1_20260812.sh`；正式入口为release内唯一`nohup bash`调用，retry=`NO`。
- 6个CPU fold worker，每fold先C再G，BLAS线程固定为2，不占GPU。
- C描述器必须只从对应PAIR原件重开C状态、checkpoint／terminal／clean SHA和source local4顺序；禁止独立policy注入。
- G bundle必须从真实G checkpoint、clean、source-LEO和binding派生并重建真实模型，封存source local4顺序和candidate训练数据配置；不得封装raw checkpoint、样本行或target数据。
- target／query／truth／role／target-fit／threshold更新均为0；任何输出覆盖、原件SHA漂移、fold／arm错配或至少2fold相同确定性异常均触发本run技术停止，不重试、不读性能。

## 发布前门与下一步

- 首轮独立审查发现真实v5的C描述器把`source_split_receipt`中的轴索引误当成物理RX／day标签，结论为`P0=0，P1=1，NO-GO`，因此未向N607发布或启动。
- 已按真实clean v4合同修复：receipt只绑定轴索引及基数；物理RX／day由clean manifest的`source_receiver_ids／source_day_ids`提供并与导出行精确核对。真实v5缺任一物理轴字段均fail-closed；legacy回退只允许确有checkpoint `split_info`的旧工件。
- 同轮闭合v5 checkpoint中空`wisig_pkl_sha256`：C／G均改由clean manifest封存的真实WiSig SHA建立数据身份；若checkpoint也有非空SHA则必须一致，真实v5两处都缺失时拒绝。
- 新增真实v5无`split_info`的C正例及缺RX／day字段负测；C／G物理轴和数据SHA聚焦`4／4`通过。完整postfreeze回归`138／138`、Task5 core回归`190／190`通过；3个生产／测试文件`py_compile`及`git diff --check`通过。
- launcher `bash -n`通过，dry-run精确12行（C6＋G6）。命令中没有cache、package、prediction、truth、score或目标适配参数；字符串`target`仅存在于生产模块名`cvsrffi.phase1_clic_target_leo`，不代表目标数据访问。
- launcher SHA-256：`26DC30E88DDAA59E637E6304AC820D5E2C9BFA00167A6B532C0B01163BB0FD55`；`git diff --check`通过。
- fresh独立复审commit`b27a24f0`：`P0=0，P1=0，ALLOW`。复审实际重跑v5聚焦`4／4`、launcher语法和dry-run`12／12`，确认禁止target／truth／score／role／cache／package参数为0，未访问N607。
- 待完成：立即交N607唯一runner完成preflight／commit archive／唯一SCP／远端静态门、唯一启动和12工件零IQ真实模型重开烟测。
- 工件闭合后，使用target confirmation v2已验证缓存派生VALIDATED_ONCE收据和known-test配置，封装一个IQ-only package，并对12臂分别发布预测；评分阶段同时报告三scene target-known DG、unknown拒识及域泛化，并只与同训练／测试数据配置的合法ADV3B02原件比较。

## N607落地、静态门与唯一启动

- 冻结commit：`74c42be1b2027ec569d7efef3f47eef1cf6b02e6`；本地dirty仅既有untracked`code/configs/phase1_clic_target_test_semantics_20260812_v1.json`与`conversation_index/`，均未进入archive、未修改。
- `git archive`严格取冻结commit，未本地解包：`clean_commit_213751415.tar`，267550720bytes，SHA256=`6DC48647285253C78847C9A575AB7FFEFC2D32B9165C79AE73D38CFDA25B93E5`。SCP恰1次（初次shell超时返回124，但客户端自然退出；随后只读核实已完整落地，未重试）至`/home/szu2070436088/2510044040/CV-SincNet/clean_commit_predictor_v1_213751415.tar`，远端bytes/SHA闭合。指定父目录原先不存在，创建`/home/szu2070436088/2510044040/releases`后在全新stage解压并原子改名为`/home/szu2070436088/2510044040/releases/phase1_clic_predictor_artifacts_20260812_v1_74c42be1`；未预建run/log/outer。
- 远端release静态文件物理SHA：launcher=`26dc30e88ddaa59e637e6304ac820d5e2c9bfa00167a6b532c0b01163bb0fd55`（与冻结SHA一致）；C entry=`71b412979c27c945191a6c2125bb8fb640d5377390a1a84c5739947e89fe087e`；G entry=`da3cd2174114927346eb5a6c6249d3fec09e0cb0827bdea2cf0daeb463520a3c`。C/G`py_compile`、C module`--help`、G`--help`、`bash -n`均PASS；launcher dry-run恰12行（C6＋G6），禁止参数flag（target/truth/score/role/cache/package）=0；run/log/outer启动前ABSENT。
- 唯一正式命令于2026-08-12T21:39:30执行：
  `nohup bash /home/szu2070436088/2510044040/releases/phase1_clic_predictor_artifacts_20260812_v1_74c42be1/code/scripts/launch_phase1_clic_predictor_artifacts12_v1_20260812.sh > /home/szu2070436088/2510044040/CV-SincNet/phase1_clic_predictor_artifacts_20260812_v1_outer.out 2>&1 &`
  - `FORMAL_INVOCATION=1`、`RETRY=NO`；outerPID=`2744797`；foldPID=`2744801,2744802,2744804,2744805,2744807,2744810`，记录于`logs/phase1_clic_predictor_artifacts_20260812_v1/pids_predictor_artifacts6.tsv`。

## 系统性技术失败与封存

- 6/6fold在C descriptor产出前立即产生完全相同的确定性异常：`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python: No module named cvsrffi.phase1_clic_target_leo`。静态阶段同一C module help曾PASS；formal运行时6份日志均仅100bytes且SHA相同`4d3148febb3ee5d1bc820f58b8afbbd6e8651f8a094a52d7af3c2cf2ba583c79`，表明运行时module resolution未闭合。该异常满足“至少2fold在完整C/G工件前相同确定性异常”的系统性停止规则。
- 保留证据：`c_predictor_state.json=0`、`c_predictor_state.train_config.json=0`、`g_deployment_bundle.zip=0`；日志6、PID行6、outer0bytes；技术异常扫描6条、无GPU进程。outer与6fold均已退出，本地SSH/SCP/TCP22清零；未重试、未修代码、未打开target cache/IQ/truth、未读取性能字段。

| fold | C descriptor | C train config | G bundle | log bytes/SHA | technical verdict |
|---|---:|---:|---:|---|---|
| F1 | 0 | 0 | 0 | 100；`4d3148febb3ee5d1bc820f58b8afbbd6e8651f8a094a52d7af3c2cf2ba583c79` | STOPPED_EARLY_SYSTEMIC_TECHNICAL_FAILURE |
| F2 | 0 | 0 | 0 | 100；`4d3148febb3ee5d1bc820f58b8afbbd6e8651f8a094a52d7af3c2cf2ba583c79` | STOPPED_EARLY_SYSTEMIC_TECHNICAL_FAILURE |
| F3 | 0 | 0 | 0 | 100；`4d3148febb3ee5d1bc820f58b8afbbd6e8651f8a094a52d7af3c2cf2ba583c79` | STOPPED_EARLY_SYSTEMIC_TECHNICAL_FAILURE |
| F4 | 0 | 0 | 0 | 100；`4d3148febb3ee5d1bc820f58b8afbbd6e8651f8a094a52d7af3c2cf2ba583c79` | STOPPED_EARLY_SYSTEMIC_TECHNICAL_FAILURE |
| F5 | 0 | 0 | 0 | 100；`4d3148febb3ee5d1bc820f58b8afbbd6e8651f8a094a52d7af3c2cf2ba583c79` | STOPPED_EARLY_SYSTEMIC_TECHNICAL_FAILURE |
| F6 | 0 | 0 | 0 | 100；`4d3148febb3ee5d1bc820f58b8afbbd6e8651f8a094a52d7af3c2cf2ba583c79` | STOPPED_EARLY_SYSTEMIC_TECHNICAL_FAILURE |

- 最终状态：`STOPPED_EARLY_SYSTEMIC_TECHNICAL_FAILURE / NO_PERFORMANCE_RESULT`。该v1不得被标记为健康artifact run，也不得在原run ID上重试；任何修复必须由主控另行本地验证、独立复审并分配fresh run ID。

## 只读故障诊断补充

- release C入口文件确实存在：`/home/szu2070436088/2510044040/releases/phase1_clic_predictor_artifacts_20260812_v1_74c42be1/code/cvsrffi/phase1_clic_target_leo.py`，bytes=`70736`，SHA256=`71b412979c27c945191a6c2125bb8fb640d5377390a1a84c5739947e89fe087e`。
- F1日志100bytes全文精确为：`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python: No module named cvsrffi.phase1_clic_target_leo`；F2–F6同内容、同bytes、同SHA。outer仍0bytes。
- 按要求使用同一release路径、同一`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`、仅只读设置`PYTHONPATH=<release>/code`执行namespace诊断（未导入target module）：`sys.path`含release/code；`import cvsrffi`成功，`cvsrffi.__path__`唯一指向release/code/cvsrffi；release目录直列也可见`phase1_clic_target_leo.py`。同环境`importlib.util.find_spec("cvsrffi.phase1_clic_target_leo")`返回该release文件的SourceFileLoader和origin，未执行模块代码。
- 因此可确认：归档物理文件及包namespace均存在，formal detached worker却报告module resolution failure；本run仅封存该运行时解析不闭合证据，不臆测根因、不修改release、不重试。后续修复必须改用绝对文件入口并建立fresh run ID，由主控重新本地验证与复审。
