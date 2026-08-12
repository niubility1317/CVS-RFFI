# Phase1 CLIC C／G predictor artifacts v1预注册报告

## 状态与目标

- 实验ID：`phase1_clic_predictor_artifacts_20260812_v1`。
- 当前状态：`LOCAL_VERIFIED / FRESH_REVIEW_PENDING / FORMAL_LAUNCH=0 / NO_PERFORMANCE_RESULT`。
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
- 待完成：提交本次修复、fresh独立P0／P1复审；若`P0=0，P1=0`，立即交N607唯一runner完成preflight／archive／SCP／远端静态门、唯一启动和12工件零IQ真实模型重开烟测。
- 工件闭合后，使用target confirmation v2已验证缓存派生VALIDATED_ONCE收据和known-test配置，封装一个IQ-only package，并对12臂分别发布预测；评分阶段同时报告三scene target-known DG、unknown拒识及域泛化，并只与同训练／测试数据配置的合法ADV3B02原件比较。
