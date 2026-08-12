# Phase1 CLIC G bundle安全封装v3预注册报告

## 状态与目标

- 实验ID：`phase1_clic_g_bundles_20260812_v3_safe_pack`。
- 当前状态：`LOCAL_VERIFYING / FORMAL_INVOCATION=0 / RETRY=NO / NO_PERFORMANCE_RESULT`。
- 目标：修复G bundle把PyTorch状态直接送入旧NumPy桥导致的N607原生segfault；不改变训练数据、测试数据、模型、权重、fold、阈值、方法或科学矩阵。
- v1六fold并发在bundle前6／6原生segfault；v2串行F1仍在bundle前exit139并停止F2。静态定位到`_state_value_bytes`唯一直接`Tensor.numpy()`入口，v3仅改为PyTorch存储字节安全封装，并保留F1G—F6G逐fold串行。

## 冻结输入与入口

- 训练输入：`runs/phase1_clic12_20260812_v5/F{1..6}G_CLIC12/{final_ssdg.pth,phase1_clic_terminal_receipt.json}`。
- clean输入：`runs/phase1_clic_postfreeze_20260812_v4/F{1..6}G_CLIC12/source_clean_proxy.npz`。
- source-LEO输入：`runs/phase1_clic_source_leo_20260812_v4/F{1..6}G_CLIC12/{source_leo.npz,source_leo.binding.json}`。
- launcher：`code/scripts/launch_phase1_clic_g_bundles6_v3_safe_pack_20260812.sh`；Conda/Python环境为N607`CVS-RFFI`，项目根`/home/szu2070436088/2510044040/CV-SincNet`，CUDA禁用，OMP／MKL／OpenBLAS各2线程。
- 正式输出：`runs/phase1_clic_g_bundles_20260812_v3_safe_pack/F{1..6}G_CLIC12/g_deployment_bundle.zip`；日志：`logs/phase1_clic_g_bundles_20260812_v3_safe_pack`；两根必须启动前均不存在。

## 本地验证与发布门

- 生产修复commit：`89d6f286`。封装器SHA-256=`463FE8416FF68EA71F59F6433E6B1D79CCE91C038F7944A9962BE92D67569C47`；仅将G状态的旧`Tensor.numpy()`改为独占连续CPU tensor上的严格有界内存块复制，不改变state schema、dtype、shape或权重字节。
- 安全封装逐字节验证覆盖bool、int64、float32、complex64共4种dtype；真实G bundle export／verify在禁用`Tensor.numpy()`和`torch.from_numpy()`时通过，单项约7秒。sample-rate／reload／G runtime标量／同runtime多row共8项通过。
- `code/tests/test_phase1_clic_postfreeze.py`完整回归`145／145`通过（56秒，仅11条既有AMP FutureWarning）；`py_compile`与`git diff --check`通过。
- v3 launcher SHA-256=`CBE8F5DD8D9F0618C0FC0A71C3EFE9112389B38EDEE141947A37F2B6CAE36ECA`；launcher等价测试SHA-256=`67ACE5661C1FAD9D31CD720565D8C1D891C2AF898FDDC3FC69242A4C255F61E3`。`bash -n`通过，dry-run矩阵与v2逐行归一化等价，launcher专测`2／2`通过。
- 新回归禁止`Tensor.numpy()`和`torch.from_numpy()`后完成真实G bundle export／verify；同时复跑sample-rate、reload、标量归一化和同runtime多row一次重建路径。
- 完整postfreeze、`py_compile`、launcher`bash -n`、dry-run六行及v2→v3矩阵等价测试必须通过；独立复审必须`P0=0，P1=0`。
- N607落地后先在独立自有烟测目录执行F1G同一完整export命令并production verify bundle。该烟测必须真实经过状态封装；若失败，不调用正式launcher。若通过，立即执行唯一正式launcher，`FORMAL_INVOCATION=1`，`RETRY=NO`。

## 健康、停止与结果边界

- 每fold成功才进入下一fold；任一非0立即停止后续fold并保留工件。正式运行不得依据性能停止，也不得读取target、truth、AUROC、unknown rejection率或DG指标。
- 预期工件为6份G bundle、六行PID／exit证据及6日志；完成后只做production verify／reload、SHA、source-only、zero-fit／zero-update技术QA。
- G bundle本身不产性能。后续target prediction必须对同一封存target LEO-weak received-IQ包运行C6／G6；正式scorer必须同时报告unknown rejection和scene／RX／class／day DG。缺任一项都不得形成性能结论。

|候选|训练／测试配置|方法变化|预期工件|target LEO-weak／unknown／DG|当前结论|
|---|---|---|---|---|---|
|F1G—F6G|与v2完全相同|无；仅状态字节桥修复|6份G bundle|本run不读取；后续同包prediction与scorer强制共同输出|待技术闭合|
