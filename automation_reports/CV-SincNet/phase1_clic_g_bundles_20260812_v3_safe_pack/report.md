# Phase1 CLIC G bundle安全封装v3预注册报告

## 状态与目标

- 实验ID：`phase1_clic_g_bundles_20260812_v3_safe_pack`。
- 当前状态：`ARTIFACTS_COMPLETE / FORMAL_INVOCATION=1 / RETRY=NO / NO_PERFORMANCE_RESULT`。
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
|F1G—F6G|与v2完全相同|无；仅状态字节桥修复|6份G bundle，实际6/6|本run不读取；后续同包prediction与scorer强制共同输出|技术闭合，性能N/A|

## N607落地与静态证据

- 冻结commit：`b3fa1c92c1f6d2c5e5b7a9b20b259bfe73585a15`。`git archive`物理tar为267653120 bytes，SHA-256=`BFCBA488DADF82F26D593C9ED8DAA2DD5517C7694D4EEE079459A4DA1B15D3F3`；未包含dirty改动或共享`conversation_index/`。SCP恰1次，远端SHA/bytes闭合。
- 原子release：`/home/szu2070436088/2510044040/releases/phase1_clic_g_bundles_20260812_v3_safe_pack_b3fa1c92`。launcher physical/canonical SHA=`CBE8F5DD8D9F0618C0FC0A71C3EFE9112389B38EDEE141947A37F2B6CAE36ECA`；exporter physical SHA=`B60484233F1946C504D7470CC444084A6437FA3D62D966D07DE2F5685A73B400`，CRLF归一化canonical SHA=`463FE8416FF68EA71F59F6433E6B1D79CCE91C038F7944A9962BE92D67569C47`。
- ManySig固定输入`/home/szu2070436088/2510044040/CV-SincNet/Dataset_WigSig/ManySig.pkl`，SHA-256=`2b0a7a7488dd3650bcae7b1d80efbcffd1598aaa671ae6b0a0df2a24dc0f694f`，bytes=`2359341461`。training v5、clean v4、source-LEO v4的F1—F6五类输入均存在。
- exporter`py_compile`、`--help`、launcher`bash -n`均PASS；`bash launcher --dry-run`精确6行F1G—F6G，C/target/truth/score/role/query/package禁用参数计数为0。GPU0—7预检均0%/1MiB；release、正式run/log/outer及smoke root启动前均ABSENT。

## F1完整export smoke门

- formal前唯一独立烟测根：`/home/szu2070436088/2510044040/CV-SincNet/.smoke_phase1_clic_g_bundles_20260812_v3_safe_pack_F1/F1G_CLIC12`。使用release exporter、正式F1G五个输入、`CUDA_VISIBLE_DEVICES=''`及线程上限2，完整export PASS；bundle bytes=`4605944`，SHA-256=`14105acb98dbcf9b7616410eed034da7d3b3d76b8e5413decf8e9905e790a592`。
- release production`verify_clic_bundle` PASS：`state_origin=checkpoint_model_exact`、`real_checkpoint_state_rebuild_verified=True`、`clean_source_runtime_access=False`、`query_fit_access=False`、`bundle_has_raw_checkpoint=False`、`bundle_has_sample_rows=False`、`single_leo_observation_required=True`，输出维度为`z_id=160,z_dom=160,q_clic=4`。无target member，不读取性能；烟测根按合同保留未清理。

## Formal launch与运行闭合

- 唯一正式命令：
  `nohup bash /home/szu2070436088/2510044040/releases/phase1_clic_g_bundles_20260812_v3_safe_pack_b3fa1c92/code/scripts/launch_phase1_clic_g_bundles6_v3_safe_pack_20260812.sh > /home/szu2070436088/2510044040/CV-SincNet/phase1_clic_g_bundles_20260812_v3_safe_pack_outer.out 2>&1 &`
  `FORMAL_INVOCATION=1`、`RETRY=NO`、outer PID=`2790915`。launcher逐fold等待并在成功后进入下一fold；F1→F6全部exit=0。

|Fold|worker PID|exit|bundle bytes|bundle SHA-256|log bytes|verify/reload|性能字段|
|---|---:|---:|---:|---|---:|---|---|
|F1G|2790921|0|4605944|`14105acb98dbcf9b7616410eed034da7d3b3d76b8e5413decf8e9905e790a592`|147|PASS/PASS|N/A|
|F2G|2790936|0|4606470|`b3ffce8ec3f7b53e0e0da4b445110a3c502fe78b56f8a085375ca19a31aa5a28`|147|PASS/PASS|N/A|
|F3G|2790941|0|4604494|`2b175eb7777b8b12a7bb7fe306d9bc7ad8764dd863e6b1b979f04e7ad49f5caf`|147|PASS/PASS|N/A|
|F4G|2791351|0|4605507|`7f2d62632c212b0fb913ede15f2e37efd8fa5dc50f73920124a26dab087d02fb`|147|PASS/PASS|N/A|
|F5G|2791356|0|4605656|`ffc3842fa1713e0493aea7f308f5e7116c67c6db7f71cdf81043960103cc91f3`|147|PASS/PASS|N/A|
|F6G|2791361|0|4605261|`be89b71081276b9ef0d30d43462690d780c994977520e13c70e9d1e204b617fc`|147|PASS/PASS|N/A|

- PID证据文件：`logs/phase1_clic_g_bundles_20260812_v3_safe_pack/pids_g_bundles6_safe_pack.tsv`，六行worker均exit=0。六个日志均为147 bytes且只含对应bundle路径JSON；outer为0 bytes（空文件SHA=`E3B0C44298FC1C149AFBF4C8996FB92427AE41E4649B934CA495991B7852B855`）。正式run包含6份zip、6份日志及完整PID表。
- 对6份正式bundle执行production`verify_clic_bundle`及一次零IQ`reload_forward(scene=leo_clear_weak)`：全部PASS，`real_checkpoint_reload_verified=True`，输出`z_id/z_dom/q_clic/tx_logits`均finite且shape分别为`(1,160)/(1,160)/(1,4)/(1,4)`；source-only、zero clean/query-fit/raw-checkpoint/sample-row访问约束均保持。未读取target、truth、AUROC、u_gap或任何性能字段。
- 完成后outer、worker及exporter进程均退出；GPU0—7均0%/1MiB；本地SSH/SCP进程与TCP22连接清零。

## 最终状态

- `ARTIFACTS_COMPLETE`：6/6 G deployment bundle、6/6日志、6/6 PID exit=0及6/6 production verify/reload技术QA闭合。该run只证明技术工件闭合，不产生性能结果；性能字段统一`N/A`，状态为`ARTIFACTS_COMPLETE / NO_PERFORMANCE_RESULT`。
