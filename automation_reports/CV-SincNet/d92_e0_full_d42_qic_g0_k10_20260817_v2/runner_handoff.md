# D92 QIC G0 v2 sole runner handoff

run_id: `d92_e0_full_d42_qic_g0_k10_20260817_v2`
role: `Luna/max sole N607 runner`
status: `ARTIFACTS_COMPLETE / TECHNICAL_G0_PASS / NO_PERFORMANCE_RESULT`
release_commit: `2392a8b79444036d66edb85bab27b2cc827ebc5b`
runtime_commit: `82ce747643af71ac3737bc0a89d18114be96f27e`

## RULES_READ / PRECHECK / SYNC

- `RULES_READ=VERIFIED`：live AGENTS.md、项目.md、Git Bash skill、failure catalog、N607 automation skill已完整读取。
- `PRECHECK=VERIFIED`：普通direct `N607`，项目根/Python/8GPU可见，v2 archive/launch/source/run/log/driver out/err及同run进程均按启动前规则核验；GPU0无compute app。
- `SYNC=VERIFIED`：archive远端218553B，SHA256=`802a52557657d6d415992192fd546fe564495ff71354a7890807645bb071113a`；launch远端6623B，SHA256=`fe9c708735c6416dec0958ece5cca1c3c26fd27a242e28e9211359fc9eed467e`，`bash -n=pass`。

## COMMAND / PID / GPU

冻结命令执行一次且仅一次：

```bash
cd /home/szu2070436088/2510044040/CV-SincNet/runs && nohup bash ./d92_qic_g0_launch_82ce7476_20260817_v2.sh >./d92_qic_g0_launch_82ce7476_20260817_v2.out 2>./d92_qic_g0_launch_82ce7476_20260817_v2.err </dev/null &
```

- 启动后source/run/logs和两臂技术artifact均生成；进程自然退出。
- 最终PID/PPID/CWD/cmdline：无存活同run PID；无延迟launcher。
- GPU0–GPU7最终均无compute app，GPU已释放。

## MARKER / SCENE TECHNICAL TABLE

固定marker与`g0_validation.status`均为`D92_QIC_G0_ACTIVE_QUANTIZATION_INTERCEPT_CLOSURE_RESOURCE_PASS`，`validation.pass=true`，三scene gates全true。

|scene|K/old/registered|active/fallback|FP16 bits|E0 residual|candidate residual|reduction|decode/full/block/LOO/Fisher/scan/requantize|candidate wall/reference wall/ratio|peak bytes|query MAC/ref/delta|state bytes/ref/delta|
|---|---|---|---:|---:|---:|---:|---|---|---:|---|---|
|`leo_clear_weak`|10/6/11|true/false|11|229.1639949631|0.0364577573|229.1275372059|1/0/0/0/0/0/0|64.911634ms/61.707970ms/1.051917|8192|3168/3168/0|8583/8583/0|
|`leo_low_elev_weak`|10/6/11|true/false|11|173.3684354920|0.0131944239|173.3552410681|1/0/0/0/0/0/0|64.196002ms/61.590854ms/1.042298|94208|3168/3168/0|8583/8583/0|
|`leo_rain_weak`|10/6/11|true/false|11|251.8618226068|0.0219630476|251.8398595592|1/0/0/0/0/0/0|63.808285ms/61.644737ms/1.035097|32768|3168/3168/0|8583/8583/0|

All scenes: `fallback_reason=null`；modified field=`intercept_fp16`；base/QIC seven query flags all false；support-only and clean/source access false；all protocol/resource gates true。QIC support upper bound为34848 MACs、357024 bytes，complexity为`O(C*K*288)+O(C*288)`。coefficient/scale/log_diag/intercept_fp32/state shape/class registry byte-exact，direct publish与class/row permutation invariance true。

## ARTIFACTS / CLEANUP / NEXT_ACTION

- 完整取回：`E:/type10-7/local_artifacts/d92_e0_full_d42_qic_g0_k10_20260817_v2/`，包含source root、两臂before/after、g0_validation、fit/resource/execution receipts、logs、driver out/err和prediction artifacts；prediction artifacts未读。
- driver out仅含固定PASS marker，driver err为空；远端工件保留未删。
- 每次SSH/SCP后本地SSH/SCP客户端均退出，无到N607的ESTABLISHED连接。
- `truth_read=false`、`scorer_read=false`、`performance_read=false`、`analyzer_run=false`、`fresh_run_retry=false`。

下一动作：主代理可基于本技术G0证据决定完整矩阵发布；本run不可重试、重启、覆盖或改写。
