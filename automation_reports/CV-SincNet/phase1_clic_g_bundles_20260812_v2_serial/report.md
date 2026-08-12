# Phase1 CLIC G bundle六fold串行v2预注册报告

## 状态与目标

- 实验ID：`phase1_clic_g_bundles_20260812_v2_serial`。
- 当前状态：`STOPPED_EARLY_TECHNICAL_FAILURE / FORMAL_INVOCATION=1 / RETRY=NO / NO_PERFORMANCE_RESULT`。
- v1正式launch=1后，单F1G真实runtime smoke成功，但6个CPU worker并发启动时6／6在任何bundle前统一native segfault。v1已封存、不得重试。
- v2是按两轮release repair上限拆出的最小独立one-shot入口：同一commit模型修复、同一F1G—F6G矩阵、同一training v5／clean v4／source-LEO v4、同一CPU线程上限，仅把6fold并发改为单进程逐fold串行；不调参、不改方法、不选fold、不接触target或性能。

## 入口与技术门

- launcher：`code/scripts/launch_phase1_clic_g_bundles6_v2_serial_20260812.sh`，SHA-256=`5F0DF6E0CB9B3F888CFF9EB24AA0C4899FC7DC7AEB6046B5D89D7286AE51BACC`。每个fold启动一个G exporter后立即wait并记录exit code，成功才进入下一fold；任何非0立即停止后续fold并保留已完成工件。
- dry-run仍精确6行F1G—F6G，禁止C／target／truth／score／role／query／package参数为0；bash-n通过；串行结构及矩阵测试`2／2`通过。
- sample-rate真实同型修复、G reload／标量／一次重建多row聚焦`5／5`，postfreeze`140／140`；fresh G-only v1代码审查已`P0=0，P1=0，ALLOW`，v2仅减少并发。
- 输出：`runs/phase1_clic_g_bundles_20260812_v2_serial/F{1..6}G_CLIC12/g_deployment_bundle.zip`；日志：`logs/phase1_clic_g_bundles_20260812_v2_serial`；启动前必须不存在。CUDA禁用，OMP／MKL／OpenBLAS各2线程；formal launch唯一1次，retry=`NO`。

## 后续与结果边界

- 成功只做production verify/reload技术QA，不读target、不报告性能。6G闭合后，target prediction v1改为C取predictor artifacts v2、G取本串行v2，仍对同一IQ-only target LEO-weak package生成12份prediction。
- target LEO-weak、unknown rejection、scene／RX／class／day DG是后续独立scorer的强制共同输出；本G-only构建run仅提供冻结predictor，不产生这些性能指标。
- fresh独立复审结论：`P0=0，P1=0，ALLOW`。确认相对v1只有run ID／TSV和6并发→逐fold串行变化；每fold立即wait、记录exit、成功才下一fold；bash-n、dry-run6和专测`2／2`通过，科学矩阵与输入未漂移。

|候选|矩阵是否保留|并发变化|预期工件|target／unknown／DG|当前结论|
|---|---|---|---|---|---|
|F1G—F6G|是，串行入口但F1失败后停止|6并发→逐fold串行|预期6份，实际0/6|本run不读取；未产生prediction|未进入scorer|STOPPED_EARLY_TECHNICAL_FAILURE|

## N607落地、静态核验与正式运行证据

- 冻结commit：`13f3deff14b31431df6d24e612560103ff160cd3`。`git archive`物理tar为267632640 bytes，SHA-256=`40CD78B4A027507B27187BA8C09A19C07E9BEFF13CDF3171163C1A58D1133B4C`；归档未包含共享`conversation_index/`。唯一SCP恰1次，远端SHA/bytes闭合。
- 原子release：`/home/szu2070436088/2510044040/releases/phase1_clic_g_bundles_20260812_v2_serial_13f3deff`。launcher远端physical/canonical SHA=`5F0DF6E0CB9B3F888CFF9EB24AA0C4899FC7DC7AEB6046B5D89D7286AE51BACC`；exporter physical SHA=`FFF7B5111D6F82E437AD001AFA90CE8D5F3C749F3E14C0A6A6D772BC637C9638`，CRLF归一化canonical SHA=`F826163A2F55FAF7CE3C9CD7F856AFE4452EAA75E52BC1364C1EECC02541C982`。
- 静态门：exporter`py_compile`、`--help`、launcher`bash -n`均PASS；`bash launcher --dry-run`精确6行F1G—F6G，C/target/truth/score/role/query/package禁用参数计数为0。
- 新release F1G只读checkpoint/terminal binding与临时runtime smoke PASS：raw`sample_rate_hz=0.0`，effective runtime=25000000.0Hz；`z_id=160,z_dom=160,q_clic=4`。未写正式bundle，未读取target或性能。
- 正式命令唯一调用：
  `nohup bash /home/szu2070436088/2510044040/releases/phase1_clic_g_bundles_20260812_v2_serial_13f3deff/code/scripts/launch_phase1_clic_g_bundles6_v2_serial_20260812.sh > /home/szu2070436088/2510044040/CV-SincNet/phase1_clic_g_bundles_20260812_v2_serial_outer.out 2>&1 &`
  `FORMAL_INVOCATION=1`，outer PID=`2774961`，`RETRY=NO`。

## 串行首波与停止证据

- 串行launcher先启动F1并等待；F1 worker PID=`2774969`以exit code `139`退出，launcher随后停止且没有启动F2。`pids_g_bundles6_serial.tsv`内容为：

```text
pid|fold|stage|log_path|exit_code
2774969|1|CLIC_G_BUNDLE_SERIAL|/home/szu2070436088/2510044040/CV-SincNet/logs/phase1_clic_g_bundles_20260812_v2_serial/F1G_CLIC12.out|139
```

- outer全文仅1行，bytes=355，SHA-256=`CD2A533B0BB052167A337711DD700F70B03885CAC5BC29F7648368DAE5A19372`，故障为release launcher line80的`Segmentation fault (core dumped)`。F1日志`F1G_CLIC12.out`为0 bytes（空文件SHA=`E3B0C44298FC1C149AFBF4C8996FB92427AE41E4649B934CA495991B7852B855`）；F1目录为空且`g_deployment_bundle.zip`计数为0，F2—F6未启动、无目录/日志/PID行。
- outer、worker均已退出；run限定目录内未发现core文件（未打开core）。停止后GPU0—7均`0%/1MiB`；本地SSH/SCP进程与TCP22连接均为0。未读取dmesg、target、IQ truth或性能字段。

## Release源码静态Tensor/NumPy转换inventory（只读）

对release内`code/export_phase1_clic_deployment_bundle.py`仅做文本扫描，未执行：

- `.numpy(`：第203行`value.detach().cpu().contiguous().numpy()`，这是唯一直接对Tensor调用`.numpy()`的可执行行。
- `torch.from_numpy`：无可执行调用；第156行仅出现在docstring文字`without torch.from_numpy()`。
- `np.asarray(`：第666、667、1089、1092—1096、1176、1177、1671、1692、1746、1755、1787、1803—1807行。Tensor分支在1746、1787先调用`.tolist()`再交给`np.asarray`；1755位于`torch.is_tensor(received_i)`为false的else分支；其余为NPZ/标量/列表字段，不是直接Tensor接收。

## 最终封存

- `ARTIFACTS_COMPLETE`未达到；最终状态为`STOPPED_EARLY_TECHNICAL_FAILURE / NO_PERFORMANCE_RESULT`，G bundle 0/6。本run不可重启或改标记为性能结果；任何后续修复必须新run ID并重新走冻结审查。
