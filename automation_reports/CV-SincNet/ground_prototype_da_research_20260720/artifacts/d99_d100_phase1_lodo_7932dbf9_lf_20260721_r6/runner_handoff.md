# D99/D100 Phase1 LODO r6 Runner Handoff

## 结论

- 实验ID：`d99_d100_phase1_lodo_7932dbf9_lf_20260721_r6`
- 最终状态：`ARTIFACTS_COMPLETE_DIAGNOSTIC_FAILURE_NOT_ANALYZABLE`
- 发布次数：1次；未重试、未改参、未访问target数据。
- 包装器PID：`1447845`，已退出；退出码：`1`。
- 开始时间：`2026-07-21 03:50:36 CST`；结束时间：`2026-07-21 03:50:38 CST`。
- GPU：物理GPU5；终态`0%`利用率、`10 MiB / 24576 MiB`，无遗留进程。
- 性能结论：运行在首个D81 episode scorer调用前失败，K1/K5/K10/K20、D81/D99/D100、NLL、floor、old/new/H、disagreement、双向rescue、receiver×pseudo-new和资源峰值均未生成；不得据此评价方法性能或晋级。

## 状态轨迹

|状态|证据|结论|
|---|---|---|
|`LOCAL_VERIFIED`|Git提交`9fc24ee5be543615fc30defb62f9ed2c40864835`；本地配置SHA-256为`8a63aaab10dceb05811af8f0aa82bfc76e45468e2603beffd419092eebe949f6`|通过|
|发布前检查|`2026-07-21 03:45:22 CST`直连preflight通过；8张GPU均为`0%/10 MiB`；r6 run/log路径不存在|通过|
|冻结输入门|r4特征档案、r1 ground bundle/base lock、D19清单和5个LF模块哈希逐项匹配预注册值|通过|
|`LANDED`|远端配置SHA匹配；output启动前不存在；唯一一次detached包装器返回`LANDED pid=1447845 gpu=5`|通过|
|执行终态|`runner.exit=1`；完整traceback落盘；output目录未创建|失败诊断|
|`ARTIFACTS_COMPLETE`|配置、完整日志、PID、起止时间、退出码、命令和包装器输出均已回收到本地并复核哈希|完成|

## 冻结版本与输入

- Git提交：`9fc24ee5be543615fc30defb62f9ed2c40864835`
- 预注册配置：`preregistered_inputs/d99_d100_phase1_lodo_7932dbf9_lf_20260721_r6.json`
- 配置SHA-256：`8a63aaab10dceb05811af8f0aa82bfc76e45468e2603beffd419092eebe949f6`
- r4 Phase1特征NPZ：`cdd8747d267336b48e8c555ce7e010206f042ff07c695af351541a97187fad03`
- r4 Phase1特征manifest：`5f363bc09503f882c66aa92805657199ca57484f627c5d2805254cad07bffa15`
- r1 ground bundle NPZ：`e69409268ad1215d440fd0555a4f8e2903214e95062f22a0c5f436fb2b799bd4`
- r1 ground manifest：`f92a1bd6e936c4e661517da73ee39cc5e94f77b796f50587dd33f6c0d2da8f0d`
- r1 base lock：`7481c351a3114432df0c06c8527b1f625c15c05b1e3aed89000098b26f022a9e`
- D19 ground manifest：`15b5e144f9af3989421d8e925c17758479c327be47e79222f6363dc63994629c`
- Phase1 checkpoint声明：`2699eedcafe8cec880828592d2d65ba3781a9948939da5cf5c82b47143d59c98`

5个实际LF模块SHA-256均与r6预注册一致：

|模块|SHA-256|
|---|---|
|`run_d99_d100_phase1_lodo.py`|`110295caa83ab0d7717e26b17b1d4ac33423337afaa8877067f64649d06c7ea1`|
|`stage2_d100_ra_cgspr_lgf.py`|`86c185ee13222bc0c97c4576984b9cd07f981201da4f0b62f8d4bc66970b4714`|
|`stage2_d81_phase1_episode_scorer.py`|`54ee742c81b60e00b6c1c36d2d6bf1f0409ad10f72a25e01c2dcd589093be55d`|
|`stage2_d99_d100_phase1_lodo.py`|`aa99b3d726338481ed7f22f4acc5cdf2cfe4b2ef420e44da6f2ff2f674841e0e`|
|`stage2_d99_ra_cgtmk_d81.py`|`c166a5e375b0b8be5c95e678e63a6f04526474cd1a01544616829106af52f56f`|

## 精确执行信息

- Conda/Python：`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`
- CWD：`/home/szu2070436088/2510044040/CV-SincNet`
- 物理GPU：5；`CUDA_VISIBLE_DEVICES=5`，因此进程内目标GPU应写为逻辑`cuda:0`。
- CPU线程：`OMP_NUM_THREADS=2`、`MKL_NUM_THREADS=2`、`OPENBLAS_NUM_THREADS=2`、`NUMEXPR_NUM_THREADS=2`。
- run目录：`/home/szu2070436088/2510044040/CV-SincNet/runs/d99_d100_phase1_lodo_7932dbf9_lf_20260721_r6`
- log目录：`/home/szu2070436088/2510044040/CV-SincNet/logs/d99_d100_phase1_lodo_7932dbf9_lf_20260721_r6`
- output目录：未创建。

冻结子命令：

```text
env OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2 CUDA_VISIBLE_DEVICES=5 PYTHONPATH=/home/szu2070436088/2510044040/CV-SincNet/runs/d99_d100_phase1_export_7932dbf9_cuda0_20260721_r4/source_7932dbf9/code /home/szu2070436088/.conda/envs/CVS-RFFI/bin/python -u /home/szu2070436088/2510044040/CV-SincNet/runs/d99_d100_phase1_export_7932dbf9_cuda0_20260721_r4/source_7932dbf9/code/scripts/run_d99_d100_phase1_lodo.py --config /home/szu2070436088/2510044040/CV-SincNet/runs/d99_d100_phase1_lodo_7932dbf9_lf_20260721_r6/input/d99_d100_phase1_lodo_7932dbf9_lf_20260721_r6.json --config-sha256 8a63aaab10dceb05811af8f0aa82bfc76e45468e2603beffd419092eebe949f6
```

## 根因

冻结配置包含：

```json
"d81_device": "cuda"
```

调用链：

```text
run_d99_d100_phase1_lodo.py
  -> run_phase1_d99_d100_lodo
  -> D81Phase1EpisodeScorer.__call__
  -> runtime_device = torch.device(self.device)
  -> stage2_d42_unified_shrinkage_lda._fit_old_only_b3_metric
  -> torch.cuda.set_device(device)
```

`torch.device("cuda")`的`type`为`cuda`但`index`为空；当前D42实现将该对象原样传给`torch.cuda.set_device`，于是抛出：

```text
ValueError: Expected a torch.device with a specified index or an integer, but got:cuda
```

因此这是设备字符串正规化缺失，不是GPU不足、CPU误跑、数据协议或模型性能问题。错误发生在第一次D81 episode打分前，没有任何可用候选指标。

## 性能指标完整性

|K-shot|D81|D99|D100|NLL|old|seen-new|H|floor|disagreement|双向rescue|receiver×pseudo-new|判定|
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
|1|未生成|未生成|未生成|未生成|未生成|未生成|未生成|未生成|未生成|未生成|未生成|不可评价|
|5|未生成|未生成|未生成|未生成|未生成|未生成|未生成|未生成|未生成|未生成|未生成|不可评价|
|10|未生成|未生成|未生成|未生成|未生成|未生成|未生成|未生成|未生成|未生成|未生成|不可评价|
|20|未生成|未生成|未生成|未生成|未生成|未生成|未生成|未生成|未生成|未生成|未生成|不可评价|

## 后续处理建议

1. 不得复用r6 run ID，也不得把本次失败视为D99/D100负性能证据。
2. 最小配置修复是将新预注册运行的`d81_device`改为`cuda:0`；在`CUDA_VISIBLE_DEVICES=5`下仍落到物理GPU5，不改变科学方法或数据协议。
3. 更稳健的代码修复是：当`runtime_device.type == "cuda"`且`runtime_device.index is None`时，显式正规化为`torch.device("cuda:0")`；需先做本地单元/窄集成验证。
4. 新运行必须使用新的run ID、配置SHA、Git提交和报告预注册；r6无重试授权，当前runner没有实施任何修复或重跑。
5. 发布门应新增一个不访问数据的设备解析检查，至少验证配置中的device可被`torch.cuda.set_device`接受，避免再次占用实验发布轮次。

## 本地回收产物

|文件|大小|SHA-256|
|---|---:|---|
|`remote_config.json`|3014 B|`8a63aaab10dceb05811af8f0aa82bfc76e45468e2603beffd419092eebe949f6`|
|`remote_log/child_command.txt`|684 B|`7d92d6cb5138947f98bb2ff9194f8fdbd0b583d3070a2f423a8fce5b5b138017`|
|`remote_log/runner.start`|24 B|`ec8040e3e939f8133c5d81337cf0dce2aae06a1ceaab065211e93f84043af8c4`|
|`remote_log/runner.end`|24 B|`4ddbeedeebb7fd2b3ae4cf33ef304835b7641ce77d3e44af887af0e79380e7c8`|
|`remote_log/runner.pid`|8 B|`d8bfc4caa5aace4fd6f0d8add5572b9c1a3e82dc4a0b555181eec30d5521748e`|
|`remote_log/runner.exit`|2 B|`4355a46b19d348dc2f57c046f8ef63d4538ebb936000f3c9ee954a27460dd865`|
|`remote_log/runner.out`|2063 B|`c2e980df09831488acf3f56a173b60ac334ae3c5626a713291acc7b56370180b`|
|`remote_log/wrapper.out`|0 B|`e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`|

最终SSH状态：本地`ssh.exe=0`；至`172.31.111.215:22`/`172.31.105.18:22`的`ESTABLISHED=0`。
