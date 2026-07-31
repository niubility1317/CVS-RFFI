# D105 Phase1 R5 N607 runner handoff

## Terminal state

- Run ID：`d105_phase1_sourceheld_9f608e8b_20260731_r5`
- Remote root：`/home/szu2070436088/2510044040/CV-SincNet/runs/d105_phase1_sourceheld_9f608e8b_20260731_r5`
- Terminal state：`STOPPED_EARLY_SYSTEMIC_TECHNICAL_FAILURE / NO_PERFORMANCE_RESULT`
- Scope：Phase1 source-held资格链路；不是formal asset，也不是D105 Target性能。
- Detached launch：仅一次，PID=`2770709`，GPU=`cuda:0`，fresh-run retry=`NO`。

该run在`tap-cache`启动阶段以exit code`2`结束。未产生严格tap、prediction、truth-open、score、gate或component工件；因此没有source-held得分、资格判定或任何性能结果。

## Frozen identity and pre-detach evidence

|检查项|结果|
|---|---|
|归档|SHA256=`dd85491e96f1cb9ea14e967694db91aec590e42273a2556492221af982ee9a67`；242851840B；4754项=`4187 regular+567 dirs`；无链接、逃逸或重复|
|工作树|`codex/stage2-da25-r1`保持clean；HEAD=`adaf89eb2f584dbeac24b65e63b6045385cd1189`|
|runtime/method|`8940e05f9fdf92d7735bba1570bb3239ee210313ecbbeffa3511b62e21685425`/`f36a0c6c4ee832b34cd98ed7664ec87707a4dbb1559c7c9b4b05dd13fbf4864e`；canonical loader验证54/54|
|D102公开输入和launcher|四个远端输入SHA均匹配；每个模式均按字符串精确检查为`444`；launcher=5624B、LF-only、`bash -n=0`|
|独立预检|54个独立pyc；source中`pyc=0`、`__pycache__=0`；D105旧`from_numpy`/`.numpy()`AST调用=0|
|checkpoint|SHA=`2699eedcafe8cec880828592d2d65ba3781a9948939da5cf5c82b47143d59c98`；`model`state dict=195 tensors；PyTorch=`2.1.0+cu121`；精确SHA绑定loader策略=`legacy_pickle_exact_frozen_sha_only`|
|预检收据|`preflight/prelaunch_verification.json`；SHA256=`737c23a55dc66d141820e7d2bb3ab503cb1fd3437b6137349ac585cb14fdf9ef`；`pass=true`|

未执行authority签名、seal、Target25或重试。

## Complete pipeline log

`logs/pipeline_stage1.log`已完整读取，共313B、4行：

```text
usage: build_d105_phase1_bundle.py [-h]
                                   {tap-runtime,tap-cache,predict-source-held,open-truth,score-source-held,derive-gate,build,seal,validate}
                                   ...
build_d105_phase1_bundle.py: error: strict tap must expose byte-bound z_id/pre_relu and z_dom
```

## Post-failure inventory and recovery integrity

- `pipeline_stage1.exit=2`；PID文件记录`2770709`，该PID已退出。
- 终态GPU检查：8张GPU均为`0%`、`1MiB`；无R5运行进程。后续`pgrep`唯一匹配为该次只读检查shell自身，而非R5 pipeline。
- `output/`仅保留空目录`output/`和`output/source_held/`；严格tap、prediction、truth-open、score、gate、component计数均为0。
- 每次SSH/SCP后都确认本机`ssh.exe=0`且至N607/bridge的ESTABLISHED TCP22连接=0。

|回收文件|大小|本地SHA256=远端SHA256|
|---|---:|---|
|`logs/pipeline_stage1.log`|313B|`9925c63cc271a0365a031c0ab682898ff7e6a0f91fbd16511cf942ed727d8880`|
|`logs/pipeline_stage1.pid`|8B|`530f02aa5e5bd976015bab53e44ad25c4ea19bbef9df2a86a28372878f2c18ea`|
|`logs/pipeline_stage1.exit`|2B|`53c234e5e8472b6ac51c1ae1cab3fe06fad053beb8ebfd8977b010655bfdd3c3`|
|`preflight/prelaunch_verification.json`|2898B|`737c23a55dc66d141820e7d2bb3ab503cb1fd3437b6137349ac585cb14fdf9ef`|

远端文件模式记录：log=`664`、pid=`664`、exit=`444`、prelaunch receipt=`400`。本交接目录是新的非覆盖回收路径；未修改主报告或Git工作树代码。
