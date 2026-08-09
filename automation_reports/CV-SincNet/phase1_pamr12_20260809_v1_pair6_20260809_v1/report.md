# P1-PAMR六折pair-only one-shot报告

目标模式：`GOAL_MODE=ACTIVE`

状态：`PREREGISTERED / NOT_LAUNCHED`

证据边界：`PHASE1_SOURCE_ONLY_OPEN_WORLD_READY_REPRESENTATION_NON_CONFIRMATORY`

## 1.目标与边界

实验ID：`phase1_pamr12_20260809_v1_pair6_20260809_v1`。日期：2026-08-09。operator：Codex主控；N607唯一runner：专属实验子代理。

本入口是两轮release-engineering修复后的最小独立one-shot。它只消费`phase1_pamr12_20260809_v1_postfreeze_v2`已经技术闭合的12组clean/LEO NPZ及`phase1_pamr12_20260809_v1`的immutable final checkpoint，串行执行6折C/G pair评分。不重新export、不重训、不运行proxy、不fit、不校准、不选参、不选择checkpoint，也不读取v1 partial。

v2在12/12 clean、12/12 LEO、12/12 proxy完成后，F1 pair进程exit139且stdout为0B，outer仅有native segmentation fault；回收artifact不含NPZ，无法在本地把根因进一步归到torch.load、NPZ解压或NumPy/OpenMP，故具体native调用点保持`ESCALATE_TO_MAIN_AGENT`，不猜测。PAIR6只做执行收缩和可观测性增强，不改变PAMR科学计算或预注册五门。

## 2.冻结版本与输入

- 实现commit：`87adbaeccc8c0a17afd4e01672aafdcf4d9fc9f8`
- v2 NPZ根：`/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_pamr12_20260809_v1_postfreeze_v2`
- 训练checkpoint根：`/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_pamr12_20260809_v1`
- 6折local4 TX顺序、source days/RX、三LEO scenario、source sat seed与原postfreeze完全相同。
- evaluator SHA256=`f11d031e57e854d730c9eb6015ed5dd14bc3e57635835164710bc4227993404d`
- launcher SHA256=`b9ffeb0f04f673056ea4ad8edbf461559ae08c657bf0f0278ca2abae4631ee8b`
- test SHA256=`eecf87e875a86c7b905b1c488e60ddea4aa661d1a46d1288a515082184cbcb11`

PAIR6保留NPZ SHA、strict-load、final-only checkpoint、exact head、local4类序、row order、physical、role、scenario、TX/RX、source profile satellite、TTA none及行级single全部绑定。raw-cosine norm/matmul/dot改用float64 ufunc add-reduce，夹具与原公式误差`<=1e-12`；四项分类floor仍完全来自冻结`tx_logits`，不受此执行路径影响。

本地验证：pycompile通过；PAMR+postfreeze focused 36 passed；launcher bash-n通过；dry-run精确6条；diff-check通过。独立复核：`P0=0 / P1=0 / ALLOW`。

## 3.N607路径与唯一命令

- release：`/home/szu2070436088/2510044040/CV-SincNet/releases/phase1_pamr_pair6_20260809_v1_87adbaec`
- run：`/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_pamr12_20260809_v1_pair6_20260809_v1`
- log：`/home/szu2070436088/2510044040/CV-SincNet/logs/phase1_pamr12_20260809_v1_pair6_20260809_v1`
- outer：`/home/szu2070436088/2510044040/CV-SincNet/logs/phase1_pamr12_20260809_v1_pair6_20260809_v1.launch.out`
- CWD：`<release>/code`
- Python：`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`

```text
cd /home/szu2070436088/2510044040/CV-SincNet/releases/phase1_pamr_pair6_20260809_v1_87adbaec/code && nohup setsid env PAIR_ONLY_RUN_ID=phase1_pamr12_20260809_v1_pair6_20260809_v1 PROJECT_ROOT=/home/szu2070436088/2510044040/CV-SincNet CODE_ROOT=/home/szu2070436088/2510044040/CV-SincNet/releases/phase1_pamr_pair6_20260809_v1_87adbaec/code PYTHON=/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python TRAIN_RUN_ROOT=/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_pamr12_20260809_v1 V2_POSTFREEZE_ROOT=/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_pamr12_20260809_v1_postfreeze_v2 PAIR_ROOT=/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_pamr12_20260809_v1_pair6_20260809_v1 LOG_ROOT=/home/szu2070436088/2510044040/CV-SincNet/logs/phase1_pamr12_20260809_v1_pair6_20260809_v1 bash /home/szu2070436088/2510044040/CV-SincNet/releases/phase1_pamr_pair6_20260809_v1_87adbaec/code/scripts/launch_phase1_pamr_pair6_20260809.sh > /home/szu2070436088/2510044040/CV-SincNet/logs/phase1_pamr12_20260809_v1_pair6_20260809_v1.launch.out 2>&1 < /dev/null & echo $!
```

CPU-only、单线程、每fold独立进程；`PYTHONFAULTHANDLER=1`且导入、NPZ load、head extract、compute、write均写flush marker。唯一启动，retry=`NO`；v1、v2及训练根均不可覆盖。

## 4.健康、停止与artifact

成功要求6/6 pair JSON、6日志和`pair_completion.tsv`完整，全部原有binding通过。首个失败写completion并继续；若两个fold出现相同exit code，则停止后续分派。任何失败均不作性能结论，不重启。

回收仅6 pair JSON、6日志、completion和manifest；不下载NPZ/checkpoint。runner不读取或解释accuracy、floor、margin或paired cosine。

## 5.预注册非补偿门

沿用postfreeze v2，不变：技术闭环；clean六折四项`G-C>=0`；18/18 LEO overall`G-C>=0`；18/18 LEO三项floor`G-C>=0`；18/18 angular margin`G-C>=0`。proxy与paired cosine仅诊断。任一失败即`REJECT_P1_PAMR_PERMANENT`，全部通过才可标`PROMOTE_P1_PAMR_FOR_PHASE1_BUNDLE_EVIDENCE`。

## 6.运行终态与结果

待artifact返回后填写。
