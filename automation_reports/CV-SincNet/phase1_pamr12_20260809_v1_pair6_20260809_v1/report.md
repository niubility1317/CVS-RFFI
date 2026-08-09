# P1-PAMR六折pair-only one-shot报告

目标模式：`GOAL_MODE=ACTIVE`

状态：`ANALYZED / REJECT_P1_PAMR_PERMANENT / NO_PERFORMANCE_RESULT / NO_PHASE3_PROMOTION`

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

- 2026-08-09已按§3精确命令唯一启动；release=`/home/szu2070436088/2510044040/CV-SincNet/releases/phase1_pamr_pair6_20260809_v1_87adbaec`，implementation commit=`87adbaeccc8c0a17afd4e01672aafdcf4d9fc9f8`，CWD=`<release>/code`。caller PID=`19722`；F1/F2 native fault PID分别为`19725`、`19850`。CPU-only，无GPU训练进程。
- `pair_completion.tsv`记录F1、F2均`FAILED`且shell `exit_code=0`；两fold日志和outer/faulthandler均记录`Fatal Python error: Segmentation fault`，最后stage=`extract_c_final_head`（`eval_phase1_pamr_pair.py` line 463→762→892→898）。launcher按两行相同shell exit停止并以exit 8结束；F3–F6未分派。shell exit 0是包装层记录，不能覆盖native SIGSEGV证据。
- 这是预注册“两个fold同exit码”技术停止；未读取或解释accuracy、floor、margin、cosine，不作性能结论，不重试。具体native根因无法从小artifact区分，保持`ESCALATE_TO_MAIN_AGENT`。
- 远端manifest SHA256=`265ade5315735a355622c1946db0927d2869424b52817e05098892410232b7b6`；artifact根=`E:\type10-7\automation_reports\CV-SincNet\phase1_pamr12_20260809_v1_pair6_20260809_v1\artifacts`，含F1/F2日志、`pair_completion.tsv`、outer、manifest共5个小文件。关键hash：F1 log=`68ede45885758b5902ca2892ebd956208fbf1d7823f3ba7e6c6d3b5d05f439f1`，F2 log=`68a0f0f2389f1e3d2e9abc1e84c9a6e1826c3278661e49c84d51f302bd593183`，completion=`c82b924998e7e114614c4c3bf92e94bacd01e6d3db1370619f62d3ceae3b2bc4`，outer=`abdbf8c54506d3940a16b8bf2d823d52c4472c721da22b64570fa8d35eebc41a`；每项远端/本地SHA与大小一致。未下载NPZ/checkpoint。
- SSH/SCP短连接已断开，TCP/22无残留；run-owned进程退出、GPU释放。根报告与Git镜像已同步更新，未提交Git；retry=`NO`。

## 7.主控非补偿裁决

|门|结果|证据|
|---|---|---|
|1.技术闭环|FAIL|完整postfreeze v1止于LEO row-view接口错配；v2止于F1 pair native SIGSEGV；最小PAIR6又在F1/F2同一`extract_c_final_head`边界SIGSEGV，6个pair JSON均未形成|
|2.clean known六折四项不退化|NOT EVALUABLE|缺少冻结pair JSON，禁止从partial proxy或训练日志替代|
|3.LEO overall 18/18不退化|NOT EVALUABLE|缺少冻结pair JSON|
|4.LEO三项floor 18/18不退化|NOT EVALUABLE|缺少冻结pair JSON|
|5.angular margin 18/18不退化|NOT EVALUABLE|final head读取阶段即崩溃，margin未计算|

最终裁决：`REJECT_P1_PAMR_PERMANENT / NO_PERFORMANCE_RESULT / NO_PHASE3_PROMOTION`。这里的“REJECT”是预注册技术闭环门失败，不是PAMR性能为负：12臂40epoch训练本身完整、PAMR coverage合同通过，但两轮完整release和一个收缩PAIR6仍不能形成合法同排证据。按照既定修复上限，不再改加载器、不绕过exact final head、不以已有proxy或训练曲线补齐缺失门，也不将本路线写成真实unknown、LEO稳健性或Phase3能力。
