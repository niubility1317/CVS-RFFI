# Phase1 ADV3B02 SAT-Anchor-SSL 8条最小验证预登记

## 当前状态

```text
run_id=phase1_adv3b02_sat_anchor_ssl8_s392002_20260822
status=PARTIAL_TECHNICAL_FAILURE / ANALYZED
seed=392002
epochs=200
matrix_rows=8
gpu_count=8
rows_per_gpu=1
```

## 目标、假设与协议

目标是在当前Phase1数据协议下停止固定比例通用伪身份监督，验证“全部`U_s`的clean-satellite配对学习＋少量严格可信U星地身份CE＋clean冻结教师锚定”能否提高未见接收机LEO鲁棒性并控制clean退化。

- 数据角色固定为`L_s/U_s/V_cal/V_select=0.07/0.63/0.15/0.15`。
- `U_s`只暴露IQ与source receiver/day metadata，TX真值隐藏。
- source receiver为RX0–RX6；target receiver为RX7–RX11；二者不相交。
- 初始化为`ADV3B02_CORE90_SOFT_E200/best_joint_safe_ssdg.pth`，除A0控制外均不得改变。
- 星地训练固定`lambda_sat_cls=0.68`、`lambda_sat_cons=0`和当前`LEO_WEAK`三段日程。
- 全部候选使用U batch256并完整覆盖52,920个`U_s`样本；不以丢样本换速度。
- source-only选择依据必须在连接target结果前冻结；最终target结果不得反馈重训、阈值或候选重排。

## 训练加速设计

1.冻结Core90教师与EMA教师只对clean `U_s`做AMP inference/no-grad前向。
2.学生clean和需要的satellite行拼接成一次前向；pair step覆盖全U satellite，非pair step只计算trusted satellite行。
3.`A3_PAIR_INTERVAL2`每两个step执行一次全U pair，trusted satellite CE仍每step执行。
4.关闭pair、anchor或trusted CE时不生成对应view、不执行对应前向。
5.每GPU只运行一条训练，避免上一轮每卡两条造成的吞吐干扰；速度比较使用同机samples/s、forward samples、optimizer step和峰值显存。

## 冻结矩阵

|GPU|候选|trusted U-sat|全U pair|clean anchor|平衡/梯度作用域|目的|
|---:|---|---|---|---|---|---|
|0|A0_CONTROL|关|关|关|无|Core90继续训练控制|
|1|A1_STRICT_USAT|adaptive no-fill|关|关|class cap|验证严格U星地身份分支|
|2|A2_USAT_ANCHOR|adaptive no-fill|关|开|class cap|控制clean退化|
|3|A3_ADAPTIVE_NO_FILL|adaptive no-fill|SimSiam每step|开|class cap|最大化U配对利用|
|4|A3_FIXED_50_FILL|固定补齐50%|SimSiam每step|开|class cap|直接验证取消固定补齐|
|5|A3_PAIR_INTERVAL2|adaptive no-fill|SimSiam每2step|开|class cap|训练加速对照|
|6|A4_CLASS_RX_CAP|adaptive no-fill|SimSiam每2step|开|class×receiver cap|改善receiver floor|
|7|A5_ADAPTER_TAIL|adaptive no-fill|SimSiam每2step|开|class×receiver cap＋U仅adapter/tail|保护强基座|

训练完成后，每条候选均自动测试epoch200 `final_ssdg.pth`的clean、`leo_clear_weak`、`leo_low_elev_weak`和`leo_rain_weak`，但晋级判断必须使用预注册的source-only指标并保持同row解释。

## 停止规则

仅因协议/query泄漏、错误代码/release、输出碰撞、OOM/NaN、确定性重复异常、缺失checkpoint、缺失prediction/四场景闭合或scorer连接错误停止对应run。不得因中间或最终性能低停止训练。

## 追踪表

|ID|来源|要求|目标文件|状态|验证|备注|
|---|---|---|---|---|---|---|
|SA-01|指导5.1–5.3|冻结Core90＋EMA双教师；`V_cal`类别条件阈值|`muse_ssdg.py`、`train_ssdg.py`、tests|verified|类别完整前缀风险扫描测试通过|阈值只读source `V_cal`，复杂度为`O(N log N)`|
|SA-02|指导5.2|取消固定50%补齐；可信可为0|routing、tests|verified|空集合、adaptive no-fill、显式fixed-fill测试通过|默认`fill_to_fraction=0`|
|SA-03|指导5.4|class cap与class×receiver cap|routing、tests|verified|确定性选择`[0,3,4]`测试通过|只做选择平衡，不做feature attraction|
|SA-04|指导6|全部U在投影空间做SimSiam clean-satellite pair|training heads、train、tests|verified|对称性、双视图梯度与pair cadence测试通过|训练期头不进入部署模型|
|SA-05|指导7|trusted U satellite CE按完整`B_U`归一化|loss、tests|verified|稀疏mask按完整batch归一化及空mask图安全测试通过|无trusted时梯度0|
|SA-06|指导8|冻结teacher clean-logit KL锚定|train、tests|verified|teacher无梯度测试通过|不做`z_id`MSE|
|SA-07|指导9|关闭通用U identity/prototype/temporal/cross-RX/nuisance|config、train、tests|verified|SAT专用分支只返回pair/satellite/anchor三项；关闭目标零额外view测试通过|保留L_s ADVB02路径|
|SA-08|指导10|A5低秩零初始化identity adapter；U梯度只进adapter/tail|model、builder、train、tests|verified|零初始化不改输出、U detach、L全梯度测试通过|adapter进入checkpoint模型状态|
|SA-09|指导11.3、用户|拼接学生前向、pair interval2、U256完整覆盖|train、speed tests|verified|单调用、卫星only、全关闭零学生调用、interval2测试通过|每GPU一条训练|
|SA-10|指导12|避免把U128收益误归因batch；主配置U256|matrix、report|verified|矩阵人工核对|不再扩大U128/U384|
|SA-11|指导13|source-only选择与target不反馈|launcher、report|verified|dry-run固定`V_cal`阈值、`V_select`选择和训练后独立target评估|target结果仅最终诊断|
|SA-12|指导14–15|发布A0–A5、fixed-fill和interval2最小矩阵|JSON、launcher、tests|verified|8行、GPU0–7一一映射、U256测试通过|A0为无U前向的M0真控制|
|SA-13|指导16|逐epoch记录trusted/agreement/confidence/margin/class/receiver覆盖、pair drift和分项梯度|telemetry、tests|verified_local|字段已进入epoch日志；真实值待checkpoint smoke|精确梯度仅首batch，pair drift按实际pair step|
|SA-14|项目完成规则|final-only、clean＋三LEO自动闭合|launcher、tests|verified_local|dry-run8条训练命令、8条联合eval、32个分场景输出通过|真实checkpoint smoke与N607启动待完成|

当前追踪计数：`verified=11`、`verified_local=2`、`pending=0`、`deferred=0`、`rejected=0`、`blocked=0`。剩余发布动作是Git提交/远端OID核对、N607真实checkpoint无query smoke和正式8行启动。

## 本地实现与验证记录

实现文件：

- `code/cvsrffi/muse_ssdg.py`：冻结教师与EMA几何融合、`V_cal`类别阈值、adaptive no-fill路由、class/class×receiver cap、完整U batch归一化CE、clean KL和pair loss。
- `code/SSDG/train_ssdg.py`：SAT专用U路径、按需学生前向、pair interval、遥测、checkpoint阈值恢复和真实教师加载。
- `code/model_dual_cvsincnet.py`与`code/post_stage_common.py`：A5零初始化低秩identity adapter及严格重建参数传递。
- `configs/phase1_adv3b02_sat_anchor_ssl8_s392002_20260822.json`与两个launcher：A0–A5固定8行、每GPU一条、final-only和clean＋三LEO闭合。

本地验证：

```text
RED: 3个生产接口导入失败，符合新增能力尚未实现的预期
GREEN-CORE: 16 passed
GREEN-FOCUSED: 46 passed
GREEN-LAUNCHER-REGRESSION: 9 passed
PY_COMPILE: PASS
BASH_DRY_RUN: 8 train commands + 8 joint eval commands + 32 scenario outputs
GIT_DIFF_CHECK: PASS
```

独立P0/P1定点审查结论：修复`V_cal`二维穷举的启动性能风险、固定50%对照语义和A0多余U前向后，未发现会导致协议越权、训练跑错、输出覆盖、无法启动或不能产生合法prediction的剩余P0/P1。未新增SHA、seal、receipt或额外审批门；如后续出现此类要求，记录为`REJECTED_EXTRA_GATE`并继续白名单最小流程。

## N607发布与启动前证据

```text
experiment_code_commit=b293b1aba5fd2b6bab6830d283f79f23d79f5fd2
local_release_archive=E:/type10-7/release_artifacts/phase1_sat_anchor_ssl_b293b1ab.tar.gz
remote_release_archive=/home/szu2070436088/2510044040/CV-SincNet/releases/incoming/phase1_sat_anchor_ssl_b293b1ab.tar.gz
release_archive_sha256=6cf5e1aebc72a155cff03b824c7eced9062e8b9140d44cc06ac7ca66e6a0abe2
remote_release_root=/home/szu2070436088/2510044040/CV-SincNet/releases/phase1_sat_anchor_ssl_b293b1ab
remote_compile=VERIFIED
remote_dry_run=VERIFIED rows=8 trains=8 joint_evals=8 scenario_outputs=32
real_checkpoint_smoke=VERIFIED input_len=256 classes=6 domains=14 trusted=1
adapter_up_grad=0.00005296
adapter_logit_grad=0.05044098
id_backbone_u_grad=NONE
frozen_teacher_grad=NONE
```

启动前资源快照：

- GPU0已有1个现有训练CUDA进程，约2046MiB；GPU1已有1个，约1072MiB。
- GPU2–GPU7无CUDA计算进程；项目根所在磁盘剩余约7.3TiB。
- 本矩阵每GPU新增1条训练，因此GPU0/1启动后为每卡2条，GPU2–7为每卡1条，不超过默认上限。
- 现有任务属于`phase1_advb02_sidfft96_guarded_20260822_v1`；本任务不终止、不修改、不覆盖该任务。

精确启动面：

```text
cwd=/home/szu2070436088/2510044040/CV-SincNet/releases/phase1_sat_anchor_ssl_b293b1ab
launcher=code/scripts/launch_phase1_adv3b02_sat_anchor_ssl8_20260822.sh
data_root=/home/szu2070436088/2510044040/CV-SincNet
run_root=/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_adv3b02_sat_anchor_ssl8_s392002_20260822
dispatcher_log=/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_adv3b02_sat_anchor_ssl8_s392002_20260822.dispatcher.log
dispatcher_pid=/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_adv3b02_sat_anchor_ssl8_s392002_20260822.dispatcher.pid
python=/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python
```

## 2026-08-23终态分析

### 终态

2026-08-23 19:10 CST只读回查确认：A0—A4共7条候选均完成E200，分别保存200条可解析`metrics_epoch.jsonl`、`final_ssdg.pth`以及clean和三种LEO弱信道评测；所有评测均为strict reconstruction，missing/unexpected/shape mismatch为0。A5在E63因连续两个完整epoch无优化器更新而终止，没有final checkpoint和性能结果。相关训练进程已全部退出，8张RTX3090均空闲。

矩阵终态为`PARTIAL_TECHNICAL_FAILURE / ANALYZED`：7条有效性能行，1条`TRAIN_FAILED / NO_PERFORMANCE_RESULT`。

### 最终结果

|候选|clean/%|clear/%|low-elev/%|rain/%|LEO均值/%|四场景floor/%|
|---|---:|---:|---:|---:|---:|---:|
|A0_CONTROL|84.847|75.215|73.128|72.113|73.486|58.467|
|A1_STRICT_USAT|85.075|75.447|73.365|72.293|73.702|58.408|
|A2_USAT_ANCHOR|84.673|75.372|73.305|72.277|73.651|59.408|
|A3_ADAPTIVE_NO_FILL|84.567|75.508|73.553|72.558|73.873|58.208|
|A3_FIXED_50_FILL|84.670|75.518|73.583|72.652|73.918|58.550|
|A3_PAIR_INTERVAL2|84.590|75.335|73.260|72.337|73.644|58.900|
|A4_CLASS_RX_CAP|84.345|75.232|73.253|72.360|73.615|58.975|
|A5_ADAPTER_TAIL|N/A|N/A|N/A|N/A|N/A|N/A|

相对A0，A1的clean/LEO/floor变化为+0.228/+0.216/-0.058pp；A2为-0.173/+0.166/+0.942pp；A3 adaptive为-0.280/+0.388/-0.258pp；A3 fixed-fill为-0.177/+0.432/+0.083pp；A3 interval2为-0.257/+0.158/+0.433pp；A4为-0.502/+0.129/+0.508pp。

### 路由与训练健康

- A1/A2/A3 adaptive/A3 interval2/A4在有效阶段平均消费约63.91/256条可信U样本；A3 fixed-fill为127.83/256，其中约63.91条来自回填。
- A3 interval2和A4的pair active比例为0.4976；每step pair的A3 adaptive/fixed-fill为1.0。
- A0—A4平均optimizer step应用率约99.91%，完整日志没有Traceback、OOM或系统性非有限更新。
- A5从E43开始出现零更新；E59的clip前梯度达到71,646.6、总loss713.54，E60和E63的`train_skipped_nonfinite_loss=1.0`且optimizer step率为0。source satellite mean从最高89.836%降至16.667%，随后触发预注册的`FASTTRUST_CONSECUTIVE_ZERO_OPTIMIZER_STEP_EPOCHS`。

### 裁决

SAT-Anchor证明adaptive no-fill的严格U星地身份监督可以稳定训练，但7条有效行的LEO增益只有0.129—0.432pp，且clean、LEO均值和floor的最优点分属不同候选。当前没有一条同时满足显著LEO提升、floor提升和clean保护；单seed证据不足以晋级为新的Phase1默认方法。
