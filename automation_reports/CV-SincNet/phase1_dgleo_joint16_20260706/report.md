# phase1_dgleo_joint16_20260706

|字段|内容|
|---|---|
|实验ID|`phase1_dgleo_joint16_20260706`|
|时间|2026-07-06|
|操作者|Codex|
|目标|落地EPOC concat星地增强基底上的DG+LEO优先source-only地面域泛化联合优化，每张GPU两实验，共16候选。|
|协议边界|Phase1/source-only地面训练；训练数据只允许`ManySig.pkl`；不使用目标接收机样本、真实`Y_unknown`、ManyTx、unknown query、target support、target统计、target阈值或target早停。|
|核心原则|以EPOC的clean+sat 2B拼接训练保持星地性能；CE、domain、ADV、cons、Fishr、prototype、zid compact、proxy/bridge/tail/source episode共同作用在拼接批次上，再在不牺牲known coverage的前提下压proxy/bridge/overflow/tail风险。|

## 用户修正后的落地原则

|原则|实现|
|---|---|
|EPOC星地基底|使用`--use_concat_sat_channel_aug --no_concat_sat_ce_only`，从第1轮开始构造clean+LEO拼接批次。|
|不只TX CE|拼接后的2B批次进入`loss_tx_l`、`loss_domain_labeled`、`loss_adv_labeled`、`loss_cons_labeled`、`loss_fishr_labeled`、`loss_proto_labeled`、`loss_zid_compact`、`loss_proxy_unknown`、`loss_source_episode`。|
|星地专项保护|sat半批次额外保留`sat_cls/sat_cons/sat_kl`可控项，主看`sat_floor`、`sat_strict_floor`、`receiver_floor`和后续LEO open-set风险。|
|协议安全|仍只使用`ManySig.pkl`；proxy unknown为源域虚拟/holdout机制，不接触真实未知类或target receiver。|

## 实验设计

|GPU|候选|组别|目的|
|---:|---|---|---|
|0|`DGLEO_J1_BASE_A`|J1|DG+LEO基础增强，验证星地视图不只走TX CE。|
|0|`DGLEO_J1_BASE_B`|J1|提高Fishr/group CE，观察receiver floor和sat floor。|
|1|`DGLEO_J2_DOMAIN_A`|J2|受限GRL+`z_dom`域分类，检查域解耦收益。|
|1|`DGLEO_J2_DOMAIN_B`|J2|更强域解耦负控，检查是否伤TX身份。|
|2|`DGLEO_J7_KD_A`|J7|old LEO teacher蒸馏保护。|
|2|`DGLEO_J7_KD_B`|J7|更强sat KD和sat CE，主看sat floor。|
|3|`DGLEO_J3_BRIDGE_A`|J3|中等bridge CVaR，压`bridge_accept_leo`。|
|3|`DGLEO_J3_BRIDGE_B`|J3|更强bridge压力，作为责任拆解。|
|4|`DGLEO_J4_PROXY_A`|J4|proxy unknown LEO中等权重。|
|4|`DGLEO_J4_PROXY_B`|J4|更密集proxy/soft unknown。|
|5|`DGLEO_J5_RADIUS_A`|J5|local radius budget中等约束。|
|5|`DGLEO_J5_RADIUS_B`|J5|更紧local radius budget。|
|6|`DGLEO_J10_BALANCED_A`|J10|主推平衡联合候选。|
|6|`DGLEO_J10_BALANCED_B`|J10|更强old LEO KD保护的平衡候选。|
|7|`DGLEO_J11_STRONG_A`|J11|强拒识上限负控。|
|7|`DGLEO_J11_STRONG_B`|J11|old保护增强的强拒识负控。|

## P0门控

|门控|要求|
|---|---|
|协议|`real_unknown_classes_in_training=0`、`target_receiver_samples_in_training=0`、`stage2_unknown_query_eval_only=1`。|
|DG|`strict UDU`、`receiver_floor`不得低于强底座同口径1-2pp。|
|LEO|`sat_floor`、`sat_strict_floor`不得退化；优先目标`sat_floor>=77%`。|
|old保持|后续Stage2 eval-only中先满足`old_acc>=80%`和`known_coverage>=80%`。|
|开集风险|在old保持后再看`proxy_vaccept_leo`、`bridge_accept_leo`、`source_overflow_leo`、`tail_accept_leo`。|
|声明|任何clean、proxy、p99、resource或后处理单项改善都不能声明部署成功。|

## 本地变更

|文件|用途|
|---|---|
|`code/SSDG/train_ssdg.py`|补齐EPOC式concat星地批次在SSDG训练中的实际接入，使联合损失作用于clean+sat拼接样本。|
|`code/scripts/launch_phase1_dgleo_joint16_20260706.sh`|新增16候选EPOC concat-sat+DG+LEO联合优化启动器。|
|`code/tests/test_phase1_dgleo_joint16_launcher.py`|新增dry-run、source-only协议和每卡两实验测试。|
|`automation_reports/CV-SincNet/phase1_dgleo_joint16_20260706/report.md`|本报告。|

## 待验证与同步

|项目|状态|
|---|---|
|本地Git状态|`E:\type10-7`和`E:\type10-7\code`不是Git仓库；代码已镜像到`github_publish/CVS-RFFI-repo`并提交`9440745 Add EPOC concat-sat DGLEO joint launcher`、`084c6d3 Use nohup for DGLEO joint launcher`；报告提交`725b6df Record DGLEO joint16 experiment report`；既有`local_artifacts/phase2_adv3b02_*`未跟踪项未改动。|
|N607预检|已通过；8张GPU均约10MiB，未发现活动训练进程。|
|本地验证|通过`python -m py_compile code/SSDG/train_ssdg.py`、`bash -n code/scripts/launch_phase1_dgleo_joint16_20260706.sh`、启动器dry-run、`ssr-gpu`环境`pytest -q code/tests/test_phase1_dgleo_joint16_launcher.py`。|
|快照|已创建`code/snapshots/phase1_dgleo_joint16_20260706/`，包含`train_ssdg.py`、启动器和测试。|
|远端同步|已同步启动器、测试和`train_ssdg.py`到N607；同步后SHA256分别为`d55e3158678b2261ea77620bf6785be7a7422046aca8032aef9c60369e0d0878`、`7bd07e38fdcdf47361db47c3615440aa2a3378e6f1fcf6b9febc40f7edcdd98f`、`27fbe66d971d232b0f3be585259cd8fc8b453baee8bc2108f01f40db5f190f55`。启动器使用`nohup "${CMD[@]}" > "${log_path}" 2>&1 &`防止SSH退出影响训练进程。|
|远端备份|远端原`code/SSDG/train_ssdg.py`已备份到`code/snapshots/phase1_dgleo_joint16_20260706/remote_before_sync_20260706_154031/SSDG/train_ssdg.py`，原SHA256为`5e7950dcf0cdb222ef92b3303c4980be83a6e66fb9305a100148343b97639f9c`。|
|远端验证|通过N607端`py_compile`、`bash -n`和单候选dry-run；dry-run显示`base=EPOC_CONCAT_SAT`、`concat_sat_mode=full_2b_core_domain`、`concat_sat_full_loss=1`。|
|远端启动|已使用`MAX_ACTIVE_PER_GPU=2 LAUNCH_SETTLE_SECONDS=12`启动16候选；每张GPU两个训练进程。|

## 同步映射

|本地文件|远端文件|
|---|---|
|`E:\type10-7\code\SSDG\train_ssdg.py`|`/home/szu2070436088/2510044040/CV-SincNet/code/SSDG/train_ssdg.py`|
|`E:\type10-7\code\scripts\launch_phase1_dgleo_joint16_20260706.sh`|`/home/szu2070436088/2510044040/CV-SincNet/code/scripts/launch_phase1_dgleo_joint16_20260706.sh`|
|`E:\type10-7\code\tests\test_phase1_dgleo_joint16_launcher.py`|`/home/szu2070436088/2510044040/CV-SincNet/code/tests/test_phase1_dgleo_joint16_launcher.py`|

## 计划启动命令

```bash
cd /home/szu2070436088/2510044040/CV-SincNet
MAX_ACTIVE_PER_GPU=2 bash code/scripts/launch_phase1_dgleo_joint16_20260706.sh
```

## 启动结果

|候选|GPU|PID|日志|
|---|---:|---:|---|
|`DGLEO_J1_BASE_A`|0|3522942|`logs/phase1_dgleo_joint16_20260706/DGLEO_J1_BASE_A.out`|
|`DGLEO_J1_BASE_B`|0|3523348|`logs/phase1_dgleo_joint16_20260706/DGLEO_J1_BASE_B.out`|
|`DGLEO_J2_DOMAIN_A`|1|3523770|`logs/phase1_dgleo_joint16_20260706/DGLEO_J2_DOMAIN_A.out`|
|`DGLEO_J2_DOMAIN_B`|1|3524176|`logs/phase1_dgleo_joint16_20260706/DGLEO_J2_DOMAIN_B.out`|
|`DGLEO_J7_KD_A`|2|3524614|`logs/phase1_dgleo_joint16_20260706/DGLEO_J7_KD_A.out`|
|`DGLEO_J7_KD_B`|2|3525431|`logs/phase1_dgleo_joint16_20260706/DGLEO_J7_KD_B.out`|
|`DGLEO_J3_BRIDGE_A`|3|3525872|`logs/phase1_dgleo_joint16_20260706/DGLEO_J3_BRIDGE_A.out`|
|`DGLEO_J3_BRIDGE_B`|3|3526278|`logs/phase1_dgleo_joint16_20260706/DGLEO_J3_BRIDGE_B.out`|
|`DGLEO_J4_PROXY_A`|4|3526715|`logs/phase1_dgleo_joint16_20260706/DGLEO_J4_PROXY_A.out`|
|`DGLEO_J4_PROXY_B`|4|3527121|`logs/phase1_dgleo_joint16_20260706/DGLEO_J4_PROXY_B.out`|
|`DGLEO_J5_RADIUS_A`|5|3527966|`logs/phase1_dgleo_joint16_20260706/DGLEO_J5_RADIUS_A.out`|
|`DGLEO_J5_RADIUS_B`|5|3528374|`logs/phase1_dgleo_joint16_20260706/DGLEO_J5_RADIUS_B.out`|
|`DGLEO_J10_BALANCED_A`|6|3528811|`logs/phase1_dgleo_joint16_20260706/DGLEO_J10_BALANCED_A.out`|
|`DGLEO_J10_BALANCED_B`|6|3529217|`logs/phase1_dgleo_joint16_20260706/DGLEO_J10_BALANCED_B.out`|
|`DGLEO_J11_STRONG_A`|7|3529655|`logs/phase1_dgleo_joint16_20260706/DGLEO_J11_STRONG_A.out`|
|`DGLEO_J11_STRONG_B`|7|3530476|`logs/phase1_dgleo_joint16_20260706/DGLEO_J11_STRONG_B.out`|

## 启动后健康检查

|检查项|结果|
|---|---|
|日志数量|16个`*.out`。|
|进程/GPU|`nvidia-smi pmon`显示0-7号GPU各2个Python训练进程。|
|EPOC concat接入|16个日志均出现`[CONFIG-CONCAT-SAT]`；16份`metrics_epoch.jsonl`均出现`concat_sat_expanded`字段。|
|训练推进|16个日志均出现`[EPOCH-BEGIN]`和`[EPOCH-END]`。|
|错误扫描|未发现`Traceback`、`RuntimeError`、`CUDA out of memory`、`unrecognized arguments`、`Killed`、`ModuleNotFoundError`、`ImportError`。|
|SSH清理|每轮SSH/SCP后本地检查均为`NO_SSH_PROCESS`、`NO_N607_OR_BRIDGE_ESTABLISHED_22`。|

## ETA快照

|时间|进度|近8轮平均epoch耗时|线性ETA|保守ETA|
|---|---|---|---|---|
|2026-07-06 16:18 CST|16个候选均在运行；最新epoch范围30-34/200，均处于label阶段。|约69-71秒/epoch。|约3.2-3.35小时。|考虑pseudo阶段、最终窗口更密集测试和checkpoint导出，预计约3.8-4.5小时。|

预计完成窗口：2026-07-06 19:35-20:45 CST。该估计基于早期label阶段速度，后续若GPU负载、I/O或最终评估变慢，需要重新刷新。

## 后续检查

启动后4-5分钟检查每个候选日志是否出现`[CONFIG-LOSS]`、`[CONFIG-SAT]`、`[CONFIG-CONCAT-SAT]`、`[EPOCH-BEGIN]`、`[LOSS-SAT-W]`、`[ZID-FEATURE-SPACE]`或对应loss记录；同时扫描Traceback、RuntimeError、CUDA OOM、NaN、unrecognized arguments和Killed。训练完成后必须导出`phase2_zid_prototypes.pt/json`，再做single-observation LEO的Stage2同row复评。
