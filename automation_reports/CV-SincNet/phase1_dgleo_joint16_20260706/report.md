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

## 运行时长估计

|时间|依据|估计|
|---|---|---|
|2026-07-06 16:40:37 CST|16个`metrics_epoch.jsonl`全量已写入行；各候选当前约50-54/200轮。|按全程均值估算，最慢剩余约2.68小时，中位剩余约2.61小时。|
|2026-07-06 16:40:37 CST|按最近5轮均值估算，近期含更重评估/IO开销。|最慢剩余约3.34小时，中位剩余约3.25小时。|

当前建议按保守口径预期约3.2-3.4小时后跑完，即大约2026-07-06 19:55-20:05 CST。若后续进入`test_eval_final_window=30`的密集评估阶段，最后30轮可能比中段更慢，完成时间可能再后移约10-25分钟。

## ETA快照

|时间|进度|近8轮平均epoch耗时|线性ETA|保守ETA|
|---|---|---|---|---|
|2026-07-06 16:18 CST|16个候选均在运行；最新epoch范围30-34/200，均处于label阶段。|约69-71秒/epoch。|约3.2-3.35小时。|考虑pseudo阶段、最终窗口更密集测试和checkpoint导出，预计约3.8-4.5小时。|

预计完成窗口：2026-07-06 19:35-20:45 CST。该估计基于早期label阶段速度，后续若GPU负载、I/O或最终评估变慢，需要重新刷新。

## 后续检查

启动后4-5分钟检查每个候选日志是否出现`[CONFIG-LOSS]`、`[CONFIG-SAT]`、`[CONFIG-CONCAT-SAT]`、`[EPOCH-BEGIN]`、`[LOSS-SAT-W]`、`[ZID-FEATURE-SPACE]`或对应loss记录；同时扫描Traceback、RuntimeError、CUDA OOM、NaN、unrecognized arguments和Killed。训练完成后必须导出`phase2_zid_prototypes.pt/json`，再做single-observation LEO的Stage2同row复评。

## 完成分析

### 协议边界

本批`phase1_dgleo_joint16_20260706`是Phase1 source-only地面训练。训练数据仍只使用`ManySig.pkl`，不使用目标接收机样本、真实`Y_unknown`、ManyTx、unknown query、target support、target统计、target阈值或target早停。因此本节只能评价闭集DG能力、星地压力鲁棒性、known特征几何、proxy unknown风险和prototype导出质量；不能声明真实`unknown_FAR`、`FPR95`、Stage2 `old_acc`、`seen_new_acc`或`H_old_new`已改善。

### 完整性与健康

|检查项|结果|
|---|---|
|候选数|16/16完成。|
|epoch|16/16均为200/200。|
|metrics|16/16有完整`metrics_epoch.jsonl`。|
|stdout|16/16有stdout日志，均有200个`[EPOCH-END]`。|
|prototype导出|16/16均有`phase2_zid_prototypes.pt`和`.json`。|
|fatal|未发现`Traceback`、`RuntimeError`、OOM、`Killed`、参数错误或导入错误。|
|非有限跳步|`J11_STRONG_A`156次、`J11_STRONG_B`151次、`J4_PROXY_B`146次非有限grad跳步；其余为0。强拒识组稳定性较差。|
|NaN分类|日志中的`nan`主要来自未激活loss、未评估轮或aux grad占位，不是fatal。|

### 泛化能力主表

|候选|overall|strict UDU|receiver floor|最弱receiver|sat mean|sat floor|best-final strict gap|结论|
|---|---:|---:|---:|---|---:|---:|---:|---|
|`J10_BALANCED_A`|88.67|83.65|69.59|rx11 69.59|78.01|77.12|1.94|星地尚可，rx11掉到70以下，final回落明显。|
|`J10_BALANCED_B`|89.73|85.62|74.80|rx11 74.80|78.67|77.68|0.06|泛化最稳，sat/floor/strict平衡较好。|
|`J11_STRONG_A`|89.41|85.11|72.38|rx11 72.38|76.54|75.56|1.23|闭集尚可，sat floor弱。|
|`J11_STRONG_B`|89.86|85.90|75.71|rx11 75.71|76.96|75.98|0.08|strict/floor强，但sat floor弱且非有限grad多。|
|`J1_BASE_A`|89.44|84.61|74.17|rx11 74.17|78.33|77.35|0.97|EPOC concat保护sat，strict中等。|
|`J1_BASE_B`|89.38|84.69|72.84|rx11 72.84|78.62|77.69|1.82|sat好，后期回落较明显。|
|`J2_DOMAIN_A`|89.11|84.14|70.01|rx11 70.01|77.92|76.96|1.30|刚过floor线，弱receiver未修复。|
|`J2_DOMAIN_B`|89.46|85.09|72.41|rx11 72.41|78.57|77.58|1.04|泛化合格，非最稳。|
|`J3_BRIDGE_A`|89.50|85.38|73.38|rx11 73.38|78.52|77.56|0.96|泛化较稳。|
|`J3_BRIDGE_B`|88.95|84.13|70.30|rx11 70.30|77.97|77.05|1.69|final回落，rx11弱。|
|`J4_PROXY_A`|89.69|84.99|70.49|rx11 70.49|78.66|77.67|0.90|sat强，rx11仍弱。|
|`J4_PROXY_B`|89.27|84.55|70.71|rx11 70.71|76.58|75.68|1.69|sat弱且非有限grad多。|
|`J5_RADIUS_A`|89.33|84.95|72.60|rx11 72.60|77.96|77.02|1.14|泛化合格。|
|`J5_RADIUS_B`|89.43|84.97|72.53|rx11 72.53|77.84|76.84|1.18|sat略弱。|
|`J7_KD_A`|89.60|85.61|72.13|rx11 72.13|78.32|77.37|0.39|KD保护有效，best-final稳定。|
|`J7_KD_B`|89.62|85.18|71.33|rx11 71.33|78.73|77.73|0.97|sat最高，receiver floor偏弱。|

泛化结论：EPOC concat-sat基底确实保护了satellite floor，多数组final sat floor在77%左右，优于强拒识/旧shell类EPOC参考。但弱点集中在rx11，receiver floor没有被根治。最稳泛化候选是`J10_BALANCED_B`和`J7_KD_A`；`J11`组闭集看起来强，但sat floor低且grad稳定性差。

### 拒识潜力主表

|候选|p95|p99|zid_tail_cvar|source_overflow|proxy_vaccept|bridge|shell|radius/inter|结论|
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
|`J10_BALANCED_A`|71.02|83.76|64.40|0.971|0.826|1.000|0.741|1.070|高风险。|
|`J10_BALANCED_B`|70.32|83.99|64.38|0.970|0.823|1.000|0.744|1.113|高风险。|
|`J11_STRONG_A`|71.95|84.04|65.59|0.978|0.834|1.000|0.739|1.078|强拒识未生效。|
|`J11_STRONG_B`|73.36|85.00|65.90|0.979|0.835|1.000|0.785|1.074|最危险之一。|
|`J1_BASE_A`|70.53|83.21|64.76|0.949|nan|nan|nan|nan|无proxy激活，source overflow仍高。|
|`J1_BASE_B`|70.33|83.15|64.08|0.960|nan|nan|nan|nan|同上。|
|`J2_DOMAIN_A`|71.18|83.83|65.03|0.949|nan|nan|nan|nan|domain未收紧tail。|
|`J2_DOMAIN_B`|69.98|84.21|64.09|0.946|nan|nan|nan|nan|p95略低但p99/overflow高。|
|`J3_BRIDGE_A`|71.45|84.23|64.77|0.958|0.836|1.000|0.769|1.076|bridge压制失败。|
|`J3_BRIDGE_B`|71.10|83.83|64.08|0.969|0.837|1.000|0.742|1.056|高风险。|
|`J4_PROXY_A`|69.61|84.04|63.93|0.957|0.827|1.000|0.751|1.080|p95低但p99/overflow/vaccept失败。|
|`J4_PROXY_B`|73.34|84.46|66.01|0.971|0.836|1.000|0.730|1.083|proxy增强反扩尾。|
|`J5_RADIUS_A`|69.33|83.22|63.45|0.969|0.836|1.000|0.772|1.063|radius项未压ratio。|
|`J5_RADIUS_B`|69.70|84.08|63.81|0.977|0.843|1.000|0.733|1.087|vaccept最差。|
|`J7_KD_A`|72.30|83.74|64.73|0.947|0.824|1.000|0.723|1.061|KD保护闭集但未收尾。|
|`J7_KD_B`|70.54|83.37|64.22|0.945|0.825|1.000|0.700|1.081|overflow最低但仍严重。|

拒识结论：没有实质改善。相对`EPOC_DISTILL_B_KDHI`，当前中位`proxy_vaccept`从0.809恶化到0.835，`source_overflow`从0.354恶化到0.965，`zid_p95`从49.79°恶化到70.78°，`zid_p99`从70.70°恶化到83.91°，`zid_tail_cvar`从52.19°恶化到64.39°。`low_density_accept_rate`很低，但不能抵消`bridge_accept=1.0`、`proxy_vaccept≈0.82-0.84`和高overflow。

### 双目标四象限

|象限|候选|判断|
|---|---|---|
|泛化提升且拒识风险下降|无|没有主推进候选。|
|泛化提升但拒识风险上升|`J10_BALANCED_B`、`J7_KD_A`、`J7_KD_B`、`J3_BRIDGE_A`、`J1_BASE_A/B`、`J2_DOMAIN_B`、`J4_PROXY_A`、`J5_RADIUS_A`|闭集/星地可用，但open-set危险。|
|泛化下降但拒识代理改善|无明显成立|没有看到proxy/source/tail体系真实改善。|
|两者都差或机制负例|`J11_STRONG_A/B`、`J4_PROXY_B`、`J5_RADIUS_B`、`J10_BALANCED_A`、`J2_DOMAIN_A`、`J3_BRIDGE_B`|作为负例或诊断，不宜推进。|

### 机制归因与失败模式

1. source episode扩大了known tail：final中位`source_overflow=0.965`，比E10的0.952还高；`source_episode`没有形成安全边界，反而把source tail长期保留在可接收域。
2. vacuum/proxy unknown没有降低接收：proxy激活组`proxy_vaccept≈0.82-0.84`，`bridge_accept=1.0`，说明virtual unknown和bridge样本仍几乎被known接收。
3. fusion/local component有导出字段：`phase2_zid_prototypes.pt`包含`fused_tx_prototypes`、`fused_tx_radii`、`fused_tx_accept_radii`、`fused_tx_mask`、`fusion_metadata`，不是空flag；但这是导出成功，不等于Stage2 unknown成功。
4. longer epoch带来tail扩张：E10到final中位`zid_p95`从59.76°升到70.78°，`zid_p99`从79.69°升到83.91°，`zid_tail_cvar`从57.77°升到64.39°，`tail_frac`从0.0526升到0.0758。
5. satellite增强保护平均星地性能，但弱receiver未修复：多数sat floor约77%，但最弱rx11仍是主瓶颈，多个候选final receiver floor只有70-72。
6. pseudo-label阶段引入后期偏置：E150后`pseudo_reliable_ratio≈0.908`，但strict UDU和receiver floor总体下降，说明大量高置信伪标签更可能强化source/receiver偏置，而不是修复跨域边界。
7. best checkpoint强但final退化：`J10_A` strict gap 1.94pp、receiver floor gap 4.38pp；`J4_PROXY_A` receiver floor gap 5.66pp；`J5_RADIUS_B` gap 5.62pp。最终checkpoint不可直接拿来Stage2声称成功。

### 候选决策

|candidate|泛化结论|拒识潜力|主要风险|可否Stage2真实unknown评估|下一步动作|
|---|---|---|---|---|---|
|`J10_BALANCED_A`|borderline，rx11<70|差|overflow/vaccept/gap|否|只做失败对照。|
|`J10_BALANCED_B`|最稳|差|overflow 0.970、vaccept 0.823|可做诊断评估，不能promote|保留为闭集DG+LEO对照。|
|`J11_STRONG_A`|sat弱|差|非有限grad、overflow|否|淘汰。|
|`J11_STRONG_B`|strict强但sat弱|差|非有限grad、vaccept高|否|淘汰。|
|`J1_BASE_A`|sat稳|差|无proxy且overflow高|可做底座对照|保留EPOC concat低压对照。|
|`J1_BASE_B`|sat稳，回落|差|overflow/p99|可做底座对照|保留对照。|
|`J2_DOMAIN_A`|弱receiver未修复|差|floor低、overflow高|否|淘汰。|
|`J2_DOMAIN_B`|泛化合格|差|p99/overflow高|诊断可评|检查domain强度上限。|
|`J3_BRIDGE_A`|泛化较稳|差|bridge=1.0|诊断可评|作为bridge失败样本。|
|`J3_BRIDGE_B`|final回落|差|gap/overflow|否|淘汰。|
|`J4_PROXY_A`|sat强，rx弱|差|p95低但p99/overflow/vaccept坏|诊断可评|用于proxy失效剖析。|
|`J4_PROXY_B`|sat弱|差|非有限grad、扩尾|否|淘汰。|
|`J5_RADIUS_A`|泛化合格|差|radius_ratio>1、overflow|诊断可评|用于radius gate重设。|
|`J5_RADIUS_B`|borderline|差|vaccept最高|否|淘汰。|
|`J7_KD_A`|稳且KD有效|差|overflow/vaccept|可做诊断评估|最适合作为下一轮闭集保护底座。|
|`J7_KD_B`|sat最高，floor偏弱|差|overflow仍高|可做诊断评估|作为sat上限对照。|

### 下一轮实验设计

|组|目标|变量|指标|成功标准|失败判据|
|---|---|---|---|---|---|
|G1 hard gate/local component dry-run|验证local component是否能硬拒tail/bridge|不重训，读取prototype，扫`accept_radius_key=p80/p90/p95`、component gate、global off|old coverage、proxy_vaccept、bridge_accept、source_overflow|old coverage>=80且proxy_vaccept<0.4、bridge<0.3|old coverage崩或bridge仍>0.8|
|G2真实Stage2 unknown query评估|确认Phase1代理与真实unknown是否一致|只评估，不回传训练；unknown query eval-only|old_acc、unknown reject、seen-new注册质量|只作为诊断，不用于训练阈值|任何用unknown调参即作废|
|G3 shell/inter-class/same-class bridge negative|直接打低密度桥区|构造shell negative、inter-class interpolation、same-class cross-domain bridge negative|bridge_accept、shell_accept、radius_ratio、p99|bridge<0.5且strict下降<1pp|strict大跌或shell仍>0.7|
|G4 core/tail/outside quarantine|把known分成core/tail/outside，tail不自动接收|tail sentinel、tail auto accept off、overflow cap hard gate|tail_frac、source_overflow、p99、old coverage|source_overflow<0.5且old coverage可控|p95降但p99/overflow不降|
|G5 source episode density gate|修复source episode扩尾|episode只采core density，不把3sigma外样本纳入正域|source_overflow、r3sigma、tail_cvar|overflow显著下降，zid_tail_cvar下降>5°|tail_cvar不降或receiver floor掉>2pp|
|G6弱receiver sat stress修复|针对rx11和LEO stress共同修复|rx11-aware sampler、sat stress on weak receiver、KD保护|receiver floor、rx11、sat floor、strict UDU|rx11>75且sat floor>=77|只涨sat mean不涨rx11|

### 最终判断

当前实验对Phase1的贡献是：验证了EPOC concat-sat可以保护一部分闭集DG和星地压力性能，尤其`J10_BALANCED_B`、`J7_KD_A/B`提供了较稳的闭集/星地底座；同时证明“把open-set代理loss直接叠加到concat-sat全损失批次”并不能自然产生可拒识表征。

当前不能声明的是：真实`unknown_FAR`、`FPR95`、Stage2 old/new/unknown成功、fusion拒识成功或部署成功。

最主要风险是：known tail显著扩张、`source_overflow≈0.95-0.98`、`proxy_vaccept≈0.82-0.84`、`bridge_accept=1.0`、弱receiver rx11未修复，以及部分候选best-final gap较大。

最值得推进的候选是：没有可直接promote的Stage2主候选。若必须选一个做诊断评估，选`J7_KD_A`或`J10_BALANCED_B`，原因是它们泛化较稳；但它们拒识代理失败，不能作为“可拒识表征”候选。

后期多数实验strict UDU下降的原因在于：pseudo阶段从E150后引入大量高置信伪标签，配合full concat sat和强teacher/sat KD继续保护闭集分类，却没有同步执行硬拒接收边界；source episode和proxy约束权重相对分类/KD太软，只形成loss惩罚而不是接收决策约束。表现为train/val维持高位、sat floor提升，但`zid_p95/p99/tail_cvar`和`source_overflow`持续扩张，rx11这个弱receiver被进一步放大，最终strict UDU和receiver floor从best点回落。
