# phase1_dgleo_osfix16_20260707

## 基本信息

|字段|值|
|---|---|
|实验ID|`phase1_dgleo_osfix16_20260707`|
|时间|2026-07-07|
|操作者|Codex Phase1地面训练实验agent|
|阶段边界|Phase1 source-only地面域泛化训练|
|数据边界|`Dataset_WigSig/ManySig.pkl`，禁止ManyTx/target/unknown进入训练|
|核心目标|保护overall/strict UDU/receiver floor/satellite floor，同时直接压低known接收域open-set风险代理指标|
|基线参照|`phase1_dgleo_directmetric16_20260706`和`phase1_dgleo_uopt24_20260707`，尤其`DGLEO_DM_P0D_RADIUS_A`、`DGLEO_UOPT_P0_CORE_C`|
|当前状态|本地实现和验证完成，等待同步并在N607启动|

## 假设

上一轮DirectMetric16/UOPT24已经证明直接指标损失和温和无标签利用能提升p95/p99/proxy_vaccept，但source_overflow、source_episode_overflow、tail/overflow接收和弱receiver/星地floor仍是主要矛盾。本轮不是继续间接调参，而是把Phase1训练中的open-set代理指标写进直接优化项：

- labeled/source侧：继续使用`direct_metric_acceptance_loss`直接约束`zid_p50/p95/p99/tail_cvar`、`source_overflow`、`proxy_vaccept`、`bridge_accept_rate`、`low_density_accept_rate`、`tail_accept_rate`、`overflow_accept_rate`、`radius_to_inter_ratio`。
- unlabeled/source侧：新增`unlabeled_known_acceptance_quarantine_loss`，对未通过伪标签门控的source无标签样本，直接降低其落入known local-component接收门的概率。
- 星地视图：保持`concat_sa`样本拼接形式，不使用CE-only；同时保留domain loss、ADV loss、sat consistency、U_s direct metric和U_s quarantine在星地增强视图上的约束。

## 本地改动

|文件|作用|
|---|---|
|`code/cvsrffi/losses.py`|新增`unlabeled_known_acceptance_quarantine_loss`，用source known anchors构建local gate，直接惩罚untrusted U_s被known接收。|
|`code/SSDG/train_ssdg.py`|接入`lambda_u_quarantine_accept`、valid-domain-only U_s direct metric、U_s p50/tail_cvar日志、quarantine日志和损失权重。|
|`code/scripts/launch_phase1_dgleo_osfix16_20260707.sh`|新增16候选launcher，每张GPU两实验，Phase1 source-only guard，concat_sa full模式。|
|`code/tests/test_unlabeled_quarantine_acceptance_loss.py`|覆盖新增quarantine损失和训练参数入口。|
|`code/tests/test_phase1_dgleo_osfix16_launcher.py`|覆盖OSFIX16协议声明、每卡两实验分配和ManyTx拒绝。|
|`automation_reports/CV-SincNet/phase1_dgleo_osfix16_20260707/local_dry_run.txt`|完整本地dry-run命令记录。|

## 本地验证

|检查|结果|
|---|---|
|根目录Git状态|`E:\type10-7`不是Git仓库，按规则使用`github_publish/CVS-RFFI-repo`作为Git承载面。|
|语法检查|`python -m py_compile code/cvsrffi/losses.py code/SSDG/train_ssdg.py`通过。|
|launcher语法|`bash -n code/scripts/launch_phase1_dgleo_osfix16_20260707.sh`通过。|
|聚焦测试|`14 passed`：DirectMetric旧测试、UOPT入口测试、新quarantine测试、新OSFIX16 launcher测试均通过。|
|完整dry-run|候选数16；GPU分布`0:2 1:2 2:2 3:2 4:2 5:2 6:2 7:2`。|
|dry-run路径|`automation_reports/CV-SincNet/phase1_dgleo_osfix16_20260707/local_dry_run.txt`|

## 实验矩阵

|candidate|GPU|组别|目标|主要直接优化压力|
|---|---:|---|---|---|
|`DGLEO_OSFIX_CORE_A`|0|P0_CORE|UOPT最优安全基线加quarantine|温和source/proxy/tail+U_s quarantine|
|`DGLEO_OSFIX_CORE_B`|0|P0_CORE|DirectMetric半径安全基线加quarantine|radius/inter和source overflow保守加强|
|`DGLEO_OSFIX_DENSITY_A`|1|P0_DENSITY|source overflow密度门|source_overflow、overflow_accept、radius/inter|
|`DGLEO_OSFIX_DENSITY_B`|1|P0_DENSITY|source overflow严格密度门|更强source_overflow和p99/tail_cvar|
|`DGLEO_OSFIX_PROXY_A`|2|P0_PROXY|proxy_vaccept CVaR安全压制|proxy_vaccept、virtual unknown接收|
|`DGLEO_OSFIX_PROXY_B`|2|P0_PROXY|proxy_vaccept CVaR严格压制|更强proxy_vaccept和tail/overflow|
|`DGLEO_OSFIX_BRIDGE_A`|3|P0_BRIDGE|bridge/low-density/shell负样本|bridge_accept、low_density_accept|
|`DGLEO_OSFIX_BRIDGE_B`|3|P0_BRIDGE|bridge/low-density严格负样本|更强bridge、low-density、radius/inter|
|`DGLEO_OSFIX_TAIL_A`|4|P0_TAIL|tail/overflow/radius clamp|tail_accept、overflow_accept、p99|
|`DGLEO_OSFIX_TAIL_B`|4|P0_TAIL|tail/overflow/radius严格clamp|更强tail、overflow、p95/p99|
|`DGLEO_OSFIX_SATOPEN_A`|5|P0_SATOPEN|星地pair open-set floor保护|sat_pair、U_s sat consistency、sat floor|
|`DGLEO_OSFIX_SATOPEN_B`|5|P0_SATOPEN|星地pair严格open-set保护|更强sat_pair、U_s sat、proxy/tail|
|`DGLEO_OSFIX_UQ_A`|6|P1_UQ|无标签quarantine安全版本|U_s untrusted known-accept压制|
|`DGLEO_OSFIX_UQ_B`|6|P1_UQ|无标签quarantine严格版本|更强U_s quarantine和ADV|
|`DGLEO_OSFIX_JOINT_A`|7|P1_JOINT|联合平衡主候选|source/proxy/bridge/tail/sat/U_s joint|
|`DGLEO_OSFIX_JOINT_B`|7|P1_JOINT|联合强约束上界|强open-set代理压制，观察泛化冲突|

## N607同步计划

|本地文件|远端目标|
|---|---|
|`code/cvsrffi/losses.py`|`/home/szu2070436088/2510044040/CV-SincNet/code/cvsrffi/losses.py`|
|`code/SSDG/train_ssdg.py`|`/home/szu2070436088/2510044040/CV-SincNet/code/SSDG/train_ssdg.py`|
|`code/scripts/launch_phase1_dgleo_osfix16_20260707.sh`|`/home/szu2070436088/2510044040/CV-SincNet/code/scripts/launch_phase1_dgleo_osfix16_20260707.sh`|
|新增测试文件|`/home/szu2070436088/2510044040/CV-SincNet/code/tests/`对应路径|

计划远端验证：

```bash
cd /home/szu2070436088/2510044040/CV-SincNet
/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python -m py_compile code/cvsrffi/losses.py code/SSDG/train_ssdg.py
bash -n code/scripts/launch_phase1_dgleo_osfix16_20260707.sh
bash code/scripts/launch_phase1_dgleo_osfix16_20260707.sh --dry-run --only=DGLEO_OSFIX_JOINT_A
```

计划启动命令：

```bash
cd /home/szu2070436088/2510044040/CV-SincNet
MAX_ACTIVE_PER_GPU=2 LAUNCH_SETTLE_SECONDS=8 bash code/scripts/launch_phase1_dgleo_osfix16_20260707.sh
```

预期输出：

- run root：`/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_dgleo_osfix16_20260707`
- log root：`/home/szu2070436088/2510044040/CV-SincNet/logs/phase1_dgleo_osfix16_20260707`
- 每候选输出：`metrics_epoch.csv/jsonl`、`latest_ssdg.pth`、`best_joint_safe_ssdg.pth`、`phase2_zid_prototypes.pt/json`

## 关键观察指标

|维度|必须跟踪|
|---|---|
|泛化|`overall_tx`、`strict_udu`、receiver floor、satellite mean/floor、best-final gap、最弱receiver|
|known紧致性|`dm_accept_zid_p50/p95/p99`、`dm_accept_zid_tail_cvar_deg`、`zid_compact_tail_cvar_deg`|
|open-set代理风险|`dm_accept_source_overflow`、`source_episode_overflow_rate`、`dm_accept_proxy_vaccept`、`dm_accept_bridge_accept_rate`、`dm_accept_low_density_accept_rate`、`dm_accept_tail_accept_rate`、`dm_accept_overflow_accept_rate`、`dm_accept_radius_to_inter_ratio`|
|无标签利用|`pseudo_selected`、`u_dm_accept_selected`、`u_dm_accept_zid_p50/p95/p99/tail_cvar`、`u_quarantine_accept_rate`、`u_quarantine_low_density_accept_rate`、`u_quarantine_rate`|
|星地视图|`sat_mean/floor`、`dm_accept_sat_pair_angle_p95_deg`、`u_dm_accept_sat_pair_angle_p95_deg`、`loss_u_sat_cons`|

## 成功标准

主推进候选必须同时满足：

- 泛化不低于当前最优量级：final strict UDU不明显回落，receiver floor和satellite floor至少不下降。
- open-set代理指标至少两个P0风险显著下降：`proxy_vaccept`、`source_overflow`、`p99/tail_cvar`、`tail/overflow_accept`、`radius_to_inter_ratio`。
- 无标签不再只是提高闭集：`u_quarantine_accept_rate`下降，且`u_dm_accept_p95/p99/tail_cvar`没有恶化。
- best-final gap可控，不能只出现best checkpoint单点强。

失败判据：

- strict UDU或satellite floor明显下降，而open-set代理改善不稳定。
- `p95`下降但`p99/tail_cvar/source_overflow`仍高。
- `proxy_vaccept`下降但`bridge/low_density/tail_accept`上升。
- U_s quarantine压制导致pseudo_selected大幅下降且receiver floor退化。
- final相对best明显回落，说明训练后期仍在破坏跨域结构。

## 当前不能声明

这是Phase1 source-only训练。本实验启动和完成后也不能直接声明真实unknown_FAR、FPR95、Stage2 old_acc、seen_new_acc或H_old_new改善；只能说明闭集DG能力、星地压力鲁棒性、known特征几何、proxy/virtual unknown风险和prototype导出质量。

## N607落地状态

更新时间：2026-07-07 10:53-11:05 CST
本地Git承载面commit：`fc83410 Add Phase1 DG-LEO OSFIX16 training route`
状态：已完成本地实现、镜像提交、N607直连preflight、文件同步、远端语法/编译/dry-run验证，并已启动16个Phase1训练候选。当前是启动健康通过，不是完成结果。

### N607 preflight与占用

|检查项|结果|
|---|---|
|直连目标|`N607`可用|
|服务器时间|`2026年 07月 07日 星期二 10:53:50 CST`|
|远端项目根|`/home/szu2070436088/2510044040/CV-SincNet`可见|
|GPU|8张RTX3090可见|
|启动前训练进程|未发现`train_ssdg.py`或同run launcher|
|启动前GPU占用|仅Xorg背景进程|
|磁盘|`/home`约11T，总用量约27%，可用约7.6T|

### 同步与远端验证

|文件|SHA256|远端验证|
|---|---|---|
|`code/cvsrffi/losses.py`|`94d5d26bcc7fc799658bf67a3949e067e4beedc3c0b04b57cb86b6c5007f6d6b`|matched|
|`code/SSDG/train_ssdg.py`|`fcf3ca59cf3f6c6758c97cfb2cb917bbfd04373152ac2ab2478d51692f83688c`|matched|
|`code/scripts/launch_phase1_dgleo_osfix16_20260707.sh`|`97a78ace17b91116b3db7107b937d99814da6f4f893238980113e00175ea89f8`|matched|
|`code/tests/test_unlabeled_quarantine_acceptance_loss.py`|`6a26d4e50c0d8c41d5602dc875cce4df79ec9722d8f611843c40436c4ffb7aba`|matched|
|`code/tests/test_phase1_dgleo_osfix16_launcher.py`|`b39287eac22d13245ae805127449dfe435c6652502dadc6e67bdcc3c5a0e72ca`|matched|

远端验证命令均通过：

```bash
cd /home/szu2070436088/2510044040/CV-SincNet
/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python -m py_compile code/cvsrffi/losses.py code/SSDG/train_ssdg.py
bash -n code/scripts/launch_phase1_dgleo_osfix16_20260707.sh
bash code/scripts/launch_phase1_dgleo_osfix16_20260707.sh --dry-run --only=DGLEO_OSFIX_JOINT_A
```

dry-run确认关键标志存在：`use_concat_sat_channel_aug=1`、`concat_sat_ce_only=0`、`direct_open_set_metric_loss=1`、`unlabeled_quarantine_accept=1`、`stage2_success_claim=0`。

### 启动命令与健康检查

实际启动命令：

```bash
cd /home/szu2070436088/2510044040/CV-SincNet
MAX_ACTIVE_PER_GPU=2 LAUNCH_SETTLE_SECONDS=8 bash code/scripts/launch_phase1_dgleo_osfix16_20260707.sh
```

launcher返回：`[OSFIX16-SUBMIT-COMPLETE]`

启动后健康检查：

|检查项|结果|
|---|---|
|GPU训练进程|16个Python训练进程，严格2个/GPU|
|日志文件|16个`.out`已生成|
|run目录|16个候选目录已生成|
|`metrics_epoch.csv`|16个已生成|
|错误扫描|未发现`Traceback`、`RuntimeError`、`Killed`、`unrecognized`|
|配置marker|`CONFIG-U-DIRECT`、`CONFIG-DM-ACCEPT`、`CONFIG-SAT`均存在|

### 候选落地表

|candidate|GPU|主PID|log|run目录|
|---|---:|---:|---|---|
|`DGLEO_OSFIX_CORE_A`|0|4057265|`logs/phase1_dgleo_osfix16_20260707/DGLEO_OSFIX_CORE_A.out`|`runs/phase1_dgleo_osfix16_20260707/DGLEO_OSFIX_CORE_A`|
|`DGLEO_OSFIX_CORE_B`|0|4057673|`logs/phase1_dgleo_osfix16_20260707/DGLEO_OSFIX_CORE_B.out`|`runs/phase1_dgleo_osfix16_20260707/DGLEO_OSFIX_CORE_B`|
|`DGLEO_OSFIX_DENSITY_A`|1|4058076|`logs/phase1_dgleo_osfix16_20260707/DGLEO_OSFIX_DENSITY_A.out`|`runs/phase1_dgleo_osfix16_20260707/DGLEO_OSFIX_DENSITY_A`|
|`DGLEO_OSFIX_DENSITY_B`|1|4058891|`logs/phase1_dgleo_osfix16_20260707/DGLEO_OSFIX_DENSITY_B.out`|`runs/phase1_dgleo_osfix16_20260707/DGLEO_OSFIX_DENSITY_B`|
|`DGLEO_OSFIX_PROXY_A`|2|4059311|`logs/phase1_dgleo_osfix16_20260707/DGLEO_OSFIX_PROXY_A.out`|`runs/phase1_dgleo_osfix16_20260707/DGLEO_OSFIX_PROXY_A`|
|`DGLEO_OSFIX_PROXY_B`|2|4059718|`logs/phase1_dgleo_osfix16_20260707/DGLEO_OSFIX_PROXY_B.out`|`runs/phase1_dgleo_osfix16_20260707/DGLEO_OSFIX_PROXY_B`|
|`DGLEO_OSFIX_BRIDGE_A`|3|4060156|`logs/phase1_dgleo_osfix16_20260707/DGLEO_OSFIX_BRIDGE_A.out`|`runs/phase1_dgleo_osfix16_20260707/DGLEO_OSFIX_BRIDGE_A`|
|`DGLEO_OSFIX_BRIDGE_B`|3|4060562|`logs/phase1_dgleo_osfix16_20260707/DGLEO_OSFIX_BRIDGE_B.out`|`runs/phase1_dgleo_osfix16_20260707/DGLEO_OSFIX_BRIDGE_B`|
|`DGLEO_OSFIX_TAIL_A`|4|4061000|`logs/phase1_dgleo_osfix16_20260707/DGLEO_OSFIX_TAIL_A.out`|`runs/phase1_dgleo_osfix16_20260707/DGLEO_OSFIX_TAIL_A`|
|`DGLEO_OSFIX_TAIL_B`|4|4061406|`logs/phase1_dgleo_osfix16_20260707/DGLEO_OSFIX_TAIL_B.out`|`runs/phase1_dgleo_osfix16_20260707/DGLEO_OSFIX_TAIL_B`|
|`DGLEO_OSFIX_SATOPEN_A`|5|4062249|`logs/phase1_dgleo_osfix16_20260707/DGLEO_OSFIX_SATOPEN_A.out`|`runs/phase1_dgleo_osfix16_20260707/DGLEO_OSFIX_SATOPEN_A`|
|`DGLEO_OSFIX_SATOPEN_B`|5|4062656|`logs/phase1_dgleo_osfix16_20260707/DGLEO_OSFIX_SATOPEN_B.out`|`runs/phase1_dgleo_osfix16_20260707/DGLEO_OSFIX_SATOPEN_B`|
|`DGLEO_OSFIX_UQ_A`|6|4063093|`logs/phase1_dgleo_osfix16_20260707/DGLEO_OSFIX_UQ_A.out`|`runs/phase1_dgleo_osfix16_20260707/DGLEO_OSFIX_UQ_A`|
|`DGLEO_OSFIX_UQ_B`|6|4063500|`logs/phase1_dgleo_osfix16_20260707/DGLEO_OSFIX_UQ_B.out`|`runs/phase1_dgleo_osfix16_20260707/DGLEO_OSFIX_UQ_B`|
|`DGLEO_OSFIX_JOINT_A`|7|4063937|`logs/phase1_dgleo_osfix16_20260707/DGLEO_OSFIX_JOINT_A.out`|`runs/phase1_dgleo_osfix16_20260707/DGLEO_OSFIX_JOINT_A`|
|`DGLEO_OSFIX_JOINT_B`|7|4064343|`logs/phase1_dgleo_osfix16_20260707/DGLEO_OSFIX_JOINT_B.out`|`runs/phase1_dgleo_osfix16_20260707/DGLEO_OSFIX_JOINT_B`|

注意：`pgrep -af`会显示DataLoader worker继承的长命令行，不能把这些worker都当作主训练PID。上表主PID来自`nvidia-smi pmon`中的GPU计算进程。

### 后续完成判定

本轮只完成启动与早期健康验证。完成后必须按同候选同epoch联合判断：

- 泛化：`overall_tx`、`strict_udu`、receiver floor、satellite mean/floor、best-final gap。
- open-set代理：`proxy_vaccept`、`source_overflow`、`bridge_accept_rate`、`low_density_accept_rate`、`tail/overflow_accept`、`radius_to_inter_ratio`、`zid_p50/p95/p99`、`zid_tail_cvar`。
- 无标签：`u_dm_accept_*`和`u_quarantine_accept_rate/low_density_accept_rate`是否改善，同时不牺牲receiver floor和satellite floor。
- 候选推进：只有同时保护泛化并降低p99/tail/proxy/source overflow风险的候选，才可进入Stage2真实unknown评估；Phase1结果本身不能声明真实unknown_FAR或FPR95改善。

### 交接前只读复查

2026-07-07交接前复查仍为健康启动状态：`nvidia-smi pmon -c 1`显示16个训练Python进程，GPU0-7均为2个训练进程；`logs/phase1_dgleo_osfix16_20260707`下16个`.out`存在；`runs/phase1_dgleo_osfix16_20260707`下16个`metrics_epoch.csv`存在；对16个日志扫描`Traceback|RuntimeError|Killed|unrecognized`未命中。

### 运行耗时估算

2026-07-07 12:10 CST只读复查：launcher总epoch为200；16个候选处于第64-67个epoch；主训练PID已运行约72.8-74.9分钟；平均每epoch约65.8-69.2秒。按当前速度线性外推，预计各候选完成200epoch的时间窗口为2026-07-07 14:35-14:47 CST，最慢候选暂为`DGLEO_OSFIX_TAIL_A`，ETA约14:46:50 CST。考虑final评估、checkpoint/prototype导出和文件flush，建议把完整可分析时间窗口按14:50-15:05 CST观察。

## 完成结果与v2复盘

更新时间：2026-07-07 16:30 CST。N607只读preflight通过，GPU0-7空闲；远端`runs/phase1_dgleo_osfix16_20260707`和`logs/phase1_dgleo_osfix16_20260707`均存在。16个候选均有200行`metrics_epoch.csv`、`latest_ssdg.pth`、`best_joint_safe_ssdg.pth`、`phase2_zid_prototypes.pt/json`。完整stdout扫描未发现`Traceback`、`RuntimeError`、`Killed`、`unrecognized arguments`、`CUDA out of memory`或`FATAL`。日志包含`CONFIG-U-DIRECT`、`CONFIG-DM-ACCEPT`、`CONFIG-SAT`和prototype export，但没有`CONFIG-PHASE1-V2`，因此本组是v2落地前的诊断证据。

### 跨组分布对照

|run|候选数|overall中位/最大|strict UDU中位/最大|sat floor中位/最大|p95中位|p99中位|tail cvar中位|dm proxy中位|旧proxy vaccept中位|source overflow中位|source episode overflow中位|bridge中位|low-density中位|tail accept中位|overflow accept中位|radius/inter中位|
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|directmetric16|16|89.56/90.08|85.29/86.44|75.32/75.96|74.92|84.78|81.35|0.1541|0.6386|0.7438|0.9709|0.0049|0.0348|0.1748|0.1262|1.1160|
|uopt24|24|89.43/89.93|84.89/86.13|75.45/75.79|73.34|83.91|80.06|0.1456|0.6430|0.7692|0.9721|0.0071|0.0303|0.1688|0.1198|1.1461|
|osfix16|16|89.47/90.19|84.92/86.40|75.29/75.90|75.34|84.36|81.01|0.1464|0.6275|0.7666|0.9729|0.0040|0.0293|0.1661|0.1189|1.1157|

### OSFIX16最终同候选表

|candidate|overall|strict UDU|rx floor|UDU rx floor|sat floor|sat strict floor|p95|p99|tail cvar|dm proxy|旧proxy|src overflow|src ep overflow|bridge|low-density|tail acc|overflow acc|ratio|U_s q loss|best p99 epoch|p99后期扩张|
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|DGLEO_OSFIX_PROXY_A|90.19|86.40|79.70|76.09|75.90|69.83|75.44|83.67|80.91|0.1430|0.6409|0.7655|0.9739|0.0024|0.0298|0.1594|0.1156|1.0973|0.0172|41|+2.25|
|DGLEO_OSFIX_DENSITY_B|90.17|86.27|79.65|75.57|75.62|69.94|74.64|84.42|80.72|0.1420|0.6209|0.7690|0.9702|0.0040|0.0295|0.1707|0.1188|1.1185|0.0208|30|+4.17|
|DGLEO_OSFIX_CORE_B|89.78|86.00|78.47|77.24|75.49|69.50|74.25|83.97|80.13|0.1428|0.6291|0.7700|0.9727|0.0046|0.0289|0.1648|0.1192|1.1266|0.0177|30|+3.85|
|DGLEO_OSFIX_SATOPEN_A|90.02|85.89|79.28|73.08|75.48|69.57|75.27|84.38|81.02|0.1443|0.6213|0.7705|0.9714|0.0033|0.0298|0.1655|0.1475|1.1173|0.0178|30|+2.96|
|DGLEO_OSFIX_TAIL_B|89.77|85.84|79.30|74.02|75.63|69.68|76.47|84.30|81.68|0.1491|0.6320|0.7687|0.9736|0.0034|0.0290|0.1691|0.1134|1.1053|0.0228|55|+3.92|
|DGLEO_OSFIX_PROXY_B|89.73|85.68|78.32|76.28|75.69|69.65|74.63|83.20|80.25|0.1505|0.6252|0.7665|0.9716|0.0075|0.0298|0.1731|0.1186|1.1867|0.0216|45|+2.26|
|DGLEO_OSFIX_UQ_B|89.97|85.55|81.72|72.57|75.32|69.37|76.95|85.18|82.27|0.1480|0.6181|0.7668|0.9724|0.0041|0.0291|0.1671|0.1310|1.1121|0.0348|30|+4.13|
|DGLEO_OSFIX_TAIL_A|89.47|84.99|80.17|75.19|75.09|68.67|73.24|84.33|80.16|0.1484|0.6163|0.7692|0.9731|0.0067|0.0294|0.1686|0.1278|1.1318|0.0208|30|+3.84|
|DGLEO_OSFIX_JOINT_B|89.37|84.86|79.94|72.96|75.00|68.71|76.66|84.89|81.84|0.1428|0.6259|0.7679|0.9717|0.0033|0.0296|0.1622|0.1189|1.1110|0.0327|62|+3.57|
|DGLEO_OSFIX_CORE_A|89.48|84.83|81.43|71.41|75.14|68.76|75.19|84.24|81.00|0.1468|0.6304|0.7628|0.9735|0.0037|0.0282|0.1613|0.1167|1.1009|0.0154|30|+3.84|
|DGLEO_OSFIX_SATOPEN_B|89.08|84.66|80.06|75.04|75.19|68.61|76.10|84.68|81.51|0.1460|0.6294|0.7636|0.9728|0.0044|0.0294|0.1581|0.1461|1.1140|0.0194|30|+4.84|
|DGLEO_OSFIX_JOINT_A|89.32|84.56|80.46|72.70|75.27|68.92|76.20|84.75|81.83|0.1490|0.6357|0.7662|0.9746|0.0052|0.0287|0.1680|0.1165|1.1226|0.0255|30|+4.18|
|DGLEO_OSFIX_BRIDGE_B|89.22|84.10|81.23|71.64|75.27|68.78|74.80|84.27|80.79|0.1457|0.6241|0.7652|0.9738|0.0030|0.0295|0.1593|0.1266|1.0940|0.0226|30|+5.41|
|DGLEO_OSFIX_DENSITY_A|88.83|84.09|78.41|68.96|75.34|68.97|75.40|84.91|81.34|0.1449|0.6331|0.7627|0.9728|0.0023|0.0281|0.1564|0.1174|1.1025|0.0196|30|+4.67|
|DGLEO_OSFIX_BRIDGE_A|88.76|83.66|79.02|69.81|74.91|68.50|74.35|83.79|80.19|0.1524|0.6143|0.7688|0.9733|0.0088|0.0287|0.1667|0.1469|1.1558|0.0212|30|+5.19|
|DGLEO_OSFIX_UQ_A|88.82|83.56|77.52|68.45|74.82|68.32|75.41|84.39|81.32|0.1499|0.6315|0.7661|0.9748|0.0061|0.0284|0.1675|0.1085|1.1213|0.0299|65|+2.67|

### 结论与v2启示

1. `DGLEO_OSFIX_PROXY_A`是本组最均衡的Phase1候选：overall 90.19、strict UDU 86.40、sat floor 75.90，同时p99 83.67、tail cvar 80.91、dm proxy 0.1430、overflow accept 0.1156都处在本组较优区间。它可作为v2的warm-start/对照候选，但仍不能直接推进为Stage2成功证据。
2. OSFIX16验证了“新dm代理可被压低，但最终拒识风险代理没有同步下降”。dm proxy中位数从directmetric16的0.1541降到0.1464，low-density/tail/overflow accept也下降；但旧`train_proxy_unknown_proxy_vaccept`仍约0.6275，`train_proxy_unknown_bridge_accept_rate`仍为1.0，说明动态direct metric软门控和旧proxy接收边界仍不等价。v2必须保留`endpoint_accept_v1`和三入口parity检查。
3. `source_episode_overflow`没有改善，甚至略高于directmetric16：directmetric16中位0.9709，uopt24为0.9721，osfix16为0.9729。这是known域过散/source-only几何矛盾未解决的核心证据。quarantine没有把“跨receiver/day稳定核心”和“尾部/域变体”拆开。
4. 无标签quarantine开始有非零梯度，但力度仍弱。OSFIX16最终`train_w_loss_u_quarantine_accept`中位约0.021，`train_w_loss_u_direct_metric_accept`中位仍为0，`u_dm_selected`约102只在后期出现。它支持v2中`u_tri_state_required`和`US_DIRECT_LOSS_IDLE`门控：不能再让U_s分支只在日志上存在、实际不改变结构。
5. 损失预算仍严重偏向闭集/KD/星地。OSFIX16最终open-set有效项约1.07，而闭集/KD/sat/domain有效项约20.89，比例约0.05。v2当前`os_eff_min_budget=0.15`是合理的下限，甚至可对P0候选提高到0.20-0.25，否则open-set目标仍会被teacher/sat/KD压住。
6. 后期tail扩张仍明显。以最均衡的`PROXY_A`为例，best p99在epoch 41，final p99又上升2.25；`DENSITY_B`、`CORE_B`、`BRIDGE_A/B`的p99后期扩张达3.85-5.41。v2的tail safety状态机必须以p95/p99/tail cvar/proxy_vaccept共同决定best/final可导出，而不是训练结束后再挑checkpoint。
7. 星地性能被保护住但没有突破。OSFIX16 sat floor最大75.90，接近directmetric16最大75.96；strict UDU最大86.40，接近directmetric16最大86.44。说明OSFIX16没有牺牲EPOC式星地/闭集能力，但也没有解决弱receiver或satellite floor的上限。
8. stronger quarantine不是单调更优。`UQ_B`旧proxy最低之一、rx floor最高81.72，但p95/p99/tail cvar最差区间，strict UDU只有85.55；`UQ_A`strict和sat floor更差。这说明只加大U_s quarantine会把部分样本推到更宽tail，v2需要core/tail/outside三态和receiver-aware local component，而不是单一“untrusted U_s远离known”。

### 对v2方案的补齐建议

- `endpoint_accept_v1`必须成为最终artifact字段，不允许把`direct_metric_acceptance_loss`内部门控当最终边界；OSFIX16已经证明dm proxy下降不等价于旧proxy下降。
- tail safety默认阈值不宜只看final绝对值，还要看“best p99 epoch到final”的扩张量。建议新增`tail_expansion_delta`门：`p99_delta>2.0`或`tail_cvar_delta>4.0`即禁止final prototype export，`p99_delta>3.5`禁止best promotion。
- `os_eff_min_budget=0.15`保留为最低门槛，P0几何修复候选建议扫`0.20/0.25`。OSFIX16的约0.05说明如果不硬门控，训练会自然回到闭集/KD/sat主方向。
- U_s三态不能只记录selected数量。必须记录`trusted_core`、`ambiguous_tail`、`outside_reject`三类计数、损失和梯度贡献；没有三态计数或`u_direct`长期为0时，候选只能标为诊断负例。
- `source_episode_overflow`需要结构性修复：receiver-aware local component、core/tail/outside quarantine、same-class bridge负样本和source episode density gate，而不是继续提高单一direct metric权重。
- v2候选优先从`DGLEO_OSFIX_PROXY_A`和`DGLEO_OSFIX_PROXY_B`派生：前者保泛化/星地最稳，后者p99最低；不要从`UQ_A/B`直接加码，因为它们证明强U_s quarantine有扩大tail的风险。

最终判断：OSFIX16对Phase1的贡献是证明“quarantine+direct metric可以改善部分dm代理，并基本保护闭集/星地性能”；不能声明真实unknown拒识改善；最主要风险仍是旧proxy_vaccept约0.62、old bridge=1.0、source episode overflow约0.97、后期p99/tail扩张和U_s direct空转。它强化了v2 hard gates的必要性，而不是替代v2。
